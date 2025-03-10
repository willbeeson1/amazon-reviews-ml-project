# stacked_classifier_search_binary_4.py

import pandas as pd
import numpy as np
import time

import sys, os

# add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# import custom preprocessing code (written by me)
from feature_engineering_pipeline import (
    preprocess_amazon_reviews,
    SentimentFeatureExtractor, 
    SentimentWeightScaler
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint

# load data
df_train = pd.read_csv("./data/Training.csv")

# 2) Basic cleaning + text stats from your pipeline module
df_train = preprocess_amazon_reviews(df_train)

# cleanup by converting + dropping unwanted columns
df_train["verified"] = df_train["verified"].astype(int)
df_train["vote"] = df_train["vote"].fillna(0)

df_train.drop(columns=["review_date", "reviewTime", 
                       "unixReviewTime", "image", 
                       "style", "reviewerName", "asin"], 
              errors="ignore", 
              inplace=True)

df_train["binary_label_4"] = (df_train["overall"] > 4).astype(int)

# combine `reviewText` and `summary` into one column for TF-IDF processing
df_train["combined_text"] = df_train["reviewText"].fillna('') + " " + df_train["summary"].fillna('')

# add sentiment features
sentiment_extractor = SentimentFeatureExtractor()
sentiment_array = sentiment_extractor.transform(df_train["combined_text"])
df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_array

# drop unwanted columns
df_train.drop(columns=["overall"], inplace=True)
df_train.drop(columns=["reviewerID"], inplace=True)
df_train.drop(columns=["summary_length"], inplace=True) 
# df_train.drop(columns=["review_hour"], inplace=True)
df_train.drop(columns=["reviewText", "summary"], inplace=True) # replaced above with combined_text


# 5) Separate features & labels
label_col = "binary_label_4"
X = df_train.drop(columns=[label_col])    # everything else as input
y = df_train[label_col].values

print("X columns after preprocessing:", X.columns)
print("y shape:", y.shape)

# 6) Define numeric + categorical columns. Adjust as needed:
numeric_cols = [
    "vote", "review_length", "num_words", "avg_word_length",
    "uppercase_ratio", "exclamation_count", "reviewer_freq",
    "review_year", "review_month", "review_day", "review_dayofweek",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]  # if you have a 'category' column

# 7) ColumnTransformer: apply TF-IDF to "reviewText" + "summary", scale numeric, one-hot cat
#    note: remainder="drop" means any columns not specified are discarded

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)

# after you finish creating review_year, review_month, review_day, etc.
df_train.drop(columns=["review_date", "reviewTime", "unixReviewTime"],
              errors="ignore", inplace=True)

def passthrough_df(X):
    """Ensures the DataFrame structure is maintained in the pipeline."""
    return X

numeric_pipeline = Pipeline([
    ("impute_num", SimpleImputer(strategy="constant", fill_value=0)),  # Handle missing values
    ("ensure_df", FunctionTransformer(passthrough_df, validate=False)),  # Keeps DataFrame format
    ("weight", SentimentWeightScaler()),  # Apply sentiment scaling
    ("scale", StandardScaler())  # Standardize numeric features
])

preprocessor = ColumnTransformer([
     # 1) Tfidf on reviewText + dimensionality reduction
    ("text", Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svd", TruncatedSVD(n_components=300, random_state=42))  # Reduce dimensions to 300
    ]), "combined_text"),
    
    # 2) Impute numeric & then scale
    ("num", numeric_pipeline, numeric_selector),

    # 3) Impute cat & then OneHot
    ("cat", Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_selector)
])

# 3. Define advanced classifier, e.g. Stacking
base_lr = LogisticRegression(solver='saga', class_weight='balanced', max_iter=10000)
base_rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=6)
base_gb = GradientBoostingClassifier(validation_fraction=0.08, n_iter_no_change=20, tol=1e-4, random_state=42)

meta_lr = LogisticRegression(solver='saga', class_weight='balanced', random_state=42, max_iter=10000) # changing to saga from lbfgs

ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=meta_lr,
    passthrough=True,
    cv=5,
    n_jobs=6
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", ensemble)
])

# 9) Define param grid/distributions for RandomizedSearchCV
#    Example: tune the Tfidf ngram_range and max_features. 
#    Also possibly tune SVD n_components or the classifier’s C
param_dist = {
    "prep__text__tfidf__ngram_range": [(1,2), (1,3)],  # Fix reference
    "prep__text__tfidf__max_features": [7000, 12000],  # Fix reference
    "prep__text__tfidf__min_df": [3, 5, 8],  # Fix reference
    "prep__text__tfidf__max_df": [0.75, 0.85, 0.90],  # Fix reference
    "prep__text__tfidf__stop_words": [None],  # Fix reference
    "prep__text__tfidf__norm": [None],
    "prep__text__tfidf__use_idf": [False, True],
    "prep__text__tfidf__sublinear_tf": [True],
    "prep__text__tfidf__token_pattern": [r"\w{2,}"],

    # SVD hyperparameters
    "prep__text__svd__n_components": [200, 300, 500],

    # Custom features
    "prep__num__weight__sentiment_weight": [1.5],  

    # RandomForest
    "clf__rf__n_estimators": randint(80, 180), # changed from 50, 150
    "clf__rf__max_depth": [None, 20, 30], # switched None to 20
    "clf__rf__min_samples_split": [2, 5, 10],
    "clf__rf__min_samples_leaf": [1, 2, 5],
    "clf__rf__max_features": ["sqrt", "log2", 0.5],

    # GradientBoosting
    "clf__gb__n_estimators": randint(60, 100), # changed from 120
    "clf__gb__learning_rate": loguniform(1e-2, 1),
    "clf__gb__max_depth": [3, 5, 6],
    "clf__gb__max_features": [0.2, "sqrt", "log2"],
    "clf__gb__subsample": [0.7, 1.0],

    # meta logistic
    "clf__final_estimator__C": loguniform(1e-3, 1.0),
    "clf__final_estimator__penalty": ["l2"],
    # "clf__final_estimator__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

    # base logistic
    "clf__lr__C": loguniform(1e-4, 1.0), 
    "clf__lr__max_iter": [10000], # added 8000
    "clf__lr__penalty": ["l2"],
    # "clf__lr__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For l2 combos
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=40,  # or however many
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=3
)

# Start timing
start_time = time.time()

print("\n=== Binary Classifier Cutoff [4] Tuning Start ===\n")
search.fit(X, y)

end_time = time.time()
print(f"\n=== Tuning Completed! Total Time Taken: {end_time - start_time:.2f} seconds (~{(end_time - start_time) / 60:.2f} min) ===\n")

print("\033[92m\n=== Binary Classifier Cutoff [4] Tuning Results ===\n\033[0m")

print("\033[92mBEST PARAMS:\033[0m")
for key, value in search.best_params_.items():
    print(f"  {key}: {value}")
print(f"\033[92m\nBEST CV Macro F1: {search.best_score_:.4f} \n\033[0m")

best_model = search.best_estimator_

print("\033[93m\nBeginning Cross Validation Prediction.....\n\033[0m")
y_pred_cv = cross_val_predict(best_model, X, y, cv=5, n_jobs=-1)  # or X_val if you have a separate holdout
cm = confusion_matrix(y, y_pred_cv)
print("Confusion matrix:", cm)

# For binary classification, you can get predicted probabilities for the positive class:
y_proba_cv = cross_val_predict(best_model, X, y, cv=5, n_jobs=-1, verbose=3, method="predict_proba")[:,1]

# ROC/AUC
fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
print("AUC:", auc(fpr, tpr))

# Accuracy, Macro F1
print("Accuracy:", accuracy_score(y, y_pred_cv))
print("Macro F1:", f1_score(y, y_pred_cv, average="macro"))

#####################################################################
# LOAD + PREPROCESS TEST, MAKE SUBMISSION
#####################################################################
print("[INFO] Loading Test.csv for final predictions...")
df_test = pd.read_csv("./data/Test.csv")

# The same pipeline steps:
df_test = preprocess_amazon_reviews(df_test)
sentiment_test = sentiment_extractor.transform(df_test["reviewText"])
df_test[["pos_count", "neg_count", "polarity_score"]] = sentiment_test
df_test["verified"] = df_test["verified"].fillna(0).astype(int)
df_test["vote"] = df_test["vote"].fillna(0)

# Drop extra columns
df_test.drop(columns=["review_date","reviewTime","unixReviewTime","image","style","reviewerName","asin"],
             errors="ignore", inplace=True)

if "reviewerID" in df_test.columns:
    # Default reviewer_freq to 1 for test data
    freq_map_test = df_test["reviewerID"].value_counts()
    df_test["reviewer_freq"] = df_test["reviewerID"].map(freq_map_test).fillna(1).astype(int)

# Apply TF-IDF Combination for test data
df_test["combined_text"] = df_test["reviewText"].fillna('') + " " + df_test["summary"].fillna('')
df_test.drop(columns=["reviewText", "summary"], inplace=True)  # Drop original text cols

# Make Predictions Using Best Model Found in Tuning
print("[INFO] Generating predictions on test data...")
test_preds = best_model.predict(df_test)

# Create Submission File
df_test["binary_split_4"] = test_preds
df_test["id"] = df_test.index

os.makedirs("results", exist_ok=True)
out_path = "results/submission_binary_split_4_from_tuner-v2.csv"
df_test[["id","binary_split_4"]].to_csv(out_path, index=False)

print(f"\n[INFO] Done! Kaggle submission => {out_path}")
