# 01_preproc_tuning_v3.py

import pandas as pd
import numpy as np

import sys, os

# add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# import custom preprocessing code
from feature_engineering_pipeline import (
    preprocess_amazon_reviews,
    SentimentFeatureExtractor
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score

# load data
df_train = pd.read_csv("./data/Training.csv")

# 2) Basic cleaning + text stats from your pipeline module
df_train = preprocess_amazon_reviews(df_train)

# 3) Add sentiment features
sentiment_extractor = SentimentFeatureExtractor()
sentiment_array = sentiment_extractor.transform(df_train["reviewText"])
df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_array

# cleanup by converting + dropping unwanted columns
df_train["verified"] = df_train["verified"].astype(int)
df_train["vote"] = df_train["vote"].fillna(0)

df_train.drop(columns=["review_date", "reviewTime", 
                       "unixReviewTime", "image", 
                       "style", "reviewerName", "asin"], 
              errors="ignore", 
              inplace=True)

# 4) Optional: add reviewer frequency, create binary_label_cutoff_1
# (Adjust as you like or if you already have it in the dataset)
if "reviewerID" in df_train.columns:
    reviewer_freq = df_train["reviewerID"].value_counts()
    df_train["reviewer_freq"] = df_train["reviewerID"].map(reviewer_freq).fillna(1).astype(int)

df_train["binary_label_cutoff_1"] = (df_train["overall"] > 1).astype(int)
df_train.drop(columns=["overall"], inplace=True)

# 5) Separate features & labels
label_col = "binary_label_cutoff_1"
X = df_train.drop(columns=[label_col])    # everything else as input
y = df_train[label_col].values

# 6) Define numeric + categorical columns. Adjust as needed:
numeric_cols = [
    "vote", "review_length", "num_words", "avg_word_length",
    "uppercase_ratio", "exclamation_count", "reviewer_freq",
    "review_year", "review_month", "review_day", "review_dayofweek", "review_hour",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]  # if you have a 'category' column

# 7) ColumnTransformer: apply TF-IDF to "reviewText", scale numeric, one-hot cat
#    note: remainder="drop" means any columns not specified are discarded

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)

# after you finish creating review_year, review_month, review_day, etc.
df_train.drop(columns=["review_date", "reviewTime", "unixReviewTime"],
              errors="ignore", inplace=True)


preprocessor = ColumnTransformer([
    # 1) Tfidf on reviewText
    ("text", TfidfVectorizer(), "reviewText"),
    
    # 2) Impute numeric & then scale
    ("num", Pipeline([
        ("impute_num", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler())
    ]), numeric_selector),

    # 3) Impute cat & then OneHot
    ("cat", Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_selector)
])

# 8) Build entire pipeline. SVD is optional; here we show an example:
pipeline = Pipeline([
    ("prep", preprocessor),
    ("svd", TruncatedSVD()),
    ("clf", LogisticRegression())
])

# 9) Define param grid/distributions for RandomizedSearchCV
#    Example: tune the Tfidf ngram_range and max_features. 
#    Also possibly tune SVD n_components or the classifier’s C
tfidf_svd_dist = {
    "prep__text__ngram_range": [(1,3)], # from (1,1), (1,2), (1,30
    "prep__text__max_features": [5000], # from 5000, 8000, 12000
    "prep__text__min_df": [4, 5],
    "prep__text__max_df": [0.85],
    # "prep__text__binary": [False, True],
    # "prep__text__stop_words": ["english", None],
    # "prep__text__norm": [None, "l1", "l2"],
    # "prep__text__use_idf": [True, False],
    # "prep__text__sublinear_tf": [True, False],
    # "prep__text__token_pattern": ["\w{1,}", "\w{2,}", "(?u)\b\w\w+\b", None],
    
    "svd__n_components": [200],  # example if you want to tune SVD dims      
}

l1_dist = {
    "clf__C": loguniform(1e-3, 4),
    "clf__penalty": ["l1"],
    "clf__class_weight": ["balanced", None],
    "clf__max_iter": [1000, 2000, 5000],
    "clf__solver": ["saga", "liblinear"],
    "clf__n_jobs": [-1] 
}

l2_dist = {
    "clf__C": loguniform(1e-3, 4),
    "clf__penalty": ["l2"],
    "clf__class_weight": ["balanced", None],
    "clf__max_iter": [1000, 2000, 5000],
    "clf__solver": ["saga", "liblinear", "lbfgs", "sag"],
    "clf__n_jobs": [-1] 
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For l2 combos
search_l2 = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={**tfidf_svd_dist, **l2_dist},
    n_iter=200,  # or however many
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

search_l2.fit(X, y)
print("Best L2 params:", search_l2.best_params_, "CV F1:", search_l2.best_score_)

search_l1 = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={**tfidf_svd_dist, **l2_dist},
    n_iter=200,  # or however many
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

search_l1.fit(X, y)
print("Best L1 params:", search_l1.best_params_, "CV F1:", search_l1.best_score_)

best_pipeline = search_l1 if search_l1.best_score_ > search_l2.best_score_ else search_l2

y_pred_cv = cross_val_predict(best_pipeline, X, y, cv=5)  # or X_val if you have a separate holdout
cm = confusion_matrix(y, y_pred_cv)
print("Confusion matrix:", cm)

# For binary classification, you can get predicted probabilities for the positive class:
y_proba_cv = cross_val_predict(best_pipeline, X, y, cv=5, method="predict_proba")[:,1]

# ROC/AUC
fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
print("AUC:", auc(fpr, tpr))

# Accuracy, Macro F1
print("Accuracy:", accuracy_score(y, y_pred_cv))
print("Macro F1:", f1_score(y, y_pred_cv, average="macro"))
