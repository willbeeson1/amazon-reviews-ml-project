#!/usr/bin/env python3
# stacked_classifier_search_bayes_binary_4.py

"""
A fully functional Python module demonstrating a BROAD Bayesian hyperparameter search
(using scikit-optimize's BayesSearchCV) for a stacked ensemble classifier that maximizes F1.
You can run this overnight on multiple CPUs to find an optimal model.

"""

import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Make sure your custom code is on the path:
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Import your custom modules
from feature_engineering_pipeline import (
    preprocess_amazon_reviews,
    SentimentFeatureExtractor,
    SentimentWeightScaler
)

###########################
# 1) Load & Preprocess Data
###########################

df_train = pd.read_csv("./data/Training.csv")

df_train = preprocess_amazon_reviews(df_train)

# Example numeric fix-ups
df_train["verified"] = df_train["verified"].astype(int)
df_train["vote"] = df_train["vote"].fillna(0)

# Drop unwanted columns
df_train.drop(
    columns=["review_date", "reviewTime", "unixReviewTime", "image", 
             "style", "reviewerName", "asin", "summary_length"],
    errors="ignore",
    inplace=True
)

# Create label for binary classification: 1 if rating > 3, else 0
df_train["binary_label_cutoff_4"] = (df_train["overall"] > 4).astype(int)

# Combine text columns
df_train["combined_text"] = df_train["reviewText"].fillna('') + " " + df_train["summary"].fillna('')

# Sentiment features
sentiment_extractor = SentimentFeatureExtractor()
sentiment_array = sentiment_extractor.transform(df_train["combined_text"])
df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_array

# A possible frequency feature for reviewerID
if "reviewerID" in df_train.columns:
    freq_map = df_train["reviewerID"].value_counts()
    df_train["reviewer_freq"] = df_train["reviewerID"].map(freq_map).fillna(1).astype(int)

# Drop text columns used for combined_text + 'overall' rating
df_train.drop(columns=["reviewText", "summary", "overall", "reviewerID"], errors="ignore", inplace=True)

X = df_train.drop(columns=["binary_label_cutoff_4"])
y = df_train["binary_label_cutoff_4"].values

print("X columns after preprocessing:", X.columns)
print("y shape:", y.shape)

########################################
# 2) Define Pipelines & ColumnSelectors
########################################

# Custom TF-IDF Vectorizer that allows for n-grams up to a maximum
class MyTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        max_ngram=3,
        max_df=1.0,
        min_df=1,
        max_features=None,
        use_idf=True,
        sublinear_tf=False,
        norm='l2',
        stop_words=None,
        token_pattern=r"\w{1,}",
        # any other Tfidf params you want
        **kwargs
    ):
        super().__init__(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            norm=norm,
            stop_words=stop_words,
            token_pattern=token_pattern,
            **kwargs
        )
        self.max_ngram = max_ngram

    def fix_ngram(self):
        self.ngram_range = (1, self.max_ngram)

    def fit(self, X, y=None):
        self.fix_ngram()
        return super().fit(X, y)

    def transform(self, X):
        self.fix_ngram()
        return super().transform(X)



def passthrough_df(X_df):
    """Ensure the DataFrame structure is maintained in the pipeline."""
    return X_df

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)

numeric_pipeline = Pipeline([
    ("impute_num", SimpleImputer(strategy="constant", fill_value=0)),
    ("ensure_df", FunctionTransformer(passthrough_df, validate=False)),
    ("weight", SentimentWeightScaler()),   # Custom weighting for sentiment features
    ("scale", StandardScaler())
])

cat_pipeline = Pipeline([
    ("impute_cat", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("text", MyTfidfVectorizer(), "combined_text"),
    ("num", numeric_pipeline, numeric_selector),
    ("cat", cat_pipeline, categorical_selector)
])

####################################
# 3) Define the Stacking Classifier
####################################

base_lr = LogisticRegression(
    solver='saga',
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)

base_rf = RandomForestClassifier(
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

base_gb = GradientBoostingClassifier(
    random_state=42
)

meta_lr = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)

ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=meta_lr,
    passthrough=True,
    cv=3,
    n_jobs=-1
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", ensemble)
])

###################################
# 4) Define Bayesian Search Space
###################################
# We use skopt's BayesSearchCV for a more thorough "Bayesian" hyperparameter optimization.

search_spaces = {
    # TF-IDF hyperparams
    "prep__text__max_ngram": Integer(1, 3), # custom one
    "prep__text__max_features": Integer(3000, 12000),
    "prep__text__min_df": Integer(2, 10),
    "prep__text__max_df": Real(0.5, 0.95),
    "prep__text__use_idf": Categorical([True, False]),
    "prep__text__sublinear_tf": Categorical([True, False]),
    "prep__text__norm": Categorical([None, 'l2']),
    "prep__text__stop_words": Categorical([None, 'english']),
    "prep__text__token_pattern": Categorical([r"\w{1,}", r"\w{2,}"]),

    # Sentiment weighting
    "prep__num__weight__sentiment_weight": Real(0.5, 3.0),  # broad range

    # RandomForest
    "clf__rf__n_estimators": Integer(50, 300),
    "clf__rf__max_depth": Integer(5, 30),

    # GradientBoosting
    "clf__gb__n_estimators": Integer(50, 300),
    "clf__gb__learning_rate": Real(1e-3, 1.0, prior='log-uniform'),
    "clf__gb__max_depth": Integer(2, 10),

    # Base Logistic
    "clf__lr__C": Real(1e-5, 10.0, prior='log-uniform'),

    # Meta Logistic
    "clf__final_estimator__C": Real(1e-5, 10.0, prior='log-uniform'),
}

##################################
# 5) Set up BayesSearch & Fit
##################################

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bayes_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_spaces,
    n_iter=200,               # Increase this to 300+ if you want an even longer search
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,                # Use all available cores
    n_points=8,               # Number of parameter sets evaluated in parallel each iteration
    random_state=42,
    verbose=3
)

start_time = time.time()
print("\n=== Binary Classifier Cutoff [4] BAYESIAN TUNING Start ===\n")
bayes_search.fit(X, y)
end_time = time.time()

print(f"\n=== Tuning Completed! Total Time Taken: {end_time - start_time:.2f} seconds "
      f"(~{(end_time - start_time)/60:.2f} min) ===\n")

print("\033[92m\n=== Binary Classifier Cutoff [4] Tuning Results ===\n\033[0m")
print("\033[92mBEST PARAMS:\033[0m")
for key, value in bayes_search.best_params_.items():
    print(f"  {key}: {value}")

print(f"\033[92m\nBEST CV Macro F1: {bayes_search.best_score_:.4f}\n\033[0m")

best_model = bayes_search.best_estimator_

###############################################
# 6) Evaluate via Cross-Val on the Whole Train
###############################################

print("\033[93m\nBeginning Cross Validation Prediction.....\n\033[0m")
y_pred_cv = cross_val_predict(best_model, X, y, cv=5, n_jobs=-1)
cm = confusion_matrix(y, y_pred_cv)
print("Confusion matrix:\n", cm)

y_proba_cv = cross_val_predict(best_model, X, y, cv=5, n_jobs=-1, method="predict_proba")[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)
print("Accuracy:", accuracy_score(y, y_pred_cv))
print("Macro F1:", f1_score(y, y_pred_cv, average="macro"))

################################################
# 7) Load + Preprocess Test Data, Make Predicts
################################################
print("[INFO] Loading Test.csv for final predictions...")
df_test = pd.read_csv("./data/Test.csv")
df_test = preprocess_amazon_reviews(df_test)

# Basic fix-ups
df_test["verified"] = df_test["verified"].fillna(0).astype(int)
df_test["vote"] = df_test["vote"].fillna(0)

if "reviewerID" in df_test.columns:
    # Map frequency for test
    freq_map_test = df_test["reviewerID"].value_counts()
    df_test["reviewer_freq"] = df_test["reviewerID"].map(freq_map_test).fillna(1).astype(int)

# Combine text
df_test["combined_text"] = df_test["reviewText"].fillna('') + " " + df_test["summary"].fillna('')
df_test.drop(columns=["reviewText", "summary"], errors="ignore", inplace=True)

# Sentiment features
sentiment_test = sentiment_extractor.transform(df_test["combined_text"])
df_test[["pos_count", "neg_count", "polarity_score"]] = sentiment_test

# Predict
print("[INFO] Generating predictions on test data with best model from Bayesian search...")
test_preds = best_model.predict(df_test)

df_test["binary_split_3"] = test_preds
df_test["id"] = df_test.index

os.makedirs("results", exist_ok=True)
out_path = "results/submission_binary_split_3_bayessearch.csv"
df_test[["id", "binary_split_3"]].to_csv(out_path, index=False)

print(f"\n[INFO] All done. Your submission is saved to: {out_path}")
