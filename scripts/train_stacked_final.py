#!/usr/bin/env python3
"""
train_stacked_final.py

Trains a stacking classifier with the known "best" hyperparams, then predicts
on Test.csv to produce a submission. Also saves the trained model.

Usage:
  python train_stacked_final.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# (1) Optionally add scripts to path if you want to reuse your pipeline code
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# (2) Import your custom preprocessing
from feature_engineering_pipeline import (
    preprocess_amazon_reviews,
    SentimentFeatureExtractor
)

# (3) Import scikit-learn stuff
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#####################################################################
# 1. LOAD & PREPROCESS TRAIN
#####################################################################
print("[INFO] Loading raw training data...")
df_train = pd.read_csv("./data/Training.csv")

print("[INFO] Applying custom preprocessing (clean_text, time features, etc.)")
df_train = preprocess_amazon_reviews(df_train)

# Add sentiment features
sentiment_extractor = SentimentFeatureExtractor()
sentiment_array = sentiment_extractor.transform(df_train["reviewText"])
df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_array

# Convert verified to int, fill NA in vote
df_train["verified"] = df_train["verified"].astype(int)
df_train["vote"] = df_train["vote"].fillna(0)

# Drop columns not used
df_train.drop(columns=["review_date","reviewTime","unixReviewTime","image","style","reviewerName","asin"],
              errors="ignore", inplace=True)

# If you want reviewer_freq
if "reviewerID" in df_train.columns:
    freq_map = df_train["reviewerID"].value_counts()
    df_train["reviewer_freq"] = df_train["reviewerID"].map(freq_map).fillna(1).astype(int)

# Create binary label cutoff=1
df_train["binary_label_cutoff_1"] = (df_train["overall"] > 1).astype(int)
df_train.drop(columns=["overall"], inplace=True)

X = df_train.drop(columns=["binary_label_cutoff_1"])
y = df_train["binary_label_cutoff_1"].values

print("Final columns of X:", X.columns)
print("Shape of X:", X.shape)


# #####################################################################
# # 2. BUILD STACKING PIPELINE & SET BEST PARAMS
# #####################################################################
print("[INFO] Building pipeline with best hyperparams from previous search...")

# a) ColumnTransformer
numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)

preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(), "reviewText"),
    ("num", Pipeline([
        ("impute_num", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler())
    ]), numeric_selector),
    ("cat", Pipeline([
        ("impute_cat", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_selector)
])

# b) Define base estimators
base_lr = LogisticRegression(solver='saga', class_weight='balanced', max_iter=2000)
base_rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1)
base_gb = GradientBoostingClassifier(random_state=42)

meta_lr = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)

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

# c) The best params from your search
best_params = {
    "clf__final_estimator__C": 2.1368329072358727,
    "clf__gb__learning_rate": 0.020589728197687916,
    "clf__gb__max_depth": 5,
    "clf__gb__n_estimators": 89,
    "clf__lr__C": 0.005415244119402535,
    "clf__rf__max_depth": None,
    "clf__rf__n_estimators": 107,
    "prep__text__max_df": 0.85,
    "prep__text__max_features": 5000,
    "prep__text__min_df": 5,
    "prep__text__ngram_range": (1,3)
}

print("Setting best_params:", best_params)
pipeline.set_params(**best_params)

#####################################################################
# 3. FIT ON FULL TRAIN
#####################################################################
print("[INFO] Fitting final pipeline on full training data...")
pipeline.fit(X, y)
print("[INFO] Training complete.")

#####################################################################
# 4. SAVE MODEL FOR FUTURE REUSE
#####################################################################


#####################################################################
# 5. LOAD + PREPROCESS TEST, MAKE SUBMISSION
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
    # up to you how you treat reviewer_freq for test – often we default to 1
    freq_map_test = df_test["reviewerID"].value_counts()
    df_test["reviewer_freq"] = df_test["reviewerID"].map(freq_map_test).fillna(1).astype(int)

print("[INFO] Generating predictions on test data...")
test_preds = pipeline.predict(df_test)

df_test["binary_split_1"] = test_preds
df_test["id"] = df_test.index

os.makedirs("results", exist_ok=True)
out_path = "results/submission_stacked_final.csv"
df_test[["id","binary_split_1"]].to_csv(out_path, index=False)
print(f"\n[INFO] Done! Kaggle submission => {out_path}")
