#!/usr/bin/env python3
"""
stacked_classifier_finetune.py

Refines the StackingClassifier search by focusing on narrower param ranges
around the best combos discovered previously.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add your scripts folder if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from feature_engineering_pipeline import (
    preprocess_amazon_reviews,
    SentimentFeatureExtractor
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint, uniform

def main():
    print("[INFO] Loading training data")
    df_train = pd.read_csv("./data/Training.csv")

    print("[INFO] Preprocessing training data (clean, sentiment, etc.)")
    df_train = preprocess_amazon_reviews(df_train)

    # Sentiment
    sentiment_extractor = SentimentFeatureExtractor()
    sentiment_array = sentiment_extractor.transform(df_train["reviewText"])
    df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_array

    # Convert verified, fill vote
    df_train["verified"] = df_train["verified"].astype(int)
    df_train["vote"] = df_train["vote"].fillna(0)

    # Drop extra columns
    df_train.drop(columns=["review_date","reviewTime","unixReviewTime","image",
                           "style","reviewerName","asin"],
                  errors="ignore", inplace=True)

    # Optional reviewer freq
    if "reviewerID" in df_train.columns:
        freq_map = df_train["reviewerID"].value_counts()
        df_train["reviewer_freq"] = df_train["reviewerID"].map(freq_map).fillna(1).astype(int)

    df_train["binary_label_cutoff_1"] = (df_train["overall"] > 1).astype(int)
    df_train.drop(columns=["overall"], inplace=True)

    X = df_train.drop(columns=["binary_label_cutoff_1"])
    y = df_train["binary_label_cutoff_1"].values

    # Build column transformer
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

    # Stacking pipeline
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

    # Narrow param dist around your best combos
    param_dist_narrow = {
        # Tfidf near best discovered combos
        "prep__text__max_df": uniform(loc=0.75, scale=0.15),
        "prep__text__max_features": randint(3000, 7001),  # around 5000
        "prep__text__min_df": randint(2, 7),
        "prep__text__ngram_range": [(1,2), (1,3)],

        # RandomForest near best
        "clf__rf__n_estimators": randint(90, 120),    # around 107
        "clf__rf__max_depth": [10, 15, None],         # None or some narrower set

        # GradientBoosting near best
        "clf__gb__n_estimators": randint(70,110),     # around 89
        "clf__gb__learning_rate": loguniform(1e-3, 5e-2), # narrower around 0.02
        "clf__gb__max_depth": [4, 5, 6],              # best was 5

        # meta logistic near best
        "clf__final_estimator__C": loguniform(0.5, 5),   # best was ~2.13

        # base logistic near best
        "clf__lr__C": loguniform(1e-4, 1e-2)  # best was ~0.0054
    }

    print("[INFO] Setting up RandomizedSearchCV with narrower param distributions")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist_narrow,
        n_iter=30,  # might do 30 or 40
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    print("[INFO] Starting narrower random search (this is still CPU-heavy, but should be faster than wide search).")
    search.fit(X, y)
    print("BEST PARAMS:", search.best_params_)
    print(f"BEST CV Macro F1: {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Evaluate final out-of-fold performance
    print("\n[INFO] Checking out-of-fold performance for final pipeline:")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    print("Confusion matrix:", cm)

    y_proba_cv = cross_val_predict(best_model, X, y, cv=5, method="predict_proba")[:,1]
    fpr, tpr, _ = roc_curve(y, y_proba_cv)
    print("AUC:", auc(fpr, tpr))
    print("Accuracy:", accuracy_score(y, y_pred_cv))
    print("Macro F1:", f1_score(y, y_pred_cv, average="macro"))

    ###############################################################################
    # 4. TRAIN FINAL ON FULL DATA, PREDICT TEST -> SUBMISSION
    ###############################################################################
    print("[INFO] Fitting best model on ALL training data for final submission.")
    # re-fit best_model on full X,y
    best_model.fit(X, y)

    print("[INFO] Loading Test.csv for final predictions...")
    df_test = pd.read_csv("./data/Test.csv")
    df_test = preprocess_amazon_reviews(df_test)

    # same sentiment
    test_sentiment = sentiment_extractor.transform(df_test["reviewText"])
    df_test[["pos_count","neg_count","polarity_score"]] = test_sentiment
    df_test["verified"] = df_test["verified"].fillna(0).astype(int)
    df_test["vote"] = df_test["vote"].fillna(0)

    df_test.drop(columns=["review_date","reviewTime","unixReviewTime","image",
                          "style","reviewerName","asin"],
                 errors="ignore", inplace=True)

    if "reviewerID" in df_test.columns:
        freq_map_test = df_test["reviewerID"].value_counts()
        df_test["reviewer_freq"] = df_test["reviewerID"].map(freq_map_test).fillna(1).astype(int)

    print("[INFO] Predicting on test data with best pipeline...")
    test_preds = best_model.predict(df_test)

    df_test["binary_split_1"] = test_preds
    df_test["id"] = df_test.index

    os.makedirs("results", exist_ok=True)
    out_path = "results/submission_kaggle_fine_tune.csv"
    df_test[["id","binary_split_1"]].to_csv(out_path, index=False)
    print(f"\n[INFO] Done! Kaggle submission => {out_path}")



if __name__ == "__main__":
    main()
