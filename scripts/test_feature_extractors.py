# /scripts/test_feature_extractors.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import vstack

# ✅ Load preprocessed data
df_train = pd.read_csv("./data/processed_train.csv")

# ✅ Define Features & Target
label_col = "binary_label_cutoff_1"
text_col = "reviewText"
numeric_cols = [
    "vote", "verified", "review_length", "num_words", "reviewer_freq",
    "review_hour", "review_weekday", "review_days_since",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]

# Drop rows with missing labels
df_train = df_train.dropna(subset=[label_col])

# Convert all column names to strings to avoid Scikit-Learn errors
df_train.columns = df_train.columns.astype(str)

# fill any empty values with empty string
df_train["reviewText"] = df_train["reviewText"].fillna("")
df_train["summary"] = df_train["summary"].fillna("")
df_train["category"] = df_train["category"].fillna("")
df_train["vote"] = df_train["vote"].fillna(0)

# Train-test split
X = df_train.drop(columns=[label_col])
y = df_train[label_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Define Parallel TF-IDF Processing
def parallel_tfidf(texts):
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1,3),
        stop_words="english"
    )
    return vectorizer.fit_transform(texts)

# Apply parallel processing to training set
text_chunks_train = np.array_split(X_train[text_col].tolist(), 8)  # Split into 8 chunks
tfidf_results_train = Parallel(n_jobs=-1)(delayed(parallel_tfidf)(chunk) for chunk in text_chunks_train)

# Combine results
tfidf_matrix_train = vstack(tfidf_results_train)

# Use TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=170, random_state=42)
X_train_text_svd = svd.fit_transform(tfidf_matrix_train)

# Convert processed features to DataFrame
df_train_svd = pd.DataFrame(X_train_text_svd, index=X_train.index)
X_train = X_train.drop(columns=[text_col]).join(df_train_svd)

# ✅ Process TF-IDF for Validation Set using the same vectorizer and SVD
text_chunks_val = np.array_split(X_val[text_col].tolist(), 8)  # Split into 8 chunks
tfidf_results_val = Parallel(n_jobs=-1)(delayed(parallel_tfidf)(chunk) for chunk in text_chunks_val)
tfidf_matrix_val = vstack(tfidf_results_val)
X_val_text_svd = svd.transform(tfidf_matrix_val)

# Convert processed features to DataFrame and merge
df_val_svd = pd.DataFrame(X_val_text_svd, index=X_val.index)
X_val = X_val.drop(columns=[text_col]).join(df_val_svd)

# ✅ Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", n_jobs=-1),
    "Perceptron": Perceptron(max_iter=2000, class_weight="balanced", n_jobs=-1),
    "SVM": SVC(kernel="linear", probability=True, class_weight="balanced"),  # Can't parallelize SVC (linear kernel)
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
}

# ✅ Define Preprocessing Pipeline for Numeric & Categorical Features
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ✅ Run Tests for Each Classifier
results = []

for clf_name, clf in classifiers.items():
    print(f"   🚀 Running {clf_name}...")

    # Train & Evaluate Model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    acc = accuracy_score(y_val, y_pred)

    results.append((clf_name, f1, acc))

    print(f"   ✅ {clf_name} => F1: {f1:.4f}, Acc: {acc:.4f}")

# ✅ Save Results for Comparison
results_df = pd.DataFrame(results, columns=["Classifier", "Macro F1", "Accuracy"])
results_df.to_csv("./results/extractor_comparison_parallel.csv", index=False)

print("\n🎯 Done! Results saved to ./results/extractor_comparison_parallel.csv")
print(results_df.sort_values(by="Macro F1", ascending=False))
