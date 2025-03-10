import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack
from joblib import Parallel, delayed

# Load dataset
df_train = pd.read_csv("./data/processed_train.csv")
label_col = "binary_label_cutoff_1"
df_train = df_train.dropna(subset=[label_col])

X = df_train.copy().drop(columns=[label_col])
y = df_train[label_col].values

# 🔥 Fix NaN issue: Fill missing values with empty string before TF-IDF
df_train["reviewText"] = df_train["reviewText"].fillna("")
df_train["summary"] = df_train["summary"].fillna("")

# Use 72 cores if dataset is large enough
num_cores = min(72, len(df_train))  

# Function to transform text in parallel with a fixed vectorizer
def transform_chunk(vectorizer, texts):
    return vectorizer.transform(texts)

# === Step 1: Fit a single TF-IDF vectorizer for reviewText (shared vocabulary) ===
tfidf_review = TfidfVectorizer(max_features=12000, ngram_range=(1,3), stop_words="english")
tfidf_review.fit(df_train["reviewText"])  # Fit on full dataset

# Split and transform in parallel
text_chunks_review = np.array_split(df_train["reviewText"].tolist(), num_cores)
tfidf_results_review = Parallel(n_jobs=-1)(delayed(transform_chunk)(tfidf_review, chunk) for chunk in text_chunks_review)
tfidf_matrix_review = vstack(tfidf_results_review)  # Now all chunks align

# === Step 2: Fit a single TF-IDF vectorizer for summary (shared vocabulary) ===
tfidf_summary = TfidfVectorizer(max_features=4000, ngram_range=(1,2), stop_words="english")
tfidf_summary.fit(df_train["summary"])  # Fit on full dataset

# Split and transform in parallel
text_chunks_summary = np.array_split(df_train["summary"].tolist(), num_cores)
tfidf_results_summary = Parallel(n_jobs=-1)(delayed(transform_chunk)(tfidf_summary, chunk) for chunk in text_chunks_summary)
tfidf_matrix_summary = vstack(tfidf_results_summary)  # Now all chunks align

# === Step 3: Apply Truncated SVD to Reduce Dimensionality ===
svd_review = TruncatedSVD(n_components=170, random_state=42)
svd_summary = TruncatedSVD(n_components=50, random_state=42)

X_text_review = svd_review.fit_transform(tfidf_matrix_review)
X_text_summary = svd_summary.fit_transform(tfidf_matrix_summary)

# Convert to DataFrame and merge back
df_train_svd_review = pd.DataFrame(X_text_review, index=df_train.index)
df_train_svd_summary = pd.DataFrame(X_text_summary, index=df_train.index)

df_train = df_train.drop(columns=["reviewText", "summary"]).join(df_train_svd_review).join(df_train_svd_summary)

# Define numeric columns
numeric_cols = [
    "vote", "verified", "review_length", "num_words", "avg_word_length",
    "uppercase_ratio", "exclamation_count", "reviewer_freq",
    "review_year", "review_month", "review_day", "review_dayofweek", "review_hour",
    "pos_count", "neg_count", "polarity_score"
]

num_pipe = StandardScaler()
preprocessor = ColumnTransformer([
    ("numeric", num_pipe, numeric_cols),
], remainder="drop")

# Define pipeline
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("svd", TruncatedSVD(n_components=100, random_state=42)),
    ("clf", LogisticRegression(
        solver="saga", class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1
    ))
])

# Define hyperparameter search
param_dist = {
    "clf__C": np.logspace(-3, 2, 10),
}

# Setup parallelized RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=12,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,  # Enables parallelism for cross-validation
    random_state=42,
    verbose=2
)

search.fit(df_train, y)

print("=== Best Hyperparameters ===")
print(search.best_params_)
print(f"CV Macro F1 = {search.best_score_:.4f}")

# Save the best pipeline
best_preproc_pipe = search.best_estimator_
