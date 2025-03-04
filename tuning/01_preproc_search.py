
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD


# Load dataset
df_train = pd.read_csv("./data/processed_train.csv")
label_col = "binary_label_cutoff_1"
df_train = df_train.dropna(subset=[label_col])

X = df_train.copy().drop(columns=[label_col])
y = df_train[label_col].values

# Ensure text columns are non-null
df_train["reviewText"] = df_train["reviewText"].fillna("")
df_train["summary"] = df_train["summary"].fillna("")
df_train["category"] = df_train["category"].fillna("")
df_train["vote"] = df_train["vote"].fillna(0)

################################################################
# 1) Fit one TfidfVectorizer on ALL review text
################################################################
full_review_text = df_train["reviewText"].tolist()  
review_vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), 
                                    stop_words="english")

print("Fitting TF-IDF for reviewText on entire dataset...")
review_vectorizer.fit(full_review_text)
print("Done fitting reviewText TF-IDF.")

################################################################
# 2) Parallel transform reviewText chunks 
################################################################
def parallel_tfidf_transform(text_chunk, vectorizer):
    return vectorizer.transform(text_chunk)

num_cores = min(72, len(df_train))  # or however many you want
text_chunks_review = np.array_split(full_review_text, num_cores)

print("Parallel transforming reviewText in chunks...")
tfidf_results_review = Parallel(n_jobs=-1)(
    delayed(parallel_tfidf_transform)(chunk, review_vectorizer) 
    for chunk in text_chunks_review
)
tfidf_matrix_review = vstack(tfidf_results_review) 
print(f"tfidf_matrix_review shape = {tfidf_matrix_review.shape}")

################################################################
# 3) Repeat for summary text
################################################################
full_summary_text = df_train["summary"].tolist()
summary_vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), 
                                     stop_words="english")

print("Fitting TF-IDF for summary on entire dataset...")
summary_vectorizer.fit(full_summary_text)
print("Done fitting summary TF-IDF.")

text_chunks_summary = np.array_split(full_summary_text, num_cores)
print("Parallel transforming summary in chunks...")
tfidf_results_summary = Parallel(n_jobs=-1)(
    delayed(parallel_tfidf_transform)(chunk, summary_vectorizer)
    for chunk in text_chunks_summary
)
tfidf_matrix_summary = vstack(tfidf_results_summary)
print(f"tfidf_matrix_summary shape = {tfidf_matrix_summary.shape}")


# Dimensionality reduction with Truncated SVD
svd_review = TruncatedSVD(n_components=170, random_state=42)
svd_summary = TruncatedSVD(n_components=50, random_state=42)

print("Starting SVD for reviewText ...") # Might take a while
X_text_review = svd_review.fit_transform(tfidf_matrix_review)
print("Finished SVD for reviewText.")

df_train_svd_review = pd.DataFrame(
    X_text_review,
    index=df_train.index,
    columns=[f"rev_svd_{i}" for i in range(X_text_review.shape[1])]
)

print("Starting SVD for summary ...") # Might take a while
X_text_summary = svd_summary.fit_transform(tfidf_matrix_summary)
print("Finished SVD for summary.")

# After computing SVD on summary
df_train_svd_summary = pd.DataFrame(
    X_text_summary,
    index=df_train.index,
    columns=[f"sum_svd_{i}" for i in range(X_text_summary.shape[1])]
)

# Now drop old text columns and join
df_train = (
    df_train.drop(columns=["reviewText", "summary"])
            .join(df_train_svd_review)
            .join(df_train_svd_summary)
)

# Convert to DataFrame and merge back
# df_train_svd_review = pd.DataFrame(X_text_review, index=df_train.index)
# df_train_svd_summary = pd.DataFrame(X_text_summary, index=df_train.index)

# df_train = df_train.drop(columns=["reviewText", "summary"]).join(df_train_svd_review).join(df_train_svd_summary)

# 2. Suppose X has columns: ["reviewText", "summary", ... numeric/sentiment features, "category", etc.]
# reviewText_cols = ["reviewText"]  # you can also do a separate Tfidf for "summary"
# summaryText_cols = ["summary"]

numeric_cols = [
    "vote", "verified", "review_length", "num_words", "avg_word_length",
    "uppercase_ratio", "exclamation_count", "reviewer_freq",
    "review_year", "review_month", "review_day", "review_dayofweek", "review_hour",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]

# separate TF-IDF vectorizers for reviewText and summary
tfidf_review = TfidfVectorizer()
tfidf_summary = TfidfVectorizer()

num_pipe = StandardScaler()

preprocessor = ColumnTransformer([
    ("tfidf_review", tfidf_review, "reviewText"),
    ("tfidf_summary", tfidf_summary, "summary"),
    # Possibly add OneHotEncoder for cat_cols
], remainder="drop")

pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(
        solver="saga", class_weight="balanced", max_iter=1000, n_jobs=-1, random_state=42
    ))
])

param_dist = {
      # TF-IDF for reviewText
    "preproc__tfidf_review__ngram_range": [(1,2), (1,3)],
    "preproc__tfidf_review__max_features": [5000, 8000, 12000],
    # "preproc__tfidf_review__min_df": [2, 5],
    # "preproc__tfidf_review__max_df": [0.7, 0.85],
    # "preproc__tfidf_review__binary": [False, True],
    # "preproc__text__stop_words": [None,"english"],
    # "preproc__tfidf_review__norm": ["l1", "l2", None],
    # "preproc__tfidf_review__use_idf": [True, False],
    # "preproc__tfidf_review__sublinear_tf": [True, False],

    # # TF-IDF for summary
    # "preproc__tfidf_summary__ngram_range": [(1,2), (1,3)],
    # "preproc__tfidf_summary__max_features": [1000, 2000, 4000],  # Usually smaller than reviewText
    # "preproc__tfidf_summary__min_df": [2, 5],
    # "preproc__tfidf_summary__max_df": [0.7, 0.85],
    # "preproc__tfidf_summary__binary": [False, True],
    # "preproc__text__stop_words": [None,"english"],

    # "preproc__text__stop_words": [None,"english"], etc.
    # Classifier
    # We'll keep C=1.0 here to keep it simple, or we can do a small range
    # "clf__C": loguniform(1e-3, 10),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold CV

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=2,  # Enough combos to see a difference
    scoring="f1_macro",
    cv=2,
    n_jobs=-1,
    random_state=42,
    verbose=2
)
search.fit(X, y)

print("=== Phase 1: Preproc best params ===")
print(search.best_params_)
print(f"CV Macro F1 = {search.best_score_:.4f}")

# Save best pipeline or best params
best_preproc_pipe = search.best_estimator_
