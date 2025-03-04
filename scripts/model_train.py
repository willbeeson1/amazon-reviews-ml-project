# /scripts/model_train.py
import pandas as pd
import numpy as np
import os as os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import Parallel, delayed
from scipy.sparse import vstack


# check for the processed data  
df_train = pd.read_csv("./data/processed_train.csv")
df_test = pd.read_csv("./data/processed_test.csv")

print(df_train.shape)  # ~29k rows
print(df_test.shape)   
print(df_train.head())  

# Define features and label
label_col = "binary_label_cutoff_1"
text_col = "reviewText"
numeric_cols = [
    "vote", "verified", "review_length", "num_words", "avg_word_length",
    "uppercase_ratio", "exclamation_count", "reviewer_freq",
    "review_year", "review_month", "review_day", "review_dayofweek", "review_hour",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]

# Drop any rows with missing target label
df_train = df_train.dropna(subset=[label_col])

# fill any empty values with empty string
df_train["reviewText"] = df_train["reviewText"].fillna("")
df_test["reviewText"] = df_test["reviewText"].fillna("")
df_train["summary"] = df_train["summary"].fillna("")
df_test["summary"] = df_test["summary"].fillna("")
df_train["category"] = df_train["category"].fillna("")
df_test["category"] = df_test["category"].fillna("")
df_train["vote"] = df_train["vote"].fillna(0)

# Parallel TF-IDF processing function
def parallel_tfidf(texts):
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1,3),
        stop_words="english"
    )
    return vectorizer.fit_transform(texts)

# Apply parallel processing to reviewText
text_chunks = np.array_split(df_train["reviewText"].tolist(), 8)  # Split into 8 chunks
tfidf_results = Parallel(n_jobs=-1)(delayed(parallel_tfidf)(chunk) for chunk in text_chunks)

# Combine results
tfidf_matrix = vstack(tfidf_results)

# Use TruncatedSVD after TF-IDF
svd = TruncatedSVD(n_components=170, random_state=42)
X_text = svd.fit_transform(tfidf_matrix)

# Replace original reviewText column with processed features
df_train_svd = pd.DataFrame(X_text, index=df_train.index)
df_train = df_train.drop(columns=["reviewText"]).join(df_train_svd)

# ✅ Process TF-IDF for TEST SET using the same vectorizer and SVD
text_chunks_test = np.array_split(df_test["reviewText"].tolist(), 8)  # Split into 8 chunks
tfidf_results_test = Parallel(n_jobs=-1)(delayed(parallel_tfidf)(chunk) for chunk in text_chunks_test)
tfidf_matrix_test = vstack(tfidf_results_test)

# Apply TruncatedSVD to test set
X_text_test = svd.transform(tfidf_matrix_test)
df_test_svd = pd.DataFrame(X_text_test, index=df_test.index)
df_test = df_test.drop(columns=["reviewText"]).join(df_test_svd)

# Convert all column names to strings to avoid Scikit-Learn errors
df_train.columns = df_train.columns.astype(str)
df_test.columns = df_test.columns.astype(str)

# Define X and y
X = df_train.drop(columns=[label_col])
y = df_train[label_col].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    # ("text_pipe", text_pipeline, text_col), - removed bc Tf-idf is parallelized separately
    ("num", numeric_pipeline, numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Define stacking classifier
base_lr = LogisticRegression(
    solver="saga", penalty="l2", class_weight="balanced", 
    max_iter=3000, C=0.1596, random_state=42, n_jobs=-1  # parallel processing
)

base_rf = RandomForestClassifier(
    n_estimators=114, max_depth=12, class_weight="balanced_subsample", 
    random_state=42, n_jobs=-1  # Fully parallel
)

base_gb = GradientBoostingClassifier(
    n_estimators=138, learning_rate=0.2666, max_depth=5, random_state=42,
    subsample=0.8,  # Speeds up training while maintaining performance
    n_iter_no_change=5  # Stops early if no improvement
)

stack_ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=LogisticRegression(
        solver="saga", penalty="l2", class_weight="balanced", max_iter=3000, 
        C=0.1596, random_state=42, n_jobs=-1  # Use multiple cores
    ),
    passthrough=True, cv=3, n_jobs=-1, verbose=2  # Full parallel execution
)

# Define final pipeline
final_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", stack_ensemble)
])

# Train the model
print("\n🚀 Training the model...")
final_pipeline.fit(X_train, y_train)

# Evaluate
y_val_pred = final_pipeline.predict(X_val)
y_val_proba = final_pipeline.predict_proba(X_val)[:,1]

val_f1  = f1_score(y_val, y_val_pred, average="macro")
val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_proba)

print("\n=== VALIDATION METRICS ===")
print(f"Macro F1 : {val_f1:.4f}")
print(f"Accuracy : {val_acc:.4f}")
print(f"ROC AUC  : {val_auc:.4f}")

# debug
print(df_test.columns)  # Check available columns before processing

# Fill missing values
for col in numeric_cols:
    df_test[col] = df_test[col].fillna(0)  # Fill NaNs in numeric columns with 0

# Predict on test set
y_test_pred = final_pipeline.predict(df_test)
df_test["binary_split_1"] = y_test_pred

# Save results for Kaggle
os.makedirs("results", exist_ok=True)
df_test["id"] = df_test.index

out_path = "results/submission_dictionary_preprocessing_1.csv"
df_test[["id","binary_split_1"]].to_csv(out_path, index=False)

print(f"\n🎉 DONE. Kaggle submission saved at: {out_path}")