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

# Define X and y
X = df_train.drop(columns=[label_col])
y = df_train[label_col].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define preprocessing

# text_pipeline = Pipeline([
#     ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1,2), stop_words="english")),
#     ("svd", TruncatedSVD(n_components=250, random_state=42))
# ])

# Update Truncated SVD (dimensionality reduction)
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1,3), stop_words="english")),
    ("svd", TruncatedSVD(n_components=170, random_state=42))  # best previous SVD value
])

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("text_pipe", text_pipeline, text_col),
    ("num", numeric_pipeline, numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Define stacking classifier
base_lr = LogisticRegression(solver="saga", penalty="l2", class_weight="balanced", max_iter=3000, C=0.1596, random_state=42)
base_rf = RandomForestClassifier(n_estimators=114, max_depth=12, class_weight="balanced_subsample", random_state=42)
base_gb = GradientBoostingClassifier(n_estimators=138, learning_rate=0.2666, max_depth=5, random_state=42)

stack_ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=LogisticRegression(
        solver="saga", penalty="l2", class_weight="balanced", max_iter=3000, 
        C=0.1596, random_state=42  # switching to saga 
    ),
    passthrough=True, cv=3, n_jobs=-1, verbose=2
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

# Fill missing values
df_test[text_col] = df_test[text_col].fillna("")  # Fill missing text values with empty string
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