import pandas as pd
import numpy as np

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

print(df_train.shape)  # Should still be ~29k rows
print(df_test.shape)   # Should match original test file
print(df_train.head())  # Verify features look right

# Define features and label
label_col = "binary_label_cutoff_1"
text_col = "reviewText"
numeric_cols = [
    "vote", "verified", "review_length", "num_words", "reviewer_freq",
    "review_hour", "review_weekday", "review_days_since",
    "pos_count", "neg_count", "polarity_score"
]
cat_cols = ["category"]

# Drop any rows with missing target label
df_train = df_train.dropna(subset=[label_col])

# Define X and y
X = df_train.drop(columns=[label_col])
y = df_train[label_col].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define preprocessing
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1,2), stop_words="english")),
    ("svd", TruncatedSVD(n_components=250, random_state=42))
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
base_lr = LogisticRegression(solver="saga", class_weight="balanced", max_iter=3000, random_state=42)
base_rf = RandomForestClassifier(n_estimators=80, max_depth=12, class_weight="balanced_subsample", random_state=42)
base_gb = GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42)

stack_ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=3000, random_state=42),
    passthrough=True,
    cv=5,
    n_jobs=-1
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

# Predict on test set
y_test_pred = final_pipeline.predict(df_test)
df_test["binary_split_1"] = y_test_pred

# Save results for Kaggle
os.makedirs("results", exist_ok=True)
df_test["id"] = df_test.index

out_path = "results/submission_dictionary_preprocessing_1.csv"
df_test[["id","binary_split_1"]].to_csv(out_path, index=False)

print(f"\n🎉 DONE. Kaggle submission saved at: {out_path}")