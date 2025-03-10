#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# scikit-optimize
# pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

###############################################################################
# 1. Load Data
###############################################################################
train_path = "./data/Training.csv"
test_path = "./data/Test.csv"

# Verify that the files exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"❌ Error: Training file not found at {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"❌ Error: Test file not found at {test_path}")

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("\n[INFO] df_train columns:", df_train.columns.tolist())
print("[INFO] df_test columns: ", df_test.columns.tolist())

###############################################################################
# 2. Label creation
###############################################################################
df_train["binary_label_cutoff_1"] = (df_train["overall"] > 1).astype(int)

###############################################################################
# 3. Helper Functions
###############################################################################
def parse_review_time(str_val):
    """Parse 'MM DD, YYYY' strings. Return NaT if fails."""
    try:
        return datetime.strptime(str_val, "%m %d, %Y")
    except:
        return pd.NaT

def advanced_time_features(df):
    """Safely create numeric time-based features from 'reviewTime' or fallback 'unixReviewTime'."""
    if "reviewTime" in df.columns:
        df["reviewTime_parsed"] = df["reviewTime"].apply(parse_review_time)
    else:
        df["reviewTime_parsed"] = pd.NaT

    if df["reviewTime_parsed"].isna().all() and "unixReviewTime" in df.columns:
        df["reviewTime_parsed"] = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")

    now = datetime.now()
    df["review_hour"]       = df["reviewTime_parsed"].dt.hour.fillna(-1).astype(int)
    df["review_weekday"]    = df["reviewTime_parsed"].dt.weekday.fillna(-1).astype(int)
    df["review_days_since"] = (now - df["reviewTime_parsed"]).dt.days.fillna(-1).astype(int)

    return df

def create_numeric_from_text(df, textcol="reviewText"):
    """Create 'review_length','num_words' from textcol. Fills with -1 if missing."""
    if textcol not in df.columns:
        df["review_length"] = -1
        df["num_words"]     = -1
    else:
        df[textcol] = df[textcol].fillna("")
        df["review_length"] = df[textcol].apply(len)
        df["num_words"]     = df[textcol].apply(lambda x: len(x.split()))
    return df

###############################################################################
# 4. Basic Preprocessing
###############################################################################
# Fill numeric columns
df_train["vote"]     = df_train.get("vote", 0).fillna(0).astype(float)
df_test["vote"]      = df_test.get("vote", 0).fillna(0).astype(float)

df_train["verified"] = df_train.get("verified", 0).fillna(0).astype(int)
df_test["verified"]  = df_test.get("verified", 0).fillna(0).astype(int)

# reviewer freq
if "reviewerID" in df_train.columns:
    freq_map_train = df_train["reviewerID"].value_counts()
    df_train["reviewer_freq"] = df_train["reviewerID"].map(freq_map_train).fillna(1).astype(int)
else:
    df_train["reviewer_freq"] = 1

if "reviewerID" in df_test.columns:
    freq_map_test = df_test["reviewerID"].value_counts()
    df_test["reviewer_freq"]  = df_test["reviewerID"].map(freq_map_test).fillna(1).astype(int)
else:
    df_test["reviewer_freq"]  = 1

# time features
df_train = advanced_time_features(df_train)
df_test  = advanced_time_features(df_test)

# text-based numeric
df_train = create_numeric_from_text(df_train, "reviewText")
df_test  = create_numeric_from_text(df_test,  "reviewText")

# fill category
if "category" not in df_train.columns:
    df_train["category"] = "unknown"
else:
    df_train["category"] = df_train["category"].fillna("unknown").astype(str)

if "category" not in df_test.columns:
    df_test["category"] = "unknown"
else:
    df_test["category"] = df_test["category"].fillna("unknown").astype(str)

# fill summary, reviewText
df_train["summary"]    = df_train.get("summary","").fillna("")
df_test["summary"]     = df_test.get("summary","").fillna("")
df_train["reviewText"] = df_train.get("reviewText","").fillna("")
df_test["reviewText"]  = df_test.get("reviewText","").fillna("")

###############################################################################
# 5. Final columns
###############################################################################
text_col = "reviewText"
summary_col = "summary"
numeric_cols = [
    "vote","verified","review_length","num_words","reviewer_freq",
    "review_hour","review_weekday","review_days_since"
]
cat_cols = ["category"]
label_col = "binary_label_cutoff_1"

final_columns = []
if text_col in df_train.columns:
    final_columns.append(text_col)
if summary_col in df_train.columns:
    final_columns.append(summary_col)
for c in cat_cols:
    if c in df_train.columns:
        final_columns.append(c)
for c in numeric_cols:
    if c in df_train.columns:
        final_columns.append(c)

df_train = df_train[final_columns + [label_col]]
df_test  = df_test[[c for c in final_columns if c in df_test.columns]]

print("\n[DEBUG] final_columns =>", final_columns)
print("[DEBUG] df_train columns =>", df_train.columns.tolist())
print("[DEBUG] df_test  columns =>", df_test.columns.tolist())

###############################################################################
# 6. Create a custom Tfidf that can handle "my_ngram" param
###############################################################################
# Because BayesSearchCV doesn't like passing a tuple for ngram_range
# We'll store "1gram" => (1,1), "2gram" => (1,2) internally
class NgramRangeTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        my_ngram="1gram",
        max_features=10000,
        stop_words="english",
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        binary=False,
        **kwargs
    ):
        super().__init__(
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            use_idf=use_idf,
            binary=binary,
            **kwargs
        )
        # custom param
        self.my_ngram = my_ngram

    def _update_ngram(self):
        """Translate 'my_ngram' => actual (1,1) or (1,2)."""
        if self.my_ngram == "1gram":
            self.ngram_range = (1,1)
        else:
            self.ngram_range = (1,2)

    def fit(self, X, y=None):
        self._update_ngram()
        return super().fit(X, y)

    def transform(self, X):
        self._update_ngram()
        return super().transform(X)

    def fit_transform(self, X, y=None):
        self._update_ngram()
        return super().fit_transform(X, y)

###############################################################################
# 7. Build pipeline
###############################################################################

# text pipeline with custom NgramRangeTfidfVectorizer
from sklearn.pipeline import Pipeline
text_pipeline = Pipeline([
    ("tfidf", NgramRangeTfidfVectorizer(  # ngram decided by "my_ngram"
        max_features=10000,
        my_ngram="1gram",  # default
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        binary=False
    )),
    ("svd", TruncatedSVD(n_components=150, random_state=42))
])

# summary tfidf
summary_tfidf = TfidfVectorizer(
    max_features=500,
    stop_words="english"
)

# numeric
from sklearn.preprocessing import StandardScaler
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

transformers_list = []
if text_col in df_train.columns:
    transformers_list.append(("text_pipe", text_pipeline, text_col))
if summary_col in df_train.columns:
    transformers_list.append(("summary_pipe", summary_tfidf, summary_col))

# cat
if all([c in df_train.columns for c in cat_cols]):
    transformers_list.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

num_ok = [c for c in numeric_cols if c in df_train.columns]
if num_ok:
    transformers_list.append(("num", numeric_pipeline, num_ok))

preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder="drop"
)

# base learners
base_lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)
base_rf = RandomForestClassifier(
    n_estimators=80,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42
)
base_gb = GradientBoostingClassifier(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

stack_ensemble = StackingClassifier(
    estimators=[
        ("lr", base_lr),
        ("rf", base_rf),
        ("gb", base_gb)
    ],
    final_estimator=LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ),
    passthrough=True,
    cv=5,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", stack_ensemble)
])

###############################################################################
# 8. Train/Val Split
###############################################################################
X = df_train.drop(columns=[label_col])
y = df_train[label_col].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

###############################################################################
# 9. Bayes Search
###############################################################################
# We'll define param space referencing our custom param "tfidf__my_ngram" 
# instead of the raw (1,2).
search_spaces = {
    "clf__rf__n_estimators": Integer(50, 175), # [!] reducing from (50, 200)
    "clf__rf__max_depth":    Integer(3, 15),
    "clf__gb__n_estimators": Integer(50, 175), # [!] reducing from (50, 200)
    "clf__gb__learning_rate": Real(1e-2, 0.2, prior="log-uniform"),
    "clf__gb__max_depth":    Integer(2, 5),
    "clf__lr__C":            Real(1e-2, 5.0, prior="log-uniform"),
    "clf__final_estimator__C": Real(1e-2, 5.0, prior="log-uniform"),

    # Tfidf
    "preprocess__text_pipe__tfidf__max_features": Integer(5000, 12000), # [!] reducing from (5000, 15000)
    "preprocess__text_pipe__tfidf__my_ngram": Categorical(["1gram","2gram"]),
    "preprocess__text_pipe__svd__n_components": Integer(100, 250) # [!] reducing from (100, 300)
}

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # [!] reducing from 5-fold

from skopt import BayesSearchCV

bayes_search = BayesSearchCV(
    estimator=final_pipeline,
    search_spaces=search_spaces,
    n_iter=20,     # (decreased from 30 to 20 to try and speed things up)
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2,
    random_state=42,
    refit=False  # We'll manually refit after
)

print("\n[INFO] Starting BayesSearchCV (30 iters, 5-fold). This can take hours.\n")
bayes_search.fit(X_train, y_train)

print("\n=== BEST RESULTS FROM BAYES SEARCH ===")
print("Best Score (f1_macro):", bayes_search.best_score_)
print("Best Params:", bayes_search.best_params_)

# Now manually set & refit
best_params = bayes_search.best_params_
final_pipeline.set_params(**best_params)
final_pipeline.fit(X_train, y_train)

###############################################################################
# 10. Evaluate on Validation
###############################################################################
y_val_pred = final_pipeline.predict(X_val)
y_val_proba = final_pipeline.predict_proba(X_val)[:,1]

val_f1  = f1_score(y_val, y_val_pred, average="macro")
val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_proba)

print("\n=== VALIDATION METRICS ===")
print(f"Macro F1 : {val_f1:.4f}")
print(f"Accuracy : {val_acc:.4f}")
print(f"ROC AUC  : {val_auc:.4f}")

###############################################################################
# 11. Train full + Submission
###############################################################################
final_pipeline.fit(X, y)

df_test_pred = final_pipeline.predict(df_test)
df_test["binary_split_1"] = df_test_pred

os.makedirs("results", exist_ok=True)
df_test["id"] = df_test.index

out_path = "results/submission_bayes_ngram_03_02_17_52.csv"
df_test[["id","binary_split_1"]].to_csv(out_path, index=False)
print(f"\n🎉 DONE. Kaggle submission => {out_path}")