import pandas as pd
import numpy as np
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

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("\n[INFO] df_train columns:", df_train.columns.tolist())
print("[INFO] df_test columns: ", df_test.columns.tolist())

###############################################################################
# 2. Label creation (just cutoff 1 for now)
###############################################################################
df_train["binary_label_cutoff_1"] = (df_train["overall"] > 1).astype(int)

###############################################################################
# 3. Helper Functions
###############################################################################

# introduce short dictionary of positive and negative words
pos_words = {"great", "excellent", "love", "amazing", "wonderful", "perfect", "best",  "awesome", "good", "like", "nice", "fine", "decent", "cool", "fantastic"}
neg_words = {"bad", "awful", "terrible", "waste", "hate", "poor", "disappointed", "sucks", "damn", "returned", "refund", "junk", "crap", "garbage", "horrible"}

# negation words/phrases
negation_words = {"not", "never", "no", "n't", "hardly", "scarcely", "rarely", "barely", "don't", "didn't", "wasn't"}
negation_prefixes = {"un", "dis", "mis", "in", "im", "il", "ir", "non"}  # reversal prefixes
    
# create features based on the sentiment dictionary
def dictionary_sentiment_features(df):
    reviews = df["reviewText"].astype(str)
    pos_list, neg_list, pol_list = [], [], []

    for text in reviews:
        tokens = re.findall(r"\b\w+\b", text.lower())  # tokenize words (remove punctuation)
        token_count = len(tokens)

        p, n = 0, 0  # positive and negative counters
        for i, word in enumerate(tokens):
            is_negated = False

            # check if word is negated ("not good")
            if word in pos_words:
                # look at the last 2 words before it
                prev_words = tokens[max(0, i - 2):i] # ("not very good"))
                if any(w in negation_words for w in prev_words):
                    is_negated = True  # mark as negated / treat as negative

            # check for negation prefixes 
            for prefix in negation_prefixes:
                if word.startswith(prefix) and word[len(prefix):] in pos_words:
                    is_negated = True
                    break  # If any prefix matches, stop checking further

            # Assign polarity
            if word in pos_words and not is_negated:
                p += 1
            elif word in pos_words and is_negated:
                n += 1  # Negated positive becomes negative
            elif word in neg_words:
                n += 1

        pos_list.append(p)
        neg_list.append(n)
        pol_list.append(p - n)  # Overall polarity score

    return np.column_stack((pos_list, neg_list, pol_list))

# 
def parse_review_time(str_val):
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