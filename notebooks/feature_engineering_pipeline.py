# scripts/feature_engineering_pipeline.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

###############################################################################
# 1. CLEANING & TEXT PREPROCESSING
###############################################################################

def clean_text(text):
    """Cleans text by removing HTML tags, URLs, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s\'\-]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

###############################################################################
# 2. TEXT STATISTICS & LEXICAL FEATURES
###############################################################################

def extract_text_statistics(df):
    """Extracts text statistics such as length, word count, and punctuation usage."""
    df['review_length'] = df['reviewText'].apply(len)
    df['summary_length'] = df['summary'].apply(len)
    df['num_words'] = df['reviewText'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['reviewText'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df['uppercase_ratio'] = df['reviewText'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df['exclamation_count'] = df['reviewText'].apply(lambda x: x.count('!'))
    return df

###############################################################################
# 3. ADVANCED SENTIMENT ANALYSIS WITH NEGATION HANDLING
###############################################################################

class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates sentiment features using an optimized negation-aware approach.
    """
    def __init__(self):
        self.pos_words = {"great", "excellent", "love", "amazing", "wonderful", 
                          "perfect", "best", "awesome", "good", "like", "nice", 
                          "fine", "decent", "cool", "fantastic"}
        self.neg_words = {"bad", "awful", "terrible", "waste", "hate", "poor", 
                          "disappointed", "sucks", "damn", "returned", "refund", 
                          "junk", "crap", "garbage", "horrible"}
        self.negation_words = {"not", "no", "never", "don't", "doesn't", 
                               "didn't", "won't", "wouldn't", "couldn't", 
                               "can't", "isn't", "aren't", "wasn't", "weren't"}
        self.negation_prefixes = {"un", "dis", "in", "im", "non", "ir"}
        
    def fit(self, X, y=None):
        return self  # No fitting needed
    
    def transform(self, X):
        pos_list, neg_list, pol_list = [], [], []

        for text in X:
            tokens = re.findall(r"\b\w+\b", text.lower())  # Tokenize
            pos_count, neg_count = 0, 0
            negation_active = False

            for i, token in enumerate(tokens):
                # Check negation context
                if token in self.negation_words:
                    negation_active = True
                    continue

                # Look at the last 3 words for negation
                if not negation_active:
                    for j in range(max(0, i-3), i):
                        if tokens[j] in self.negation_words:
                            negation_active = True
                            break
                
                # Prefix negation check
                for prefix in self.negation_prefixes:
                    if token.startswith(prefix) and len(token) > len(prefix) + 1:
                        base_word = token[len(prefix):]
                        if base_word in self.pos_words:
                            neg_count += 1
                            break
                        elif base_word in self.neg_words:
                            pos_count += 1
                            break
                
                # Apply sentiment counting
                if token in self.pos_words:
                    pos_count += 1 if not negation_active else 0
                    neg_count += 1 if negation_active else 0
                elif token in self.neg_words:
                    neg_count += 1 if not negation_active else 0
                    pos_count += 1 if negation_active else 0

                if token in self.pos_words or token in self.neg_words or token in {".", ",", ";", "!", "?"}:
                    negation_active = False

            pos_list.append(pos_count)
            neg_list.append(neg_count)
            pol_list.append(pos_count - neg_count)

        return np.column_stack((pos_list, neg_list, pol_list))

###############################################################################
# 4. APPLY ALL PREPROCESSING STEPS
###############################################################################

def preprocess_amazon_reviews(df):
    """Runs the entire preprocessing pipeline on the dataset."""
    df = df.copy()
    df["reviewText"] = df["reviewText"].fillna("").astype(str).apply(clean_text)
    df["summary"] = df["summary"].fillna("").astype(str).apply(clean_text)
    
    df = process_time_features(df)
    df = extract_text_statistics(df)
    
    return df

# Example usage
if __name__ == "__main__":
    train_path = "./data/Training.csv"
    test_path = "./data/Test.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df_train = preprocess_amazon_reviews(df_train)
    df_test = preprocess_amazon_reviews(df_test)
    
    sentiment_extractor = SentimentFeatureExtractor()
    sentiment_train = sentiment_extractor.transform(df_train["reviewText"])
    sentiment_test = sentiment_extractor.transform(df_test["reviewText"])

    df_train[["pos_count", "neg_count", "polarity_score"]] = sentiment_train
    df_test[["pos_count", "neg_count", "polarity_score"]] = sentiment_test

    # Add reviewer frequency (number of reviews per reviewer)
    if "reviewerID" in df_train.columns:
        reviewer_freq_map = df_train["reviewerID"].value_counts()
        df_train["reviewer_freq"] = df_train["reviewerID"].map(reviewer_freq_map).fillna(1).astype(int)

    if "reviewerID" in df_test.columns:
        reviewer_freq_map_test = df_test["reviewerID"].value_counts()
        df_test["reviewer_freq"] = df_test["reviewerID"].map(reviewer_freq_map_test).fillna(1).astype(int)

        # Add binary labels for all 4 cutoffs (used in training)
    for cutoff in [1, 2, 3, 4]:
        df_train[f"binary_label_cutoff_{cutoff}"] = (df_train["overall"] > cutoff).astype(int)

    print("\nProcessed training data:")
    print(df_train.head())

    # Save processed data
    df_train.to_csv("./data/processed_train.csv", index=False)
    df_test.to_csv("./data/processed_test.csv", index=False)

    df_train = pd.read_csv("./data/processed_train.csv")
    df_test = pd.read_csv("./data/processed_test.csv")

    print(df_train.columns)  # Make sure 'num_words' is present
