import re
import string
import pandas as pd
import spacy
import os
import joblib
import nltk
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List



# Ensure required NLTK data is available
for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}") if resource == "stopwords" else nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# Load SpaCy English model for lemmatization
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# NLTK stopwords
STOPWORDS = set(stopwords.words("english"))

# --- Shared Utilities ---
def filter_short_speeches(df: pd.DataFrame, text_col: str = "speech", min_words: int = 15) -> tuple[pd.DataFrame, int]:
    initial_rows = len(df)
    filtered_df = df[df[text_col].apply(lambda x: len(str(x).split()) >= min_words)].copy()
    removed_short_speeches_count = initial_rows - len(filtered_df)
    return filtered_df, removed_short_speeches_count

def remove_duplicates(df: pd.DataFrame, text_col: str = "speech") -> tuple[pd.DataFrame, int]:
    initial_rows = len(df)
    deduplicated_df = df.drop_duplicates(subset=[text_col]).copy()
    removed_duplicate_speeches_count = initial_rows - len(deduplicated_df)
    return deduplicated_df, removed_duplicate_speeches_count

# --- Reusable Cleaning Components ---
def lowercase(text: str) -> str:
    return text.lower()

def remove_punctuation_digits(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}\d]", "", text)

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [w for w in tokens if w not in STOPWORDS]

def lemmatize(tokens: List[str]) -> List[str]:
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# --- TF-IDF-Specific Cleaning ---
def clean_text_for_tfidf(text: str) -> str:
    text = lowercase(text)
    text = remove_punctuation_digits(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

# --- BERT-Specific Cleaning ---
def clean_text_for_bert(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# --- Batch Processing ---
def preprocess_df_for_tfidf(df: pd.DataFrame, text_col: str = "speech") -> tuple[pd.DataFrame, int, int]:
    df, removed_short_speeches_count = filter_short_speeches(df, text_col)
    df, removed_duplicate_speeches_count = remove_duplicates(df, text_col)
    
    df[text_col] = df[text_col].apply(clean_text_for_tfidf)
    return df, removed_short_speeches_count, removed_duplicate_speeches_count 

def preprocess_df_for_bert(df: pd.DataFrame, text_col: str = "speech") -> pd.DataFrame:
    df = filter_short_speeches(df, text_col)
    df = remove_duplicates(df, text_col)
    df[text_col] = df[text_col].apply(clean_text_for_bert)
    return df

def encode_labels(labels, encoder_path):

    encoder_path = Path(encoder_path)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    if encoder_path.exists():
        le = joblib.load(encoder_path)
    else:
        le = LabelEncoder().fit(labels)
        joblib.dump(le, encoder_path)

    return le, le.transform(labels) #le.transform(labels) → immediate NumPy array of ints, ready for training.

