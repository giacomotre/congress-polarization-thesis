import re
import string
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List
import nltk


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
def filter_short_speeches(df: pd.DataFrame, text_col: str = "speech", min_words: int = 15) -> pd.DataFrame:
    """Filter speeches with fewer than min_words words."""
    return df[df[text_col].apply(lambda x: len(str(x).split()) >= min_words)].copy()

def remove_duplicates(df: pd.DataFrame, text_col: str = "speech") -> pd.DataFrame:
    """Remove duplicate speeches."""
    return df.drop_duplicates(subset=[text_col]).copy()

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
def preprocess_df_for_tfidf(df: pd.DataFrame, text_col: str = "speech") -> pd.DataFrame:
    df = filter_short_speeches(df, text_col)
    df = remove_duplicates(df, text_col)
    df[text_col] = df[text_col].apply(clean_text_for_tfidf)
    return df

def preprocess_df_for_bert(df: pd.DataFrame, text_col: str = "speech") -> pd.DataFrame:
    df = filter_short_speeches(df, text_col)
    df = remove_duplicates(df, text_col)
    df[text_col] = df[text_col].apply(clean_text_for_bert)
    return df

