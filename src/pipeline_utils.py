import re
import string
import pandas as pd
import spacy
import os
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Tuple

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

# --- Reusable Cleaning Components ---
def lowercase(text: str) -> str:
    if not isinstance(text, str):
        return "" # Handle non-string input gracefully
    return text.lower()

def remove_punctuation_digits(text: str) -> str:
    if not isinstance(text, str):
        return "" # Handle non-string input gracefully
    return re.sub(rf"[{re.escape(string.punctuation)}\d]", "", text)

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return [] # Handle non-string input gracefully
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [w for w in tokens if w not in STOPWORDS]

def lemmatize(tokens: List[str]) -> List[str]:
    if not tokens: # Handle empty list
        return []
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
    if not isinstance(text, str):
        return "" # Handle non-string input gracefully
    return re.sub(r"\s+", " ", text).strip()

# --- Consistent Label Encoding using a Map (Simplified) ---
def encode_labels_with_map(df: pd.DataFrame, party_map: Dict[str, int], party_col: str = "party", label_col: str = "label") -> pd.DataFrame:
    """
    Encodes party labels to integers using a predefined map.
    Assumes all parties in df[party_col] are present in party_map.

    Args:
        df (pd.DataFrame): Input DataFrame with a party column.
        party_map (Dict[str, int]): Dictionary mapping party strings to integer labels.
        party_col (str): The name of the column containing party strings.
        label_col (str): The name for the new column containing integer labels.

    Returns:
        pd.DataFrame: DataFrame with the new integer label column.
    Raises:
        ValueError: If the party_col is not found in the DataFrame.
        KeyError: If a party in df[party_col] is not found in party_map
                  (this indicates the assumption of pre-cleaned data was violated).
    """
    if party_col not in df.columns:
        raise ValueError(f"Column '{party_col}' not found in the DataFrame.")

    df_processed = df.copy()

    # Apply the mapping.
    # If a party in df_processed[party_col] is not a key in party_map,
    # this will raise a KeyError if we don't handle it (e.g. by using .get with a default).
    # However, based on your request, we assume all parties are in the map.
    # If an unmapped party *does* appear, it's better to error out to highlight the data issue.
    try:
        df_processed[label_col] = df_processed[party_col].map(party_map)
    except Exception as e: # Catching a generic exception, but a more specific one might be better if .map has a typical error for missing keys
        # Re-raise or handle more specifically if .map itself doesn't raise KeyError directly for missing values.
        # Pandas .map() will insert NaN for keys not found. We need to check for this.
        print(f"Error during mapping. This might be due to an unexpected party not in party_map: {e}")
        # Add a check for NaNs which indicate unmapped parties
        if df_processed[label_col].isnull().any():
            unmapped_parties = df_processed[df_processed[label_col].isnull()][party_col].unique()
            raise KeyError(f"Unmapped parties found: {unmapped_parties}. "
                           f"Ensure all parties in '{party_col}' are keys in party_map or clean the data.")
        raise # Re-raise original error if it wasn't about NaNs

    # Ensure the new 'label' column is of integer type
    # If any NaNs were introduced by .map (i.e., unmapped parties), .astype(int) would fail.
    # The check above should prevent this if unmapped parties exist.
    df_processed[label_col] = df_processed[label_col].astype(int)
    
    # Optional: Remove the original 'party' column if you only need the integer 'label'
    # df_processed = df_processed.drop(columns=[party_col])

    return df_processed

# --- Utility for Loading Trained Pipelines ---
def load_pipeline(model_dir: str, model_type: str, congress_year: str, seed: int):
    """Loads a trained pipeline from a joblib file."""
    model_filename = f"{model_dir}/tfidf_{model_type}_{congress_year}_seed{seed}_pipeline.joblib"
    if not os.path.exists(model_filename):
        print(f"Error: Model file not found at {model_filename}")
        return None
    try:
        pipeline = joblib.load(model_filename)
        print(f"Successfully loaded pipeline from {model_filename}")
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline from {model_filename}: {e}")
        return None


# --- Keep the old encode_labels if it's used elsewhere for LabelEncoder functionality ---
# Or remove it if this new function is the only label encoding needed.
# Assuming for now you only need the map-based encoding for the RoBERTa pipeline,
# you might not need the old function. If you need both, rename one.
# Let's rename the old one or remove it if not used elsewhere.
# Given the previous error, it seems the old one was only called in the pipeline,
# so we can likely remove it. If other scripts use it, it should be kept and renamed.

# Removing the old encode_labels function based on context that the pipeline was trying to use the map
# If needed elsewhere, copy the old function code and give it a new name like encode_labels_with_encoder