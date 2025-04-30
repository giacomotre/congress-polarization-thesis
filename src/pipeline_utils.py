import re
import string
import pandas as pd
import spacy
import os
# import joblib # No longer needed for this encode_labels approach
import nltk
from pathlib import Path
# from sklearn.preprocessing import LabelEncoder # No longer needed for this encode_labels approach
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

# --- Shared Utilities ---
def filter_short_speeches(df: pd.DataFrame, text_col: str = "speech", min_words: int = 15) -> tuple[pd.DataFrame, int]:
    initial_rows = len(df)
    # Ensure the text column is treated as string to avoid errors with non-string types
    filtered_df = df[df[text_col].astype(str).apply(lambda x: len(x.split()) >= min_words)].copy()
    removed_short_speeches_count = initial_rows - len(filtered_df)
    return filtered_df, removed_short_speeches_count

def remove_duplicates(df: pd.DataFrame, text_col: str = "speech") -> tuple[pd.DataFrame, int]:
    initial_rows = len(df)
    # Ensure the text column is treated as string before dropping duplicates
    deduplicated_df = df.astype({text_col: 'str'}).drop_duplicates(subset=[text_col]).copy()
    removed_duplicate_speeches_count = initial_rows - len(deduplicated_df)
    return deduplicated_df, removed_duplicate_speeches_count

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

# --- Batch Processing ---
def preprocess_df_for_tfidf(df: pd.DataFrame, text_col: str = "speech") -> tuple[pd.DataFrame, int, int]:
    # This function seems to expect the OLD return types of filter_short_speeches and remove_duplicates
    # If those utils now return tuples, this function's calls to them need to be updated as well
    # For now, assuming the caller of preprocess_df_for_tfidf handles getting the correct DF
    # However, let's update the calls within this function to handle the tuple return
    df_filtered_short, removed_short_speeches_count = filter_short_speeches(df.copy(), text_col)
    df_deduplicated, removed_duplicate_speeches_count = remove_duplicates(df_filtered_short, text_col)

    df_deduplicated[text_col] = df_deduplicated[text_col].apply(clean_text_for_tfidf)
    return df_deduplicated, removed_short_speeches_count, removed_duplicate_speeches_count

def preprocess_df_for_bert(df: pd.DataFrame, text_col: str = "speech") -> pd.DataFrame:
    # This function also seems to expect the OLD return types. Update calls.
    df_filtered_short, _ = filter_short_speeches(df.copy(), text_col) # Ignore count if not needed
    df_deduplicated, _ = remove_duplicates(df_filtered_short, text_col) # Ignore count if not needed

    df_deduplicated[text_col] = df_deduplicated[text_col].apply(clean_text_for_bert)
    return df_deduplicated

# --- Consistent Label Encoding using a Map ---
def encode_labels_with_map(df: pd.DataFrame, party_map: Dict[str, int], party_col: str = "party", label_col: str = "label") -> pd.DataFrame:
    """
    Encodes party labels to integers using a predefined map and handles unmapped parties.

    Args:
        df (pd.DataFrame): Input DataFrame with a party column.
        party_map (Dict[str, int]): Dictionary mapping party strings to integer labels.
        party_col (str): The name of the column containing party strings.
        label_col (str): The name for the new column containing integer labels.

    Returns:
        pd.DataFrame: DataFrame with the new integer label column,
                    and rows with parties not in the map removed.
    Raises:
        ValueError: If the party_col is not found in the DataFrame.
    """
    if party_col not in df.columns:
        raise ValueError(f"Column '{party_col}' not found in the DataFrame.")

    initial_rows = len(df)

    # Apply the mapping. .map() will replace values not in party_map with NaN.
    df_processed = df.copy() # Work on a copy to avoid modifying original df in place outside this function
    df_processed[label_col] = df_processed[party_col].map(party_map)

    # Drop rows where the party was not in the map (resulting in NaN in the label column)
    df_processed.dropna(subset=[label_col], inplace=True)

    removed_unmapped_count = initial_rows - len(df_processed)
    if removed_unmapped_count > 0:
        print(f"  - encode_labels_with_map: Removed {removed_unmapped_count} rows with parties not in the provided PARTY_MAP.")

    # Ensure the new 'label' column is of integer type
    df_processed[label_col] = df_processed[label_col].astype(int)

    # Optional: Remove the original 'party' column if you only need the integer 'label'
    # df_processed = df_processed.drop(columns=[party_col]) # Decide if you want to keep or drop original party column

    return df_processed


# --- Keep the old encode_labels if it's used elsewhere for LabelEncoder functionality ---
# Or remove it if this new function is the only label encoding needed.
# Assuming for now you only need the map-based encoding for the RoBERTa pipeline,
# you might not need the old function. If you need both, rename one.
# Let's rename the old one or remove it if not used elsewhere.
# Given the previous error, it seems the old one was only called in the pipeline,
# so we can likely remove it. If other scripts use it, it should be kept and renamed.

# Removing the old encode_labels function based on context that the pipeline was trying to use the map
# If needed elsewhere, copy the old function code and give it a new name like encode_labels_with_encoder