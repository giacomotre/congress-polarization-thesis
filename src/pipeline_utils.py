import re
import string
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import joblib

# --- Consistent Label Encoding using a Map (Simplified) ---
# In pipeline_utils.py

import pandas as pd
from typing import Dict
import numpy as np # Make sure numpy is imported

# --- Consistent Label Encoding using a Map (Robust Version + Debugging) ---
def encode_labels_with_map(df: pd.DataFrame, party_map: Dict[str, int], party_col: str = "party", label_col: str = "label") -> pd.DataFrame:
    """
    Encodes party labels to integers using a predefined map and handles unmapped parties by removing them.
    Includes debugging prints.

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

    # Work on a copy to avoid modifying original df potentially passed by reference
    df_processed = df.copy()
    initial_rows = len(df_processed)

    # --- Add Debugging ---
    try:
        unique_vals = df_processed[party_col].unique()
        print(f"  DEBUG: encode_labels_with_map received unique '{party_col}' values: {unique_vals}")
        print(f"  DEBUG: encode_labels_with_map received '{party_col}' dtype: {df_processed[party_col].dtype}")
    except Exception as e:
        print(f"  DEBUG: Error getting unique values/dtype: {e}")
    # --- End Debugging ---

    # Apply the mapping directly to the original party column type.
    # This assumes your keys in party_map ('D', 'R') will match the actual data.
    # If the column contains non-string data that should map, the map needs adjustment.
    df_processed[label_col] = df_processed[party_col].map(party_map) # REMOVED .astype(str)

    # --- Handle potential NaN values from unmapped parties ---
    original_rows = len(df_processed)
    # Check which rows have NaN in the new label column (indicates mapping failed)
    nan_mask = df_processed[label_col].isnull()
    rows_dropped = nan_mask.sum()

    if rows_dropped > 0:
        # Find which actual party values from the original column caused the drop
        # Get unique non-null values from the original party column
        all_unmapped_values_in_col = df[party_col][nan_mask].dropna().unique()

        print(f"  - encode_labels_with_map: Removed {rows_dropped} rows with unmapped parties (values found: {list(all_unmapped_values_in_col)[:10]}...). Ensure PARTY_MAP keys match data types and values.")
        # Drop rows where the party was not in the map
        df_processed = df_processed[~nan_mask].reset_index(drop=True)

    # If all rows were dropped, return the empty DataFrame before trying astype
    if df_processed.empty:
        print(f"  - encode_labels_with_map: Warning - DataFrame is empty after removing rows with unmapped parties.")
        if label_col not in df_processed.columns:
             df_processed[label_col] = pd.Series(dtype='object')
        return df_processed

    # Ensure the new 'label' column is of integer type
    # Use nullable integer type to be safe
    df_processed[label_col] = df_processed[label_col].astype(float).astype(pd.Int64Dtype())

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