import joblib
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import utility functions and config loader (assuming they are accessible)
from config_loader import load_config # Make sure config_loader is available
from pipeline_utils import load_pipeline, encode_labels_with_map, clean_text_for_tfidf # Needed if you need to re-use text cleaning or mapping

# ------ Loading Unified Config -------
# Assume config is in the same relative location as in all_models.py
CONFIG_PATH_UNIFIED = Path(__file__).parent / "config" / "config.yaml"

try:
    unified_config = load_config(CONFIG_PATH_UNIFIED)
    print("Loaded unified config:", json.dumps(unified_config, indent=4))
except FileNotFoundError:
    print(f"Error: Unified config file not found at {CONFIG_PATH_UNIFIED}. Please create it.")
    exit() # Exit if config file is missing

# --- Extract Common Parameters ---
common_params = unified_config.get('common', {})
data_params = common_params.get('data_params', {})
CONGRESS_YEAR_START = data_params.get('congress_year_start', 75)
CONGRESS_YEAR_END = data_params.get('congress_year_end', 112)
SEEDS = common_params.get('split_params', {}).get('seeds', [42]) # Get seeds used for training
PARTY_MAP = common_params.get('party_map', {}) # Assuming party mapping is in the common section

# Ensure PARTY_MAP is not empty and contains expected parties
if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
     print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")
     # You might want to handle this more robustly, e.g., exit if party mapping is critical.

# Create a reverse mapping from encoded label back to party name
REVERSE_PARTY_MAP = {v: k for k, v in PARTY_MAP.items()}

# Define the directory where models are saved
MODEL_DIR = "models"
# Define directory to save explainability results
EXPLAINABILITY_DIR = "explainability_results"
os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)


def get_top_features(pipeline, model_type: str, n: int = 20):
    """
    Extracts and returns the top N features for each class from a trained pipeline.
    Handles different model types (LR, SVM, Bayes).
    """
    if pipeline is None:
        return None

    # Get the TF-IDF vectorizer and the trained model from the pipeline
    tfidf_vectorizer = pipeline.named_steps['tfidf']
    model = pipeline.named_steps[model_type] # Access the model step by its name

    # Get feature names (words/n-grams)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    n_features = len(feature_names)

    top_features_by_class = {}

    if model_type in ['lr', 'svm']:
        # Logistic Regression and Linear SVC have 'coef_' attribute
        # For binary classification, coef_ is shape (1, n_features) or (n_features,)
        # For multi-class, coef_ is shape (n_classes, n_features)
        coefficients = model.coef_

        if coefficients.ndim == 1: # Binary classification
            # Reshape for consistency if needed
            coefficients = coefficients.reshape(1, -1)

        n_classes = coefficients.shape[0]

        for i in range(n_classes):
            class_label_encoded = model.classes_[i] # Get the encoded label for this class
            class_name = REVERSE_PARTY_MAP.get(class_label_encoded, f"Class_{class_label_encoded}") # Map back to party name

            # Get coefficients for this class (row i)
            class_coefficients = coefficients[i, :]

            # Get indices of top positive and negative coefficients
            # Top positive coefficients indicate features most associated with this class vs others
            # Top negative coefficients indicate features most associated with other classes vs this one
            # For binary LR/SVM, positive coef means favoring the positive class (usually 1), negative favors the negative class (usually 0).
            # Let's consider positive coefficients as indicative of the current class's unique language.
            top_positive_indices = class_coefficients.argsort()[-n:][::-1]
            top_negative_indices = class_coefficients.argsort()[:n]

            top_features_by_class[f"{class_name} (Pro-Language)"] = [(feature_names[idx], class_coefficients[idx]) for idx in top_positive_indices]
            # For binary classification, negative coefficients for class 1 are essentially positive coefficients for class 0
            # For multi-class, negative coefficients mean less likely for this class compared to the intercept/average.
            # Let's also show features that strongly differentiate *against* this class, which might be pro-the other class in binary cases.
            if n_classes > 1:
                 top_features_by_class[f"{class_name} (Anti-Language)"] = [(feature_names[idx], class_coefficients[idx]) for idx in top_negative_indices]
            else: # Binary case, negative coefs for class 1 are positive for class 0
                 other_class_label_encoded = 1 - class_label_encoded # Assuming 0 and 1 are the encoded labels
                 other_class_name = REVERSE_PARTY_MAP.get(other_class_label_encoded, f"Class_{other_class_label_encoded}")
                 top_features_by_class[f"{other_class_name} (Pro-Language)"] = [(feature_names[idx], -class_coefficients[idx]) for idx in top_negative_indices] # Flip sign for the other class perspective


    elif model_type == 'bayes':
        # ComplementNB has 'feature_log_prob_' attribute
        # Shape is (n_classes, n_features)
        feature_log_prob = model.feature_log_prob_
        n_classes = feature_log_prob.shape[0]

        for i in range(n_classes):
            class_label_encoded = model.classes_[i] # Get the encoded label for this class
            class_name = REVERSE_PARTY_MAP.get(class_label_encoded, f"Class_{class_label_encoded}") # Map back to party name

            # Get log probabilities for this class (row i)
            class_log_prob = feature_log_prob[i, :]

            # Sort features by log probability (higher means more likely in this class)
            top_indices = class_log_prob.argsort()[-n:][::-1]

            top_features_by_class[class_name] = [(feature_names[idx], class_log_prob[idx]) for idx in top_indices]

    else:
        print(f"Explainability not implemented for model type: {model_type}")
        return None

    return top_features_by_class

# --- Main Execution for Explainability ---
if __name__ == "__main__":
    congress_years = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END)]
    models_to_explain = ['lr', 'svm', 'bayes']
    n_top_features = 30

    for year in congress_years:
        for seed in SEEDS:
            print(f"\nAnalyzing explainability for Congress Year: {year}, Seed: {seed}")
            for model_type in models_to_explain:
                print(f"  Loading and analyzing {model_type.upper()} model...")

                # Call the load_pipeline function from pipeline_utils
                pipeline = load_pipeline(MODEL_DIR, model_type, year, seed)

                if pipeline:
                   # ... (rest of the logic in the main block to get and save top features)
                    top_features = get_top_features(pipeline, model_type, n=n_top_features)

                    if top_features:
                        print(f"\n  Top {n_top_features} Features for {model_type.upper()} (Congress {year}, Seed {seed}):")

                        # Prepare data for saving
                        explainability_data = []

                        for class_name, features in top_features.items():
                            print(f"\n    --- {class_name} ---")
                            for feature, score in features:
                                print(f"      {feature}: {score:.4f}")
                                explainability_data.append({
                                    'year': year,
                                    'seed': seed,
                                    'model_type': model_type,
                                    'class': class_name,
                                    'feature': feature,
                                    'score': score
                                })

                        # Save the explainability data to a CSV file
                        explainability_df = pd.DataFrame(explainability_data)
                        output_filename = f"{EXPLAINABILITY_DIR}/top_features_{model_type}_{year}_seed{seed}.csv"
                        explainability_df.to_csv(output_filename, index=False)
                        print(f"\n  Explainability results saved to {output_filename}")
                    else:
                        print(f"  Could not get top features for {model_type.upper()}.")
                else:
                    print(f"  Skipping explainability for {model_type.upper()} due to pipeline loading error.")


    print("\n--- Explainability analysis finished ---")
    print(f"Results saved in the '{EXPLAINABILITY_DIR}' directory.")