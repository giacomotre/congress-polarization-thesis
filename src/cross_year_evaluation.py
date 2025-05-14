# cross_year_evaluation.py (Modified for comprehensive testing with manual CV and cuDF/CuPy)

import os
import pandas as pd
import joblib
import nltk # Keep if needed by pipeline_utils
import json
import time
import numpy as np
from pathlib import Path
# from collections import Counter # Not used in updated all_models.py

# Import RAPIDS components
import cudf
import cupy

# Import necessary components from sklearn and cuml
from sklearn.pipeline import Pipeline # Keep Pipeline as a concept/container, even if saving components separately
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# Use KFold and ParameterGrid for manual tuning
from sklearn.model_selection import KFold, ParameterGrid
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.naive_bayes import ComplementNB
from cuml.svm import LinearSVC
from cuml.linear_model import LogisticRegression

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map, clean_text_for_tfidf # Assuming load_pipeline is or will be in pipeline_utils if you use it

# ------ Loading Unified Config -------
# Adjust the path as needed based on your project structure
CONFIG_PATH_UNIFIED = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

try:
    unified_config = load_config(CONFIG_PATH_UNIFIED)
    print("Loaded unified config:", json.dumps(unified_config, indent=4))
except FileNotFoundError:
    print(f"Error: Unified config file not found at {CONFIG_PATH_UNIFIED}. Please create it.")
    exit() # Exit if config file is missing

# --- Extract Common Parameters ---
common_params = unified_config.get('common', {})
split_params = common_params.get('split_params', {})
# Use TEST_SIZE and VALIDATION_SIZE for the within-year split *for tuning*
TEST_SIZE = split_params.get('test_size', 0.15)
VALIDATION_SIZE = split_params.get('validation_size', 0.25)
DEFAULT_RANDOM_STATE = split_params.get('random_state', 42)
SEEDS = split_params.get('seeds', [DEFAULT_RANDOM_STATE])

data_params = common_params.get('data_params', {})
CONGRESS_YEAR_START = data_params.get('congress_year_start', 75)
CONGRESS_YEAR_END = data_params.get('congress_year_end', 112) # Include END year for testing

PARTY_MAP = common_params.get('party_map', {})

if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
     print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")

# Define detailed log path for comprehensive cross-year evaluation
COMPREHENSIVE_CROSS_YEAR_LOG_PATH = "logs/tfidf_comprehensive_cross_year_performance_detailed.csv"

# Define directory for saving trained TF-IDF vectorizers and models
MODEL_DIR_PATH = Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True) # Ensure models directory exists

# Remove existing log file to start fresh and write header
if os.path.exists(COMPREHENSIVE_CROSS_YEAR_LOG_PATH):
    os.remove(COMPREHENSIVE_CROSS_YEAR_LOG_PATH)
    print(f"Deleted existing comprehensive cross-year detailed log file: {COMPREHENSIVE_CROSS_YEAR_LOG_PATH}")
with open(COMPREHENSIVE_CROSS_YEAR_LOG_PATH, "w") as f:
    f.write("seed,train_year,test_year,accuracy,f1_score,auc\n")


# --- Define Model Configurations Dictionary ---
model_configs = {
    'bayes': unified_config.get('bayes', {}),
    'svm': unified_config.get('svm', {}),
    'lr': unified_config.get('logistic_regression', {})
}

# Function to train a model (TF-IDF and Model) for a specific year using manual CV
def train_model_for_year(train_year: str, model_type: str, model_config: dict, random_state: int, party_map: dict):
    """
    Loads data for train_year, performs manual CV tuning, trains the final
    TF-IDF and model with best params, saves them, and returns the trained objects.
    """
    print(f"\n --- Training {model_type.upper()} model for year {train_year} with seed {random_state} ---")
    start_time_total = time.time()

    # --- Data Loading (Train Year) ---
    train_input_path = f"data/merged/house_db/house_merged_{train_year}.csv"
    if not os.path.exists(train_input_path):
        print(f"⚠️  Skipping training for {train_year} (seed {random_state}): CSV file not found at {train_input_path}.")
        return None, None # Return None for both TF-IDF and Model

    print(f"Loading training data for year {train_year}...")
    train_df_full = pd.read_csv(train_input_path)
    print("Training data loaded.")

    if train_df_full.empty or not all(col in train_df_full.columns for col in ['speech', 'party', 'speakerid']):
         print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Data empty or missing required columns.")
         return None, None
    train_df_full.dropna(subset=['speech', 'party', 'speakerid'], inplace=True)
    if train_df_full.empty:
        print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Data empty after NaNs drop.")
        return None, None


    # --- Data Splitting (Train Year - for Tuning - Leave-out Speaker) ---
    print(f"Performing within-year split for tuning on {train_year} data...")
    unique_train_speakers = train_df_full['speakerid'].unique()

    if len(unique_train_speakers) < 2:
        print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Not enough unique speakers ({len(unique_train_speakers)}) for split.")
        return None, None

    # Adjust test_size if number of speakers is very small
    current_test_size = TEST_SIZE if len(unique_train_speakers) * TEST_SIZE >= 1 else 1/len(unique_train_speakers)

    train_val_speaker, _ = train_test_split( # We only need train/val speakers for tuning
        unique_train_speakers,
        test_size=current_test_size, # Proportion for the temporary 'test' part of this split
        random_state=random_state
    )

    if len(train_val_speaker) < 2:
         print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Not enough train/val speakers ({len(train_val_speaker)}) for further split.")
         return None, None

    # Adjust validation_size based on remaining data
    current_validation_size = VALIDATION_SIZE / (1 - TEST_SIZE) if (1 - TEST_SIZE) > 0 else VALIDATION_SIZE
    current_validation_size = current_validation_size if len(train_val_speaker) * current_validation_size >= 1 else 1/len(train_val_speaker)


    train_speaker_tune, val_speaker_tune = train_test_split(
        train_val_speaker,
        test_size=current_validation_size,
        random_state=random_state
    )

    train_df_tune = train_df_full[train_df_full["speakerid"].isin(train_speaker_tune)].reset_index(drop=True)
    val_df_tune = train_df_full[train_df_full["speakerid"].isin(val_speaker_tune)].reset_index(drop=True)


    X_train_tune_pd = train_df_tune["speech"]
    y_train_tune_pd = train_df_tune["party"]
    X_val_tune_pd = val_df_tune["speech"]
    y_val_tune_pd = val_df_tune["party"]


    # --- Encoding (Train Year - for Tuning) ---
    print("Encoding labels for tuning data...")
    train_data_to_encode_tune = pd.DataFrame({'party': y_train_tune_pd, 'speech': X_train_tune_pd})
    train_encoded_df_tune = encode_labels_with_map(train_data_to_encode_tune, party_map)
    X_train_tune_pd_aligned = train_encoded_df_tune['speech'].reset_index(drop=True)
    y_train_encoded_pd_aligned_tune = train_encoded_df_tune['label'].reset_index(drop=True)

    val_data_to_encode_tune = pd.DataFrame({'party': y_val_tune_pd, 'speech': X_val_tune_pd})
    val_encoded_df_tune = encode_labels_with_map(val_data_to_encode_tune, party_map)
    X_val_tune_pd_aligned = val_encoded_df_tune['speech'].reset_index(drop=True)
    y_val_encoded_pd_aligned_tune = val_encoded_df_tune['label'].reset_index(drop=True)

    # Combine ALIGNED train + val for manual CV
    X_train_val_combined_pd_aligned = pd.concat([X_train_tune_pd_aligned, X_val_tune_pd_aligned], ignore_index=True)
    y_train_val_encoded_pd_aligned_tune = pd.concat([y_train_encoded_pd_aligned_tune, y_val_encoded_pd_aligned_tune], ignore_index=True)


    if X_train_val_combined_pd_aligned.empty or y_train_val_encoded_pd_aligned_tune.empty:
         print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Combined train/validation data is empty after encoding.")
         return None, None
    if len(np.unique(y_train_val_encoded_pd_aligned_tune)) < 2:
         print(f"⚠️  Skipping training for {train_year} (seed {random_state}): Fewer than 2 unique classes in training labels for tuning.")
         return None, None

    # --- Manual Cross-Validation Loop for Tuning ---
    n_splits = 5 # Or get from config
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1
    best_params = None
    # results = [] # To store results for each param combination if needed

    print(f"Starting manual {n_splits}-fold cross-validation for hyperparameter tuning on {train_year} data...")
    start_time_tuning = time.time()

    # Define Hyperparameter Grid (using model_config, same as updated all_models.py)
    param_combinations = {
        'tfidf__max_features': model_config.get("tfidf_max_features_grid", [10000]),
        'tfidf__ngram_range': [tuple(nr) for nr in model_config.get("ngram_range_grid", [[1, 2]])],
    }
    model_specific_grid = {}
    if model_type == 'lr':
        model_specific_grid['model__C'] = model_config.get('lr_C_grid', [1.0])
    # Add other model params here ('model__param_name')

    param_combinations.update(model_specific_grid)
    grid = ParameterGrid(param_combinations)


    for params in grid:
        # print(f"  Testing params: {params}")
        fold_scores = []

        # Extract params for TF-IDF and Model
        tfidf_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('tfidf__')}
        model_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('model__')}

        fold_num = 0
        # Use the combined train/val data for the KFold split
        for train_idx, val_idx in kf.split(X_train_val_combined_pd_aligned, y_train_val_encoded_pd_aligned_tune):
            fold_num += 1
            # print(f"    Fold {fold_num}/{n_splits}")

            # Get pandas folds for this iteration
            X_train_fold_pd = X_train_val_combined_pd_aligned.iloc[train_idx]
            y_train_fold_pd = y_train_val_encoded_pd_aligned_tune.iloc[train_idx]
            X_val_fold_pd = X_train_val_combined_pd_aligned.iloc[val_idx]
            y_val_fold_pd = y_train_val_encoded_pd_aligned_tune.iloc[val_idx]

            try:
                # Clean text data for the folds
                X_train_fold_cleaned = X_train_fold_pd.apply(clean_text_for_tfidf)
                X_val_fold_cleaned = X_val_fold_pd.apply(clean_text_for_tfidf)

                # Convert folds to GPU data (cuDF for X, CuPy for y is preferred by cuML models)
                X_train_fold_cudf = cudf.Series(X_train_fold_cleaned)
                y_train_fold_cupy = cupy.asarray(y_train_fold_pd.to_numpy(dtype=np.int32)) # Target as CuPy array
                X_val_fold_cudf = cudf.Series(X_val_fold_cleaned)
                y_val_fold_cupy = cupy.asarray(y_val_fold_pd.to_numpy(dtype=np.int32)) # Target as CuPy array


                # Instantiate cuML components with current params
                tfidf_vectorizer = TfidfVectorizer(**tfidf_params) # Pass extracted tfidf params

                if model_type == 'bayes':
                    model = ComplementNB(**model_params)
                elif model_type == 'svm':
                    model = LinearSVC(**model_params)
                elif model_type == 'lr':
                    model = LogisticRegression(penalty='l2', **model_params)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Fit TF-IDF and transform
                X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold_cudf)
                X_val_tfidf = tfidf_vectorizer.transform(X_val_fold_cudf)

                # Fit Model
                model.fit(X_train_tfidf, y_train_fold_cupy)

                # Predict on validation fold
                y_pred_val_gpu = model.predict(X_val_tfidf)

                # Evaluate (convert predictions and true labels to CPU/NumPy for sklearn metric)
                y_pred_val_cpu = cupy.asnumpy(y_pred_val_gpu)
                y_val_fold_cpu = cupy.asnumpy(y_val_fold_cupy)
                score = accuracy_score(y_val_fold_cpu, y_pred_val_cpu)
                fold_scores.append(score)

                # Clean up GPU memory explicitly if needed
                del X_train_fold_cudf, y_train_fold_cupy, X_val_fold_cudf, y_val_fold_cupy
                del X_train_tfidf, X_val_tfidf, y_pred_val_gpu
                del model, tfidf_vectorizer # Delete model/vectorizer from this fold
                cupy.get_default_memory_pool().free_all_blocks()


            except Exception as fold_e:
                 print(f"    Error in Fold {fold_num} for params {params}: {fold_e}")
                 fold_scores.append(0) # Assign 0 score if fold fails


        # Average score across folds for this parameter set
        avg_score = np.mean(fold_scores) if fold_scores else 0
        print(f"  Params: {params} -> Avg CV Score: {avg_score:.4f}")
        # results.append({'params': params, 'score': avg_score}) # Append to results list if needed

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    tuning_time = time.time() - start_time_tuning
    print(f"Manual hyperparameter tuning complete in {tuning_time:.2f} seconds on {train_year} data.")
    if best_params:
        print(f"Best parameters found: {best_params}")
        print(f"Best cross-validation accuracy: {best_score:.4f}")
    else:
        print(f"Warning: No best parameters found for {train_year} (tuning may have failed).")
        return None, None # Return None if tuning failed


    # --- Final Model Training (on Full Train Year Data with Best Params) ---
    print(f"Training final model on full {train_year} data using best parameters...")
    start_time_final_train = time.time()

    # Get full train_year data (including speakers left out for tuning validation)
    # Use the original full train_df_full loaded at the start of the function
    X_train_full_year_pd = train_df_full["speech"]
    y_train_full_year_pd = train_df_full["party"]

    # Encode full year labels
    train_full_year_data_to_encode = pd.DataFrame({'party': y_train_full_year_pd, 'speech': X_train_full_year_pd})
    train_full_year_encoded_df = encode_labels_with_map(train_full_year_data_to_encode, party_map)
    X_train_full_year_pd_aligned = train_full_year_encoded_df['speech'].reset_index(drop=True)
    y_train_full_year_encoded_pd_aligned = train_full_year_encoded_df['label'].reset_index(drop=True)

    if X_train_full_year_pd_aligned.empty or y_train_full_year_encoded_pd_aligned.empty:
        print(f"⚠️  Skipping final training for {train_year} (seed {random_state}): Full train data empty after encoding.")
        return None, None

    # Convert full train data to cuDF/CuPy
    X_train_full_year_cudf = cudf.Series(X_train_full_year_pd_aligned.apply(clean_text_for_tfidf))
    y_train_full_year_cupy = cupy.asarray(y_train_full_year_encoded_pd_aligned.to_numpy(dtype=np.int32))


    # Instantiate final components with best params
    best_tfidf_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('tfidf__')}
    best_model_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('model__')}

    final_tfidf_vectorizer = TfidfVectorizer(**best_tfidf_params)

    if model_type == 'bayes':
        final_model = ComplementNB(**best_model_params)
    elif model_type == 'svm':
        final_model = LinearSVC(**best_model_params)
    elif model_type == 'lr':
        final_model = LogisticRegression(penalty='l2', **best_model_params)
    else:
        # This should not happen if model_type was handled above
        raise ValueError(f"Unknown model type for final model instantiation: {model_type}")


    # Fit final TF-IDF and Model on the full train_year data
    X_train_full_year_tfidf = final_tfidf_vectorizer.fit_transform(X_train_full_year_cudf)
    final_model.fit(X_train_full_year_tfidf, y_train_full_year_cupy)

    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds on {train_year} data.")

    # --- Saving the trained components (TF-IDF Vectorizer and Model) ---
    print(f"Saving the final TF-IDF vectorizer and {model_type.upper()} model for {train_year} (seed {random_state})...")
    tfidf_filename = MODEL_DIR_PATH / f"tfidf_{model_type}_{train_year}_seed{random_state}_vectorizer.joblib"
    model_filename = MODEL_DIR_PATH / f"tfidf_{model_type}_{train_year}_seed{random_state}_model.joblib"
    try:
        joblib.dump(final_tfidf_vectorizer, tfidf_filename)
        joblib.dump(final_model, model_filename)
        print(f"Final TF-IDF vectorizer saved to {tfidf_filename}")
        print(f"Final Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving the final components for {train_year} (seed {random_state}): {e}")
        # Decide if you want to return None here if saving fails
        # For this comprehensive run, it's better to have the objects to evaluate
        # immediately, even if saving failed. But saving is good practice.


    # Clean up full train year data from GPU memory
    del X_train_full_year_cudf, y_train_full_year_cupy, X_train_full_year_tfidf
    cupy.get_default_memory_pool().free_all_blocks()


    # Return the trained components
    return final_tfidf_vectorizer, final_model


# Function to evaluate trained components on data from a specific test year
def evaluate_model_on_year(trained_tfidf_vectorizer, trained_model, test_year: str, model_type: str, party_map: dict):
    """
    Loads data for test_year, uses the pre-trained TF-IDF vectorizer and model
    to predict, calculates metrics, and returns them.
    """
    if trained_tfidf_vectorizer is None or trained_model is None:
        print(f"  Skipping evaluation on {test_year}: Trained components are missing.")
        return None, None, None # Return None for metrics

    print(f"\n  Evaluating model (trained on earlier year) on test data ({test_year})...")
    start_time_test = time.time()

    # --- Data Loading (Test Year) ---
    test_input_path = f"data/merged/house_db/house_merged_{test_year}.csv"
    if not os.path.exists(test_input_path):
        print(f"⚠️  Skipping evaluation on {test_year}: CSV file not found at {test_input_path}.")
        return None, None, None

    print(f"  Loading test data for year {test_year}...")
    test_df_full = pd.read_csv(test_input_path)
    print("  Test data loaded.")

    if test_df_full.empty or not all(col in test_df_full.columns for col in ['speech', 'party', 'speakerid']):
         print(f"⚠️  Skipping evaluation on {test_year}: Data empty or missing required columns.")
         return None, None, None
    test_df_full.dropna(subset=['speech', 'party', 'speakerid'], inplace=True)
    if test_df_full.empty:
        print(f"⚠️  Skipping evaluation on {test_year}: Data empty after NaNs drop.")
        return None, None, None


    # --- Encoding (Test Year) ---
    print("  Encoding labels for test data...")
    test_data_to_encode = pd.DataFrame({'party': test_df_full["party"], 'speech': test_df_full["speech"]})
    test_encoded_df = encode_labels_with_map(test_data_to_encode, party_map)

    X_test_pd_aligned = test_encoded_df['speech'].reset_index(drop=True)
    y_test_encoded_pd_aligned = test_encoded_df['label'].reset_index(drop=True)

    if X_test_pd_aligned.empty or y_test_encoded_pd_aligned.empty:
        print(f"⚠️  Skipping evaluation on {test_year}: Test data empty after encoding & alignment.")
        return None, None, None

    # Convert test data to cuDF/CuPy
    X_test_cudf = cudf.Series(X_test_pd_aligned.apply(clean_text_for_tfidf))
    y_test_encoded_cpu = y_test_encoded_pd_aligned.to_numpy(dtype=np.int32)


    # --- Testing on Test Year Data ---
    # Use the loaded/trained components for transformation and prediction
    try:
        X_test_tfidf = trained_tfidf_vectorizer.transform(X_test_cudf)
        y_test_pred_gpu = trained_model.predict(X_test_tfidf)
        evaluation_time = time.time() - start_time_test
        print(f"  Evaluation complete in {evaluation_time:.2f} seconds.")

        y_test_pred_cpu = cupy.asnumpy(y_test_pred_gpu)

    except Exception as e:
        print(f"  Error during evaluation on {test_year}: {e}")
        return None, None, None # Return None for metrics


    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_test_encoded_cpu, y_test_pred_cpu)
    f1_weighted = f1_score(y_test_encoded_cpu, y_test_pred_cpu, average='weighted')

    # (AUC calculation - needs predict_proba or decision_function from trained_model)
    auc = None
    try:
        if hasattr(trained_model, "predict_proba"):
            probability_scores_gpu = trained_model.predict_proba(X_test_tfidf)
            probability_scores_cpu = cupy.asnumpy(probability_scores_gpu)
            if len(np.unique(y_test_encoded_cpu)) == 2:
                 auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu[:, 1])
            else:
                 from sklearn.preprocessing import LabelBinarizer
                 lb = LabelBinarizer().fit(y_test_encoded_cpu)
                 y_test_encoded_onehot_cpu = lb.transform(y_test_encoded_cpu)
                 if y_test_encoded_onehot_cpu.shape[1] == probability_scores_cpu.shape[1]:
                      auc = roc_auc_score(y_test_encoded_onehot_cpu, probability_scores_cpu, average='macro', multi_class='ovr')
                 elif probability_scores_cpu.ndim == 1 and y_test_encoded_onehot_cpu.shape[1] == 2: # Handle binary case where predict_proba might return 1D array
                      auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu) # type: ignore # Use the raw score for binary
                 else:
                      print(f"  AUC shape mismatch Warning for {model_type} on {test_year}.")


        elif hasattr(trained_model, "decision_function"):
            decision_scores_gpu = trained_model.decision_function(X_test_tfidf)
            decision_scores_cpu = cupy.asnumpy(decision_scores_gpu)
            if len(np.unique(y_test_encoded_cpu)) == 2:
                 auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu)
            else:
                 # Handle multi-class decision_function
                 auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu, multi_class='ovr', average='macro') # type: ignore


    except Exception as e:
        print(f"  Could not calculate ROC-AUC for {model_type} on {test_year}: {e}")
        auc = None

    # Clean up test data from GPU memory
    del X_test_cudf, X_test_tfidf, y_test_pred_gpu
    cupy.get_default_memory_pool().free_all_blocks()

    return accuracy, f1_weighted, auc


# ------ Main Execution for Comprehensive Cross-Year Evaluation -------
if __name__ == "__main__":
    # Generate list of all years to consider for training and testing
    all_years_str = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END + 1)] # Include END year for testing

    models_to_run = ['lr', 'svm', 'bayes'] # Add or remove models here

    # Dictionary to store trained models for re-use across test years
    # Structure: trained_components[seed][train_year][model_type] = (tfidf_vectorizer, model)
    trained_components = {}


    for seed in SEEDS:
        print(f"\n--- Starting comprehensive cross-year runs for seed: {seed} ---")
        trained_components[seed] = {} # Initialize for this seed

        for train_year in all_years_str:
            print(f"\nProcessing Train Year: {train_year} (Seed {seed})")
            trained_components[seed][train_year] = {} # Initialize for this train_year/seed

            # --- Train Models for the current train_year and seed ---
            # This happens ONCE per train_year/seed
            print(f"Training models for train_year: {train_year}")

            for model_type in models_to_run:
                 model_config = model_configs.get(model_type)
                 if model_config is None:
                     print(f"Warning: Configuration for model '{model_type}' not found. Skipping training for {train_year}.")
                     continue

                 # Train the model and get the trained components
                 tfidf_vectorizer, model = train_model_for_year(
                     train_year, model_type, model_config, seed, PARTY_MAP
                 )

                 # Store the trained components if training was successful
                 if tfidf_vectorizer is not None and model is not None:
                     trained_components[seed][train_year][model_type] = (tfidf_vectorizer, model)
                 else:
                     print(f"Skipping evaluation for {model_type} (Train {train_year}) as training failed.")


            # --- Evaluate Trained Models on all subsequent years ---
            print(f"\nEvaluating models trained on {train_year} (Seed {seed}) on subsequent years...")
            try:
                 train_year_index = all_years_str.index(train_year)
            except ValueError:
                 print(f"Error: Train year {train_year} not found in the list of all years. Skipping evaluations for this year.")
                 continue # Skip to next train year

            # Iterate through all years from the train year itself to the end for testing
            # This includes testing on the training year itself, which gives the "within-year" performance
            for test_year in all_years_str[train_year_index:]:
                 print(f"\n  Evaluating on test_year: {test_year} (Train {train_year}, Seed {seed})")

                 for model_type in models_to_run:
                     # Retrieve the trained components for this train_year/seed/model_type
                     components = trained_components[seed][train_year].get(model_type)

                     if components: # Check if components were successfully trained and stored
                         tfidf_vectorizer_trained, model_trained = components

                         # Evaluate the trained model on the current test_year data
                         accuracy, f1_weighted, auc = evaluate_model_on_year(
                             tfidf_vectorizer_trained, model_trained, test_year, model_type, PARTY_MAP
                         )

                         # --- Log results ---
                         if accuracy is not None: # Only log if evaluation was successful
                             print(f"--- Results ({model_type.upper()}): Train {train_year} -> Test {test_year} (Seed {seed}) ---")
                             print(f"Accuracy: {accuracy:.4f}")
                             print(f"Weighted F1 Score: {f1_weighted:.4f}")
                             if auc is not None:
                                print(f"ROC-AUC          : {auc:.4f}")
                             print("-" * 25)

                             with open(COMPREHENSIVE_CROSS_YEAR_LOG_PATH, "a") as f:
                                 f.write(f"{seed},{train_year},{test_year},{accuracy:.4f},{f1_weighted:.4f},{auc if auc is not None else 'NA'}\n")
                         else:
                             print(f"  Skipping logging for {model_type} (Train {train_year} -> Test {test_year}, Seed {seed}) due to evaluation error.")

                     else:
                         print(f"  Skipping evaluation for {model_type} (Train {train_year} -> Test {test_year}, Seed {seed}) as trained components are missing.")


    print("\n--- Comprehensive cross-year evaluation script finished ---")
    print(f"Results saved in the '{COMPREHENSIVE_CROSS_YEAR_LOG_PATH}' file.")

    # Note: You will need a separate script or notebook to load COMPREHENSIVE_CROSS_YEAR_LOG_PATH
    # and generate plots (e.g., heatmap, decay curves) for visualization.