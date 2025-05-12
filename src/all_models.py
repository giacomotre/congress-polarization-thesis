import os
import pandas as pd
import joblib
# import nltk # nltk was imported but not directly used here, pipeline_utils handles its own
import json
import time
import numpy as np
from pathlib import Path
# from collections import Counter # Counter was imported but not directly used

# Import RAPIDS components
import cudf
import cupy

# Import necessary components from sklearn and cuml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
#from sklearn.model_selection import GridSearchCV incompatability with cuML
from sklearn.model_selection import KFold, ParameterGrid
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.naive_bayes import ComplementNB
from cuml.svm import LinearSVC
from cuml.linear_model import LogisticRegression

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map, clean_text_for_tfidf #
from plotting_utils import plot_performance_metrics, plot_confusion_matrix #

# ------ Loading Unified Config -------
CONFIG_PATH_UNIFIED = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

try:
    unified_config = load_config(CONFIG_PATH_UNIFIED)
    print("Loaded unified config:", json.dumps(unified_config, indent=4))
except FileNotFoundError:
    print(f"Error: Unified config file not found at {CONFIG_PATH_UNIFIED}. Please create it.")
    exit()

# --- Extract Common Parameters ---
common_params = unified_config.get('common', {})
split_params = common_params.get('split_params', {})
TEST_SIZE = split_params.get('test_size', 0.15)
VALIDATION_SIZE = split_params.get('validation_size', 0.25)
DEFAULT_RANDOM_STATE = split_params.get('random_state', 42)
SEEDS = split_params.get('seeds', [DEFAULT_RANDOM_STATE]) #

data_params = common_params.get('data_params', {})
CONGRESS_YEAR_START = data_params.get('congress_year_start', 75) #
CONGRESS_YEAR_END = data_params.get('congress_year_end', 112) #

PARTY_MAP = common_params.get('party_map', {}) #
if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
     print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")


# --- Define Model Configurations Dictionary ---
model_configs = {
    'bayes': unified_config.get('bayes', {}), #
    'svm': unified_config.get('svm', {}), #
    'lr': unified_config.get('logistic_regression', {}) #
}

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

detailed_log_paths = {
    'bayes': "logs/tfidf_bayes_performance_detailed.csv",
    'svm': "logs/tfidf_svm_performance_detailed.csv",
    'lr': "logs/tfidf_lr_performance_detailed.csv"
}

# This dictionary will also be used for plot output directories
model_plotting_info = {
    'bayes': {"avg_log_path": detailed_log_paths['bayes'].replace("_detailed.csv", "_avg.csv"), "output_dir": "plots/bayes"},
    'svm': {"avg_log_path": detailed_log_paths['svm'].replace("_detailed.csv", "_avg.csv"), "output_dir": "plots/svm"},
    'lr': {"avg_log_path": detailed_log_paths['lr'].replace("_detailed.csv", "_avg.csv"), "output_dir": "plots/lr"}
}


for model_type, log_path in detailed_log_paths.items():
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing detailed log file: {log_path}")
    with open(log_path, "w") as f:
        f.write("seed,year,accuracy,f1_score,auc\n")

# Function to run a single model pipeline
def run_model_pipeline(
    X_train_pd: pd.Series, y_train_encoded_pd: pd.Series, # ALREADY ENCODED
    X_val_pd: pd.Series, y_val_encoded_pd: pd.Series,   # ALREADY ENCODED
    X_test_pd: pd.Series, y_test_encoded_pd: pd.Series, # ALREADY ENCODED
    model_type: str, model_config: dict, random_state: int, congress_year: str, party_map: dict,
    model_plot_output_dir: str
):
    print(f"\n --- Running {model_type.upper()} pipeline [Manual Tuning] for Congress {congress_year} with seed {random_state} ---")
    timing = {}
    start_time_total = time.time()

    # 1. Define Hyperparameter Grid (using model_config)
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

    # 2. Prepare Full Training Data (Combine ALREADY ENCODED train + val sets)
    if not X_val_pd.empty:
        X_train_val_combined_pd_aligned = pd.concat([X_train_pd, X_val_pd], ignore_index=True)
        # --> Use the already encoded y Series <--
        y_train_val_encoded_pd_aligned = pd.concat([y_train_encoded_pd, y_val_encoded_pd], ignore_index=True)
    else:
        X_train_val_combined_pd_aligned = X_train_pd.copy()
        # --> Use the already encoded y Series <--
        y_train_val_encoded_pd_aligned = y_train_encoded_pd.copy()

    if X_train_val_combined_pd_aligned.empty or y_train_val_encoded_pd_aligned.empty:
         print("Error: Combined train/validation data is empty after encoding.")
         return None # Or handle appropriately

    # 3. Manual Cross-Validation Loop
    n_splits = 5 # Or get from config
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1
    best_params = None
    results = [] # To store results for each param combination

    print(f"Starting manual {n_splits}-fold cross-validation for hyperparameter tuning...")
    start_time_tuning = time.time()

    for params in grid:
        print(f"  Testing params: {params}")
        fold_scores = []

        # Extract params for TF-IDF and Model
        tfidf_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('tfidf__')}
        model_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('model__')}

        fold_num = 0
        for train_idx, val_idx in kf.split(X_train_val_combined_pd_aligned, y_train_val_encoded_pd_aligned):
            fold_num += 1
            # print(f"    Fold {fold_num}/{n_splits}")

            # Get pandas folds for this iteration
            X_train_fold_pd = X_train_val_combined_pd_aligned.iloc[train_idx]
            y_train_fold_pd = y_train_val_encoded_pd_aligned.iloc[train_idx]
            X_val_fold_pd = X_train_val_combined_pd_aligned.iloc[val_idx]
            y_val_fold_pd = y_train_val_encoded_pd_aligned.iloc[val_idx]

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
                    # Ensure correct params are passed, e.g. C from model_params
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
                y_val_fold_cpu = cupy.asnumpy(y_val_fold_cupy) # or use y_val_fold_pd.to_numpy()
                score = accuracy_score(y_val_fold_cpu, y_pred_val_cpu)
                fold_scores.append(score)
                
                # Clean up GPU memory explicitly if needed, though loop replacement should help
                del X_train_fold_cudf, y_train_fold_cupy, X_val_fold_cudf, y_val_fold_cupy
                del X_train_tfidf, X_val_tfidf, y_pred_val_gpu, model, tfidf_vectorizer
                cupy.get_default_memory_pool().free_all_blocks()


            except Exception as fold_e:
                 print(f"    Error in Fold {fold_num} for params {params}: {fold_e}")
                 # Decide how to handle fold errors: append a bad score (0?), skip fold, or stop?
                 fold_scores.append(0) # Example: assign 0 score if fold fails

        # Average score across folds for this parameter set
        avg_score = np.mean(fold_scores) if fold_scores else 0
        print(f"  Params: {params} -> Avg CV Score: {avg_score:.4f}")
        results.append({'params': params, 'score': avg_score})

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    tuning_time = time.time() - start_time_tuning
    print(f"Manual hyperparameter tuning complete in {tuning_time:.2f} seconds.")
    if best_params:
        print(f"Best parameters found: {best_params}")
        print(f"Best cross-validation accuracy: {best_score:.4f}")
    else:
        print("Warning: No best parameters found (tuning may have failed).")
        return None # Or handle error

    # --- Final Model Training (using best_params) ---
    print("Training final model using best parameters found...")
    start_time_final_train = time.time()

    # Prepare full aligned train+val data (already done above)
    X_train_val_cleaned_pd = X_train_val_combined_pd_aligned.apply(clean_text_for_tfidf)
    X_train_val_final_cudf = cudf.Series(X_train_val_cleaned_pd)
    y_train_val_final_cupy = cupy.asarray(y_train_val_encoded_pd_aligned.to_numpy(dtype=np.int32))

    # Instantiate final components with best params
    best_tfidf_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('tfidf__')}
    best_model_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('model__')}

    final_tfidf = TfidfVectorizer(**best_tfidf_params)
    if model_type == 'bayes':
        final_model = ComplementNB(**best_model_params)
    elif model_type == 'svm':
        final_model = LinearSVC(**best_model_params)
    elif model_type == 'lr':
        final_model = LogisticRegression(penalty='l2', **best_model_params)
    else: # Add other models if needed
        raise ValueError(f"Unknown model type: {model_type}") # Should not happen

    # Fit final TF-IDF and Model
    X_train_val_final_tfidf = final_tfidf.fit_transform(X_train_val_final_cudf)
    final_model.fit(X_train_val_final_tfidf, y_train_val_final_cupy)

    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds.")

    # --- Saving the trained pipeline (Now need to save TF-IDF and Model separately) ---
    # Option 1: Save separately
    print(f"Saving the final TF-IDF and {model_type.upper()} model...")
    model_dir_path = Path("models")
    model_dir_path.mkdir(parents=True, exist_ok=True)
    tfidf_filename = model_dir_path / f"tfidf_{model_type}_{congress_year}_seed{random_state}_vectorizer.joblib"
    model_filename = model_dir_path / f"tfidf_{model_type}_{congress_year}_seed{random_state}_model.joblib"
    try:
        joblib.dump(final_tfidf, tfidf_filename)
        joblib.dump(final_model, model_filename)
        print(f"Final TF-IDF saved to {tfidf_filename}")
        print(f"Final Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving the final components: {e}")

    # Option 2: Create a simple dictionary or custom wrapper class to hold both and save that
    # final_pipeline_components = {'tfidf': final_tfidf, 'model': final_model}
    # joblib.dump(final_pipeline_components, ...)

    # --- Testing on Test Data ---
    print("Preparing test data for evaluation...")
    # Encode test labels
    test_data_to_encode = pd.DataFrame({'party': y_test_pd, 'speech': X_test_pd})
    test_encoded_df = encode_labels_with_map(test_data_to_encode, party_map)
    X_test_pd_aligned = test_encoded_df['speech'].reset_index(drop=True)
    y_test_encoded_pd_aligned = test_encoded_df['label'].reset_index(drop=True)

    if X_test_pd_aligned.empty:
        print("Error: Test data is empty after encoding.")
        return None # Or handle appropriately

    X_test_cleaned_pd = X_test_pd_aligned.apply(clean_text_for_tfidf)
    X_test_cudf = cudf.Series(X_test_cleaned_pd)
    y_test_encoded_cpu = y_test_encoded_pd_aligned.to_numpy(dtype=np.int32)

    print("Evaluating final model on test cuDF data...")
    start_time_test = time.time()
    # Use the saved/trained final components
    X_test_final_tfidf = final_tfidf.transform(X_test_cudf)
    y_test_pred_gpu = final_model.predict(X_test_final_tfidf)
    evaluation_time = time.time() - start_time_test
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    y_test_pred_cpu = cupy.asnumpy(y_test_pred_gpu)

    # --- Calculate Metrics ---
    final_accuracy = accuracy_score(y_test_encoded_cpu, y_test_pred_cpu)
    final_f1_weighted = f1_score(y_test_encoded_cpu, y_test_pred_cpu, average='weighted')
    print(f"\n--- Final Test Results ({model_type.upper()}) ---")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted:.4f}")

    # (AUC calculation - needs predict_proba or decision_function from final_model)
    auc = None
    try:
        if hasattr(final_model, "predict_proba"):
            probability_scores_gpu = final_model.predict_proba(X_test_final_tfidf)
            probability_scores_cpu = cupy.asnumpy(probability_scores_gpu)
            # ... (rest of AUC logic) ...
            if len(np.unique(y_test_encoded_cpu)) == 2: auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu[:, 1]) # type: ignore
            else:
                 from sklearn.preprocessing import LabelBinarizer
                 lb = LabelBinarizer().fit(y_test_encoded_cpu); y_test_encoded_onehot_cpu = lb.transform(y_test_encoded_cpu)
                 if y_test_encoded_onehot_cpu.shape[1] == probability_scores_cpu.shape[1]: auc = roc_auc_score(y_test_encoded_onehot_cpu, probability_scores_cpu, average='macro', multi_class='ovr')
                 elif probability_scores_cpu.ndim == 1 and y_test_encoded_onehot_cpu.shape[1] == 2 : auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu) # type: ignore
                 else: print(f"AUC shape mismatch Warning.")

        elif hasattr(final_model, "decision_function"):
            decision_scores_gpu = final_model.decision_function(X_test_final_tfidf)
            decision_scores_cpu = cupy.asnumpy(decision_scores_gpu)
            # ... (rest of AUC logic) ...
            if len(np.unique(y_test_encoded_cpu)) == 2: auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu) # type: ignore
            else: auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu, multi_class='ovr', average='macro') # type: ignore

    except Exception as e: print(f"Could not calculate ROC-AUC: {e}"); auc = None

    # --- Logging ---
    # (Confusion matrix, classification report, logging remain similar)
    # ...

    # ---> ADD THIS BLOCK BACK <---
    print("Calculating Confusion Matrix...")
    cm_cpu = confusion_matrix(y_test_encoded_cpu, y_test_pred_cpu)
    cm_list = cm_cpu.tolist() # For JSON

    try:
        reverse_party_map = {v: k for k, v in party_map.items()}
        unique_labels_in_test_cpu = sorted(list(np.unique(y_test_encoded_cpu)))
        target_names = [reverse_party_map.get(i, str(i)) for i in unique_labels_in_test_cpu]
    except Exception as e:
        print(f"Could not get target names: {e}")
        target_names = [str(i) for i in sorted(list(np.unique(y_test_encoded_cpu)))]

    if auc is not None: print(f"ROC-AUC          : {auc:.4f}") # Already there
    print("Confusion Matrix:\n", cm_cpu) # Add this print
    try:
        print("\nClassification Report:") # Already there
        print(classification_report(y_test_encoded_cpu, y_test_pred_cpu, target_names=target_names, zero_division=0)) # Already there
    except Exception as e:
        print(f"Could not print Classification Report: {e}") # Already there
    # ---> END OF BLOCK TO ADD BACK <---


    print("-" * 25)
    current_detailed_log_path = detailed_log_paths[model_type]
    with open(current_detailed_log_path, "a") as f:
         f.write(f"{random_state},{congress_year},{final_accuracy:.4f},{final_f1_weighted:.4f},{auc if auc is not None else 'NA'}\n")

    # (JSON logging - adapt structure if needed since pipeline isn't saved directly)
    result_json = {
         "seed": random_state, "year": congress_year, "accuracy": round(final_accuracy, 4),
         "f1_score": round(final_f1_weighted, 4), "auc": round(auc, 4) if auc is not None else "NA",
         # "confusion_matrix": cm_list, "labels": target_names, # Add back CM calculation if needed
         "best_params": best_params, # Log best params found
         "timing": {
             "tuning_sec": round(tuning_time, 2), "final_train_sec": round(final_train_time, 2),
             "evaluation_sec": round(evaluation_time, 2), "total_pipeline_sec": round(time.time() - start_time_total, 2)
         }
     }
    # ... (Save JSON, plot confusion matrix if calculated)

    # Clean up final model components from memory
    del X_train_val_final_cudf, y_train_val_final_cupy, X_train_val_final_tfidf
    del X_test_cudf, X_test_final_tfidf, y_test_pred_gpu
    del final_tfidf, final_model
    cupy.get_default_memory_pool().free_all_blocks()


    return result_json # Or return path to saved models/results

# (The rest of the __main__ block in all_models.py remains the same)

# ------ Main Execution -------
if __name__ == "__main__":
    # Ensure end year from config is inclusive for the range
    congress_years_to_process = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END + 1)] #
    models_to_run = ['bayes', 'svm', 'lr']

    for seed in SEEDS: #
        print(f"\n--- Starting runs for seed: {seed} ---")
        for year_str in congress_years_to_process:
            print(f"\nProcessing Congress Year: {year_str}")
            input_csv_path = Path(f"data/merged/house_db/house_merged_{year_str}.csv")

            if not input_csv_path.exists():
                print(f"⚠️  Skipping Congress {year_str} (seed {seed}): CSV file not found at {input_csv_path}.")
                continue
            try:
                print("Loading data...")
                df_full = pd.read_csv(input_csv_path)
                print(f"Data loaded. Shape: {df_full.shape}")

                if df_full.empty or not all(col in df_full.columns for col in ['speech', 'party', 'speakerid']):
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Data empty or missing required columns.")
                    continue
                df_full.dropna(subset=['speech', 'party', 'speakerid'], inplace=True)
                if df_full.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Data empty after NaNs drop.")
                    continue

                print("Performing leave-out speaker split...")
                start_time_split = time.time()
                unique_speakers = df_full['speakerid'].unique()

                if len(unique_speakers) < 2: # Min for train_test_split
                     print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Not enough unique speakers ({len(unique_speakers)}).")
                     continue
                
                # Adjust test_size if number of speakers is very small
                current_test_size = TEST_SIZE if len(unique_speakers) * TEST_SIZE >= 1 else 1/len(unique_speakers)

                train_val_speakers, test_speakers = train_test_split(
                    unique_speakers, test_size=current_test_size, random_state=seed)

                # Adjust validation_size if number of train_val_speakers is very small
                if len(train_val_speakers) < 2 : # Cannot split for validation
                    train_speakers = train_val_speakers
                    val_speakers = np.array([]) # Empty array for val_speakers
                else:
                    current_validation_size = VALIDATION_SIZE if len(train_val_speakers) * VALIDATION_SIZE >= 1 else 1/len(train_val_speakers)
                    train_speakers, val_speakers = train_test_split(
                        train_val_speakers, test_size=current_validation_size, random_state=seed)
                
                train_df = df_full[df_full["speakerid"].isin(train_speakers)].reset_index(drop=True)
                val_df = df_full[df_full["speakerid"].isin(val_speakers)].reset_index(drop=True) if len(val_speakers) > 0 else pd.DataFrame(columns=df_full.columns)
                test_df = df_full[df_full["speakerid"].isin(test_speakers)].reset_index(drop=True)

                split_time = time.time() - start_time_split
                print(f"Split complete in {split_time:.2f} secs. Train spk: {len(train_speakers)}, Val spk: {len(val_speakers)}, Test spk: {len(test_speakers)}")
                print(f"  - Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

                if train_df.empty or test_df.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Train or Test DataFrame is empty after split.")
                    continue

                # Initial X, y from pandas DataFrames
                X_train_pd_orig = train_df["speech"]
                y_train_pd_orig = train_df["party"]
                X_val_pd_orig = val_df["speech"] if not val_df.empty else pd.Series(dtype='object')
                y_val_pd_orig = val_df["party"] if not val_df.empty else pd.Series(dtype='object')
                X_test_pd_orig = test_df["speech"]
                y_test_pd_orig = test_df["party"]

                print("Encoding labels and aligning X data with filtered labels...") #
                start_time_encode = time.time()

                # --- Data Alignment after encode_labels_with_map ---
                # Train
                train_data_to_encode = pd.DataFrame({'party': y_train_pd_orig, 'speech': X_train_pd_orig})
                train_encoded_df = encode_labels_with_map(train_data_to_encode, PARTY_MAP) #
                X_train_pd_aligned = train_encoded_df['speech'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='object')
                y_train_encoded_pd_aligned = train_encoded_df['label'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='int')

                # Validation
                if not X_val_pd_orig.empty:
                    val_data_to_encode = pd.DataFrame({'party': y_val_pd_orig, 'speech': X_val_pd_orig})
                    val_encoded_df = encode_labels_with_map(val_data_to_encode, PARTY_MAP) #
                    X_val_pd_aligned = val_encoded_df['speech'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = val_encoded_df['label'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='int')
                else:
                    X_val_pd_aligned = pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = pd.Series(dtype='int')
                
                # Test
                test_data_to_encode = pd.DataFrame({'party': y_test_pd_orig, 'speech': X_test_pd_orig})
                test_encoded_df = encode_labels_with_map(test_data_to_encode, PARTY_MAP) #
                X_test_pd_aligned = test_encoded_df['speech'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='object')
                y_test_encoded_pd_aligned = test_encoded_df['label'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='int')
                
                encode_time = time.time() - start_time_encode
                print(f"Labels encoded and X data aligned in {encode_time:.2f} seconds.")
                print(f"  Post-encoding/alignment - Train: {len(X_train_pd_aligned)}, Val: {len(X_val_pd_aligned)}, Test: {len(X_test_pd_aligned)}")


                if X_train_pd_aligned.empty or y_train_encoded_pd_aligned.empty or X_test_pd_aligned.empty or y_test_encoded_pd_aligned.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Train or Test data empty after label encoding & alignment.")
                    continue
                if len(np.unique(y_train_encoded_pd_aligned)) < 2 :
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Fewer than 2 unique classes in training labels after encoding. GridSearchCV requires at least 2.")
                    continue

                for model_type_to_run in models_to_run:
                    current_model_config = model_configs.get(model_type_to_run)
                    if current_model_config is None:
                        print(f"Warning: Config for model '{model_type_to_run}' not found. Skipping.")
                        continue
                    
                    # Get the specific plot output directory for this model
                    current_model_plot_dir = model_plotting_info[model_type_to_run]["output_dir"]
                    Path(current_model_plot_dir).mkdir(parents=True, exist_ok=True)


                    run_model_pipeline(
                        X_train_pd_aligned, y_train_encoded_pd_aligned,
                        X_val_pd_aligned, y_val_encoded_pd_aligned,
                        X_test_pd_aligned, y_test_encoded_pd_aligned,
                        model_type=model_type_to_run,
                        model_config=current_model_config,
                        random_state=seed,
                        congress_year=year_str,
                        party_map=PARTY_MAP,
                        model_plot_output_dir=current_model_plot_dir
                    )
            except Exception as e:
                print(f"❌ An error occurred during processing for Congress {year_str} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    print("\n--- Calculating and plotting averaged results ---")
    for model_type_to_plot in models_to_run:
        plot_info = model_plotting_info.get(model_type_to_plot)
        if plot_info is None:
             print(f"Warning: Plotting info for model '{model_type_to_plot}' not found. Skipping.")
             continue

        detailed_log_for_avg = detailed_log_paths[model_type_to_plot]
        avg_log_path_for_plot = plot_info["avg_log_path"]
        output_dir_for_plot = plot_info["output_dir"]
        Path(output_dir_for_plot).mkdir(parents=True, exist_ok=True)

        try:
            if not Path(detailed_log_for_avg).exists() or Path(detailed_log_for_avg).stat().st_size == 0:
                print(f"Error: Detailed log empty/not found for {model_type_to_plot.upper()} at {detailed_log_for_avg}.")
                continue
            
            df_detailed_metrics = pd.read_csv(detailed_log_for_avg)
            if df_detailed_metrics.empty or 'year' not in df_detailed_metrics.columns:
                 print(f"Error: Detailed log for {model_type_to_plot.upper()} empty/malformed after reading.")
                 continue

            df_detailed_metrics['auc'] = pd.to_numeric(df_detailed_metrics['auc'], errors='coerce')
            df_avg_metrics = df_detailed_metrics.groupby('year')[['accuracy', 'f1_score', 'auc']].mean(numeric_only=True).reset_index()
            
            if df_avg_metrics.empty:
                print(f"Warning: No data to average for {model_type_to_plot.upper()}. Skipping plotting.")
                continue

            df_std_metrics = df_detailed_metrics.groupby('year')[['accuracy', 'f1_score', 'auc']].std(numeric_only=True).reset_index().rename(
                columns={'accuracy':'accuracy_std', 'f1_score':'f1_score_std', 'auc':'auc_std'})
            df_avg_metrics = df_avg_metrics.merge(df_std_metrics, on='year', how='left')

            try: # Attempt to sort by year as integer
                df_avg_metrics['year_int'] = df_avg_metrics['year'].astype(int)
                df_avg_metrics = df_avg_metrics.sort_values('year_int').drop('year_int', axis=1)
            except ValueError: # Fallback to string sort if 'year' cannot be int
                print(f"Warning: 'year' column for {model_type_to_plot.upper()} not purely numeric. Sorting as string.")
                df_avg_metrics = df_avg_metrics.sort_values('year')

            df_avg_metrics.to_csv(avg_log_path_for_plot, index=False)
            print(f"Saved averaged metrics for {model_type_to_plot.upper()} to {avg_log_path_for_plot}")

            print(f"\nGenerating performance plots for {model_type_to_plot.upper()} using {avg_log_path_for_plot}...")
            plot_performance_metrics(avg_log_path_for_plot, output_dir=output_dir_for_plot) #

        except FileNotFoundError:
            print(f"Error: Detailed performance log not found for {model_type_to_plot.upper()} at {detailed_log_for_avg}.")
        except Exception as e:
            print(f"An error occurred during avg calculation or plotting for {model_type_to_plot.upper()}: {e}")
            import traceback
            traceback.print_exc()
    print("\n--- Script finished ---")