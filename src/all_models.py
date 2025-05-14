import os
import pandas as pd
import joblib
import json
import time
import numpy as np
from pathlib import Path

# Import RAPIDS components
import cudf
import cupy

from sklearn.pipeline import Pipeline # Pipeline is imported but not directly used for model definition here
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.model_selection import KFold, ParameterGrid
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.naive_bayes import ComplementNB
from cuml.svm import LinearSVC
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score as cuml_accuracy_score
from cuml.metrics import confusion_matrix as cuml_confusion_matrix
from cuml.metrics.roc_auc import roc_auc_score as cuml_roc_auc_score

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map 
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

# --- Define model function ---
def run_model_pipeline(
    X_train_pd_cleaned: pd.Series, y_train_encoded_pd: pd.Series, # Assuming X_train_pd_cleaned is ALREADY CLEANED
    X_val_pd_cleaned: pd.Series, y_val_encoded_pd: pd.Series,   # Assuming X_val_pd_cleaned is ALREADY CLEANED
    X_test_pd_cleaned: pd.Series, y_test_encoded_pd: pd.Series, # Assuming X_test_pd_cleaned is ALREADY CLEANED
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

    # 2. Prepare Full Training Data (Combine ALREADY CLEANED & ENCODED train + val sets)
    if not X_val_pd_cleaned.empty:
        X_train_val_combined_pd_cleaned = pd.concat([X_train_pd_cleaned, X_val_pd_cleaned], ignore_index=True)
        y_train_val_encoded_pd_aligned = pd.concat([y_train_encoded_pd, y_val_encoded_pd], ignore_index=True)
    else:
        X_train_val_combined_pd_cleaned = X_train_pd_cleaned.copy()
        y_train_val_encoded_pd_aligned = y_train_encoded_pd.copy()

    if X_train_val_combined_pd_cleaned.empty or y_train_val_encoded_pd_aligned.empty:
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
        for train_idx, val_idx in kf.split(X_train_val_combined_pd_cleaned, y_train_val_encoded_pd_aligned):
            fold_num += 1
            # print(f"       Fold {fold_num}/{n_splits}")

            # Get pandas folds for this iteration (text is already cleaned)
            X_train_fold_pd = X_train_val_combined_pd_cleaned.iloc[train_idx]
            y_train_fold_pd = y_train_val_encoded_pd_aligned.iloc[train_idx]
            X_val_fold_pd = X_train_val_combined_pd_cleaned.iloc[val_idx]
            y_val_fold_pd = y_train_val_encoded_pd_aligned.iloc[val_idx]

            try:
                # Text is already cleaned, directly convert to cuDF
                X_train_fold_cudf = cudf.Series(X_train_fold_pd)
                y_train_fold_cupy = cupy.asarray(y_train_fold_pd.to_numpy(dtype=np.int32))
                X_val_fold_cudf = cudf.Series(X_val_fold_pd)
                y_val_fold_cupy = cupy.asarray(y_val_fold_pd.to_numpy(dtype=np.int32))

                # Instantiate cuML components with current params
                tfidf_vectorizer = TfidfVectorizer(**tfidf_params)

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

                # Evaluate
                y_pred_val_cpu = cupy.asnumpy(y_pred_val_gpu)
                y_val_fold_cpu = cupy.asnumpy(y_val_fold_cupy)
                score = cuml_accuracy_score(y_val_fold_cpu, y_pred_val_cpu)
                fold_scores.append(score)
                
                del X_train_fold_cudf, y_train_fold_cupy, X_val_fold_cudf, y_val_fold_cupy
                del X_train_tfidf, X_val_tfidf, y_pred_val_gpu, model, tfidf_vectorizer
                cupy.get_default_memory_pool().free_all_blocks()

            except Exception as fold_e:
                print(f"       Error in Fold {fold_num} for params {params}: {fold_e}")
                fold_scores.append(0) # Assign 0 score if fold fails

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
        return None

    # --- Final Model Training (using best_params and ALREADY CLEANED combined data) ---
    print("Training final model using best parameters found...")
    start_time_final_train = time.time()

    # Data is X_train_val_combined_pd_cleaned and y_train_val_encoded_pd_aligned
    X_train_val_final_cudf = cudf.Series(X_train_val_combined_pd_cleaned)
    y_train_val_final_cupy = cupy.asarray(y_train_val_encoded_pd_aligned.to_numpy(dtype=np.int32))

    best_tfidf_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('tfidf__')}
    best_model_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('model__')}

    final_tfidf = TfidfVectorizer(**best_tfidf_params)
    if model_type == 'bayes':
        final_model = ComplementNB(**best_model_params)
    elif model_type == 'svm':
        final_model = LinearSVC(**best_model_params)
    elif model_type == 'lr':
        final_model = LogisticRegression(penalty='l2', **best_model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    X_train_val_final_tfidf = final_tfidf.fit_transform(X_train_val_final_cudf)
    final_model.fit(X_train_val_final_tfidf, y_train_val_final_cupy)

    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds.")

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

    # --- Testing on Test Data (X_test_pd_cleaned is ALREADY CLEANED) ---
    print("Preparing test data for evaluation...")
    # y_test_encoded_pd comes in as already encoded from main.
    # X_test_pd_cleaned comes in as already cleaned from main.

    if X_test_pd_cleaned.empty: # X_test_pd_cleaned is already aligned and cleaned
        print("Error: Test data (X_test_pd_cleaned) is empty.")
        return None

    # Test text is already cleaned
    X_test_cudf = cudf.Series(X_test_pd_cleaned)
    y_test_encoded_cpu = y_test_encoded_pd.to_numpy(dtype=np.int32) # y_test_encoded_pd is passed in

    print("Evaluating final model on test cuDF data...")
    start_time_test = time.time()
    X_test_final_tfidf = final_tfidf.transform(X_test_cudf)
    y_test_pred_gpu = final_model.predict(X_test_final_tfidf)
    evaluation_time = time.time() - start_time_test
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    y_test_pred_cpu = cupy.asnumpy(y_test_pred_gpu)

    final_accuracy = cuml_accuracy_score(y_test_encoded_cpu, y_test_pred_cpu)
    final_f1_weighted = f1_score(y_test_encoded_cpu, y_test_pred_cpu, average='weighted')
    print(f"\n--- Final Test Results ({model_type.upper()}) ---")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted:.4f}")

    auc = None
    try:
        if hasattr(final_model, "predict_proba"):
            probability_scores_gpu = final_model.predict_proba(X_test_final_tfidf)
            probability_scores_cpu = cupy.asnumpy(probability_scores_gpu)
            if len(np.unique(y_test_encoded_cpu)) == 2: auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu[:, 1])
            else:
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer().fit(y_test_encoded_cpu); y_test_encoded_onehot_cpu = lb.transform(y_test_encoded_cpu)
                if y_test_encoded_onehot_cpu.shape[1] == probability_scores_cpu.shape[1]: auc = roc_auc_score(y_test_encoded_onehot_cpu, probability_scores_cpu, average='macro', multi_class='ovr')
                elif probability_scores_cpu.ndim == 1 and y_test_encoded_onehot_cpu.shape[1] == 2 : auc = roc_auc_score(y_test_encoded_cpu, probability_scores_cpu)
                else: print(f"AUC shape mismatch Warning.")
        elif hasattr(final_model, "decision_function"):
            decision_scores_gpu = final_model.decision_function(X_test_final_tfidf)
            decision_scores_cpu = cupy.asnumpy(decision_scores_gpu)
            if len(np.unique(y_test_encoded_cpu)) == 2: auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu)
            else: auc = roc_auc_score(y_test_encoded_cpu, decision_scores_cpu, multi_class='ovr', average='macro')
    except Exception as e: print(f"Could not calculate ROC-AUC: {e}"); auc = None

    print("Calculating Confusion Matrix...")
    cm_gpu = cuml_confusion_matrix(y_test_encoded_cpu, y_test_pred_cpu)
    cm_cpu = cupy.asnumpy(cm_gpu) # Only convert to CPU if needed for printing/sklearn.classification_report
    cm_list = cm_cpu.tolist()

    try:
        reverse_party_map = {v: k for k, v in party_map.items()}
        unique_labels_in_test_cpu = sorted(list(np.unique(y_test_encoded_cpu)))
        target_names = [reverse_party_map.get(i, str(i)) for i in unique_labels_in_test_cpu]
    except Exception as e:
        print(f"Could not get target names: {e}")
        target_names = [str(i) for i in sorted(list(np.unique(y_test_encoded_cpu)))]

    if auc is not None: print(f"ROC-AUC          : {auc:.4f}")
    print("Confusion Matrix:\n", cm_cpu)
    try:
        print("\nClassification Report:")
        print(classification_report(y_test_encoded_cpu, y_test_pred_cpu, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"Could not print Classification Report: {e}")

    print("-" * 25)
    current_detailed_log_path = detailed_log_paths[model_type]
    with open(current_detailed_log_path, "a") as f:
        f.write(f"{random_state},{congress_year},{final_accuracy:.4f},{final_f1_weighted:.4f},{auc if auc is not None else 'NA'}\n")

    result_json = {
        "seed": random_state, "year": congress_year, "accuracy": round(final_accuracy, 4),
        "f1_score": round(final_f1_weighted, 4), "auc": round(auc, 4) if auc is not None else "NA",
        "confusion_matrix": cm_list, "labels": target_names, # Added back
        "best_params": best_params,
        "timing": {
            "tuning_sec": round(tuning_time, 2), "final_train_sec": round(final_train_time, 2),
            "evaluation_sec": round(evaluation_time, 2), "total_pipeline_sec": round(time.time() - start_time_total, 2)
        }
    }
    # Save JSON for this run (optional, good for detailed tracking)
    # json_log_dir = Path("logs/json_results") / model_type / year_str
    # json_log_dir.mkdir(parents=True, exist_ok=True)
    # json_log_path = json_log_dir / f"results_seed{random_state}.json"
    # with open(json_log_path, "w") as jf:
    #    json.dump(result_json, jf, indent=4)
    # print(f"Saved detailed JSON results to {json_log_path}")

    # Plot confusion matrix for this specific run
    # plot_confusion_matrix(cm_cpu, target_names, model_plot_output_dir,
    #                       filename_prefix=f"{model_type}_{congress_year}_seed{random_state}_cm")


    del X_train_val_final_cudf, y_train_val_final_cupy, X_train_val_final_tfidf
    del X_test_cudf, X_test_final_tfidf, y_test_pred_gpu
    del final_tfidf, final_model
    cupy.get_default_memory_pool().free_all_blocks()

    return result_json




# ------ Main Execution -------
if __name__ == "__main__":
    congress_years_to_process = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END + 1)]
    models_to_run = ['bayes', 'svm', 'lr']

    for seed in SEEDS:
        print(f"\n--- Starting runs for seed: {seed} ---")
        for year_str in congress_years_to_process:
            print(f"\nProcessing Congress Year: {year_str}")
            input_csv_path = Path(f"data/merged/house_db/house_cleaned_{year_str}.csv")

            if not input_csv_path.exists():
                print(f"⚠️  Skipping Congress {year_str} (seed {seed}): CSV file not found at {input_csv_path}.")
                continue
            try:
                print("Loading data...")
                # ASSUMPTION: 'speech' column in this CSV is ALREADY CLEANED
                df_full = pd.read_csv(input_csv_path)
                print(f"Data loaded. Shape: {df_full.shape}")

                if df_full.empty or not all(col in df_full.columns for col in ['speech', 'party', 'speakerid']):
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Data empty or missing required columns.")
                    continue
                df_full.dropna(subset=['speech', 'party', 'speakerid'], inplace=True)
                if df_full.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Data empty after NaNs drop.")
                    continue
                
                # --- Leave-out-speaker split ---
                print("Performing leave-out speaker split...")
                start_time_split = time.time()
                unique_speakers = df_full['speakerid'].unique()

                if len(unique_speakers) < 2:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Not enough unique speakers ({len(unique_speakers)}).")
                    continue
                
                current_test_size = TEST_SIZE if len(unique_speakers) * TEST_SIZE >= 1 else 1/len(unique_speakers)
                train_val_speakers, test_speakers = train_test_split(
                    unique_speakers, test_size=current_test_size, random_state=seed)

                if len(train_val_speakers) < 2 :
                    train_speakers = train_val_speakers
                    val_speakers = np.array([])
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

                X_train_pd_cleaned_orig = train_df["speech"]
                y_train_pd_orig = train_df["party"]
                X_val_pd_cleaned_orig = val_df["speech"] if not val_df.empty else pd.Series(dtype='object')
                y_val_pd_orig = val_df["party"] if not val_df.empty else pd.Series(dtype='object')
                X_test_pd_cleaned_orig = test_df["speech"]
                y_test_pd_orig = test_df["party"]
                
                # --- Encoding ---
                print("Encoding labels and aligning X data (already cleaned) with filtered labels...")
                start_time_encode = time.time()

                train_data_to_encode = pd.DataFrame({'party': y_train_pd_orig, 'speech': X_train_pd_cleaned_orig})
                train_encoded_df = encode_labels_with_map(train_data_to_encode, PARTY_MAP)
                X_train_pd_cleaned_aligned = train_encoded_df['speech'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='object')
                y_train_encoded_pd_aligned = train_encoded_df['label'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='int')

                if not X_val_pd_cleaned_orig.empty:
                    val_data_to_encode = pd.DataFrame({'party': y_val_pd_orig, 'speech': X_val_pd_cleaned_orig})
                    val_encoded_df = encode_labels_with_map(val_data_to_encode, PARTY_MAP)
                    X_val_pd_cleaned_aligned = val_encoded_df['speech'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = val_encoded_df['label'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='int')
                else:
                    X_val_pd_cleaned_aligned = pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = pd.Series(dtype='int')
                
                test_data_to_encode = pd.DataFrame({'party': y_test_pd_orig, 'speech': X_test_pd_cleaned_orig})
                test_encoded_df = encode_labels_with_map(test_data_to_encode, PARTY_MAP)
                X_test_pd_cleaned_aligned = test_encoded_df['speech'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='object')
                y_test_encoded_pd_aligned = test_encoded_df['label'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='int')
                
                encode_time = time.time() - start_time_encode
                print(f"Labels encoded and X data aligned in {encode_time:.2f} seconds.")
                print(f"  Post-encoding/alignment - Train: {len(X_train_pd_cleaned_aligned)}, Val: {len(X_val_pd_cleaned_aligned)}, Test: {len(X_test_pd_cleaned_aligned)}")

                if X_train_pd_cleaned_aligned.empty or y_train_encoded_pd_aligned.empty or X_test_pd_cleaned_aligned.empty or y_test_encoded_pd_aligned.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Train or Test data empty after label encoding & alignment.")
                    continue
                if len(np.unique(y_train_encoded_pd_aligned)) < 2 :
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Fewer than 2 unique classes in training labels after encoding. Model fitting requires at least 2.")
                    continue
                
                #--- Loop for GridSearch ---
                for model_type_to_run in models_to_run:
                    current_model_config = model_configs.get(model_type_to_run)
                    if current_model_config is None:
                        print(f"Warning: Config for model '{model_type_to_run}' not found. Skipping.")
                        continue
                    
                    current_model_plot_dir = model_plotting_info[model_type_to_run]["output_dir"]
                    Path(current_model_plot_dir).mkdir(parents=True, exist_ok=True)

                    # --- Actual Processing --- 
                    run_model_pipeline(
                        X_train_pd_cleaned_aligned, y_train_encoded_pd_aligned,
                        X_val_pd_cleaned_aligned, y_val_encoded_pd_aligned,
                        X_test_pd_cleaned_aligned, y_test_encoded_pd_aligned,
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

            try: 
                df_avg_metrics['year_int'] = df_avg_metrics['year'].astype(int)
                df_avg_metrics = df_avg_metrics.sort_values('year_int').drop('year_int', axis=1)
            except ValueError: 
                print(f"Warning: 'year' column for {model_type_to_plot.upper()} not purely numeric. Sorting as string.")
                df_avg_metrics = df_avg_metrics.sort_values('year')

            df_avg_metrics.to_csv(avg_log_path_for_plot, index=False)
            print(f"Saved averaged metrics for {model_type_to_plot.upper()} to {avg_log_path_for_plot}")

            print(f"\nGenerating performance plots for {model_type_to_plot.upper()} using {avg_log_path_for_plot}...")
            plot_performance_metrics(avg_log_path_for_plot, output_dir=output_dir_for_plot)

        except FileNotFoundError:
            print(f"Error: Detailed performance log not found for {model_type_to_plot.upper()} at {detailed_log_for_avg}.")
        except Exception as e:
            print(f"An error occurred during avg calculation or plotting for {model_type_to_plot.upper()}: {e}")
            import traceback
            traceback.print_exc()
    print("\n--- Script finished ---")