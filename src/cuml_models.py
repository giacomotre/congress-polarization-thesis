import os
os.environ['CUDF_USE_KVIKIO'] = '0' # Disable KvikIO/cuFile

import pandas as pd
import joblib
import json
import time
import pickle
import numpy as np
from pathlib import Path

# Import RAPIDS components
import cudf
import cupy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ParameterGrid

#cuml components
from cuml.feature_extraction.text import TfidfVectorizer # Keeping cuML TfidfVectorizer
from cuml.naive_bayes import ComplementNB
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score as cuml_accuracy_score
from cuml.metrics import confusion_matrix as cuml_confusion_matrix
from cuml.metrics import roc_auc_score as cuml_roc_auc_score # Add this
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import classification_report # Add this line


# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map
from plotting_utils import plot_performance_metrics

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
SEEDS = split_params.get('seeds', [DEFAULT_RANDOM_STATE])

data_params = common_params.get('data_params', {})
CONGRESS_YEAR_START = data_params.get('congress_year_start', 75)
CONGRESS_YEAR_END = data_params.get('congress_year_end', 112)

PARTY_MAP = common_params.get('party_map', {})
if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
    print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")


# --- Define Model Configurations Dictionary ---
model_configs = {
    'bayes': unified_config.get('bayes', {}),
    'lr': unified_config.get('logistic_regression', {})
}

#define outputs path 
os.makedirs("logs", exist_ok=True)

detailed_log_paths = {
    'bayes': "logs/tfidf_bayes_performance_detailed.csv",
    'lr': "logs/tfidf_lr_performance_detailed.csv"
}

model_plotting_info = {
    'bayes': {"avg_log_path": detailed_log_paths['bayes'].replace("_detailed.csv", "_avg.csv"), "output_dir": "plots/bayes"},
    'lr': {"avg_log_path": detailed_log_paths['lr'].replace("_detailed.csv", "_avg.csv"), "output_dir": "plots/lr"}
}

timing_log_paths = {
    'bayes': "logs/bayes_timing_log.csv",
    'lr': "logs/lr_timing_log.csv"
}

# detail csv file header
detailed_csv_header_columns = ["seed", "year", "accuracy", "f1_score", "auc"]
detailed_csv_header = ",".join(detailed_csv_header_columns) + "\n"

for model_type, log_path in detailed_log_paths.items():
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing detailed log file: {log_path}")
    with open(log_path, "w") as f:
        f.write(detailed_csv_header)

#timing csv header
timing_csv_header_columns = [
    "seed", "year",
    "timing_tuning_sec", "timing_final_train_sec",
    "timing_evaluation_sec", "timing_total_pipeline_sec"
]
timing_csv_header = ",".join(timing_csv_header_columns) + "\n"

for model_type, log_path in timing_log_paths.items(): # Using your globally defined timing_log_paths
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing timing log file: {log_path}")
    with open(log_path, "w") as f:
        f.write(timing_csv_header)

# --- Define model function ---
def run_model_pipeline(
    X_train_pd: pd.Series, y_train_encoded_pd: pd.Series,
    X_val_pd: pd.Series, y_val_encoded_pd: pd.Series,
    X_test_pd: pd.Series, y_test_encoded_pd: pd.Series,
    model_type: str, model_config: dict, random_state: int, congress_year: str, party_map: dict,
    model_plot_output_dir: str, 
    fixed_vocabulary_dict: cudf.Series
):
    print(f"\n --- Running {model_type.upper()} pipeline [Manual Tuning] for Congress {congress_year} with seed {random_state} ---")
    timing = {}
    start_time_total = time.time()
    
    # loading optimization config
    model_specific_grid = {}
    
    param_combinations = {
        'tfidf__use_idf': model_config.get("tfidf_use_idf_grid", [True, False]), 
        'tfidf__norm': model_config.get("tfidf_norm_grid", ['l1', 'l2']), 
    }
    
    model_specific_grid = {}
    
    if model_type == 'bayes':
        model_specific_grid['model__alpha'] = model_config.get('bayes_alpha_grid', [1.0]) # Add this
    elif model_type == 'lr':
        model_specific_grid['model__C'] = model_config.get('lr_C_grid', [1.0])
        model_specific_grid['model__max_iter'] = model_config.get('lr_max_iter_grid', [1000]) 
        model_specific_grid['model__penalty'] = model_config.get('lr_penalty_grid', ['l1', 'l2'])
        model_specific_grid['model__class_weight'] = model_config.get('lr_class_weight_grid', [None, 'balanced'])
        

    param_combinations.update(model_specific_grid)
    grid = ParameterGrid(param_combinations)
    
    # concatenating training and validation
    if not X_val_pd.empty:
        X_train_val_combined_pd = pd.concat([X_train_pd, X_val_pd], ignore_index=True)
        y_train_val_encoded_pd_aligned = pd.concat([y_train_encoded_pd, y_val_encoded_pd], ignore_index=True)
    else:
        X_train_val_combined_pd = X_train_pd.copy()
        y_train_val_encoded_pd_aligned = y_train_encoded_pd.copy()

    if X_train_val_combined_pd.empty or y_train_val_encoded_pd_aligned.empty:
        print("Error: Combined train/validation data is empty after encoding.")
        return None
    
    # --- Cross-Validation - Tuning ---
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1
    best_params = None
    results = []

    print(f"Starting manual {n_splits}-fold cross-validation for hyperparameter tuning...")
    start_time_tuning = time.time()

    for params in grid:
        print(f"  Testing params: {params}")
        fold_scores = []
        #tfidf_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('tfidf__')} before fixed vocabulary
        model_params_from_grid = {k.split('__')[1]: v for k, v in params.items() if k.startswith('model__')}

        fold_num = 0
        for train_idx, val_idx in kf.split(X_train_val_combined_pd, y_train_val_encoded_pd_aligned): #how split devide this -> more advance
            fold_num += 1
            X_train_fold_pd = X_train_val_combined_pd.iloc[train_idx]
            y_train_fold_pd = y_train_val_encoded_pd_aligned.iloc[train_idx]
            X_val_fold_pd = X_train_val_combined_pd.iloc[val_idx]
            y_val_fold_pd = y_train_val_encoded_pd_aligned.iloc[val_idx]

            # To be defined in the try block
            model_instance = None
            X_train_tfidf_gpu, X_val_tfidf_gpu = None, None
            y_train_fold_cpu = None
            y_pred_val_gpu = None
            
            #conversion to GPU data type
            try:
                X_train_fold_cudf = cudf.Series(X_train_fold_pd) #cudf is a GPU DataFrame library that mirrors the pandas API.
                y_train_fold_cupy = cupy.asarray(y_train_fold_pd.to_numpy(dtype=np.int32)) # CuPy for Arrays: cupy is a GPU array library that mirrors the NumPy API
                X_val_fold_cudf = cudf.Series(X_val_fold_pd)
                y_val_fold_cupy = cupy.asarray(y_val_fold_pd.to_numpy(dtype=np.int32))

                #cv_tfidf_vectorizer = TfidfVectorizer(**tfidf_params) # cuml.TfidfVectorizer
                current_tfidf_params_for_cv = {k.split('__')[1]: v for k, v in params.items() if k.startswith('tfidf__')}
                cv_tfidf_vectorizer = TfidfVectorizer(
                    vocabulary=fixed_vocabulary_dict,
                    ngram_range=(1, 2),       # The fixed vocab defines the n-grams CHANGEEEE
                    lowercase=False,          # Assuming SpaCy handled this
                    stop_words=None,          # Assuming SpaCy handled this
                    **current_tfidf_params_for_cv # Add this if you ARE tuning other TF-IDF params
                )
                X_train_tfidf_gpu = cv_tfidf_vectorizer.fit_transform(X_train_fold_cudf)
                X_val_tfidf_gpu = cv_tfidf_vectorizer.transform(X_val_fold_cudf)

                current_score = 0.0
                y_pred_val_cpu_fold = None # Prediction on validation fold
                
                #model fitting
                if model_type == 'bayes':
                    model_instance = ComplementNB(**model_params_from_grid)
                    model_instance.fit(X_train_tfidf_gpu, y_train_fold_cupy)
                    y_pred_val_gpu = model_instance.predict(X_val_tfidf_gpu)
                    y_pred_val_cpu_fold = cupy.asnumpy(y_pred_val_gpu)
                    current_score = cuml_accuracy_score(cupy.asnumpy(y_val_fold_cupy), y_pred_val_cpu_fold)

                elif model_type == 'lr':
                    model_instance = LogisticRegression(**model_params_from_grid)
                    model_instance.fit(X_train_tfidf_gpu, y_train_fold_cupy)
                    y_pred_val_gpu = model_instance.predict(X_val_tfidf_gpu)
                    y_pred_val_cpu_fold = cupy.asnumpy(y_pred_val_gpu)
                    current_score = cuml_accuracy_score(cupy.asnumpy(y_val_fold_cupy), y_pred_val_cpu_fold)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                fold_scores.append(current_score)

            except Exception as fold_e:
                print(f"       Error in Fold {fold_num} for params {params}: {fold_e}")
                import traceback
                traceback.print_exc()
                fold_scores.append(0)
            finally: # Ensure cleanup
                del X_train_fold_cudf, y_train_fold_cupy, X_val_fold_cudf, y_val_fold_cupy
                del X_train_tfidf_gpu, X_val_tfidf_gpu
                if y_train_fold_cpu is not None: del y_train_fold_cpu
                if y_pred_val_gpu is not None: del y_pred_val_gpu
                if 'cv_tfidf_vectorizer' in locals(): del cv_tfidf_vectorizer
                if model_instance is not None: del model_instance
                cupy.get_default_memory_pool().free_all_blocks()

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

    #--- Training final model ---

    print("Training final model using best parameters found...")
    start_time_final_train = time.time()

    X_train_val_final_cudf = cudf.Series(X_train_val_combined_pd)
    y_train_val_final_cupy = cupy.asarray(y_train_val_encoded_pd_aligned.to_numpy(dtype=np.int32))

    best_tfidf_params_final = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('tfidf__')}
    best_model_params_final = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('model__')}

    #final_tfidf_vectorizer = TfidfVectorizer(**best_tfidf_params_final) # cuml.TfidfVectorizer old tfidf
    final_tfidf_vectorizer = TfidfVectorizer(
    vocabulary=fixed_vocabulary_dict, # This is passed to run_model_pipeline
    ngram_range=(1, 2), #CHANGEEE
    lowercase=False,
    stop_words=None,
    **best_tfidf_params_final # This will include the best 'use_idf' and 'norm'
    ) 
    X_train_val_final_tfidf_gpu = final_tfidf_vectorizer.fit_transform(X_train_val_final_cudf)
    final_model_instance = None # To ensure it's defined for del

    if model_type == 'bayes':
        final_model_instance = ComplementNB(**best_model_params_final)
        final_model_instance.fit(X_train_val_final_tfidf_gpu, y_train_val_final_cupy)
        
        global congress_feature_importance_bayes
        if hasattr(final_model_instance, 'feature_log_prob_'):
            coefficients = final_model_instance.feature_log_prob_[1] - final_model_instance.feature_log_prob_[0]
            feature_names = final_tfidf_vectorizer.get_feature_names().to_pandas()
            feature_importance = dict(zip(feature_names, coefficients))
            
            
            congress_seed_key = f"{congress_year}_{random_state}"
            congress_feature_importance_bayes[congress_seed_key] = feature_importance

    elif model_type == 'lr':
        final_model_instance = LogisticRegression(**best_model_params_final)
        final_model_instance.fit(X_train_val_final_tfidf_gpu, y_train_val_final_cupy)
        
        global congress_feature_importance_lr
        coefficients = final_model_instance.coef_[0]
        feature_names = final_tfidf_vectorizer.get_feature_names().to_pandas()
        feature_importance = dict(zip(feature_names, coefficients))
        
        congress_seed_key = f"{congress_year}_{random_state}"
        congress_feature_importance_lr[congress_seed_key] = feature_importance
            
        print(f"Extracted feature importance for Congress {congress_year}, seed {random_state} (LR)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds.")

    """print(f"Saving the final TF-IDF and {model_type.upper()} model...")
    model_dir_path = Path("models")
    model_dir_path.mkdir(parents=True, exist_ok=True)
    tfidf_filename = model_dir_path / f"tfidf_{model_type}_{congress_year}_seed{random_state}_vectorizer.joblib"
    model_filename = model_dir_path / f"tfidf_{model_type}_{congress_year}_seed{random_state}_model.joblib"
    try:
        joblib.dump(final_tfidf_vectorizer, tfidf_filename)
        joblib.dump(final_model_instance, model_filename)
        print(f"Final TF-IDF saved to {tfidf_filename}")
        print(f"Final Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving the final components: {e}")"""

    print("Preparing test data for evaluation...")
    if X_test_pd.empty:
        print("Error: Test data (X_test_pd) is empty.")
        # Cleanup before returning
        if 'X_train_val_final_tfidf_gpu' in locals(): del X_train_val_final_tfidf_gpu
        if 'final_tfidf_vectorizer' in locals(): del final_tfidf_vectorizer
        if 'final_model_instance' in locals(): del final_model_instance
        if 'X_train_val_final_cudf' in locals(): del X_train_val_final_cudf
        if 'y_train_val_final_cupy' in locals(): del y_train_val_final_cupy
        cupy.get_default_memory_pool().free_all_blocks()
        return None

    # Ensure y_test is a CuPy array for cuML metrics
    y_test_encoded_gpu_eval = cupy.asarray(y_test_encoded_pd.to_numpy(dtype=cupy.int32))
    X_test_cudf_eval = cudf.Series(X_test_pd) # X_test_pd should be the raw text data
    
    # --- Test final model ---
    print("Evaluating final model on test data...")
    start_time_test = time.time()
    
    # Transform test data using the trained TF-IDF vectorizer
    X_test_final_tfidf_gpu_eval = final_tfidf_vectorizer.transform(X_test_cudf_eval)

    # Initialize metrics
    final_accuracy_eval = 0.0
    final_f1_weighted_eval = 0.0 # now calculate not available
    auc_eval = None
    cm_cpu_eval = None # Initialize cm_cpu_eval
    probability_scores_gpu = None # Initialize for cleanup
    decision_scores_gpu = None # Initialize for cleanup
    classification_report_dict = {} # Initialize for classification report

    # Get predictions (already on GPU)
    y_test_pred_gpu_eval = final_model_instance.predict(X_test_final_tfidf_gpu_eval)
    
    # Transfer necessary data from GPU (CuPy arrays) to CPU (NumPy arrays) for scikit-learn metrics
    y_test_encoded_cpu = cupy.asnumpy(y_test_encoded_gpu_eval)
    y_test_pred_cpu = cupy.asnumpy(y_test_pred_gpu_eval)

    # --- Calculate target_names_eval (Moved Earlier) ---
    try:
        reverse_party_map_eval = {v: k for k, v in party_map.items()}
        unique_labels_in_test_cpu_eval = sorted(list(np.unique(y_test_encoded_cpu))) # Use y_test_encoded_cpu directly
        target_names_eval = [reverse_party_map_eval.get(i, str(i)) for i in unique_labels_in_test_cpu_eval]
    except Exception as e:
        print(f"Could not get target names for classification report: {e}")
        # Fallback if party_map is problematic or labels are unexpected
        if 'y_test_encoded_cpu' in locals() and y_test_encoded_cpu.size > 0:
             target_names_eval = [str(i) for i in sorted(list(np.unique(y_test_encoded_cpu)))]
        else: # Absolute fallback if y_test_encoded_cpu isn't even available
            target_names_eval = [] 
            print("Warning: y_test_encoded_cpu not available for target_names_eval.")
            
    # Calculate metrics using cuML and scikit-learn
    try:
        # Accuracy (cuML)
        # Ensure y_test_pred_gpu_eval is available before using it
        if 'y_test_pred_gpu_eval' in locals() and y_test_pred_gpu_eval is not None:
            final_accuracy_gpu = cuml_accuracy_score(y_test_encoded_gpu_eval, y_test_pred_gpu_eval)
            final_accuracy_eval = cupy.asnumpy(final_accuracy_gpu).item() if final_accuracy_gpu is not None else 0.0
        else:
            print("Warning: y_test_pred_gpu_eval not available for accuracy calculation.")
            final_accuracy_eval = 0.0

        # F1 Score (Weighted) using scikit-learn
        final_f1_weighted_eval = sklearn_f1_score(y_test_encoded_cpu, y_test_pred_cpu, average='weighted', zero_division=0)
        
        # Confusion Matrix (cuML)
        if 'y_test_pred_gpu_eval' in locals() and y_test_pred_gpu_eval is not None:
            cm_gpu_eval_calc = cuml_confusion_matrix(y_test_encoded_gpu_eval, y_test_pred_gpu_eval, convert_dtype=True)
            cm_cpu_eval = cupy.asnumpy(cm_gpu_eval_calc) if cm_gpu_eval_calc is not None else np.array([])
        else:
            print("Warning: y_test_pred_gpu_eval not available for confusion matrix calculation.")
            cm_cpu_eval = np.array([])

        # ROC-AUC Score (Binary Classification)
        if hasattr(final_model_instance, "predict_proba"):
            probability_scores_gpu = final_model_instance.predict_proba(X_test_final_tfidf_gpu_eval)
            auc_gpu = cuml_roc_auc_score(y_test_encoded_gpu_eval, probability_scores_gpu[:, 1])
            auc_eval = cupy.asnumpy(auc_gpu).item() if auc_gpu is not None else None
        elif hasattr(final_model_instance, "decision_function"):
            decision_scores_gpu = final_model_instance.decision_function(X_test_final_tfidf_gpu_eval)
            auc_gpu = cuml_roc_auc_score(y_test_encoded_gpu_eval, decision_scores_gpu)
            auc_eval = cupy.asnumpy(auc_gpu).item() if auc_gpu is not None else None
        else:
            print("Model has neither 'predict_proba' nor 'decision_function'. ROC-AUC cannot be calculated.")
            auc_eval = None
        
        # --- Per-Class Metrics (Classification Report) ---
        if target_names_eval: # Only proceed if we have target names
            report_str = classification_report(y_test_encoded_cpu, y_test_pred_cpu, target_names=target_names_eval, zero_division=0)
            print("\nClassification Report:\n", report_str)
            classification_report_dict = classification_report(y_test_encoded_cpu, y_test_pred_cpu, target_names=target_names_eval, zero_division=0, output_dict=True)
        else:
            print("Warning: target_names_eval is empty. Skipping classification report.")
            classification_report_dict = {}

    except Exception as e:
        print(f"Error during cuML and scikit metrics calculation: {e}")
        import traceback # Good for debugging
        traceback.print_exc() # Good for debugging
        final_accuracy_eval = 0.0
        final_f1_weighted_eval = 0.0
        auc_eval = None
        cm_cpu_eval = np.array([])
        classification_report_dict = {}

    # Cleanup GPU arrays used for metrics if no longer needed
    if 'y_test_pred_gpu_eval' in locals(): del y_test_pred_gpu_eval
    if probability_scores_gpu is not None: del probability_scores_gpu
    if decision_scores_gpu is not None: del decision_scores_gpu
    
    evaluation_time = time.time() - start_time_test
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    print(f"\n--- Final Test Results ({model_type.upper()}) ---")
    print(f"Accuracy: {final_accuracy_eval:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted_eval:.4f}")
    # The classification_report string is already printed above if generated

    cm_list_eval = cm_cpu_eval.tolist() if cm_cpu_eval is not None and cm_cpu_eval.size > 0 else []
    # target_names_eval is already defined and handled before this print block

    if auc_eval is not None: print(f"ROC-AUC          : {auc_eval:.4f}")
    else: print("ROC-AUC          : NA")

    if cm_cpu_eval is not None and cm_cpu_eval.size > 0: print("Confusion Matrix:\n", cm_cpu_eval)

    print("-" * 25)

    result_json = {
        "seed": random_state, "year": congress_year, "accuracy": round(final_accuracy_eval, 4),
        "f1_score": round(final_f1_weighted_eval, 4), "auc": round(auc_eval, 4) if auc_eval is not None else "NA",
        "confusion_matrix": cm_list_eval, 
        "labels": target_names_eval, # target_names_eval already defined
        "classification_report": classification_report_dict, # Add the report dictionary
        "best_params": best_params,
        "timing": {
            "tuning_sec": round(tuning_time, 2),
            "final_train_sec": round(final_train_time, 2),
            "evaluation_sec": round(evaluation_time, 2),
            "total_pipeline_sec": round(time.time() - start_time_total, 2)
        }
    }
    
    current_timing_log_path = timing_log_paths[model_type]
    with open(current_timing_log_path, "a") as f:
        f.write(
            f"{result_json['seed']},"
            f"{result_json['year']},"
            f"{result_json['timing']['tuning_sec']},"
            f"{result_json['timing']['final_train_sec']},"
            f"{result_json['timing']['evaluation_sec']},"
            f"{result_json['timing']['total_pipeline_sec']}\n"
        )
    
    current_detailed_log_path = detailed_log_paths[model_type]
    with open(current_detailed_log_path, "a") as f:
        f.write(
            f"{result_json['seed']},"       # Assuming seed and year are not problematic
            f"{result_json['year']},"
            f"{result_json['accuracy']},"
            f"{result_json['f1_score']},"
            f"{result_json['auc']},"
            f"{result_json['best_params']}\n"
        )

    # Cleanup remaining major variables from this pipeline run
    if 'X_train_val_final_cudf' in locals(): del X_train_val_final_cudf
    if 'y_train_val_final_cupy' in locals(): del y_train_val_final_cupy
    if 'X_train_val_final_tfidf_gpu' in locals(): del X_train_val_final_tfidf_gpu
    
    if 'X_test_cudf_eval' in locals(): del X_test_cudf_eval
    if 'X_test_final_tfidf_gpu_eval' in locals(): del X_test_final_tfidf_gpu_eval
    if 'y_test_encoded_gpu_eval' in locals(): del y_test_encoded_gpu_eval # Clean up the GPU array for y_test

    if 'final_tfidf_vectorizer' in locals(): del final_tfidf_vectorizer
    if 'final_model_instance' in locals(): del final_model_instance

    cupy.get_default_memory_pool().free_all_blocks()

    return result_json

def save_feature_importance(feature_dict, model_type, output_dir="feature_importance"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"congress_feature_importance_{model_type}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(feature_dict, f)
    
    print(f"Feature importance dictionary saved to: {filename}")
    return filename

# ------ Main Execution -------
if __name__ == "__main__":
    
    # (os.environ setting should still be at the top, just in case it affects other parts of cuDF)
    # (Ensure 'import pandas as pd' and 'import cudf' are at the script's top)

    cuml_vocab_load_path = Path("data/vocabulary_dumps/1_word/global_vocabulary_processed_bigram_100_min_df_cuml_from_sklearn.parquet")

    if not cuml_vocab_load_path.exists():
        print(f"ERROR: Fixed vocabulary file not found at {cuml_vocab_load_path}")
        print("Please run the global vocabulary generation script first.")
        exit()

    print(f"Attempting to read Parquet file with pandas (as primary method for vocab): {cuml_vocab_load_path}")
    try:
        pandas_df_vocab = pd.read_parquet(cuml_vocab_load_path)
        print(f"Successfully read Parquet file with pandas. Shape: {pandas_df_vocab.shape}")

        print("Converting pandas DataFrame vocabulary to cuDF Series...")
        # Assuming the vocabulary is in the 'term' column as per your saving script
        fixed_cuml_vocabulary_terms = cudf.Series(pandas_df_vocab['term'])
        print(f"Successfully converted to cuDF Series. Length: {len(fixed_cuml_vocabulary_terms)}")

    except Exception as e:
        print(f"Error reading Parquet file with pandas or converting to cuDF: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- --- --- --- --- 
    
    congress_years_to_process = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END + 1)]
    models_to_run = ['bayes', 'lr',] 
    congress_feature_importance_bayes = {}
    congress_feature_importance_lr = {}

    for seed in SEEDS:
        print(f"\n--- Starting runs for seed: {seed} ---")
        for year_str in congress_years_to_process:
            print(f"\nProcessing Congress Year: {year_str}")
            input_csv_path = Path(f"data/processed/house_db/house_cleaned_{year_str}.csv")

            if not input_csv_path.exists():
                print(f"⚠️  Skipping Congress {year_str} (seed {seed}): CSV file not found at {input_csv_path}.")
                continue
            try:
                print("Loading data...")
                df_full = pd.read_csv(input_csv_path)
                print(f"Data loaded. Shape: {df_full.shape}")

                #controll
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

                if len(unique_speakers) < 2:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Not enough unique speakers ({len(unique_speakers)}).")
                    continue

                current_test_size = TEST_SIZE if len(unique_speakers) * TEST_SIZE >= 1 else 1/len(unique_speakers)
                train_val_speakers, test_speakers = train_test_split(
                    unique_speakers, test_size=current_test_size, random_state=seed)

                if len(train_val_speakers) < 2 :
                    train_speakers = train_val_speakers
                    val_speakers = np.array([]) # Ensure it's an empty array for consistency
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

                X_train_pd_orig = train_df["speech"]
                y_train_pd_orig = train_df["party"]
                X_val_pd_orig = val_df["speech"] if not val_df.empty else pd.Series(dtype='object')
                y_val_pd_orig = val_df["party"] if not val_df.empty else pd.Series(dtype='object')
                X_test_pd_orig = test_df["speech"]
                y_test_pd_orig = test_df["party"]

                print("Encoding labels and aligning X data (already cleaned) with filtered labels...")
                start_time_encode = time.time()

                train_data_to_encode = pd.DataFrame({'party': y_train_pd_orig, 'speech': X_train_pd_orig})
                train_encoded_df = encode_labels_with_map(train_data_to_encode, PARTY_MAP)
                X_train_pd_aligned = train_encoded_df['speech'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='object')
                y_train_encoded_pd_aligned = train_encoded_df['label'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='int')

                if not X_val_pd_orig.empty:
                    val_data_to_encode = pd.DataFrame({'party': y_val_pd_orig, 'speech': X_val_pd_orig})
                    val_encoded_df = encode_labels_with_map(val_data_to_encode, PARTY_MAP)
                    X_val_pd_aligned = val_encoded_df['speech'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = val_encoded_df['label'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='int')
                else:
                    X_val_pd_aligned = pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = pd.Series(dtype='int')

                test_data_to_encode = pd.DataFrame({'party': y_test_pd_orig, 'speech': X_test_pd_orig})
                test_encoded_df = encode_labels_with_map(test_data_to_encode, PARTY_MAP)
                X_test_pd_aligned = test_encoded_df['speech'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='object')
                y_test_encoded_pd_aligned = test_encoded_df['label'].reset_index(drop=True) if not test_encoded_df.empty else pd.Series(dtype='int')

                encode_time = time.time() - start_time_encode
                print(f"Labels encoded and X data aligned in {encode_time:.2f} seconds.")
                print(f"  Post-encoding/alignment - Train: {len(X_train_pd_aligned)}, Val: {len(X_val_pd_aligned)}, Test: {len(X_test_pd_aligned)}")

                if X_train_pd_aligned.empty or y_train_encoded_pd_aligned.empty or X_test_pd_aligned.empty or y_test_encoded_pd_aligned.empty:
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Train or Test data empty after label encoding & alignment.")
                    continue
                if len(np.unique(y_train_encoded_pd_aligned)) < 2 :
                    print(f"⚠️  Skipping Congress {year_str} (seed {seed}): Fewer than 2 unique classes in training labels after encoding. Model fitting requires at least 2.")
                    continue

                for model_type_to_run in models_to_run:
                    current_model_config = model_configs.get(model_type_to_run)
                    if current_model_config is None:
                        print(f"Warning: Config for model '{model_type_to_run}' not found. Skipping.")
                        continue

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
                        model_plot_output_dir=current_model_plot_dir,
                        fixed_vocabulary_dict=fixed_cuml_vocabulary_terms
                    )
            except Exception as e:
                print(f"❌ An error occurred during processing for Congress {year_str} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # General cleanup at the end of processing a year, if any dataframes are large and still in scope
                if 'df_full' in locals(): del df_full
                if 'train_df' in locals(): del train_df
                if 'val_df' in locals(): del val_df
                if 'test_df' in locals(): del test_df
                cupy.get_default_memory_pool().free_all_blocks()

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
    
    # Save both dictionaries
    print("Saving feature importance dictionaries...")
    save_feature_importance(congress_feature_importance_bayes, "bayes")
    save_feature_importance(congress_feature_importance_lr, "lr")

    print(f"Bayes combinations saved: {len(congress_feature_importance_bayes)}")
    print(f"LR combinations saved: {len(congress_feature_importance_lr)}")
    
    print("\n--- Script finished ---")