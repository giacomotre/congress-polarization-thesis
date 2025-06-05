import os
import pandas as pd
import joblib
import json
import time
import pickle
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import classification_report # Add this line

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map
from plotting_utils import plot_performance_metrics

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import csv

# ------ Loading Unified Config -------
@dataclass
class ExperimentConfig:
    """Configuration for the congress speech classification experiment"""
    congress_year_start: int
    congress_year_end: int
    test_size: float
    validation_size: float
    random_state: int
    seeds: List[int]
    party_map: Dict[str, int]
    
    @classmethod
    def from_config_file(cls, config_path: Path):
        """Load configuration from YAML file"""
        try:
            unified_config = load_config(config_path)
            print("✅ Loaded unified config successfully")
        except FileNotFoundError:
            print(f"❌ Error: Config file not found at {config_path}")
            raise
        
        # Extract parameters (same logic as your original code)
        common = unified_config.get('common', {})
        split_params = common.get('split_params', {})
        data_params = common.get('data_params', {})
        party_map = common.get('party_map', {})
        
        # Validate party map
        if not party_map or not all(party in party_map for party in ['D', 'R']):
            print("Warning: config.party_map missing or incomplete. Using default.")
            party_map = {'D': 0, 'R': 1}
        
        instance = cls(
            congress_year_start=data_params.get('congress_year_start', 76),
            congress_year_end=data_params.get('congress_year_end', 111),
            test_size=split_params.get('test_size', 0.15),
            validation_size=split_params.get('validation_size', 0.25),
            random_state=split_params.get('random_state', 42),
            seeds=split_params.get('seeds', [42]),
            party_map=party_map
        )
        
        instance._raw_config = unified_config
        
        return instance
    
    def get_congress_years(self):
        """Get list of congress years as strings"""
        return [f"{i:03}" for i in range(self.congress_year_start, self.congress_year_end + 1)]
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return self._raw_config.get(model_name, {})

@dataclass
class ResultsManager:
    """Gestisce tutto il logging e salvataggio dei risultati degli esperimenti."""
    base_log_dir: str = "logs"
    base_plot_dir: str = "plots"
    
    def __post_init__(self):
        """Chiamato automaticamente dopo __init__. Inizializza le directory."""
        self._create_directories()
        self._initialize_log_files()
    
    def _create_directories(self):
        """Crea le directory necessarie se non esistono"""
        os.makedirs(self.base_log_dir, exist_ok=True)
        os.makedirs(self.base_plot_dir, exist_ok=True)
        print(f"✅ Created directories: {self.base_log_dir}, {self.base_plot_dir}")
    
    def get_detailed_log_path(self, model_type: str) -> str:
        """Ottieni il path del file di log dettagliato per un modello"""
        return f"{self.base_log_dir}/tfidf_{model_type}_performance_detailed.csv"
    
    def get_avg_log_path(self, model_type: str) -> str:
        """Ottieni il path del file di log delle medie per un modello"""
        detailed_path = self.get_detailed_log_path(model_type)
        return detailed_path.replace("_detailed.csv", "_avg.csv")
    
    def get_timing_log_path(self, model_type: str) -> str:
        """Ottieni il path del file di log dei tempi per un modello"""
        return f"{self.base_log_dir}/{model_type}_timing_log.csv"
    
    def get_plot_output_dir(self, model_type: str) -> str:
        """Ottieni la directory per i plot di un modello"""
        return f"{self.base_plot_dir}/{model_type}"
    
    def get_model_plotting_info(self, model_type: str) -> Dict[str, str]:
        """Ottieni le informazioni per il plotting di un modello."""
        return {
            "avg_log_path": self.get_avg_log_path(model_type),
            "output_dir": self.get_plot_output_dir(model_type)
        }
    
    def _get_detailed_csv_header(self) -> List[str]:
        """Definisce le colonne del CSV dettagliato"""
        return ["seed", "year", "accuracy", "f1_score", "auc", "model_C", "tfidf_norm"]
    
    def _get_timing_csv_header(self) -> List[str]:
        """Definisce le colonne del CSV dei tempi"""
        return [
            "seed", "year",
            "timing_tuning_sec", "timing_final_train_sec", 
            "timing_evaluation_sec", "timing_total_pipeline_sec"
        ]
    
    def _initialize_log_files(self):
        """Inizializza i file CSV con gli header."""
        models = ['svm']  # Puoi espandere questa lista
        
        for model_type in models:
            # Inizializza file dettagliato
            detailed_path = self.get_detailed_log_path(model_type)
            self._initialize_csv_file(
                detailed_path, 
                self._get_detailed_csv_header(),
                "detailed log"
            )
            
            # Inizializza file timing
            timing_path = self.get_timing_log_path(model_type)
            self._initialize_csv_file(
                timing_path,
                self._get_timing_csv_header(), 
                "timing log"
            )
            
            # Crea directory plot
            plot_dir = self.get_plot_output_dir(model_type)
            os.makedirs(plot_dir, exist_ok=True)
    
    def _initialize_csv_file(self, file_path: str, header_columns: List[str], file_type: str):
        """Inizializza un singolo file CSV con header."""
        # Rimuovi file esistente se presente
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️  Deleted existing {file_type} file: {file_path}")
        
        # Crea nuovo file con header
        header_line = ",".join(header_columns) + "\n"
        with open(file_path, "w") as f:
            f.write(header_line)
        print(f"📝 Initialized {file_type} file: {file_path}")
    
    def save_experiment_result(self, result_json: Dict[str, Any], model_type: str):
        """Salva i risultati di un esperimento nel file CSV dettagliato."""
        detailed_log_path = self.get_detailed_log_path(model_type)
        
        try:
            # Estrai parametri dai best_params
            best_params = result_json.get('best_params', {})
            model_C = self._extract_parameter(best_params, ['model__C', 'svm_C'], default=1.0)
            tfidf_norm = self._extract_parameter(best_params, ['tfidf__norm', 'tfidf_norm'], default='l2')
            
            # Prepara la riga da scrivere
            row_data = [
                result_json['seed'],
                result_json['year'], 
                result_json['accuracy'],
                result_json['f1_score'],
                result_json['auc'],
                model_C,
                tfidf_norm
            ]
            
            # Scrivi nel CSV
            with open(detailed_log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            print(f"💾 Saved result for {model_type} - Year: {result_json['year']}, Seed: {result_json['seed']}")
            
        except Exception as e:
            print(f"❌ Error saving result to {detailed_log_path}: {e}")
            raise
    
    def _extract_parameter(self, params_dict: Dict, possible_keys: List[str], default: Any) -> Any:
        """Estrae un parametro da un dizionario provando multiple chiavi."""
        for key in possible_keys:
            if key in params_dict:
                return params_dict[key]
        return default

# --- Define model function ---
def run_model_pipeline(
    X_train_pd: pd.Series, 
    y_train_encoded_pd: pd.Series,
    X_val_pd: pd.Series, 
    y_val_encoded_pd: pd.Series,
    X_test_pd: pd.Series, 
    y_test_encoded_pd: pd.Series,
    config: 'ExperimentConfig',  
    model_type: str,  
    model_config: Dict[str, Any],  
    random_state: int,  
    congress_year: str,  
    model_plot_output_dir: str,
    fixed_vocabulary_dict: Dict[str, int],
    results_manager: 'ResultsManager',
    congress_feature_importance: Dict[str, Dict[str, float]]
) -> Optional[Dict[str, Any]]:
    
    print(f"\n --- Running {model_type.upper()} pipeline [Fixed Vocabulary] for Congress {congress_year} with seed {random_state} ---")
    
    timing = {}
    start_time_total = time.time()
    
    # --- loading optimization config ---
    model_specific_grid = {}
    
    param_combinations = {
        'tfidf__use_idf': model_config.get("tfidf_use_idf_grid", [True, False]), 
        'tfidf__norm': model_config.get("tfidf_norm_grid", ['l1', 'l2']), 
    }
    
    if model_type == 'svm':
        model_specific_grid['model__C'] = model_config.get('C_grid', [1.0]) 
        
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
        fold_num = 0
        
        for train_idx, val_idx in kf.split(X_train_val_combined_pd, y_train_val_encoded_pd_aligned): #how split devide this -> more advance
            fold_num += 1
            X_train_fold_pd = X_train_val_combined_pd.iloc[train_idx]
            y_train_fold_pd = y_train_val_encoded_pd_aligned.iloc[train_idx]
            X_val_fold_pd = X_train_val_combined_pd.iloc[val_idx]
            y_val_fold_pd = y_train_val_encoded_pd_aligned.iloc[val_idx]

            # To be defined in the try block
            model_instance = None
            
            current_tfidf_params_for_cv = {k.split('__')[1]: v for k, v in params.items() if k.startswith('tfidf__')}
            try:
                cv_tfidf_vectorizer = TfidfVectorizer(
                    vocabulary=fixed_vocabulary_dict,
                    ngram_range=(1, 2),       # The fixed vocab defines the n-grams CHANGEEEE
                    lowercase=False,          # Assuming SpaCy handled this
                    stop_words=None,          # Assuming SpaCy handled this
                    **current_tfidf_params_for_cv # Add this if you ARE tuning other TF-IDF params
                )
                X_train_tfidf = cv_tfidf_vectorizer.fit_transform(X_train_fold_pd)
                X_val_tfidf = cv_tfidf_vectorizer.transform(X_val_fold_pd)

                current_score = 0.0
                y_pred_val = None # Prediction on validation fold
                
                #model fitting
                model_params_from_grid = {k.split('__')[1]: v for k, v in params.items() if k.startswith('model__')}
                if model_type == 'svm':
                    svm_max_iter = model_config.get('max_iter', 5000) # Default to 5000 if not in config
                    model_instance = LinearSVC(**model_params_from_grid,  max_iter=svm_max_iter)
                    model_instance.fit(X_train_tfidf, y_train_fold_pd)
                    y_pred_val = model_instance.predict(X_val_tfidf)
                    current_score = accuracy_score(y_val_fold_pd, y_pred_val)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                fold_scores.append(current_score)

            except Exception as fold_e:
                print(f"       Error in Fold {fold_num} for params {params}: {fold_e}")
                import traceback
                traceback.print_exc()
                fold_scores.append(0)
            finally: # Ensure cleanup
                del X_train_fold_pd, y_train_fold_pd, X_val_fold_pd, y_val_fold_pd
                if 'cv_tfidf_vectorizer' in locals(): del cv_tfidf_vectorizer

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
        #arrived here
    #--- Training final model ---

    print("Training final model using best parameters found...")
    start_time_final_train = time.time()

    tuned_tfidf_params_final = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('tfidf__')}
    best_model_params_final = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('model__')}

    final_tfidf_vectorizer = TfidfVectorizer(
    vocabulary=fixed_vocabulary_dict, # This is passed to run_model_pipeline
    ngram_range=(1, 2), #CHANGEEEE
    lowercase=False,
    stop_words=None,
    **tuned_tfidf_params_final # This will include the best 'use_idf' and 'norm'
    )  
    X_train_val_final_tfidf = final_tfidf_vectorizer.fit_transform(X_train_val_combined_pd)
    final_model_instance = None # To ensure it's defined for del

    if model_type == 'svm':
        svm_max_iter = model_config.get('max_iter', 5000) # Default to 5000 if not in config
        final_model_instance = LinearSVC(**best_model_params_final, max_iter=svm_max_iter)
        final_model_instance.fit(X_train_val_final_tfidf, y_train_val_encoded_pd_aligned)
        
        # Extract coefficients for feature importance analysis
        coefficients = final_model_instance.coef_[0]  # Shape: (n_features,)
        feature_names = final_tfidf_vectorizer.get_feature_names_out()
        feature_importance = dict(zip(feature_names, coefficients))
        
        # Create a unique key for this congress-seed combination
        congress_seed_key = f"{congress_year}_{random_state}"
        congress_feature_importance[congress_seed_key] = feature_importance
        
        print(f"Extracted feature importance for Congress {congress_year}, seed {random_state}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds.")

    print("Preparing test data for evaluation...")
    if X_test_pd.empty:
        print("Error: Test data (X_test_pd) is empty.")
        # Cleanup before returning
        if 'X_train_val_final_tfidf' in locals(): del X_train_val_final_tfidf
        if 'final_tfidf_vectorizer' in locals(): del final_tfidf_vectorizer
        if 'final_model_instance' in locals(): del final_model_instance
        if 'X_train_val_combined_pd' in locals(): del X_train_val_combined_pd
        if 'y_train_val_encoded_pd_aligned' in locals(): del y_train_val_encoded_pd_aligned
        return None
    
    # --- Test final model ---
    print("Evaluating final model on test data...")
    start_time_test = time.time()
    
    # Transform test data using the trained TF-IDF vectorizer
    X_test_final_tfidf_eval = final_tfidf_vectorizer.transform(X_test_pd)

    # Initialize metrics
    final_accuracy_eval = 0.0
    final_f1_weighted_eval = 0.0 # now calculate not available
    auc_eval = None
    cm_eval = None 
    probability_scores = None # Initialize for cleanup
    decision_scores = None # Initialize for cleanup
    classification_report_dict = {} # Initialize for classification report

    # Get predictions (already on GPU)
    y_test_pred_eval = final_model_instance.predict(X_test_final_tfidf_eval)

    # --- Calculate target_names_eval (Moved Earlier) ---
    try:
        reverse_party_map_eval = {v: k for k, v in config.party_map.items()}
        unique_labels_in_test_cpu_eval = sorted(list(np.unique(y_test_encoded_pd))) # Use y_test_encoded_pd directly
        target_names_eval = [reverse_party_map_eval.get(i, str(i)) for i in unique_labels_in_test_cpu_eval]
    except Exception as e:
        print(f"Could not get target names for classification report: {e}")
        # Fallback if party_map is problematic or labels are unexpected
        if 'y_test_encoded_pd' in locals() and y_test_encoded_pd.size > 0:
            target_names_eval = [str(i) for i in sorted(list(np.unique(y_test_encoded_pd)))]
        else: # Absolute fallback if y_test_encoded_pd isn't even available
            target_names_eval = [] 
            print("Warning: y_test_encoded_pd not available for target_names_eval.")
            
    # Calculate metrics using cuML and scikit-learn
    try:
        # Accuracy (cuML)
        # Ensure y_test_pred_eval is available before using it
        if 'y_test_pred_eval' in locals() and y_test_pred_eval is not None:
            final_accuracy= accuracy_score(y_test_encoded_pd, y_test_pred_eval)
            final_accuracy_eval = final_accuracy
        else:
            print("Warning: y_test_pred_eval not available for accuracy calculation.")
            final_accuracy_eval = 0.0

        # F1 Score (Weighted) using scikit-learn
        final_f1_weighted_eval = sklearn_f1_score(y_test_encoded_pd, y_test_pred_eval, average='weighted', zero_division=0)
        
        # Confusion Matrix (cuML)
        if 'y_test_pred_eval' in locals() and y_test_pred_eval is not None:
            cm_eval_calc = confusion_matrix(y_test_encoded_pd, y_test_pred_eval)
            cm_eval = cm_eval_calc 
        else:
            print("Warning: y_test_pred_eval not available for confusion matrix calculation.")
            cm_eval = np.array([])

        # ROC-AUC Score (Binary Classification)
        if hasattr(final_model_instance, "predict_proba"):
            probability_scores = final_model_instance.predict_proba(X_test_final_tfidf_eval)
            auc = roc_auc_score(y_test_encoded_pd, probability_scores[:, 1])
            auc_eval = auc
        elif hasattr(final_model_instance, "decision_function"): #svc has a decision function
            decision_scores = final_model_instance.decision_function(X_test_final_tfidf_eval)
            auc = roc_auc_score(y_test_encoded_pd, decision_scores)
            auc_eval = auc
        else:
            print("Model has neither 'predict_proba' nor 'decision_function'. ROC-AUC cannot be calculated.")
            auc_eval = None
        
        # --- Per-Class Metrics (Classification Report) ---
        if target_names_eval: # Only proceed if we have target names
            report_str = classification_report(y_test_encoded_pd, y_test_pred_eval, target_names=target_names_eval, zero_division=0)
            print("\nClassification Report:\n", report_str)
            classification_report_dict = classification_report(y_test_encoded_pd, y_test_pred_eval, target_names=target_names_eval, zero_division=0, output_dict=True)
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
        cm_eval = np.array([])
        classification_report_dict = {}

    # Cleanup GPU arrays used for metrics if no longer needed
    if 'y_test_pred_eval' in locals(): del y_test_pred_eval
    if probability_scores is not None: del probability_scores
    if decision_scores is not None: del decision_scores
    
    evaluation_time = time.time() - start_time_test
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    print(f"\n--- Final Test Results ({model_type.upper()}) ---")
    print(f"Accuracy: {final_accuracy_eval:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted_eval:.4f}")
    # The classification_report string is already printed above if generated

    cm_list_eval = cm_eval.tolist() if cm_eval is not None and cm_eval.size > 0 else []
    # target_names_eval is already defined and handled before this print block

    if auc_eval is not None: print(f"ROC-AUC          : {auc_eval:.4f}")
    else: print("ROC-AUC          : NA")

    if cm_eval is not None and cm_eval.size > 0: print("Confusion Matrix:\n", cm_eval)
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
    
    results_manager.save_experiment_result(result_json, model_type)

    # Cleanup remaining major variables from this pipeline run
    if 'X_train_val_combined_pd' in locals(): del X_train_val_combined_pd
    if 'y_train_val_encoded_pd_aligned' in locals(): del y_train_val_encoded_pd_aligned
    if 'X_train_val_final_tfidf' in locals(): del X_train_val_final_tfidf
    
    if 'X_test_pd' in locals(): del X_test_pd
    if 'X_test_final_tfidf_eval' in locals(): del X_test_final_tfidf_eval
    if 'y_test_encoded_pd' in locals(): del y_test_encoded_pd # Clean up the GPU array for y_test

    if 'final_tfidf_vectorizer' in locals(): del final_tfidf_vectorizer
    if 'final_model_instance' in locals(): del final_model_instance

    return result_json

    # After your main loops complete, save the dictionary
def save_feature_importance(congress_feature_importance, model_type, output_dir="feature_importance"):
    """
    Save the feature importance dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"congress_feature_importance_{model_type}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(congress_feature_importance, f)
    
    print(f"Feature importance dictionary saved to: {filename}")
    return filename


# ------ Main Execution -------
if __name__ == "__main__":
    print("Starting Congress Speech Classification Analysis")
    
    # NEW: Load configuration using the class
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    config = ExperimentConfig.from_config_file(config_path)
    
    results_manager = ResultsManager()
    print("✅ Results manager initialized")
    
    # Test: stampa la configurazione SVM
    svm_config = config.get_model_config('svm')
    print(f"SVM config: {svm_config}")
    
    model_configs = {
        'svm': config.get_model_config('svm'),
    }
    
    congress_feature_importance = {}

    # --- load fixed vocabulary ---
    sklearn_vocab_load_path = Path("data/vocabulary_dumps/1_word/global_vocabulary_processed_bigram_100_min_df_sklearn_from_sklearn.joblib") # Adjust if your path is different

    if not sklearn_vocab_load_path.exists():
        print(f"ERROR: Fixed vocabulary file not found at {sklearn_vocab_load_path}")
        print("Please run the global vocabulary generation script first.")
        exit()
    
    print(f"Loading fixed scikit-learn vocabulary from {sklearn_vocab_load_path}...")
    fixed_sklearn_vocabulary = joblib.load(sklearn_vocab_load_path)
    print(f"Loaded fixed vocabulary with {len(fixed_sklearn_vocabulary)} terms.")
    # --- --- --- --- --- 
    
    congress_years_to_process = config.get_congress_years()
    models_to_run = ['svm'] 
    
    
    for seed in config.seeds:
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

                current_test_size = config.test_size if len(unique_speakers) * config.test_size >= 1 else 1/len(unique_speakers)
                train_val_speakers, test_speakers = train_test_split(
                    unique_speakers, test_size=current_test_size, random_state=seed)

                if len(train_val_speakers) < 2 :
                    train_speakers = train_val_speakers
                    val_speakers = np.array([]) # Ensure it's an empty array for consistency
                else:
                    current_validation_size = config.validation_size if len(train_val_speakers) * config.validation_size >= 1 else 1/len(train_val_speakers)
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
                train_encoded_df = encode_labels_with_map(train_data_to_encode, config.party_map)
                X_train_pd_aligned = train_encoded_df['speech'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='object')
                y_train_encoded_pd_aligned = train_encoded_df['label'].reset_index(drop=True) if not train_encoded_df.empty else pd.Series(dtype='int')

                if not X_val_pd_orig.empty:
                    val_data_to_encode = pd.DataFrame({'party': y_val_pd_orig, 'speech': X_val_pd_orig})
                    val_encoded_df = encode_labels_with_map(val_data_to_encode, config.party_map)
                    X_val_pd_aligned = val_encoded_df['speech'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = val_encoded_df['label'].reset_index(drop=True) if not val_encoded_df.empty else pd.Series(dtype='int')
                else:
                    X_val_pd_aligned = pd.Series(dtype='object')
                    y_val_encoded_pd_aligned = pd.Series(dtype='int')

                test_data_to_encode = pd.DataFrame({'party': y_test_pd_orig, 'speech': X_test_pd_orig})
                test_encoded_df = encode_labels_with_map(test_data_to_encode, config.party_map)
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
                    
                    current_model_plot_dir = results_manager.get_plot_output_dir(model_type_to_run)
                    Path(current_model_plot_dir).mkdir(parents=True, exist_ok=True)

                    run_model_pipeline(
                        X_train_pd_aligned, y_train_encoded_pd_aligned,
                        X_val_pd_aligned, y_val_encoded_pd_aligned, 
                        X_test_pd_aligned, y_test_encoded_pd_aligned,
                        config,  # L'oggetto config completo
                        model_type=model_type_to_run,  # es: 'svm' 
                        model_config=current_model_config,  # es: svm_config
                        random_state=seed,
                        congress_year=year_str,
                        model_plot_output_dir=current_model_plot_dir,
                        fixed_vocabulary_dict=fixed_sklearn_vocabulary,
                        results_manager=results_manager,
                        congress_feature_importance=congress_feature_importance
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
                
    print("\n--- Calculating and plotting averaged results ---")
    for model_type_to_plot in models_to_run:
        # ✅ NUOVO: Usa ResultsManager invece delle variabili globali
        plot_info = results_manager.get_model_plotting_info(model_type_to_plot)
        detailed_log_for_avg = results_manager.get_detailed_log_path(model_type_to_plot)
        avg_log_path_for_plot = plot_info["avg_log_path"]
        output_dir_for_plot = plot_info["output_dir"]
        
        # ✅ ASSICURATI che la directory esista
        Path(output_dir_for_plot).mkdir(parents=True, exist_ok=True)

        try:
            # ✅ Usa la variabile corretta detailed_log_for_avg (che ora viene da ResultsManager)
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
            
    print("Saving feature importance dictionary...")
    save_feature_importance(congress_feature_importance, model_type="svm")  # or whatever model_type you're using
    print(f"Total congress-seed combinations saved: {len(congress_feature_importance)}")

    print("\n--- Script finished ---")