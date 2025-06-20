import os
import pandas as pd
import joblib
import json
import time
import numpy as np
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import classification_report

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map

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
TEST_SIZE = 0.15  # Fixed 15% as per requirements
DEFAULT_RANDOM_STATE = split_params.get('random_state', 42)

# Define the 3 seeds for cross-period evaluation
#CROSS_PERIOD_SEEDS = [42, 123, 456]  # You can modify these as needed
CROSS_PERIOD_SEEDS = split_params.get('seeds', [DEFAULT_RANDOM_STATE])

data_params = common_params.get('data_params', {})
CONGRESS_YEAR_START = data_params.get('congress_year_start', 76)
CONGRESS_YEAR_END = data_params.get('congress_year_end', 112)

PARTY_MAP = common_params.get('party_map', {})
if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
    print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")

# --- Define Model Configuration ---
model_config = unified_config.get('svm', {})
DEFAULT_MAX_ITER = model_config.get('max_iter', 5000)

# Path to optimal hyperparameters CSV file
OPTIMAL_PARAMS_CSV_PATH = "logs/tfidf_svm_performance_detailed.csv"  # Adjust path as needed

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
CROSS_PERIOD_LOG_PATH = "logs/cross_period_evaluation_results.csv"

# CSV header for cross-period results
cross_period_header = "seed,train_year,test_year,accuracy,f1_score,auc,tn,fp,fn,tp\n"

# Initialize log file
if os.path.exists(CROSS_PERIOD_LOG_PATH):
    os.remove(CROSS_PERIOD_LOG_PATH)
    print(f"Deleted existing cross-period log file: {CROSS_PERIOD_LOG_PATH}")

with open(CROSS_PERIOD_LOG_PATH, "w") as f:
    f.write(cross_period_header)

# --- Helper Functions ---
def load_optimal_hyperparameters():
    """Load optimal hyperparameters from CSV file with individual parameter columns"""
    try:
        if not Path(OPTIMAL_PARAMS_CSV_PATH).exists():
            print(f"ERROR: Optimal parameters file not found at {OPTIMAL_PARAMS_CSV_PATH}")
            return None
        
        # Read CSV - should now have individual parameter columns
        df = pd.read_csv(OPTIMAL_PARAMS_CSV_PATH)
        print(f"Loaded optimal hyperparameters from {OPTIMAL_PARAMS_CSV_PATH}")
        print(f"CSV columns: {list(df.columns)}")
        print(f"CSV shape: {df.shape}")
        print("First few rows:")
        print(df.head())
        
        # Expected columns: seed, year, accuracy, f1_score, auc, model_C, tfidf_norm
        required_columns = ['seed', 'year', 'model_C', 'tfidf_norm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Create a dictionary for quick lookup: (seed, year) -> best_params
        params_dict = {}
        
        for idx, row in df.iterrows():
            try:
                seed = int(row['seed'])
                year = str(int(row['year'])).zfill(3)  # Convert to 3-digit format
                
                # Extract parameters directly from columns
                optimal_params = {
                    'svm_C': float(row['model_C']),
                    'tfidf_norm': str(row['tfidf_norm'])
                }
                
                params_dict[(seed, year)] = optimal_params
                print(f"Loaded params for seed {seed}, year {year}: {optimal_params}")
                
            except Exception as e:
                print(f"Warning: Could not parse row {idx}: {e}")
                print(f"Row data: {row.to_dict()}")
                continue
        
        print(f"Successfully loaded parameters for {len(params_dict)} seed-year combinations")
        return params_dict
        
    except Exception as e:
        print(f"ERROR loading optimal hyperparameters: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_hyperparameters_for_seed_year(params_dict, seed, year):
    """Get optimal hyperparameters for a specific seed and year"""
    key = (seed, year)
    
    if params_dict is None or key not in params_dict:
        print(f"Warning: No optimal parameters found for seed {seed}, year {year}. Using defaults.")
        # Return reasonable defaults
        return {
            'tfidf_norm': 'l2',
            'svm_C': 1.0
        }
    
    optimal_params = params_dict[key]
    print(f"Using parameters for seed {seed}, year {year}: {optimal_params}")
    return optimal_params

def set_all_seeds(seed_value):
    """Set random seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    # Note: sklearn uses numpy's random state

def extract_unique_speaker_id(speakerid):
    """Extract the last 6 digits of speaker ID as unique identifier"""
    return str(speakerid)[-6:]

def load_congress_data(year_str):
    """Load data for a specific congress year"""
    input_csv_path = Path(f"data/processed/house_db/house_cleaned_{year_str}.csv")
    
    if not input_csv_path.exists():
        print(f"⚠️  Data file not found for Congress {year_str}")
        return None
    
    try:
        df = pd.read_csv(input_csv_path)
        
        # Basic validation
        if df.empty or not all(col in df.columns for col in ['speech', 'party', 'speakerid']):
            print(f"⚠️  Invalid data structure for Congress {year_str}")
            return None
        
        # Clean data
        df.dropna(subset=['speech', 'party', 'speakerid'], inplace=True)
        
        if df.empty:
            print(f"⚠️  No data remaining after cleaning for Congress {year_str}")
            return None
        
        # Add unique speaker ID column
        df['unique_speaker_id'] = df['speakerid'].apply(extract_unique_speaker_id)
        
        print(f"✅ Loaded Congress {year_str}: {len(df)} speeches, {len(df['unique_speaker_id'].unique())} unique speakers")
        return df
        
    except Exception as e:
        print(f"❌ Error loading Congress {year_str}: {e}")
        return None

def train_model_on_congress(train_data, fixed_vocabulary_dict, random_state, optimal_params):
    """Train SVM model on training congress data using optimal hyperparameters"""
    print("Training model on source congress...")
    
    # Use all data from training congress (no train/val split within source)
    X_train = train_data["speech"]
    y_train = train_data["party"]
    
    # Encode labels
    train_data_to_encode = pd.DataFrame({'party': y_train, 'speech': X_train})
    train_encoded_df = encode_labels_with_map(train_data_to_encode, PARTY_MAP)
    
    if train_encoded_df.empty:
        print("⚠️  No valid data after encoding")
        return None, None
    
    X_train_encoded = train_encoded_df['speech'].reset_index(drop=True)
    y_train_encoded = train_encoded_df['label'].reset_index(drop=True)
    
    if len(np.unique(y_train_encoded)) < 2:
        print("⚠️  Insufficient classes for training")
        return None, None
    
    # Create TF-IDF vectorizer with optimal hyperparameters for this seed-year
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=fixed_vocabulary_dict,
        ngram_range=(1, 2),  # Matching your baseline
        lowercase=False,
        stop_words=None,
        norm=optimal_params['tfidf_norm']
    )
    
    # Transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_encoded)
    
    # Train SVM model with optimal C parameter
    svm_model = LinearSVC(
        C=optimal_params['svm_C'],
        max_iter=DEFAULT_MAX_ITER,
        random_state=random_state
    )
    
    svm_model.fit(X_train_tfidf, y_train_encoded)
    
    print(f"Model trained successfully on {len(X_train_encoded)} samples")
    print(f"Using parameters: C={optimal_params['svm_C']}, norm='{optimal_params['tfidf_norm']}'")
    return svm_model, tfidf_vectorizer

def evaluate_on_test_congress(model, vectorizer, test_data, training_speaker_ids, random_state):
    """Evaluate model on test congress with speaker exclusion"""
    print("Evaluating on target congress...")
    
    # Step 1: Exclude speakers who were in training data
    test_data['unique_speaker_id'] = test_data['speakerid'].apply(extract_unique_speaker_id)
    filtered_test_data = test_data[~test_data['unique_speaker_id'].isin(training_speaker_ids)].copy()
    
    print(f"Original test data: {len(test_data)} speeches")
    print(f"After speaker exclusion: {len(filtered_test_data)} speeches")
    print(f"Excluded {len(test_data) - len(filtered_test_data)} speeches from {len(training_speaker_ids)} training speakers")
    
    if filtered_test_data.empty:
        print("⚠️  No test data remaining after speaker exclusion")
        return None
    
    # Step 2: Randomly sample 15% of filtered data
    sample_size = max(1, int(len(filtered_test_data) * TEST_SIZE))
    
    if sample_size >= len(filtered_test_data):
        # If 15% would be the entire dataset, use all
        test_sample = filtered_test_data.copy()
    else:
        test_sample = filtered_test_data.sample(n=sample_size, random_state=random_state)
    
    print(f"Final test set size: {len(test_sample)} speeches ({sample_size} requested)")
    
    # Step 3: Prepare test data
    X_test = test_sample["speech"]
    y_test = test_sample["party"]
    
    # Encode test labels
    test_data_to_encode = pd.DataFrame({'party': y_test, 'speech': X_test})
    test_encoded_df = encode_labels_with_map(test_data_to_encode, PARTY_MAP)
    
    if test_encoded_df.empty:
        print("⚠️  No valid test data after encoding")
        return None
    
    X_test_encoded = test_encoded_df['speech'].reset_index(drop=True)
    y_test_encoded = test_encoded_df['label'].reset_index(drop=True)
    
    if len(np.unique(y_test_encoded)) < 2:
        print("⚠️  Insufficient classes in test data")
        return None
    
    # Step 4: Transform test data and make predictions
    X_test_tfidf = vectorizer.transform(X_test_encoded)
    y_pred = model.predict(X_test_tfidf)
    
    # Step 5: Calculate metrics (matching your baseline)
    try:
        # Accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # F1 Score (weighted)
        f1_score = sklearn_f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        # AUC Score
        auc_score = None
        try:
            if hasattr(model, "decision_function"):
                decision_scores = model.decision_function(X_test_tfidf)
                auc_score = roc_auc_score(y_test_encoded, decision_scores)
        except Exception as auc_e:
            print(f"Could not calculate AUC: {auc_e}")
            auc_score = None
        
        # Additional metrics for verification (not logged but printed)
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Print results
        print(f"Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        if auc_score is not None:
            print(f"  AUC: {auc_score:.4f}")
        else:
            print(f"  AUC: N/A")
        print(f"  Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1_score, 4),
            'auc': round(auc_score, 4) if auc_score is not None else "NA",
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]), 
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def run_cross_period_evaluation():
    """Main function to run cross-period evaluation"""
    
    # Load fixed vocabulary
    sklearn_vocab_load_path = Path("data/vocabulary_dumps/1_word/global_vocabulary_processed_bigram_100_min_df_sklearn_from_sklearn.joblib")
    
    if not sklearn_vocab_load_path.exists():
        print(f"ERROR: Fixed vocabulary file not found at {sklearn_vocab_load_path}")
        print("Please run the global vocabulary generation script first.")
        return
    
    print(f"Loading fixed scikit-learn vocabulary from {sklearn_vocab_load_path}...")
    fixed_sklearn_vocabulary = joblib.load(sklearn_vocab_load_path)
    print(f"Loaded fixed vocabulary with {len(fixed_sklearn_vocabulary)} terms.")
    
    # Load optimal hyperparameters
    optimal_params_dict = load_optimal_hyperparameters()
    if optimal_params_dict is None:
        print("ERROR: Could not load optimal hyperparameters. Exiting.")
        return
    
    # Generate congress years to process
    congress_years_to_process = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END + 1)]
    print(f"Congress years to process: {congress_years_to_process}")
    
    # Calculate total combinations
    total_combinations = len(congress_years_to_process) * (len(congress_years_to_process) - 1) * len(CROSS_PERIOD_SEEDS)
    print(f"Total combinations to process: {total_combinations}")
    
    combination_count = 0
    
    # Main evaluation loop
    for current_seed in CROSS_PERIOD_SEEDS:
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION WITH SEED: {current_seed}")
        print(f"{'='*60}")
        
        # Set all random seeds for this run
        set_all_seeds(current_seed)
        
        for train_year in congress_years_to_process:
            print(f"\n{'---'*20}")
            print(f"TRAINING CONGRESS: {train_year}")
            print(f"{'---'*20}")
            
            # Get optimal hyperparameters for this seed-year combination
            optimal_params = get_hyperparameters_for_seed_year(optimal_params_dict, current_seed, train_year)
            
            # Load training data
            train_data = load_congress_data(train_year)
            if train_data is None:
                print(f"⚠️  Skipping training year {train_year} - no data available")
                continue
            
            # Extract unique speaker IDs from training data
            training_speaker_ids = set(train_data['unique_speaker_id'].unique())
            print(f"Training speakers: {len(training_speaker_ids)} unique speakers")
            
            # Train model on this congress with optimal parameters
            model, vectorizer = train_model_on_congress(train_data, fixed_sklearn_vocabulary, current_seed, optimal_params)
            if model is None or vectorizer is None:
                print(f"⚠️  Failed to train model on Congress {train_year}")
                continue
            
            # Test on all other congresses
            for test_year in congress_years_to_process:
                if test_year == train_year:
                    continue  # Skip same-period testing
                
                combination_count += 1
                print(f"\n--- Combination {combination_count}/{total_combinations}: {train_year} → {test_year} ---")
                
                # Load test data
                test_data = load_congress_data(test_year)
                if test_data is None:
                    print(f"⚠️  Skipping test year {test_year} - no data available")
                    continue
                
                try:
                    # Evaluate model on test congress
                    results = evaluate_on_test_congress(
                        model, vectorizer, test_data, training_speaker_ids, current_seed
                    )
                    
                    if results is not None:
                        # Log results
                        with open(CROSS_PERIOD_LOG_PATH, "a") as f:
                            f.write(f"{current_seed},{train_year},{test_year},"
                                f"{results['accuracy']},{results['f1_score']},{results['auc']},"
                                f"{results['tn']},{results['fp']},{results['fn']},{results['tp']}\n")
                        
                        print(f"✅ Completed: {train_year} → {test_year}")
                    else:
                        print(f"❌ Failed: {train_year} → {test_year}")
                        
                except Exception as e:
                    print(f"❌ Error in {train_year} → {test_year}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Cleanup for memory management
            del model, vectorizer, train_data
            if 'test_data' in locals():
                del test_data
    
    print(f"\n{'='*60}")
    print("CROSS-PERIOD EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {CROSS_PERIOD_LOG_PATH}")
    
    # Display summary statistics
    try:
        results_df = pd.read_csv(CROSS_PERIOD_LOG_PATH)
        if not results_df.empty:
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total combinations processed: {len(results_df)}")
            print(f"Average accuracy: {results_df['accuracy'].mean():.4f}")
            print(f"Average F1 score: {results_df['f1_score'].mean():.4f}")
            
            # Convert 'NA' to NaN for numeric operations
            results_df['auc_numeric'] = pd.to_numeric(results_df['auc'], errors='coerce')
            auc_mean = results_df['auc_numeric'].mean()
            if not pd.isna(auc_mean):
                print(f"Average AUC: {auc_mean:.4f}")
            else:
                print("Average AUC: N/A (no valid AUC scores)")
                
            print(f"\nResults by seed:")
            seed_summary = results_df.groupby('seed')[['accuracy', 'f1_score']].agg(['mean', 'std']).round(4)
            print(seed_summary)
        else:
            print("No results to summarize")
    except Exception as e:
        print(f"Error generating summary: {e}")

# ------ Main Execution -------
if __name__ == "__main__":
    print("Starting Cross-Period Evaluation Script")
    print(f"Seeds to use: {CROSS_PERIOD_SEEDS}")
    print(f"Congress range: {CONGRESS_YEAR_START} to {CONGRESS_YEAR_END}")
    print(f"Test size: {TEST_SIZE * 100}%")
    print(f"Optimal parameters will be loaded from: {OPTIMAL_PARAMS_CSV_PATH}")
    
    # Run the evaluation
    run_cross_period_evaluation()
    
    print("\n--- Script finished ---")