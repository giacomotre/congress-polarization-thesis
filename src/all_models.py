import os
import pandas as pd
import joblib
import nltk
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

# Import necessary components from sklearn and cuml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # or use dask_ml's version if suitable for your data handling
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from dask_ml.model_selection import GridSearchCV # Keep train_test_split from sklearn if you prefer
from cuml.feature_extraction.text import TfidfVectorizer # Using cuML's TF-IDF
from cuml.naive_bayes import ComplementNB # Using cuML's ComplementNB
from cuml.svm import LinearSVC # Using cuML's LinearSVC
from cuml.linear_model import LogisticRegression # Using cuML's LogisticRegression

# Import utility functions
from config_loader import load_config
from pipeline_utils import encode_labels_with_map, clean_text_for_tfidf
from plotting_utils import plot_performance_metrics, plot_confusion_matrix

# ------ Loading Unified Config -------
CONFIG_PATH_UNIFIED = Path(__file__).parent.parent / "config" / "config.yaml"

try:
    unified_config = load_config(CONFIG_PATH_UNIFIED)
    print("Loaded unified config:", json.dumps(unified_config, indent=4))
except FileNotFoundError:
    print(f"Error: Unified config file not found at {CONFIG_PATH_UNIFIED}. Please create it.")
    exit() # Exit if config file is missing

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

PARTY_MAP = common_params.get('party_map', {}) # Assuming party mapping is in the common section

# Ensure PARTY_MAP is not empty and contains expected parties
if not PARTY_MAP or not all(party in PARTY_MAP for party in ['D', 'R']):
     print("Warning: PARTY_MAP in config is missing or incomplete. Ensure D and R are mapped.")


# --- Define Model Configurations Dictionary ---
# Access model-specific configs using keys
model_configs = {
    'bayes': unified_config.get('bayes', {}),
    'svm': unified_config.get('svm', {}),
    'lr': unified_config.get('logistic_regression', {}) # Key matches the section name in config.yaml
}

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define detailed log paths for each model type (using consistent naming)
detailed_log_paths = {
    'bayes': "logs/tfidf_bayes_performance_detailed.csv",
    'svm': "logs/tfidf_svm_performance_detailed.csv",
    'lr': "logs/tfidf_lr_performance_detailed.csv"
}

# Remove existing detailed log files to start fresh and write headers
for model_type, log_path in detailed_log_paths.items():
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Deleted existing detailed log file: {log_path}")
    with open(log_path, "w") as f:
        f.write("seed,year,accuracy,f1_score,auc\n")

# Function to run a single model pipeline
# Now accepts model_config which is a subset of the unified config
def run_model_pipeline(X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, model_type: str, model_config: dict, random_state: int, congress_year: str, party_map: dict):
    print(f"\n --- Running {model_type.upper()} pipeline for Congress {congress_year} with seed {random_state} ---")
    timing = {}
    start_time_total = time.time()

    # --- Pipeline ---
    # Text cleaning and Vectorization (common steps in pipeline, but params from model_config)
    cleaning_step = FunctionTransformer(lambda x: x.apply(clean_text_for_tfidf), validate=False)

    # Use TF-IDF parameters from the specific model's configuration subset
    tfidf_max_features = model_config.get("tfidf_max_features", 10000)
    tfidf_ngram_range = tuple(model_config.get("ngram_range", [1, 2]))
    tfidf_step = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)

    # Define the model step based on model_type
    model_step = None
    model_param_grid = {} # Parameter grid specific to the model step

    if model_type == 'bayes':
        model_step = ComplementNB()
        # Add parameters for ComplementNB tuning if needed, using 'bayes__' prefix
        # model_param_grid['bayes__alpha'] = model_config.get('bayes_alpha_grid', [0.1, 0.5, 1.0]) # Example
    elif model_type == 'svm':
        model_step = LinearSVC()
        # Add parameters for LinearSVC tuning if needed, using 'svm__' prefix
        # model_param_grid['svm__C'] = model_config.get('svm_C_grid', [0.1, 1.0, 10.0]) # Example
    elif model_type == 'lr':
        # Use cuML's LogisticRegression with L2 penalty
        model_step = LogisticRegression(penalty='l2')
        # Add parameters for LogisticRegression tuning (e.g., 'C'), using 'lr__' prefix
        model_param_grid['lr__C'] = model_config.get('lr_C_grid', [0.01, 0.1, 1.0, 10.0])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create the pipeline
    pipeline = Pipeline([
        ('cleaning', cleaning_step),
        ('tfidf', tfidf_step),
        (model_type, model_step) # Use model_type as step name
    ])

    print(f"Pipeline created with Cleaning, TF-IDF, and {model_type.upper()}.")

    # --- Optimization (Hyperparameter Tuning) ---
    print("Starting hyperparameter tuning using GridSearchCV...")
    start_time_tuning = time.time()

    # Define the full parameter grid for tuning (combining TF-IDF and model params)
    # Get TF-IDF tuning grids from the specific model's config subset
    param_grid = {
        'tfidf__max_features': model_config.get("tfidf_max_features_grid", [5000, 10000, 20000]),
        'tfidf__ngram_range': [tuple(nr) for nr in model_config.get("ngram_range_grid", [[1, 1], [1, 2]])],
    }
    # Add model-specific parameters to the grid
    param_grid.update(model_param_grid)

    # Create the GridSearchCV object
    # Use accuracy as scoring metric for simplicity, can be changed
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train_encoded)

    tuning_time = time.time() - start_time_tuning
    print(f"Hyperparameter tuning complete in {tuning_time:.2f} seconds.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # The best pipeline is now stored in grid_search.best_estimator_
    best_pipeline = grid_search.best_estimator_

    # --- Final Model Training (on Combined Train/Validation Data) ---
    print("Training final model (best pipeline) on combined train/validation data...")
    start_time_final_train = time.time()

    # Combine the ORIGINAL train and validation data (X)
    X_train_val_combined = pd.concat([X_train, X_val])

    # Combine the ENCODED train and validation labels (y)
    y_train_val_combined_encoded = pd.concat([pd.Series(y_train_encoded).reset_index(drop=True),
                                            pd.Series(y_val_encoded).reset_index(drop=True)])

    # Train the best pipeline on the combined ORIGINAL training and validation data and ENCODED labels.
    final_pipeline = best_pipeline.fit(X_train_val_combined, y_train_val_combined_encoded)
    
    # --- Saving the trained pipeline ---
    print(f"Saving the trained {model_type.upper()} pipeline...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True) # Ensure the models directory exists

    # Create a descriptive filename
    model_filename = f"{model_dir}/tfidf_{model_type}_{congress_year}_seed{random_state}_pipeline.joblib"

    try:
        joblib.dump(final_pipeline, model_filename) # for model explainability, the .joblib file is the model artifact.
        print(f"Trained pipeline saved to {model_filename}")
    except Exception as e:
        print(f"Error saving the trained pipeline: {e}")

    final_train_time = time.time() - start_time_final_train
    print(f"Final model training complete in {final_train_time:.2f} seconds.")

    # --- Testing on Test Data ---
    print("Evaluating final model on test data...")
    start_time_test = time.time()

    y_test_pred_final = final_pipeline.predict(X_test)

    # Evaluate the predictions against the encoded test labels
    final_accuracy = accuracy_score(y_test_encoded, y_test_pred_final)
    final_f1_weighted = f1_score(y_test_encoded, y_test_pred_final, average='weighted')

    evaluation_time = time.time() - start_time_test
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    print(f"\n--- Final Test Results ({model_type.upper()}) ---")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted:.4f}")

    # --- Calculate Additional Metrics (ROC-AUC, Confusion Matrix) ---
    auc = None
    try:
        if model_type == 'bayes':
            # ComplementNB provides predict_proba
            probability_scores = final_pipeline.predict_proba(X_test)
            if len(set(y_test_encoded)) == 2:
                auc = roc_auc_score(y_test_encoded, probability_scores[:, 1])
            else:
                # For multi-class, calculate OVR AUC
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                y_test_encoded_onehot = lb.fit_transform(y_test_encoded)
                if y_test_encoded_onehot.shape[1] == probability_scores.shape[1]:
                    auc = roc_auc_score(y_test_encoded_onehot, probability_scores, average='macro') # Or 'weighted'


        elif model_type in ['svm', 'lr']:
            # LinearSVC and LogisticRegression provide decision_function
            # Check if predict_proba is available for AUC calculation, especially for LR
            if hasattr(final_pipeline, 'predict_proba'):
                 probability_scores = final_pipeline.predict_proba(X_test)
                 if len(set(y_test_encoded)) == 2:
                     auc = roc_auc_score(y_test_encoded, probability_scores[:, 1])
                 else:
                     from sklearn.preprocessing import LabelBinarizer
                     lb = LabelBinarizer()
                     y_test_encoded_onehot = lb.fit_transform(y_test_encoded)
                     if y_test_encoded_onehot.shape[1] == probability_scores.shape[1]:
                         auc = roc_auc_score(y_test_encoded_onehot, probability_scores, average='macro')

            elif hasattr(final_pipeline, 'decision_function'):
                decision_scores = final_pipeline.decision_function(X_test)
                if len(set(y_test_encoded)) == 2:
                     auc = roc_auc_score(y_test_encoded, decision_scores)
                else:
                     auc = roc_auc_score(y_test_encoded, decision_scores, multi_class='ovr')


    except Exception as e:
        print(f"Could not calculate ROC-AUC for {model_type}: {e}")
        auc = None


    cm = confusion_matrix(y_test_encoded, y_test_pred_final).tolist()

    # Get target names for confusion matrix from the common party_map
    try:
        reverse_party_map = {v: k for k, v in party_map.items()}
        unique_encoded_labels = sorted(list(set(y_test_encoded)))
        target_names = [reverse_party_map[i] for i in unique_encoded_labels]
    except Exception as e:
        print(f"Could not get target names: {e}")
        target_names = [str(i) for i in sorted(list(set(y_test_encoded)))] # Fallback


    if auc is not None:
        print(f"ROC-AUC          : {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_test_pred_final))
    try:
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_test_pred_final, target_names=target_names))
    except Exception as e:
        print(f"Could not print Classification Report: {e}")


    print("-" * 25)

    # --- Log results ---
    # Determine the correct detailed log path based on model type
    detailed_log_path = detailed_log_paths[model_type]

    with open(detailed_log_path, "a") as f:
        f.write(f"{random_state},{congress_year},{final_accuracy:.4f},{final_f1_weighted:.4f},{auc if auc is not None else 'NA'}\n")

    result_json = {
        "seed": random_state,
        "year": congress_year,
        "accuracy": round(final_accuracy, 4),
        "f1_score": round(final_f1_weighted, 4),
        "auc": round(auc, 4) if auc is not None else "NA",
        "confusion_matrix": cm,
        "labels": target_names,
        "timing": {
            "tuning_sec": round(tuning_time, 2),
            "final_train_sec": round(final_train_time, 2),
            "evaluation_sec": round(evaluation_time, 2),
            "total_pipeline_sec": round(time.time() - start_time_total, 2)
        }
    }

    # Save JSON results with seed in the filename
    json_log_path = f"logs/tfidf_{model_type}_results_{congress_year}_seed{random_state}.json"
    with open(json_log_path, "w") as jf:
        json.dump(result_json, jf, indent=4)

    # Plot confusion matrix for this run
    try:
        plot_confusion_matrix(json_log_path)
    except Exception as e:
        print(f"Error plotting confusion matrix for {model_type} {congress_year} seed {random_state}: {e}")

    return result_json

# ------ Main Execution -------
if __name__ == "__main__":
    congress_years = [f"{i:03}" for i in range(CONGRESS_YEAR_START, CONGRESS_YEAR_END)]

    # Define the list of models to run
    models_to_run = ['bayes', 'svm', 'lr'] # Add or remove models here

    for seed in SEEDS:
        print(f"\n--- Starting runs for seed: {seed} ---")
        for year in congress_years:
            print(f"\nProcessing Congress Year: {year}")
            input_path = f"data/merged/house_db/house_merged_{year}.csv"

            if not os.path.exists(input_path):
                print(f"⚠️  Skipping Congress {year} (seed {seed}): CSV file not found at {input_path}.")
                continue

            try:
                # ------ Data Loading -------
                print("Loading data...")
                df = pd.read_csv(input_path)
                print("Data loaded.")

                # ------ Data Splitting - Leave-out speaker approach -------
                print("Performing leave-out speaker split...")
                start_time_split = time.time()

                unique_speakers = df['speakerid'].unique()

                train_val_speaker, test_speaker = train_test_split(
                    unique_speakers,
                    test_size=TEST_SIZE,
                    random_state=seed
                )

                train_speaker, val_speaker = train_test_split(
                    train_val_speaker,
                    test_size=VALIDATION_SIZE,
                    random_state=seed
                )

                train_df = df[df["speakerid"].isin(train_speaker)].reset_index(drop=True)
                val_df = df[df["speakerid"].isin(val_speaker)].reset_index(drop=True)
                test_df = df[df["speakerid"].isin(test_speaker)].reset_index(drop=True)

                split_time = time.time() - start_time_split
                print(f"Split complete in {split_time:.2f} seconds.")
                print(f"  - Train speakers: {len(train_speaker)}, Samples: {len(train_df)}")
                print(f"  - Validation speakers: {len(val_speaker)}, Samples: {len(val_df)}")
                print(f"  - Test speakers: {len(test_speaker)}, Samples: {len(test_df)}")


                # Separate features (X) and labels (y)
                X_train = train_df["speech"]
                y_train = train_df["party"]
                X_val = val_df["speech"]
                y_val = val_df["party"]
                X_test = test_df["speech"]
                y_test = test_df["party"]


                # ------ Encoding -------
                print("Encoding labels using encode_labels_with_map...")
                start_time_encode = time.time()

                train_df_encoded = pd.DataFrame({'party': y_train})
                val_df_encoded = pd.DataFrame({'party': y_val})
                test_df_encoded = pd.DataFrame({'party': y_test})

                y_train_encoded = encode_labels_with_map(train_df_encoded, PARTY_MAP)['label']
                y_val_encoded = encode_labels_with_map(val_df_encoded, PARTY_MAP)['label']
                y_test_encoded = encode_labels_with_map(test_df_encoded, PARTY_MAP)['label']

                y_train_encoded = pd.Series(y_train_encoded).reset_index(drop=True)
                y_val_encoded = pd.Series(y_val_encoded).reset_index(drop=True)
                y_test_encoded = pd.Series(y_test_encoded).reset_index(drop=True)


                encode_time = time.time() - start_time_encode
                print(f"Labels encoded in {encode_time:.2f} seconds.")

                # --- Run each specified model ---
                for model_type in models_to_run:
                    # Get the configuration subset for the current model
                    model_config = model_configs.get(model_type)
                    if model_config is None:
                        print(f"Warning: Configuration for model '{model_type}' not found in config.yaml. Skipping.")
                        continue

                    run_model_pipeline(X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded,
                                       model_type=model_type, model_config=model_config, random_state=seed, congress_year=year, party_map=PARTY_MAP)


            except Exception as e:
                print(f"❌ An error occurred during processing for Congress {year} with seed {seed}: {e}")

    # --- Calculate Averages per Year across Seeds and Generate Plots ---
    print("\n--- Calculating and plotting averaged results ---")

    # Dictionary to map model types to their average log paths and plot output directories
    model_plotting_info = {
        'bayes': {"avg_log_path": "logs/tfidf_bayes_performance_avg.csv", "output_dir": "plots/bayes"},
        'svm': {"avg_log_path": "logs/tfidf_svm_performance_avg.csv", "output_dir": "plots/svm"},
        'lr': {"avg_log_path": "logs/tfidf_lr_performance_avg.csv", "output_dir": "plots/lr"}
    }

    for model_type in models_to_run:
        info = model_plotting_info.get(model_type)
        if info is None:
             print(f"Warning: Plotting info for model '{model_type}' not found. Skipping plotting for this model.")
             continue

        detailed_log_path = detailed_log_paths[model_type]
        avg_log_path = info["avg_log_path"]
        output_dir = info["output_dir"]

        try:
            df_detailed = pd.read_csv(detailed_log_path)
            # Ensure 'auc' column is numeric, converting 'NA' to NaN for averaging
            df_detailed['auc'] = pd.to_numeric(df_detailed['auc'], errors='coerce')

            # Calculate mean metrics per year, grouped by the 'year' column
            df_avg = df_detailed.groupby('year')[['accuracy', 'f1_score', 'auc']].mean(numeric_only=True).reset_index()

            # Optional: Calculate standard deviation if you want to visualize variability later
            df_std = df_detailed.groupby('year')[['accuracy', 'f1_score', 'auc']].std(numeric_only=True).reset_index()
            df_std.rename(columns={'accuracy':'accuracy_std', 'f1_score':'f1_score_std', 'auc':'auc_std'}, inplace=True)
            df_avg = df_avg.merge(df_std, on='year')


            # Sort the averaged results by year for correct plotting order
            df_avg['year_int'] = df_avg['year'].astype(int)
            df_avg = df_avg.sort_values('year_int').drop('year_int', axis=1)

            # Save the averaged results to a new CSV file
            df_avg.to_csv(avg_log_path, index=False)
            print(f"Saved averaged performance metrics for {model_type.upper()} to {avg_log_path}")

            # Generate Plots using Averaged Data
            print(f"\nGenerating performance plots for {model_type.upper()} using averaged data...")
            plot_performance_metrics(avg_log_path, output_dir=output_dir)


        except FileNotFoundError:
            print(f"Error: Detailed performance log not found for {model_type.upper()} at {detailed_log_path}. Cannot calculate averages or plot.")
        except Exception as e:
            print(f"An error occurred during calculating averages or plotting for {model_type.upper()}: {e}")


    print("\n--- Script finished ---")