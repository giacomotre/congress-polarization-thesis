import os
import pandas as pd
import joblib
import nltk
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from config_loader import load_config
from pipeline_utils import encode_labels_with_map, clean_text_for_tfidf
from plotting_utils import plot_performance_metrics, plot_confusion_matrix


# ------ Loading Config -------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "svm_config.yaml"
config = load_config(CONFIG_PATH)
print("Loaded config:", json.dumps(config, indent=4))

MAX_FEATURES = config.get("tfidf_max_features", 10000)
NGRAM_RANGE = tuple(config.get("ngram_range", [1, 2]))
TEST_SIZE = config['split_params']['test_size']
VALIDATION_SIZE = config['split_params']['validation_size']
RANDOM_STATE = config['split_params']['random_state']
PARTY_MAP = config['party_map']


# ------ Main Loop -------
def run_tfidf_pipeline(congress_year: str, config: dict):
    print(f"\n Running pipeline for Congress {congress_year}")
    timing = {}
    start_total = time.time()

    # ------ Data Loading -------   
    input_path = f"data/merged/house_db/house_merged_{congress_year}.csv"
    df = pd.read_csv(input_path)

    # ------ Data Splitting - Leave-out speaker approach -------
    print("Performing leave-out speaker split...")
    start_time = time.time()

    unique_speakers = df['speakerid'].unique()

    # Split speakers into train_val and test sets using config values
    train_val_speaker, test_speaker = train_test_split(
        unique_speakers,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_speaker, val_speaker = train_test_split(
        train_val_speaker,
        test_size=VALIDATION_SIZE, 
        random_state=RANDOM_STATE
    )

    # Create dataframes based on speaker IDs
    train_df = df[df["speakerid"].isin(train_speaker)].reset_index(drop=True)
    val_df = df[df["speakerid"].isin(val_speaker)].reset_index(drop=True)
    test_df = df[df["speakerid"].isin(test_speaker)].reset_index(drop=True)

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

    # Class Distribution Logging 
    print("  - Class distribution after split:")
    train_counts = Counter(train_df['party'])
    val_counts = Counter(val_df['party'])
    test_counts = Counter(test_df['party'])
    print(f"    Train: {dict(train_counts)}")
    print(f"    Validation: {dict(val_counts)}")
    print(f"    Test: {dict(test_counts)}")

    split_time = time.time() - start_time
    print(f"Split complete in {split_time:.2f} seconds.")
    
    # ------ Encoding -------
    print("Encoding labels using encode_labels_with_map...")

    # Create temporary DataFrames to use your function
    train_df_encoded = pd.DataFrame({'party': y_train})
    val_df_encoded = pd.DataFrame({'party': y_val})
    test_df_encoded = pd.DataFrame({'party': y_test})

    # Apply the encoding function
    y_train_encoded = encode_labels_with_map(train_df_encoded, PARTY_MAP)['label']
    y_val_encoded = encode_labels_with_map(val_df_encoded, PARTY_MAP)['label']
    y_test_encoded = encode_labels_with_map(test_df_encoded, PARTY_MAP)['label']

    # Convert back to Series if needed for consistency with scikit-learn inputs
    y_train_encoded = pd.Series(y_train_encoded)
    y_val_encoded = pd.Series(y_val_encoded)
    y_test_encoded = pd.Series(y_test_encoded)

    print("Labels encoded.") 
    
    # --- Pipeline ---
    #Text cleaning, Vectorization and

    # Define the cleaning step using FunctionTransformer
    #FuntionTransformer is an utility that let you integrate into the pipeline a user func
    # This wraps your clean_text_for_tfidf function for use in a pipeline.
    # We use a lambda function and .apply() assuming clean_text_for_tfidf works on a single string
    # and X_train etc are pandas Series. validate=False might be needed for Series input.
    cleaning_step = FunctionTransformer(lambda x: x.apply(clean_text_for_tfidf), validate=False)

    # Define the TF-IDF vectorization step
    tfidf_step = TfidfVectorizer() # Parameters like max_features, ngram_range will be tuned

    # Define the SVM model step
    svm_step = LinearSVC() # Parameters like C can also be tuned

    # Create the pipeline chaining the steps
    pipeline = Pipeline([
        ('cleaning', cleaning_step),
        ('tfidf', tfidf_step),
        ('svm', svm_step)
    ])

    print("Pipeline created with Cleaning, TF-IDF, and LinearSVC.")

    # --- Optimization (Hyperparameter Tuning) ---
    print("Starting hyperparameter tuning using GridSearchCV...")
    start_time_tuning = time.time()

    # Define the parameter grid for tuning
    # Parameters are referenced by step_name__parameter_name
    param_grid = {
        'tfidf__max_features': [5000, 10000, 20000], # Example max_features values
        'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)], # Example ngram_range values
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1) # n_jobs=-1 uses all available cores

    grid_search.fit(X_train, y_train_encoded)

    print(f"Hyperparameter tuning complete in {time.time() - start_time_tuning:.2f} seconds.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # The best pipeline is now stored in grid_search.best_estimator_
    best_pipeline = grid_search.best_estimator_

    # --- 7. Final Model Training (on Combined Train/Validation Data) ---
    print("Training final model (best pipeline) on combined train/validation data...")
    start_time_final_train = time.time()

    # Combine the ORIGINAL train and validation data (X)
    X_train_val_combined = pd.concat([X_train, X_val]) # Assuming X_train, X_val are Series/DataFrames

    # Combine the ENCODED train and validation labels (y)
    y_train_val_combined_encoded = pd.concat([pd.Series(y_train_encoded).reset_index(drop=True),
                                            pd.Series(y_val_encoded).reset_index(drop=True)]) # Concatenate encoded Series

    # Train the best pipeline on the combined ORIGINAL training and validation data and ENCODED labels.
    final_pipeline = best_pipeline.fit(X_train_val_combined, y_train_val_combined_encoded)

    print(f"Final model training complete in {time.time() - start_time_final_train:.2f} seconds.")

    # --- Testing on Test Data ---
    print("Evaluating final model on test data...")
    start_time_test = time.time()

    y_test_pred_final = final_pipeline.predict(X_test)

    # Evaluate the predictions against the encoded test labels
    final_accuracy = accuracy_score(y_test_encoded, y_test_pred_final)
    # Use average='weighted' for f1_score in multi-class classification
    final_f1_weighted = f1_score(y_test_encoded, y_test_pred_final, average='weighted')

    print(f"Evaluation complete in {time.time() - start_time_test:.2f} seconds.")

    print("\n--- Final Test Results ---")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Weighted F1 Score: {final_f1_weighted:.4f}")

    # Optional: Print more detailed metrics
    # You might need the original label mapping to interpret the confusion matrix and classification report
    # Assuming you have a reverse_party_map or similar to get original labels from encoded integers
    # reverse_party_map = {v: k for k, v in party_map.items()}
    # target_names = [reverse_party_map[i] for i in sorted(reverse_party_map.keys())]

    # print("\nClassification Report:")
    # print(classification_report(y_test_encoded, y_test_pred_final, target_names=target_names))

    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test_encoded, y_test_pred_final))

    print("-" * 25)

    # ------ Metrics -------

    # --- Calculate Additional Metrics (ROC-AUC, Confusion Matrix) ---

    auc = None # Initialize AUC to None, will check for binary later

    # To calculate ROC-AUC, we need the decision function scores from the SVM step
    # We need to pass the data through the pipeline's transformation steps first (cleaning, vectorization)
    # then get the decision scores from the final SVM estimator.
    try:
        # Apply transformations up to the step before the estimator (SVM)
        # final_pipeline.transform(X_test) gets output after tfidf_step
        X_test_transformed_for_svm = final_pipeline.transform(X_test)

        # Access the SVM step within the pipeline and get decision function scores
        # This assumes your SVM step in the pipeline is named 'svm'
        decision_scores = final_pipeline.named_steps['svm'].decision_function(X_test_transformed_for_svm)

        # ROC-AUC is primarily for binary classification or multi-class (one-vs-rest)
        # The standard roc_auc_score expects binary labels (0 or 1) and confidence scores/decision function for the positive class.
        # For multi-class decision_function returns shape (n_samples, n_classes)
        # We'll stick to the binary check as in your original code for simplicity.
        if len(set(y_test_encoded)) == 2:
            # If binary, decision_function returns (n_samples,) or (n_samples, 1)
            # roc_auc_score needs the score for the positive class (usually column 1 if shape is (n_samples, 2))
            # LinearSVC decision_function for binary returns shape (n_samples,) which is correct input
            auc = roc_auc_score(y_test_encoded, decision_scores)
        else:
            # If multi-class and you want OVR AUC, you'd need a different approach
            # For this example, we'll set AUC to None for multi-class, matching your original logic
            auc = None # Or handle multi-class AUC calculation if needed

    except Exception as e:
        print(f"Could not calculate ROC-AUC or Decision Function: {e}")
        auc = None # Ensure auc is None if calculation fails

    cm = confusion_matrix(y_test_encoded, y_test_pred_final).tolist()

    # --- Timing ---
    evaluation_time = time.time() - start_time_test
    print(f"Metric calculation complete in {evaluation_time:.2f} seconds.")
    # Update the timing dictionary (assuming timing dict is defined)
    if 'timing' in locals() or 'timing' in globals():
        timing["evaluation_sec"] = round(evaluation_time, 2)
        # Make sure total_sec is updated after all steps are done
        # timing["total_sec"] = round(time.time() - start_total, 2)

    # --- Print Metrics ---
    print("\n--- Final Test Results ---")
    print(f"Accuracy : {final_accuracy:.4f}") # Use final_accuracy from previous step
    print(f"Weighted F1 Score: {final_f1_weighted:.4f}") # Use final_f1_weighted from previous step
    if auc is not None:
        print(f"ROC-AUC  : {auc:.4f}")
    # Corrected: Use y_test_encoded in the confusion_matrix printout
    print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_test_pred_final))
    # Optional: Print classification report
    # Assuming target_names can be derived from your party_map and encoded labels
    try:
        # Need reverse map to get class names from encoded integers
        reverse_party_map = {v: k for k, v in PARTY_MAP.items()}
        # Ensure target_names are in the order of encoded labels (0, 1, 2...)
        unique_encoded_labels = sorted(list(set(y_test_encoded)))
        target_names = [reverse_party_map[i] for i in unique_encoded_labels]
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_test_pred_final, target_names=target_names))
    except Exception as e:
        print(f"Could not print Classification Report: {e}")

    print("-" * 25)
    
    # --- Log results ---
    log_path = "logs/tfidf_svm_performance.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("year,accuracy,f1_score,auc\n") # Added auc column
    with open(log_path, "a") as f:
        # Corrected: Use final_accuracy, final_f1_weighted, and auc (or 'NA')
        f.write(f"{congress_year},{final_accuracy:.4f},{final_f1_weighted:.4f},{auc if auc is not None else 'NA'}\n")

    result_json = {
        "year": congress_year,
        "accuracy": round(final_accuracy, 4),
        "f1_score": round(final_f1_weighted, 4), # Use weighted F1
        "auc": round(auc, 4) if auc is not None else "NA",
        "confusion_matrix": cm # Use cm calculated with encoded labels
    }

    # Make sure total_sec timing is updated at the very end of the script
    if 'timing' in locals() or 'timing' in globals():
        timing["total_sec"] = round(time.time() - start_total, 2)
        result_json.update(timing)


    with open(f"logs/tfidf_results_{congress_year}.json", "w") as jf:
        json.dump(result_json, jf, indent=4)

    plot_confusion_matrix(f"logs/tfidf_results_{congress_year}.json")

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "svm_config.yaml"
    config = load_config(config_path)
    congress_years = [f"{i:03}" for i in range(75, 112)]
    for year in congress_years:
        try:
            run_tfidf_pipeline(year, config)
        except FileNotFoundError:
            print(f"⚠️  Skipping Congress {year}: CSV file not found.")
    plot_performance_metrics("logs/tfidf_svm_performance.csv")