import os
import pandas as pd
import joblib
import nltk
import json
import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pipeline_utils import preprocess_df_for_tfidf
from config_loader import load_config
from pipeline_utils import encode_labels
from plotting_utils import plot_performance_metrics, plot_confusion_matrix

#loading config
config_path = Path(__file__).parent.parent / "config" / "svm_config.yaml"
config = load_config(config_path)
print("Loaded config:", json.dumps(config, indent=4))
max_features = config.get("tfidf_max_features", 10000)
ngram_range = tuple(config.get("ngram_range", [1, 2]))
test_size = config.get("test_size", 0.2)
random_state = config.get("random_state", 42)

def run_tfidf_pipeline(congress_year: str, config: dict):
    print(f"\n Running pipeline for Congress {congress_year}")
    timing = {}
    start_total = time.time()

    # Paths
    input_path = f"data/merged/house_db/house_merged_{congress_year}.csv"
    processed_path = f"data/processed/speeches_tfidf_{congress_year}.csv"
    
    # Corrected: Initialize counts before the if/else block
    removed_short_speeches_count = 0
    removed_duplicate_speeches_count = 0

    if Path(processed_path).exists():
        print("Found cached clean CSV – loading instead of preprocessing")
        clean_df = pd.read_csv(processed_path)
    else:
        # Load and preprocess
        df = pd.read_csv(input_path)
        print("Preprocessing speeches...")
        start = time.time()
        clean_df, removed_short_speeches_count, removed_duplicate_speeches_count = preprocess_df_for_tfidf(df, text_col="speech") # added clip count
        timing["preprocessing_sec"] = round(time.time() - start, 2)
        print(f"Preprocessing complete. {len(clean_df)} speeches after cleaning.")

        # Save cleaned text
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        clean_df.to_csv(processed_path, index=False)

    # ------ Leave-out speaker approach split -------

    unique_speakers = clean_df['speakerid'].unique()

    train_val_speaker, test_speaker = train_test_split(unique_speakers,
                                                    test_size=config.get("test_size", 0.2),
                                                    random_state=config.get("random_state", 42)
                                                    )

    validation_size = config.get("validation_size", 0.25)

    train_speaker, val_speaker = train_test_split(train_val_speaker,
                                                test_size= validation_size,
                                                random_state=config.get("random_state", 42)
                                                )

    train_df = clean_df[clean_df["speakerid"].isin(train_speaker)]
    val_df = clean_df[clean_df["speakerid"].isin(val_speaker)]
    test_df = clean_df[clean_df["speakerid"].isin(test_speaker)]

    # Corrected lines: Assign the Series directly
    X_train_df = train_df["speech"]
    y_train_df = train_df["party"]
    X_val_df = val_df["speech"]
    y_val_df = val_df["party"]
    X_test_df = test_df["speech"]
    y_test_df = test_df["party"]


    le, y_train = encode_labels(labels=y_train_df, encoder_path=f"models/label_encoder_{congress_year}.pkl")
    _, y_val = encode_labels(labels=y_val_df, encoder_path=f"models/label_encoder_{congress_year}.pkl")
    _, y_test = encode_labels(labels=y_test_df, encoder_path=f"models/label_encoder_{congress_year}.pkl")

    # Vectorize
    print("Vectorizing text with TF-IDF...")
    start = time.time()
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    # Corrected lines: Use the Series directly
    X_train_vec = tfidf.fit_transform(X_train_df)
    X_val_vec = tfidf.transform(X_val_df)
    X_test_vec = tfidf.transform(X_test_df)
    timing["vectorization_sec"] = round(time.time() - start, 2)
    print("Vectorization complete.")

    # Train + optimizing
    print("Training Linear SVM model...")
    start = time.time()

    best_accuracy = 0
    best_params = {}

    for current_max_features in [5000, 10000, 20000]:
        for current_ngram_range in [(1,1), (1, 2), (2, 2)]:
            print(f"Testing parameters: max_features={current_max_features}, ngram_range={current_ngram_range}")

            tfidf_tuned = TfidfVectorizer(max_features=current_max_features, ngram_range=current_ngram_range)
            # Corrected lines: Use the Series directly
            X_train_vec_tuned = tfidf_tuned.fit_transform(X_train_df)
            X_val_vec_tuned = tfidf_tuned.transform(X_val_df)

            #Train with current parameters
            svm_tuned = LinearSVC()
            svm_tuned.fit(X_train_vec_tuned, y_train)

            # Evaluate on validation set
            y_val_pred = svm_tuned.predict(X_val_vec_tuned)
            current_accuracy = accuracy_score(y_val, y_val_pred)

            print(f"Validation Accuracy: {current_accuracy:.4f}")

            # Check if current parameters are better
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_params = {"max_features": current_max_features, "ngram_range": current_ngram_range}

    print(f"Best parameters found: {best_params} with Validation Accuracy: {best_accuracy:.4f}")

    timing["training_sec"] = round(time.time() - start, 2)
    print("Model training and optimization complete.")

    # ------ Testing set  -------
    start = time.time()

    # Combine train and validation data
    # Corrected lines: Concatenate the Series directly
    X_train_val_df_combined = pd.concat([X_train_df, X_val_df])
    y_train_val_df_combined = pd.concat([y_train_df, y_val_df])


    _, y_train_val = encode_labels(labels=y_train_val_df_combined, encoder_path=f"models/label_encoder_{congress_year}.pkl") # Re-encode combined labels

    # Re-vectorize combined data and test data with best parameters
    tfidf_final = TfidfVectorizer(max_features=best_params["max_features"], ngram_range=best_params["ngram_range"])
    # Corrected lines: Use the combined and test Series directly
    X_train_val_vec = tfidf_final.fit_transform(X_train_val_df_combined)
    X_test_vec_final = tfidf_final.transform(X_test_df)

    # Train final model on combined data
    svm_final = LinearSVC()
    svm_final.fit(X_train_val_vec, y_train_val)

    # Evaluate on the test set
    y_test_pred_final = svm_final.predict(X_test_vec_final)
    final_accuracy = accuracy_score(y_test, y_test_pred_final)
    final_f1 = f1_score(y_test, y_test_pred_final, average="weighted")

    # ------ Metrics  -------

    #ROC-AUC
    auc = "NA"
    # Corrected lines: Use svm_final and X_test_vec_final for decision_function
    if len(set(y_test)) == 2:                              # binary task only
        decision_scores = svm_final.decision_function(X_test_vec_final)   # shape (n_samples,)
        auc = roc_auc_score(y_test, decision_scores)  # y_test is already 0/1 ints

    timing["evaluation_sec"] = round(time.time() - start, 2)

    # Print metrics
    print(f"Accuracy : {final_accuracy:.3f}")
    print(f"F1 Score : {final_f1:.3f}")
    if auc != "NA":
        print(f"ROC-AUC  : {auc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_final))

    # Log results
    log_path = "logs/tfidf_svm_performance.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("year,accuracy,f1_score,auc\n")
    with open(log_path, "a") as f:
        # Corrected lines: Use final_accuracy and final_f1
        f.write(f"{congress_year},{final_accuracy:.4f},{final_f1:.4f},{auc if auc else 'NA'}\n")

    # le.classes_ is an ndarray like array(['D', 'R'], dtype='<U1')
    label_names = le.classes_.tolist()

    # Save JSON log
    # Corrected lines: Use y_test and y_test_pred_final for confusion_matrix
    cm = confusion_matrix(y_test, y_test_pred_final).tolist()
    result_json = {
        "removed_short_speeches": removed_short_speeches_count,
        "removed_duplicate_speeches": removed_duplicate_speeches_count,
        "year": congress_year,
        "accuracy": round(final_accuracy, 4),
        "f1_score": round(final_f1, 4),
        "auc": round(auc, 4) if auc else "NA",
        "confusion_matrix": cm,
        "labels": label_names
    }

    timing["total_sec"] = round(time.time() - start_total, 2)
    result_json.update(timing)

    with open(f"logs/tfidf_results_{congress_year}.json", "w") as jf:
        json.dump(result_json, jf, indent=4)

    plot_confusion_matrix(f"logs/tfidf_results_{congress_year}.json")

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "svm_config.yaml"
    config = load_config(config_path)
    congress_years = [f"{i:03}" for i in range(79, 81)]
    for year in congress_years:
        try:
            run_tfidf_pipeline(year, config)
        except FileNotFoundError:
            print(f"⚠️  Skipping Congress {year}: CSV file not found.")
    plot_performance_metrics("logs/tfidf_svm_performance.csv")