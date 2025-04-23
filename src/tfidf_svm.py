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

    X = clean_df["speech"]
    le, y = encode_labels(
    labels       = clean_df["party"],
    encoder_path = f"models/label_encoder_{congress_year}.pkl",
    )
    

    # Train/test split
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    timing["split_sec"] = round(time.time() - start, 2)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Vectorize
    print("Vectorizing text with TF-IDF...")
    start = time.time()
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    timing["vectorization_sec"] = round(time.time() - start, 2)
    print("Vectorization complete.")

    # Train
    print("Training Linear SVM model...")
    start = time.time()
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    timing["training_sec"] = round(time.time() - start, 2)
    print("Model training complete.")

    # Evaluate
    start = time.time()

    # Hard‑label predictions (still needed for accuracy / F1 / confusion matrix)
    y_pred = svm.predict(X_test_vec)

    # Classic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    #ROC‑AUC
    auc = "NA"
    if len(set(y)) == 2:                              # binary task only
        decision_scores = svm.decision_function(X_test_vec)   # shape (n_samples,)
        auc = roc_auc_score(y_test, decision_scores)  # y_test is already 0/1 ints

    timing["evaluation_sec"] = round(time.time() - start, 2)

    # Print metrics
    print(f"Accuracy : {accuracy:.3f}")
    print(f"F1 Score : {f1:.3f}")
    if auc != "NA":
        print(f"ROC‑AUC  : {auc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Log results
    log_path = "logs/tfidf_svm_performance.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("year,accuracy,f1_score,auc\n")
    with open(log_path, "a") as f:
        f.write(f"{congress_year},{accuracy:.4f},{f1:.4f},{auc if auc else 'NA'}\n")

    # le.classes_ is an ndarray like array(['D', 'R'], dtype='<U1')
    label_names = le.classes_.tolist()          

    # Save JSON log
    cm = confusion_matrix(y_test, y_pred).tolist()
    result_json = {
        "removed_short_speeches": removed_short_speeches_count,
        "removed_duplicate_speeches": removed_duplicate_speeches_count,
        "year": congress_year,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
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