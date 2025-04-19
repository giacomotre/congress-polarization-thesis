import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocessing.pipeline_utils import preprocess_df_for_tfidf

def run_tfidf_pipeline(congress_year: str):
    print(f"\n Running pipeline for Congress {congress_year}")

    # Paths
    input_path = f"data/house_merged_{congress_year}.csv"
    processed_path = f"data/processed/speeches_tfidf_{congress_year}.csv"
    vectorizer_path = f"models/tfidf_vectorizer_{congress_year}.pkl"
    model_path = f"models/svm_classifier_{congress_year}.pkl"
    X_train_path = f"data/processed/X_train_vec_{congress_year}.pkl"
    X_test_path = f"data/processed/X_test_vec_{congress_year}.pkl"

    # Load and preprocess
    df = pd.read_csv(input_path)
    clean_df = preprocess_df_for_tfidf(df, text_col="speech")

    # Save cleaned text
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    clean_df.to_csv(processed_path, index=False)

    X = clean_df["speech"]
    y = clean_df["party"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Vectorize
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Save vectorized matrices and model
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(X_train_vec, X_train_path)
    joblib.dump(X_test_vec, X_test_path)

    # Train
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    joblib.dump(svm, model_path)

    # Evaluate
    y_pred = svm.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    auc = ""
    if len(set(y)) == 2:
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_pred_bin = lb.transform(y_pred)
        auc = roc_auc_score(y_test_bin, y_pred_bin)
        print(f"AUC: {auc:.3f}")

    # Log results
    log_path = "logs/tfidf_svm_performance.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("year,accuracy,f1_score,auc\n")
    with open(log_path, "a") as f:
        f.write(f"{congress_year},{accuracy:.4f},{f1:.4f},{auc if auc else 'NA'}\n")

if __name__ == "__main__":
    congress_years = [f"{i:03}" for i in range(79, 112)]
    for year in congress_years:
        try:
            run_tfidf_pipeline(year)
        except FileNotFoundError:
            print(f"⚠️  Skipping Congress {year}: CSV file not found.")
