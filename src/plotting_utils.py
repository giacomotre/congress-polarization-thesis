import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_performance_metrics(csv_filepath: str, output_dir: str = "plots"):
    """
    Plots accuracy, F1-score, and AUC from a CSV log file across years as separate plots.

    Args:
        csv_filepath (str): The path to the CSV file containing performance metrics.
        output_dir (str): Directory to save the plot. Defaults to "plots".
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: Performance metrics CSV not found at {csv_filepath}")
        return

    df = pd.read_csv(csv_filepath)

    if df.empty:
        print(f"No data to plot in {csv_filepath}")
        return

    # Ensure 'year' column is treated as numeric for correct plotting order
    df['year'] = pd.to_numeric(df['year'])

    # --- Plot Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(df['year'], df['accuracy'], marker='o', linestyle='-')
    plt.title('Accuracy Across Congress Years')
    plt.xlabel('Congress Year')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(df['year']) # Ensure all years are shown on x-axis
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path_acc = os.path.join(output_dir, "accuracy_across_years.png")
    plt.savefig(plot_path_acc)
    print(f"Accuracy plot saved to {plot_path_acc}")
    plt.close()

    # --- Plot F1 Score ---
    plt.figure(figsize=(8, 5))
    plt.plot(df['year'], df['f1_score'], marker='o', linestyle='-', color='orange')
    plt.title('F1 Score Across Congress Years')
    plt.xlabel('Congress Year')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.xticks(df['year']) # Ensure all years are shown on x-axis
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path_f1 = os.path.join(output_dir, "f1_score_across_years.png")
    plt.savefig(plot_path_f1)
    print(f"F1 Score plot saved to {plot_path_f1}")
    plt.close()

    # --- Plot AUC ---
    if 'auc' in df.columns and df['auc'].dtype != object:
        df['auc'] = pd.to_numeric(df['auc'], errors='coerce')
        plt.figure(figsize=(8, 5))
        plt.plot(df['year'], df['auc'], marker='o', linestyle='-', color='green')
        plt.title('AUC Across Congress Years')
        plt.xlabel('Congress Year')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.xticks(df['year']) # Ensure all years are shown on x-axis
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_path_auc = os.path.join(output_dir, "auc_across_years.png")
        plt.savefig(plot_path_auc)
        print(f"AUC plot saved to {plot_path_auc}")
        plt.close()


def plot_confusion_matrix(json_filepath: str, output_dir: str = "plots"):
    """
    Plots a confusion matrix from a JSON log file.

    Args:
        json_filepath (str): The path to the JSON file containing the confusion matrix.
        output_dir (str): Directory to save the plot. Defaults to "plots".
    """
    if not os.path.exists(json_filepath):
        print(f"Error: JSON log not found at {json_filepath}")
        return

    with open(json_filepath, 'r') as f:
        results = json.load(f)

    if "confusion_matrix" not in results or "labels" not in results:
        print(f"Error: 'confusion_matrix' or 'labels' not found in {json_filepath}")
        return

    cm = results["confusion_matrix"]
    labels = results["labels"]
    year = results.get("year", "Unknown Year") # Get year for title

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for Congress {year}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"confusion_matrix_{year}.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix plot for {year} saved to {plot_path}")
    plt.close()

if __name__ == '__main__':
    # Example Usage (assuming you have these files)
    # plot_performance_metrics("logs/tfidf_svm_performance.csv")
    # plot_confusion_matrix("logs/tfidf_results_079.json")
    pass # Remove pass and uncomment lines above to test