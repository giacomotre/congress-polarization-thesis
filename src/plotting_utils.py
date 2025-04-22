import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                

PLOT_DIR = "outputs\plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_lines_from_log(log_path: str = "logs/tfidf_svm_performance.csv") -> None:
    
    if not os.path.isfile(log_path):
        print(f"[plot_lines_from_log] ‼ No log found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    df["year"] = df["year"].astype(int)
    df.sort_values("year", inplace=True)

    # Common kwargs for consistent style
    common = dict(linewidth=2, markeredgecolor="black")

    # ---- Accuracy ----
    plt.figure(figsize=(8, 5))
    plt.plot(df["year"], df["accuracy"], marker="o", **common)
    _style_axes("Accuracy Over Years", "Congress Year", "Accuracy")
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_line.png"), dpi=300)
    plt.close()

    # ---- F1 Score ----
    plt.figure(figsize=(8, 5))
    plt.plot(df["year"], df["f1_score"], marker="s", color="forestgreen", **common)
    _style_axes("F1 Score Over Years", "Congress Year", "F1 Score")
    plt.savefig(os.path.join(PLOT_DIR, "f1_score_line.png"), dpi=300)
    plt.close()

    # ---- AUC ----
    plt.figure(figsize=(8, 5))
    plt.plot(df["year"], pd.to_numeric(df["auc"], errors="coerce"),
             marker="^", color="firebrick", **common)
    _style_axes("AUC Over Years", "Congress Year", "AUC")
    plt.savefig(os.path.join(PLOT_DIR, "auc_line.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred,
                        labels=None,
                        fname: str = "confusion_matrix.png") -> None:
    """
    Saves a publication‑ready confusion‑matrix heat‑map.
    Call this right after you have y_true/y_pred in your pipeline.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(y_true)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"})
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
    plt.close()

# ---- helper ----
def _style_axes(title, xlabel, ylabel):
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

