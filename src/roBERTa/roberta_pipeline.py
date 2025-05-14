import os
import time
import json
import pandas as pd
from pathlib import Path
from collections import Counter # Import Counter for class distribution logging

import torch
import torch.nn as nn
from torch.optim import AdamW # Explicitly import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split # Ensure this is imported
import numpy as np # Import numpy for potential use in confusion matrix handling

from plotting_utils import plot_performance_metrics, plot_confusion_matrix

try:
    from config_loader import load_config
    from pipeline_utils import encode_labels_with_map
    from dataset import CongressSpeechDataset
    from roberta_model import RobertaClassifier
    from engine import train_epoch, evaluate_epoch
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure dataset.py, roberta_model.py, engine.py, config_loader.py, and pipeline_utils.py are accessible in your Python path.")
    exit()

# --- Direct Imports for Config Loading ---
config_path = Path(__file__).parent.parent / "config" / "roberta_config.yaml"
try:
    config = load_config(config_path)
    print("Loaded configuration parameters:")
    print(json.dumps(config, indent=4)) # Print loaded config for verification
except FileNotFoundError:
    print(f"Error: Config file not found at {config_path}")
    exit() # Exit if config file is missing
except Exception as e:
    print(f"Error loading config file: {e}")
    exit()


# --- Directory Paths (Defined directly in script) ---
DATA_DIR = Path("data/merged/house_db") # Directory where your congress CSVs are
OUTPUT_DIR = Path("roberta_output")     # Base output directory
LOG_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure output directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --- Main Pipeline Function ---

def run_roberta_pipeline(congress_year: str, config: dict):
    print(f"\n--- Running RoBERTa pipeline for Congress {congress_year} ---")
    start_total_time = time.time()

    # --- Configuration ---
    MODEL_NAME = config['model_params']['model_name']
    NUM_LABELS = config['model_params']['num_labels']
    MAX_TOKEN_LEN = config['model_params']['max_token_len']
    BATCH_SIZE = config['model_params']['batch_size']
    NUM_EPOCHS = config['model_params']['num_epochs']
    LEARNING_RATE = float(config['model_params']['learning_rate']) # Ensure float for scientific notation
    WEIGHT_DECAY = float(config['model_params'].get('weight_decay', 0.0)) # Use .get for optional params

    TEST_SIZE = config['split_params']['test_size']
    VALIDATION_SIZE = config['split_params']['validation_size']
    RANDOM_STATE = config['split_params']['random_state']

    MIN_WORD_COUNT = config['filter_params']['min_word_count']
    PARTY_MAP = config['filter_params']['party_map']
    REVERSE_PARTY_MAP = {v: k for k, v in PARTY_MAP.items()} # Create reverse map for logging

    # Path 
    csv_file_path = DATA_DIR / f"house_merged_{congress_year}.csv"
    if not csv_file_path.exists():
        print(f"⚠️  Skipping Congress {congress_year}: File not found at {csv_file_path}")
        return

    performance_log_path = LOG_DIR / "roberta_performance.csv"
    year_results_json_path = LOG_DIR / f"roberta_results_{congress_year}.json"
    model_save_path = MODELS_DIR / f"roberta_classifier_{congress_year}.pth" # Optional: path to save model

    # Device Setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    start_time = time.time()
    df = pd.read_csv(csv_file_path)

    # Encode party label
    df_processed = encode_labels_with_map(df, party_map=PARTY_MAP)
    print(f"  - Mapped party labels. Final sample count for split: {len(df_processed)}")

    preprocessing_time = time.time() - start_time
    print(f"Preprocessing complete in {preprocessing_time:.2f} seconds.")

    # --- Leave-out Speaker Split ---
    print("Performing leave-out speaker split...")
    start_time = time.time()

    unique_speakers = df_processed['speakerid'].unique()
    if len(unique_speakers) < 2: # Need at least 2 speakers for train/test split
        print(f"⚠️  Skipping Congress {congress_year}: Not enough unique speakers ({len(unique_speakers)}) for split.")
        return

    # Split speakers into train_val and test sets using config values
    train_val_speaker, test_speaker = train_test_split(
        unique_speakers,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Split train_val speakers into train and validation sets using config values
    if len(train_val_speaker) < 2: # Need at least 2 speakers in train_val for validation split
        print(f"⚠️  Skipping Congress {congress_year}: Not enough unique speakers ({len(train_val_speaker)}) in train_val set for validation split.")
        return

    train_speaker, val_speaker = train_test_split(
        train_val_speaker,
        test_size=VALIDATION_SIZE, # This is relative to train_val_speaker size
        random_state=RANDOM_STATE
    )

    # Create dataframes based on speaker IDs
    train_df = df_processed[df_processed["speakerid"].isin(train_speaker)].reset_index(drop=True)
    val_df = df_processed[df_processed["speakerid"].isin(val_speaker)].reset_index(drop=True)
    test_df = df_processed[df_processed["speakerid"].isin(test_speaker)].reset_index(drop=True)

    print(f"  - Train speakers: {len(train_speaker)}, Samples: {len(train_df)}")
    print(f"  - Validation speakers: {len(val_speaker)}, Samples: {len(val_df)}")
    print(f"  - Test speakers: {len(test_speaker)}, Samples: {len(test_df)}")

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

    # Check if any split is empty or has only one class (for binary classification)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print(f"⚠️  Skipping Congress {congress_year}: One or more data splits resulted in zero samples.")
        return
    # Check if both classes are present in the training data AFTER the split
    if len(train_df['label'].unique()) < 2:
        # Identify the single class present for better logging
        single_class = train_df['party'].iloc[0] if not train_df.empty else 'N/A'
        print(f"⚠️  Skipping Congress {congress_year}: Only one class ({single_class}) present in training data after split.")
        return

    # --- Dataset and DataLoader Creation ---
    print("Creating Datasets and DataLoaders...")
    start_time = time.time()

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create dataset instances using the split DataFrames and config values
    train_dataset = CongressSpeechDataset(train_df, tokenizer, party_map=PARTY_MAP, max_token_len=MAX_TOKEN_LEN)
    val_dataset = CongressSpeechDataset(val_df, tokenizer, party_map=PARTY_MAP, max_token_len=MAX_TOKEN_LEN)
    test_dataset = CongressSpeechDataset(test_df, tokenizer, party_map=PARTY_MAP, max_token_len=MAX_TOKEN_LEN)

    # Create DataLoader instances using config batch size
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataloader_time = time.time() - start_time
    print(f"DataLoader creation complete in {dataloader_time:.2f} seconds.")


    # --- Model, Optimizer, Criterion Setup ---
    print("Setting up model, optimizer, and criterion...")
    start_time = time.time()

    model = RobertaClassifier(MODEL_NAME, NUM_LABELS)
    model.to(device) # Move model to GPU

    # Use config values for optimizer parameters
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Calculate and Use Weighted Loss for Imbalance ---
    # Calculate class weights for CrossEntropyLoss based on training data distribution
    train_labels = train_df['label'].values # Get numerical labels from the training DataFrame
    class_counts = Counter(train_labels)
    # Ensure counts for all expected labels are present, even if 0
    # This is important for creating the weight tensor in the correct order (0, 1)
    sorted_labels = sorted(PARTY_MAP.values()) # Get the sorted numerical labels (e.g., [0, 1])
    class_counts_sorted = [class_counts.get(label, 0) for label in sorted_labels]


    # Calculate weights: inverse frequency (total_samples / class_count)
    total_samples = sum(class_counts_sorted)
    # Avoid division by zero if a class is missing entirely in the training set
    # If a class has 0 samples in the training set, its weight should be 0
    # A common approach is to use 1 / count, then normalize.
    # Let's use the total_samples / count approach, handling zero counts.
    class_weights = [total_samples / count if count > 0 else 0 for count in class_counts_sorted]

    # Convert weights to a PyTorch tensor and move to device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Use weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print(f"  - Using weighted CrossEntropyLoss with weights: {class_weights_tensor.cpu().numpy()}")
    # -------------------------------------------


    setup_time = time.time() - start_time
    print(f"Setup complete in {setup_time:.2f} seconds.")


    # --- Training Loop ---
    print("Starting training...")
    training_start_time = time.time()

    # best_val_metric = -float('inf') # Track best validation metric for potential early stopping

    for epoch in range(NUM_EPOCHS): # Use config num_epochs
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluate on validation set after each epoch
        val_metrics = evaluate_epoch(model, val_dataloader, criterion, device) # Pass criterion
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Validation F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"  Validation AUC: {val_metrics['auc'] if isinstance(val_metrics['auc'], str) else val_metrics['auc']:.4f}")

        # Optional: Add logic here for early stopping or saving best model
        # based on validation_metrics['accuracy'] or another metric

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")


    training_time = time.time() - training_start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")

    # --- Final Evaluation on Test Set ---
    print("Evaluating on test set...")
    start_time = time.time()
    # Pass criterion to evaluate_epoch for test set as well
    test_metrics = evaluate_epoch(model, test_dataloader, criterion, device)
    evaluation_time = time.time() - start_time
    print(f"Evaluation complete in {evaluation_time:.2f} seconds.")

    print("\n--- Test Metrics ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"AUC     : {test_metrics['auc'] if isinstance(test_metrics['auc'], str) else test_metrics['auc']:.4f}")
    print("Confusion Matrix:")
    # Safely get label names from config party_map for printing
    # Need to handle the case where predicted/true labels might be a subset of all possible labels
    # The confusion_matrix from evaluate_epoch returns labels found in the data, use those if available
    cm_labels = test_metrics.get('labels', list(PARTY_MAP.values())) # Use labels from metrics if available, else default to mapped integers
    party_names = {v: k for k, v in PARTY_MAP.items()} # Reverse map for printing
    cm_label_names_print = [f"{party_names.get(l, str(l))}({l})" for l in sorted(cm_labels)] # Map integers back to party names for printing


    print(f"  Predicted -> {'  '.join(cm_label_names_print)}")
    # Print confusion matrix rows - need to align with cm_labels order
    cm_matrix_data = test_metrics['confusion_matrix'] # This is a list of lists from evaluate_epoch
    # Ensure cm_matrix_data has the expected shape based on cm_labels
    expected_cm_shape = (len(cm_labels), len(cm_labels))

    # Add a check for the shape before attempting to print row by row
    if len(cm_matrix_data) != expected_cm_shape[0] or (len(cm_matrix_data) > 0 and len(cm_matrix_data[0]) != expected_cm_shape[1]):
        print("Warning: Confusion matrix shape mismatch. Printing raw matrix data.")
        print(cm_matrix_data)
    else:
        # Print rows aligned with sorted cm_labels
        for i, true_label_int in enumerate(sorted(cm_labels)):
            row_data = cm_matrix_data[i]
            print(f"True {party_names.get(true_label_int, str(true_label_int))}({true_label_int}) | {'  '.join(map(str, row_data))}")


    # --- Logging Results ---
    print("Logging results...")
    # Log performance metrics to CSV
    log_data = {
        "year": congress_year,
        "accuracy": test_metrics['accuracy'],
        "f1_score": test_metrics['f1_score'],
        "auc": test_metrics['auc'] if not isinstance(test_metrics['auc'], str) else 'NA'
    }

    # Append to CSV log file (create if it doesn't exist)
    pd.DataFrame([log_data]).to_csv(
        performance_log_path,
        mode='a', # Append mode
        header=not performance_log_path.exists(), # Write header only if file is new
        index=False
    )
    print(f"Performance metrics appended to {performance_log_path}")

    # Log full results (including confusion matrix) to JSON
    full_results = {
        "year": congress_year,
        "test_metrics": test_metrics,
        "preprocessing_time_sec": round(preprocessing_time, 2),
        "split_time_sec": round(split_time, 2),
        "dataloader_time_sec": round(dataloader_time, 2),
        "setup_time_sec": round(setup_time, 2),
        "training_time_sec": round(training_time, 2),
        "evaluation_time_sec": round(evaluation_time, 2),
        "total_pipeline_time_sec": round(time.time() - start_total_time, 2),
        "config_params": config # Log the loaded parameters from config
    }

    with open(year_results_json_path, 'w') as f:
        json.dump(full_results, f, indent=4)
    print(f"Full results saved to {year_results_json_path}")

    # Plot confusion matrix for the year
    plot_confusion_matrix(year_results_json_path, output_dir=PLOTS_DIR)


    total_pipeline_time = time.time() - start_total_time
    print(f"--- Pipeline for Congress {congress_year} finished in {total_pipeline_time:.2f} seconds ---\n")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the path to your RoBERTa config file
    # ADJUST THIS PATH based on your repository structure
    config_path = Path(__file__).parent.parent / "config" / "roberta_config.yaml"

    # Load the configuration
    try:
        pipeline_config = load_config(config_path)
        print("Loaded configuration parameters:")
        # Print only the parameters loaded from config
        print(json.dumps(pipeline_config, indent=4))
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit() # Exit if config file is missing
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()


    # Define the list of congress years to process - GENERATE FROM RANGE IN CONFIG
    try:
        start_year = pipeline_config['processing_range']['start_year']
        end_year = pipeline_config['processing_range']['end_year']
        # Generate the list of years (inclusive) and format as 3-digit strings
        congress_years_to_process = [str(year).zfill(3) for year in range(start_year, end_year + 1)]
        print(f"Processing years from {start_year} to {end_year}: {congress_years_to_process}")

    except KeyError as e:
        print(f"Error: Missing 'processing_range' or sub-keys ('start_year', 'end_year') in config file.")
        print("Please ensure your roberta_config.yaml has a 'processing_range' section with 'start_year' and 'end_year'.")
        exit()
    except Exception as e:
        print(f"Error generating years from config range: {e}")
        exit()


    # Optional: Clear previous logs/plots if starting fresh (add to config?)
    # if LOG_DIR.exists():
    #      for f in LOG_DIR.glob("*"): f.unlink()
    # if PLOTS_DIR.exists():
    #      for f in PLOTS_DIR.glob("*"): f.unlink()


    for year in congress_years_to_process:
        # Pass the loaded config parameters to the pipeline function
        run_roberta_pipeline(year, pipeline_config)

    # After processing all years, plot the performance metrics across years
    # Directory paths are accessed from the variables defined at the top
    performance_csv_path = LOG_DIR / "roberta_performance.csv"
    plots_output_dir = PLOTS_DIR

    if performance_csv_path.exists():
        print(f"\n--- Plotting overall performance from {performance_csv_path} ---")
        plot_performance_metrics(performance_csv_path, output_dir=plots_output_dir)
    else:
        print("\nNo performance log found to plot.")
