import os
import time
import json
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split # Ensure this is imported

# Import your custom modules
from dataset import CongressSpeechDataset # Assuming dataset.py is in the same directory or accessible
from roberta_model import RobertaClassifier     # Assuming model.py is in the same directory or accessible
from engine import train_epoch, evaluate_epoch # Assuming engine.py is in the same directory or accessible
from config_loader import load_config
from pipeline_utils import filter_short_speeches, remove_duplicates, encode_labels_with_map

# --- Direct Imports for Config Loader and Utilities ---
config_path = Path(__file__).parent.parent / "config" / "roberta_config.yaml"
config = load_config(config_path)

# Import plotting functions
from plotting_utils import plot_performance_metrics, plot_confusion_matrix # Assuming plotting_utils.py is accessible


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
# Pass the loaded config dictionary containing only parameters
def run_roberta_pipeline(congress_year: str, config: dict):
    print(f"\n--- Running RoBERTa pipeline for Congress {congress_year} ---")
    start_total_time = time.time()

    # --- Configuration (Access from config dictionary) ---
    # Access parameters from the passed config dictionary
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


    # --- Paths (Defined directly in script, not from config) ---
    # Use the variables defined at the top of the script
    csv_file_path = DATA_DIR / f"house_merged_{congress_year}.csv"
    if not csv_file_path.exists():
        print(f"⚠️  Skipping Congress {congress_year}: File not found at {csv_file_path}")
        return

    performance_log_path = LOG_DIR / "roberta_performance.csv"
    year_results_json_path = LOG_DIR / f"roberta_results_{congress_year}.json"
    model_save_path = MODELS_DIR / f"roberta_classifier_{congress_year}.pth" # Optional: path to save model

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    start_time = time.time()
    df = pd.read_csv(csv_file_path)
    initial_count = len(df)
    print(f"  - Loaded {initial_count} initial rows.")

    # filter_short_speeches also returns a tuple (df, count)
    df_filtered, _ = filter_short_speeches(df, min_words=MIN_WORD_COUNT) # Unpack and ignore count if not needed here

    # remove_duplicates also returns a tuple (df, count)
    df_filtered, removed_duplicate_count = remove_duplicates(df_filtered) # Unpack the DataFrame and the count
    print(f"  - Removed {removed_duplicate_count} duplicate speeches.") # Optional: print the count

    # Now df_filtered is the actual DataFrame
    # Handle missing speeches or parties BEFORE mapping and splitting
    df_filtered.dropna(subset=['speech', 'party'], inplace=True)
    print(f"  - Removed rows with missing speech or party. Remaining: {len(df_filtered)}")

    # Map party labels to integers using imported utility function and config party_map
    df_processed = encode_labels_with_map(df_filtered, party_map=PARTY_MAP)
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

    split_time = time.time() - start_time
    print(f"Split complete in {split_time:.2f} seconds.")

    # Check if any split is empty
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print(f"⚠️  Skipping Congress {congress_year}: One or more data splits resulted in zero samples.")
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
    criterion = nn.CrossEntropyLoss() # Standard for classification

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
        val_metrics = evaluate_epoch(model, val_dataloader, criterion, device)
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
    for i, true_label_int in enumerate(sorted(cm_labels)):
        # Assuming cm_matrix_data rows correspond to sorted cm_labels
         row_data = cm_matrix_data[i] if i < len(cm_matrix_data) else [0] * len(cm_labels) # Handle potential size mismatch
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
