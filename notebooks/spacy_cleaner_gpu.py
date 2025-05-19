import multiprocessing as mp # Import the multiprocessing library

# Set the start method to 'spawn'
# This MUST be done in the `if __name__ == '__main__':` block
# or at the very top level of your script before other relevant imports if not using the block.
# For scripts, placing it under `if __name__ == '__main__':` is the safest.

# ... (other imports like spacy, pandas, etc.)

if __name__ == '__main__': # Ensures this runs only when script is executed directly
    try:
        # Attempt to set the start method. This might need to be done very early.
        # If it's already been set by another import, this might raise a RuntimeError,
        # in which case it's often okay if it was already set to 'spawn'.
        mp.set_start_method('spawn', force=True) # 'force=True' can be helpful
        print("Multiprocessing start method set to 'spawn'.") # Optional: for confirmation
    except RuntimeError as e:
        # This can happen if the start method has already been set (e.g., by another library
        # or if this script is imported as a module after the parent has set it).
        # Check if it was already set to spawn.
        if mp.get_start_method() == 'spawn':
            print("Multiprocessing start method was already 'spawn'.")
        else:
            print(f"Could not set multiprocessing start method to 'spawn': {e}. Current method: {mp.get_start_method()}")
            # Depending on the error, you might still proceed or exit if it's critical.

import spacy
from tqdm import tqdm # For progress bars
import pandas as pd
from pathlib import Path
import logging # For better error messages and info

# --- Get the directory of the current script ---
# This ensures that paths are relative to the script's location,
# regardless of where you run the script from.
SCRIPT_DIR = Path(__file__).resolve().parent # This will be '.../congress-polarization-thesis/notebooks'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Load SpaCy Model ---
try:
    spacy.prefer_gpu() # Call this before loading if you have a working GPU setup
    nlp = spacy.load("en_core_web_trf")
    logging.info("SpaCy transformer model 'en_core_web_trf' loaded successfully.")
    if spacy.require_gpu(): # Check if SpaCy is actually using GPU
        logging.info("SpaCy is configured to use GPU if available and compatible.")
    else:
        logging.info("SpaCy is using CPU.")
except OSError:
    logging.error("Transformer model 'en_core_web_trf' not found. Attempting to download.")
    try:
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
        logging.info("SpaCy transformer model 'en_core_web_trf' downloaded and loaded successfully.")
    except Exception as e:
        logging.critical(f"Could not download or load 'en_core_web_trf'. Exiting. Error: {e}")
        exit()
except Exception as e:
    logging.critical(f"An error occurred during SpaCy setup: {e}. Exiting.")
    exit()


# --- Core Preprocessing Function (modified to handle potential empty docs) ---
def process_spacy_doc(doc):
    # (Your process_spacy_doc function remains the same)
    if not doc or not doc.has_annotation("SENT_START"):
        return []
    ents_to_remove_char_spans = []
    if doc.has_annotation("ENT_IOB"):
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "PERCENTAGE"]:
                ents_to_remove_char_spans.append((ent.start_char, ent.end_char))
    processed_tokens = []
    for token in doc:
        token_in_entity_to_remove = False
        for start_char, end_char in ents_to_remove_char_spans:
            if token.idx >= start_char and (token.idx + len(token.text)) <= end_char:
                token_in_entity_to_remove = True
                break
        if token_in_entity_to_remove:
            continue
        if (not token.is_stop and
            not token.is_punct and
            not token.is_digit and
            not token.like_num and
            token.text.strip() != ''):
            lemma = token.lemma_.lower().strip()
            if lemma:
                processed_tokens.append(lemma)
    return processed_tokens

# --- Main Processing Loop ---
CONGRESS_RANGE = range(76, 77)
batch_size = 64
n_processes = -1

# --- Define base and output directories RELATIVE TO THE SCRIPT LOCATION ---
# SCRIPT_DIR is '.../congress-polarization-thesis/notebooks'
# So, '../data/' from SCRIPT_DIR correctly points to '.../congress-polarization-thesis/data/'
base_dir = SCRIPT_DIR / "../data/merged"
cleaned_dir = SCRIPT_DIR / "../data/processed" # This will also be '.../congress-polarization-thesis/data/processed'

# Ensure the cleaned directory exists
cleaned_dir.mkdir(parents=True, exist_ok=True)

for congress_num in tqdm(CONGRESS_RANGE, desc="Processing Congresses"):
    year_str = f"{congress_num:03}"
    # Now house_file path will be correctly resolved:
    # e.g., .../congress-polarization-thesis/notebooks/../data/merged/house_db/house_merged_076.csv
    # which simplifies to .../congress-polarization-thesis/data/merged/house_db/house_merged_076.csv
    house_file = base_dir / f"house_db/house_merged_{year_str}.csv"

    logging.info(f"--Processing Congress {year_str} from {house_file} --")

    if not house_file.exists():
        logging.warning(f"File not found, skipping: {house_file}")
        continue

    try:
        df = pd.read_csv(house_file)
    except Exception as e:
        logging.error(f"Error reading {house_file}: {e}. Skipping.")
        continue

    if "speech" not in df.columns:
        logging.warning(f"'speech' column not found in {house_file}. Skipping cleaning for this file.")
        continue
    
    speeches_to_clean = df["speech"].fillna("").astype(str).tolist()

    if not speeches_to_clean:
        logging.info(f"No speeches to clean in Congress {year_str} (column was empty or all NaNs).")
        continue
    
    logging.info(f"Cleaning {len(speeches_to_clean)} speeches for Congress {year_str}...")

    processed_tokens_for_current_df = []
    for doc in tqdm(nlp.pipe(speeches_to_clean, batch_size=batch_size, n_process=n_processes), 
                    total=len(speeches_to_clean), desc=f"Cleaning speeches in Congress {year_str}", leave=False):
        cleaned_tokens = process_spacy_doc(doc)
        processed_tokens_for_current_df.append(cleaned_tokens)
    
    if len(processed_tokens_for_current_df) == len(df):
        df["cleaned_speech_tokens"] = processed_tokens_for_current_df
        df["cleaned_speech_str"] = [" ".join(tokens) for tokens in processed_tokens_for_current_df]
        logging.info(f"Finished cleaning speeches for Congress {year_str}.")
        
        output_cleaned_file = cleaned_dir / f"house_cleaned_{year_str}.csv"
        columns_to_keep = ["speech_id", 'speakerid', 'party', 'cleaned_speech_str'] 
        columns_present_in_df = [col for col in columns_to_keep if col in df.columns]
        
        df_to_save = df[columns_present_in_df].copy()
        if 'cleaned_speech_str' in df_to_save.columns:
            df_to_save.rename(columns={'cleaned_speech_str': 'speech'}, inplace=True)
        elif 'cleaned_speech_tokens' in df_to_save.columns and 'cleaned_speech_str' not in columns_to_keep:
             df_to_save.rename(columns={'cleaned_speech_tokens': 'speech'}, inplace=True)

        try:
            df_to_save.to_csv(output_cleaned_file, index=False)
            logging.info(f"Saved cleaned data for Congress {year_str} to {output_cleaned_file}")
        except Exception as e:
            logging.error(f"Error saving cleaned data for Congress {year_str} to {output_cleaned_file}: {e}")
    else:
        logging.error(f"Mismatch in length between processed speeches ({len(processed_tokens_for_current_df)}) "
                      f"and DataFrame rows ({len(df)}) for Congress {year_str}. Skipping save for this file.")

logging.info("\nAll specified Congresses processed.")