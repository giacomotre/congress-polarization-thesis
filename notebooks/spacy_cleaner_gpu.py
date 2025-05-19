import spacy
from tqdm import tqdm # For progress bars
import pandas as pd
# import numpy as np # Not strictly needed for this version
from pathlib import Path
import logging # For better error messages and info

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Load SpaCy Model ---
try:
    # Consider GPU if available and model supports it well.
    # spacy.prefer_gpu() # Call this before loading if you have a working GPU setup
    # For very large datasets, disabling unnecessary pipeline components can speed things up
    # if you only need tokenization, lemmatization, and NER.
    # Example: nlp = spacy.load("en_core_web_trf", disable=["parser", "attribute_ruler"])
    nlp = spacy.load("en_core_web_trf")
    logging.info("SpaCy transformer model 'en_core_web_trf' loaded successfully.")
    if spacy.require_gpu():
        logging.info("SpaCy is using GPU.")
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
        exit() # Critical error, cannot proceed
except Exception as e: # Catch other potential errors during spacy.prefer_gpu() or load
    logging.critical(f"An error occurred during SpaCy setup: {e}. Exiting.")
    exit()


# --- Core Preprocessing Function (modified to handle potential empty docs) ---
def process_spacy_doc(doc):
    """
    Processes a single SpaCy Doc object to extract cleaned, lemmatized tokens
    after removing specified entities, stop words, punctuation, and digits.
    """
    if not doc or not doc.has_annotation("SENT_START"): # Check if doc is empty or lacks basic annotation
        return []

    ents_to_remove_char_spans = []
    if doc.has_annotation("ENT_IOB"): # Only access doc.ents if NER ran
        for ent in doc.ents:
            # Added GPE, DATE, CARDINAL, PERCENTAGE as per your script
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
            not token.like_num and # Broader number check (e.g., "ten")
            token.text.strip() != ''): # Ensure no empty strings
            
            lemma = token.lemma_.lower().strip()
            if lemma: # Ensure lemma is not empty
                processed_tokens.append(lemma)
            
    return processed_tokens

# --- Main Processing Loop ---
# Adjust your range as needed. range(76, 107) would process Congresses 76 through 106.
# For testing one Congress, range(76, 77) is fine.
CONGRESS_RANGE = range(76, 77) # Example: Process only Congress 76

batch_size = 64  # Adjust based on your RAM and model (transformers might need smaller batches)
n_processes = -1 # Use all available CPU cores, or set to a specific number (e.g., 2, 4)

# Define base and output directories
base_dir = Path("../data/merged")
cleaned_dir = Path("../data/processed")
cleaned_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

# Outer loop for processing each Congress file
for congress_num in tqdm(CONGRESS_RANGE, desc="Processing Congresses"):
    year_str = f"{congress_num:03}" # e.g., 076, 077
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
    
    # Prepare speeches for cleaning: fill NaNs with empty strings and convert to list
    # It's crucial to handle NaNs before passing to nlp.pipe
    speeches_to_clean = df["speech"].fillna("").astype(str).tolist()

    if not speeches_to_clean:
        logging.info(f"No speeches to clean in Congress {year_str} (column was empty or all NaNs).")
        continue
    
    logging.info(f"Cleaning {len(speeches_to_clean)} speeches for Congress {year_str}...")

    # This list will store processed tokens for EACH speech in the CURRENT DataFrame
    processed_tokens_for_current_df = []
    
    # Use nlp.pipe() for efficient batch processing.
    # tqdm can be wrapped around nlp.pipe to show progress for the current file.
    for doc in tqdm(nlp.pipe(speeches_to_clean, batch_size=batch_size, n_process=n_processes), 
                    total=len(speeches_to_clean), desc=f"Cleaning speeches in Congress {year_str}", leave=False):
        cleaned_tokens = process_spacy_doc(doc)
        processed_tokens_for_current_df.append(cleaned_tokens)
    
    # --- Assign cleaned tokens back to the DataFrame ---
    # Ensure the number of processed speeches matches the number of rows in the DataFrame
    if len(processed_tokens_for_current_df) == len(df):
        # You have two main options for storing the cleaned text:
        # 1. As a list of tokens (what process_spacy_doc returns)
        #    This is good if you want to do further per-token analysis later.
        #    When saved to CSV, it will look like "[ 'word1', 'word2', ... ]"
        df["cleaned_speech_tokens"] = processed_tokens_for_current_df

        # 2. As a single string of space-separated tokens
        #    This is often more convenient if your next step is TF-IDF, as TfidfVectorizer
        #    typically expects a list of strings (documents).
        df["cleaned_speech_str"] = [" ".join(tokens) for tokens in processed_tokens_for_current_df]
        
        logging.info(f"Finished cleaning speeches for Congress {year_str}.")
        
        # --- Save the processed DataFrame ---
        output_cleaned_file = cleaned_dir / f"house_cleaned_{year_str}.csv"
        
        # Define columns to keep. Choose one of the cleaned speech columns.
        # If your TF-IDF expects strings, use 'cleaned_speech_str'.
        # Let's assume you want to save the string version and rename it to 'speech'
        # to replace the original, plus other key identifiers.
        columns_to_keep = ["speech_id", 'speakerid', 'party', 'cleaned_speech_str'] 
        
        # Filter df for only existing columns among those you want to keep
        columns_present_in_df = [col for col in columns_to_keep if col in df.columns]
        if 'cleaned_speech_str' not in columns_present_in_df and 'cleaned_speech_tokens' in columns_present_in_df and 'cleaned_speech_str' in columns_to_keep:
            logging.warning("'cleaned_speech_str' requested but not found, 'cleaned_speech_tokens' might be available.")
            # Decide fallback or error. For now, let's make sure we select what's available.
        
        df_to_save = df[columns_present_in_df].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Optional: Rename the cleaned speech column if you want it to be 'speech' in the output CSV
        if 'cleaned_speech_str' in df_to_save.columns:
            df_to_save.rename(columns={'cleaned_speech_str': 'speech'}, inplace=True)
        elif 'cleaned_speech_tokens' in df_to_save.columns and 'cleaned_speech_str' not in columns_to_keep: # if you decided to keep tokens
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