import warnings

# Suppress the specific FutureWarning related to _register_pytree_node
# This is quite specific to the message you're seeing.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.utils._pytree._register_pytree_node` is deprecated.*"
)

import multiprocessing as mp
import spacy
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# ---- Global constants and function definitions ----
# These are safe to define at the top level as they don't execute process-creating code on import.
SCRIPT_DIR = Path(__file__).resolve().parent

def process_spacy_doc(doc):
    """
    Processes a single SpaCy Doc object to extract cleaned, lemmatized tokens
    after removing specified entities, stop words, punctuation, digits,
    currency symbols, and the phrase "mr speaker".
    """
    if not doc or not doc.has_annotation("SENT_START"):
        return []

    ents_to_remove_char_spans = []
    if doc.has_annotation("ENT_IOB"):
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "PERCENTAGE"]:
                ents_to_remove_char_spans.append((ent.start_char, ent.end_char))

    # Initial pass to get lemmas, filtering out entities, stop words, punct, digits, currency
    intermediate_lemmas = []
    for token in doc:
        token_in_entity_to_remove = False
        for start_char, end_char in ents_to_remove_char_spans:
            if token.idx >= start_char and (token.idx + len(token.text)) <= end_char:
                token_in_entity_to_remove = True
                break
        
        if token_in_entity_to_remove:
            continue

# Updated function to handle te expection

        if (not token.is_stop and
            not token.is_punct and
            # not token.is_digit and # is_alpha will handle this
            # not token.like_num and # is_alpha will handle this
            not token.is_currency and
            token.is_alpha and  # <-- ADD THIS: Ensures only A-Z
            len(token.lemma_) > 1 and # <-- ADD THIS: Min length 2
            token.text.strip() != ''):
            
            lemma = token.lemma_.lower().strip()
            if lemma and lemma.isalpha() and len(lemma) > 1: # Double check lemma too
                intermediate_lemmas.append(lemma)
    
    # Second pass: Remove "mr speaker" / "mister speaker" sequences
    final_tokens = []
    i = 0
    while i < len(intermediate_lemmas):
        current_lemma = intermediate_lemmas[i]
        # Normalize "mr." to "mr" for checking, also handle "mister"
        is_mr_or_mister = (current_lemma.rstrip('.') == "mr" or current_lemma == "mister")

        if is_mr_or_mister and \
           (i + 1 < len(intermediate_lemmas)) and \
           intermediate_lemmas[i+1] == "speaker":
            i += 2  # Skip both "mr/mister" and "speaker"
            continue
        
        final_tokens.append(current_lemma)
        i += 1
            
    return final_tokens

# ---- Main execution block ----
if __name__ == '__main__':
    # 1. Set multiprocessing start method (and call freeze_support)
    try:
        mp.set_start_method('spawn', force=True)
        # Call freeze_support() as suggested by the error message for spawn method
        mp.freeze_support()
        print("Multiprocessing start method set to 'spawn' and freeze_support() called.")
    except RuntimeError as e:
        current_method = mp.get_start_method(allow_none=True)
        if current_method == 'spawn':
            print(f"Multiprocessing start method was already 'spawn'. Current context: {current_method}")
            # If freeze_support() needs to be called only once, ensure it is.
            # However, freeze_support() is usually safe to call multiple times if needed,
            # but typically it's called once in the main entry point.
            # For simplicity here, if spawn is already set, we assume freeze_support might have been handled
            # or is not the primary issue if this block is re-entered (which it shouldn't be).
        else:
            print(f"Could not set multiprocessing start method to 'spawn': {e}. Current method: {current_method}")
            exit() # Exit if we can't set it and it's not already set, or handle as appropriate

    # 2. Setup Logging for the main process execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 3. Load SpaCy Model and other initializations for the main process
    nlp = None # Initialize to None
    try:
        spacy.prefer_gpu() # Uncomment if you have a GPU and want to use it
        if spacy.prefer_gpu():
            print("Attempted to prefer GPU for SpaCy.")
        else:
            print("GPU not available or not preferred for SpaCy.")

        nlp = spacy.load("en_core_web_trf")
        logging.info("SpaCy transformer model 'en_core_web_trf' loaded successfully (main process).")
        # Check actual GPU usage after loading the model for TRF models
        if "transformer" in nlp.pipe_names:
             if nlp.get_pipe("transformer").model.ops.device_type == "gpu":
                 logging.info("SpaCy transformer is using GPU.")
             else:
                 logging.info("SpaCy transformer is using CPU.")

    except Exception as e:
        logging.critical(f"Could not load SpaCy model or configure GPU: {e}")
        exit()

    # --- Main Processing Loop ---
    CONGRESS_RANGE = range(76, 78) # Example: 76-112
    batch_size = 512 #VRAM usage 10 out of 40, could increase to 256)
    n_processes = 1 # Use -1 for all cores, or a specific number > 1 for multiprocessing

    base_dir = SCRIPT_DIR / "../data/merged"
    cleaned_dir = SCRIPT_DIR / "../data/processed"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    for congress_num in tqdm(CONGRESS_RANGE, desc="Processing Congresses"):
        year_str = f"{congress_num:03}"
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
            logging.warning(f"'speech' column not found in {house_file}. Skipping.")
            continue
        
        speeches_to_clean = df["speech"].fillna("").astype(str).tolist()

        if not speeches_to_clean:
            logging.info(f"No speeches to clean in Congress {year_str}.")
            continue
        
        logging.info(f"Cleaning {len(speeches_to_clean)} speeches for Congress {year_str}...")

        processed_tokens_for_current_df = []
        # The nlp.pipe call is what starts the child processes. It MUST be within this block.
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
                logging.error(f"Error saving data for Congress {year_str}: {e}")
        else:
            logging.error(f"Mismatch in length for Congress {year_str}. Skipping save.")

    logging.info("\nAll specified Congresses processed.")