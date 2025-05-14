import re
import string
from typing import List, Set

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
# Ensure NLTK resources are available.
# It's good practice to handle potential Lookuprrors if these haven't been downloaded.
try:
    STOPWORDS_CORPUS = nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    STOPWORDS_CORPUS = nltk.corpus.stopwords.words('english')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# SpaCy import
import spacy

# --- Global Initializations (Load/Compile ONCE) ---

# 1. Pre-compiled regular expression
PUNCTUATION_DIGITS_RE = re.compile(rf"[{re.escape(string.punctuation)}\d]+") # Added + for one or more occurrences

# 2. Stopwords as a set for efficient lookup
STOPWORDS_SET: Set[str] = set(STOPWORDS_CORPUS)

# 3. Load spaCy model once, disable unnecessary pipes for speed
# Choose the smallest model adequate for your lemmatization needs (e.g., "en_core_web_sm")
# Disabling "parser" and "ner" can significantly speed up processing if only lemmatization is needed.
try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# --- Reusable Cleaning Components ---
def lowercase(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()

def remove_punctuation_digits(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Use the pre-compiled regex
    return PUNCTUATION_DIGITS_RE.sub("", text)

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    # Use the global STOPWORDS_SET for faster lookups
    return [w for w in tokens if w not in STOPWORDS_SET]

def lemmatize(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    # Efficiently create a Doc from pre-tokenized words
    # The `NLP.vocab` is needed, and `words` takes the list of tokens.
    # We tell spaCy there's a space after each token, which is common.
    doc = spacy.tokens.Doc(NLP.vocab, words=tokens, spaces=[True]*len(tokens))

    # Process the doc with the loaded pipeline (which now only has tagger, lemmatizer etc.)
    # This step might be implicit if the necessary components are already enabled and
    # haven't been run. For explicit control or if components were added dynamically:
    # for pipe_name in NLP.pipe_names:
    # if pipe_name not in NLP.disabled:
    # doc = NLP.get_pipe(pipe_name)(doc)
    # Simpler: just accessing .lemma_ will trigger necessary processing if not done.
    # The line above constructing the Doc doesn't run the pipeline; attribute access does.

    return [token.lemma_ for token in doc if token.lemma_.strip()] # Ensure lemma is not just whitespace

# --- TF-IDF-Specific Cleaning ---
def clean_text_for_tfidf(text: str) -> str:
    text = lowercase(text)
    text = remove_punctuation_digits(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

# --- For processing multiple documents (MUCH FASTER with spaCy) ---
def clean_texts_for_tfidf_batch(texts: List[str]) -> List[str]:
    """
    Cleans a list of texts more efficiently, leveraging spaCy's nlp.pipe
    for the lemmatization part and applying other steps in sequence.
    """
    processed_texts = []
    # Initial Python-based cleaning (can be done in a loop or list comprehension)
    for text_content in texts:
        text_content = lowercase(text_content)
        text_content = remove_punctuation_digits(text_content)
        tokens = tokenize(text_content)
        tokens = remove_stopwords(tokens)
        processed_texts.append(tokens) # Keep as token lists for now

    # Lemmatize using nlp.pipe for efficiency on multiple documents
    # We need to provide the tokenized texts to nlp.pipe if we want to use our own tokenizer.
    # SpaCy's nlp.pipe is most efficient with raw strings, but then it would use its own tokenizer.
    # To use our tokenizer and then spaCy's lemmatizer efficiently:
    lemmatized_texts_joined = []
    # Construct Doc objects for nlp.pipe from pre-tokenized text
    docs_for_pipe = (spacy.tokens.Doc(NLP.vocab, words=token_list, spaces=[True]*len(token_list) if token_list else []) for token_list in processed_texts)

    # Adjust batch_size based on your system's memory and the average document length
    for doc in NLP.pipe(docs_for_pipe, batch_size=50):
        lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_.strip()]
        lemmatized_texts_joined.append(" ".join(lemmatized_tokens))

    return lemmatized_texts_joined


# --- Example Usage ---
if __name__ == "__main__":
    sample_text_list = [
        "This is the first Sample sentence with 123 numbers, punctuation!! and some stop words like 'the' and 'is'.",
        "Another example here; we are testing the new optimized functions for speed and correctness.",
        "The quick brown fox jumps over the lazy dog. Running, ran, better, best.",
        "", # Empty string
        "Just oneWord",
        "12345 !@#$%" # Only punctuation and digits
    ]

    print("--- Cleaning one by one using clean_text_for_tfidf ---")
    for i, text in enumerate(sample_text_list):
        cleaned = clean_text_for_tfidf(text)
        print(f"Original {i+1}: '{text}'")
        print(f"Cleaned {i+1}: '{cleaned}'\n")

    print("\n--- Cleaning a batch using clean_texts_for_tfidf_batch ---")
    cleaned_batch_results = clean_texts_for_tfidf_batch(sample_text_list)
    for i, (original, cleaned) in enumerate(zip(sample_text_list, cleaned_batch_results)):
        print(f"Original {i+1}: '{original}'")
        print(f"Cleaned {i+1}: '{cleaned}'\n")

    # For a proper speed test, you'd use a much larger dataset and the `timeit` module
    # import timeit
    #
    # # Create a larger dataset for timing
    # large_corpus = sample_text_list * 1000
    #
    # def wrapper_single():
    #     for text in large_corpus:
    #         clean_text_for_tfidf(text)
    #
    # def wrapper_batch():
    #     clean_texts_for_tfidf_batch(large_corpus)
    #
    # # Time the execution (number can be adjusted)
    # # Be patient, this can take a while with a large corpus
    # print("Timing clean_text_for_tfidf (one by one)...")
    # time_single = timeit.timeit(wrapper_single, number=1)
    # print(f"Time taken for single processing: {time_single:.4f} seconds")
    #
    # print("Timing clean_texts_for_tfidf_batch...")
    # time_batch = timeit.timeit(wrapper_batch, number=1)
    # print(f"Time taken for batch processing: {time_batch:.4f} seconds")
    #
    # if time_batch > 0 : # Avoid division by zero
    # print(f"Batch processing was approximately {time_single/time_batch:.2f} times faster.")