import spacy
# Load a SpaCy model (e.g., en_core_web_sm for a small English model)
# For better NER accuracy, you might consider a medium (md) or large (lg) model
# Make sure you've downloaded it: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_trf") # Or "en_core_web_md", "en_core_web_lg"
except OSError:
    print("Downloading language model for the spaCy POS tagger\n"
        "(don't worry, this will only happen once)")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_speech(speech_text):
    # 1. Process the text with SpaCy
    # This runs the entire pipeline (tokenization, tagging, parsing, NER, etc.)
    doc = nlp(speech_text)

    # --- Steps that involve filtering based on NER results ---
    # Create a list of tokens to keep after NER filtering
    # This needs to be done carefully to not disrupt token indices if modifying on the fly
    # It's often easier to build a new list of desired tokens.

    # Get the character spans of PERSON and ORG entities
    ents_to_remove_char_spans = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "PERCENTAGE"]:
            ents_to_remove_char_spans.append((ent.start_char, ent.end_char))

    processed_tokens = []
    for token in doc:
        # Check if the token falls within any of the entity spans to be removed
        token_in_entity_to_remove = False
        for start_char, end_char in ents_to_remove_char_spans:
            if token.idx >= start_char and (token.idx + len(token.text)) <= end_char:
                token_in_entity_to_remove = True
                break
        
        if token_in_entity_to_remove:
            continue # Skip this token if it's part of a PERSON or ORG entity

        # 2. Lemmatize (already available: token.lemma_)
        # 3. Remove stop words (token.is_stop)
        # 4. Remove all digits (token.is_digit or token.like_num)
        # 5. Punctuation Removal (token.is_punct)
        # 6. Tokenization (already done by SpaCy, we are iterating through tokens)

        if (not token.is_stop and
            not token.is_punct and
            not token.is_digit and # Simple digit check
            not token.like_num and   # Broader number check (e.g., "ten")
            token.text.strip() != ''): # Ensure no empty strings after potential modifications
            
            # Get the lemma
            lemma = token.lemma_.lower().strip() # Lowercase and strip whitespace
            processed_tokens.append(lemma)
            
    return processed_tokens

# Example usage:
your_speech_transcript = """
Good morning, Mr. Chairman and members of the Committee. 
My name is Dr. Jane Doe, and I represent the FutureTech Organization. 
We believe that Congress should invest 100 million dollars in renewable energy by 2025. 
This is crucial for America's future. John Smith from Acme Corp agrees.
It is also crucial for Californiam increasing 25% of taxes by June
"""

cleaned_tokens = preprocess_speech(your_speech_transcript)
print(cleaned_tokens)
# Expected output (order might vary slightly depending on exact stop word list and lemmatization):
# ['good', 'morning', 'chairman', 'member', 'committee', 'believe', 'congress', 
#  'invest', 'million', 'dollar', 'renewable', 'energy', 'crucial', 'america', 'future', 'agree']
# Note: "Jane Doe", "FutureTech Organization", "John Smith", "Acme Corp" "100", "2025" are removed.
# "Mr." might be removed as punctuation depending on the model's tokenization or if you add custom logic.