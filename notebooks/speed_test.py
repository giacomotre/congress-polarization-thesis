import time
import spacy

# Load models
print("Loading en_core_web_lg (CPU)...")
nlp_lg = spacy.load("en_core_web_lg")

print("Loading en_core_web_trf (GPU)...")
spacy.require_gpu()
nlp_trf = spacy.load("en_core_web_trf")

# Create sample texts
text = "Apple is looking at buying U.K. startup for $1 billion." * 10  # repeat for length
texts = [text for _ in range(100)]  # batch of 100

def benchmark(nlp, texts, model_name):
    print(f"\nRunning benchmark for {model_name}...")
    start = time.time()
    docs = list(nlp.pipe(texts, batch_size=16))
    end = time.time()
    print(f"{model_name} took {end - start:.2f} seconds for {len(texts)} documents.")

# Benchmark both models
benchmark(nlp_lg, texts, "en_core_web_lg (CPU)")
benchmark(nlp_trf, texts, "en_core_web_trf (GPU)")
