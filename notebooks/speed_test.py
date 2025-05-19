import time
import spacy

# Load models
print("Loading en_core_web_lg (CPU)...")
nlp_lg = spacy.load("en_core_web_lg")

print("Loading en_core_web_trf (GPU)...")
spacy.require_gpu()
nlp_trf = spacy.load("en_core_web_trf")

# Create sample texts
text = "Apple is looking at buying a U.K. startup for $1 billion. " * 10  # simulate length
texts = [text for _ in range(100)]  # 100 documents

def benchmark(nlp, texts, model_name, batch_size=32, warmup=False):
    print(f"\nRunning benchmark for {model_name}...")
    
    if warmup:
        _ = list(nlp.pipe(["Warmup text"] * 5, batch_size=batch_size))
    
    start = time.perf_counter()
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    end = time.perf_counter()
    
    total_time = end - start
    print(f"{model_name} took {total_time:.2f} seconds for {len(texts)} documents.")
    print(f"→ {total_time / len(texts):.4f} seconds per document.")

# Benchmark both models
benchmark(nlp_lg, texts, "en_core_web_lg (CPU)", batch_size=32)
benchmark(nlp_trf, texts, "en_core_web_trf (GPU)", batch_size=32, warmup=True)
