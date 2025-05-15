import cupy
import cupyx.scipy.sparse
from cuml.svm import LinearSVC
from cuml.feature_extraction.text import TfidfVectorizer # To generate a sparse matrix
import cudf # Only if TfidfVectorizer needs cudf Series for input

print("--- Minimal Reproducible Example for cuML LinearSVC ---")

# 1. Create tiny text data
text_data = [
    "some example text here",
    "more example text for testing",
    "rapids cuml is fun",
    "another document for the sparse matrix",
    "short text",
    "example example example",
    "text text text",
    "gpu accelerated computing",
    "final piece of text",
    "one more for ten samples"
]
# For cuML TF-IDF, it's often easiest to feed it a cuDF Series
try:
    text_cudf_series = cudf.Series(text_data)
except Exception as e:
    print(f"Error creating cudf.Series: {e}. Ensure cudf is correctly imported and environment is set.")
    exit()

# 2. Create a sparse CuPy matrix using cuML's TfidfVectorizer
print("\nStep 1: Creating sparse X matrix with TfidfVectorizer...")
try:
    vectorizer = TfidfVectorizer(max_features=5) # Keep max_features small for MRE
    X_sparse = vectorizer.fit_transform(text_cudf_series)
    print(f"X_sparse type: {type(X_sparse)}")
    print(f"X_sparse shape: {X_sparse.shape}")
    print(f"X_sparse dtype: {X_sparse.dtype}")
    # print("X_sparse (first 5 rows, dense view for inspection):")
    # print(X_sparse[:5].todense())
except Exception as e:
    print(f"Error during TfidfVectorizer: {e}")
    X_sparse = None # Ensure it's defined for later checks

if X_sparse is None:
    print("Could not create X_sparse. Exiting.")
    exit()

# 3. Create a tiny 1D CuPy int32 label array
y_labels = cupy.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=cupy.int32)
print(f"\ny_labels type: {type(y_labels)}")
print(f"y_labels shape: {y_labels.shape}")
print(f"y_labels dtype: {y_labels.dtype}")
print(f"y_labels sample: {y_labels[:5]}")


# 4. Try to fit cuml.svm.LinearSVC() with the sparse data
print("\nStep 2: Fitting LinearSVC with SPARSE X data...")
try:
    svm_sparse = LinearSVC(random_state=42) # Added random_state for reproducibility
    svm_sparse.fit(X_sparse, y_labels)
    print("SUCCESS: LinearSVC fitted with SPARSE X data.")
except ValueError as ve:
    print(f"!!! VALUERROR (Sparse X): {ve}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"!!! OTHER ERROR (Sparse X): {e}")
    import traceback
    traceback.print_exc()

# 5. Convert sparse matrix to dense CuPy array
print("\nStep 3: Converting X_sparse to DENSE X data...")
try:
    X_dense = X_sparse.todense()
    print(f"X_dense type: {type(X_dense)}")
    print(f"X_dense shape: {X_dense.shape}")
    print(f"X_dense dtype: {X_dense.dtype}")
    # print("X_dense (first 5 rows for inspection):")
    # print(X_dense[:5])
except Exception as e:
    print(f"Error converting X_sparse to dense: {e}")
    X_dense = None # Ensure it's defined

if X_dense is None:
    print("Could not create X_dense. Skipping dense fit test.")
else:
    # 6. Try to fit cuml.svm.LinearSVC() with the dense data
    print("\nStep 4: Fitting LinearSVC with DENSE X data...")
    try:
        svm_dense = LinearSVC(random_state=42) # Added random_state
        svm_dense.fit(X_dense, y_labels)
        print("SUCCESS: LinearSVC fitted with DENSE X data.")
    except ValueError as ve:
        print(f"!!! VALUERROR (Dense X): {ve}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"!!! OTHER ERROR (Dense X): {e}")
        import traceback
        traceback.print_exc()

print("\n--- MRE Script Finished ---")