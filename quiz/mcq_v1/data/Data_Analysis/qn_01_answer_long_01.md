# Handling High-Dimensional Sparse Data with PCA

## Question 1
**What technique would you use to handle high-dimensional sparse data when performing PCA?**

- A. Standard PCA with normalization
- B. Truncated SVD (also known as LSA)
- C. Kernel PCA with RBF kernel
- D. Factor Analysis

## Detailed Explanation

### A. Standard PCA with normalization - INCORRECT

Standard PCA computes the covariance matrix after centering the data (subtracting the mean). For sparse matrices, this operation presents a significant problem:

```python
# Standard PCA implementation
from sklearn.decomposition import PCA
import numpy as np
from scipy import sparse

# Create a sparse matrix
X_sparse = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print("Original sparsity: ", 1.0 - X_sparse.nnz / X_sparse.shape[0] / X_sparse.shape[1])

# Attempt standard PCA
try:
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_sparse.toarray())  # Must convert to dense first
    print("PCA transformed shape:", X_transformed.shape)
except Exception as e:
    print(f"Error: {e}")
```

**Why it's incorrect:**
- Centering a sparse matrix destroys its sparsity by turning many zeros into non-zero values
- The covariance matrix computation becomes memory-intensive as the dimensions grow
- For very high-dimensional data, converting to dense format may be impossible due to memory constraints
- Normalization doesn't solve the fundamental issue of sparsity loss during centering

### B. Truncated SVD (also known as LSA) - CORRECT

Truncated SVD is specifically designed for sparse matrices and is implemented in scikit-learn's `TruncatedSVD`:

```python
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import sparse

# Create a sparse matrix
X_sparse = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print("Original sparsity: ", 1.0 - X_sparse.nnz / X_sparse.shape[0] / X_sparse.shape[1])

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_transformed = svd.fit_transform(X_sparse)
print("Truncated SVD transformed shape:", X_transformed.shape)
print("Explained variance ratio:", svd.explained_variance_ratio_)
```

**Why it's correct:**
- Truncated SVD doesn't center the data, preserving the sparsity of the original matrix
- It works directly with sparse matrices without requiring conversion to dense format
- It's computationally efficient for high-dimensional sparse data
- It's the technique behind Latent Semantic Analysis (LSA) used in text mining
- The algorithm only computes the top k singular values and vectors, making it memory-efficient

### C. Kernel PCA with RBF kernel - INCORRECT

Kernel PCA with RBF (Radial Basis Function) kernel applies a non-linear transformation:

```python
from sklearn.decomposition import KernelPCA
import numpy as np
from scipy import sparse

# Create a sparse matrix
X_sparse = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

# Must convert to dense for Kernel PCA
X_dense = X_sparse.toarray()

# Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_transformed = kpca.fit_transform(X_dense)
print("Kernel PCA transformed shape:", X_transformed.shape)
```

**Why it's incorrect:**
- Kernel PCA requires computing a kernel matrix of size n×n (where n is the number of samples)
- It doesn't leverage the sparse structure of the data
- The RBF kernel requires converting to a dense representation first
- It's computationally expensive for large datasets and doesn't scale well
- The non-linear mapping may not be necessary for sparse data where the primary goal is dimensionality reduction

### D. Factor Analysis - INCORRECT

Factor Analysis is a statistical method that models the covariance structure:

```python
from sklearn.decomposition import FactorAnalysis
import numpy as np
from scipy import sparse

# Create a sparse matrix
X_sparse = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

# Must convert to dense for Factor Analysis
X_dense = X_sparse.toarray()

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_transformed = fa.fit_transform(X_dense)
print("Factor Analysis transformed shape:", X_transformed.shape)
```

**Why it's incorrect:**
- Like PCA, Factor Analysis requires computing statistics that destroy the sparsity
- It assumes a specific statistical model that may not be appropriate for sparse data
- It's not designed to work directly with sparse matrices
- It's more focused on modeling the covariance structure rather than efficiently reducing dimensionality

## Practical Implementation

For high-dimensional sparse data (like text data using TF-IDF), a typical pipeline would be:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Sample text data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=1000),  # Creates sparse matrix
    TruncatedSVD(n_components=100),      # Reduces dimensions while preserving sparsity
    Normalizer()                         # Optional normalization after SVD
)

# Apply transformation
X_transformed = pipeline.fit_transform(corpus)
print("Final shape:", X_transformed.shape)
```

## Summary

When dealing with high-dimensional sparse data, **Truncated SVD** is the most appropriate technique because:

1. It preserves the sparsity structure during computation
2. It avoids the memory-intensive operation of converting to dense format
3. It's specifically optimized for sparse matrices
4. It's computationally efficient even for very high dimensions
5. It's the foundation for Latent Semantic Analysis (LSA) in text mining applications