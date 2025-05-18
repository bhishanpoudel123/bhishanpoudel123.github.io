# Data Science Study Guide - Question 09

## **Question:**
What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?

## **Answer Choices:**
1. **`pd.get_dummies(df, sparse=True)`**  
2. **`pd.Categorical(df['col']).codes` in combination with sklearn's `OneHotEncoder(sparse=True)`**  
3. **Use `pd.factorize()` on all categorical columns followed by scipy's sparse matrices**  
4. **Convert to category dtype then use `df['col'].cat.codes` with sklearn's `OneHotEncoder(sparse=True)`**  

---

## **Correct Answer:** `Convert to category dtype then use df['col'].cat.codes with sklearn's OneHotEncoder(sparse=True)`

### **Explanation:**
For **high-cardinality categorical variables**, memory-efficient encoding is essential.  
- **Using pandas' `category` dtype** reduces memory usage significantly.
- **`.cat.codes` converts categorical values to integer codes**, avoiding unnecessary string storage.
- **Applying `OneHotEncoder(sparse=True)` ensures efficient sparse matrix representation**, reducing memory usage compared to dense encoding.

Python implementation:
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Example dataset
df = pd.DataFrame({'category': ['apple', 'banana', 'cherry', 'apple', 'banana', 'date']})

# Convert to category dtype and extract codes
df['category'] = df['category'].astype('category')
codes = df['category'].cat.codes.values.reshape(-1, 1)

# Apply sparse OneHotEncoder
encoder = OneHotEncoder(sparse=True)
encoded = encoder.fit_transform(codes)

print(encoded)  # Sparse matrix representation
```

---

## **Why Other Choices Are Incorrect?**
### **1. `pd.get_dummies(df, sparse=True)`**
- **Creates too many columns for high-cardinality features**, increasing memory overhead.
- Even in sparse mode, can result in **large sparse matrices that are inefficient**.

Python demonstration:
```python
df_encoded = pd.get_dummies(df, sparse=True)
print(df_encoded)
```
Efficient for small datasets but **not scalable** for high-cardinality categories.

### **2. `pd.Categorical(df['col']).codes` with `OneHotEncoder(sparse=True)`**
- While `Categorical` reduces memory, **not setting dtype as `category` first can lead to unnecessary conversions**.
- Using `.cat.codes` directly from `category` dtype **is better**.

### **3. `pd.factorize()` on categorical columns followed by scipy's sparse matrices**
- **Factorize creates integer labels**, but manual handling of sparse matrices requires additional complexity.
- **Scipy sparse matrices are efficient but require extra transformation steps**.

Python example:
```python
import numpy as np
from scipy.sparse import csr_matrix

factorized_values, _ = pd.factorize(df['category'])
sparse_matrix = csr_matrix((np.ones_like(factorized_values), (np.arange(len(factorized_values)), factorized_values)))

print(sparse_matrix)
```
Useful for **low-level control**, but **sklearn's OneHotEncoder is more streamlined**.

