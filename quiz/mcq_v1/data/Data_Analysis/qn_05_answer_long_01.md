# Data Science Study Guide - Question 05

## **Question:**
What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?

## **Answer Choices:**
1. **`sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X).kneighbors(X)`**  
2. **`sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X).kneighbors(X)`**  
3. **`sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X).kneighbors(X)`**  
4. **Depends on data dimensionality, size, and structure**  

---

## **Correct Answer:** `Depends on data dimensionality, size, and structure`

### **Explanation:**
The efficiency of different k-nearest neighbor algorithms depends on key dataset characteristics:
- **Brute force (`brute`)**: Fast for **small datasets** but inefficient for large-scale data.
- **kd-tree (`kd_tree`)**: Best for **low-dimensional data (<20 dimensions)**.
- **Ball tree (`ball_tree`)**: More efficient for **higher-dimensional data or non-Euclidean metrics**.

Choosing the best method depends on:
- **Size of the dataset** (Brute works well for small data)
- **Dimensionality** (kd-tree is better for low dimensions, ball-tree for higher ones)
- **Distribution and metric used** (Ball-tree handles various distance metrics more effectively)

Python implementation for **kd-tree** (best for low-dimensional structured data):
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Example dataset
X = np.random.rand(1000, 5)  # 1000 points, 5 dimensions

# Apply k-NN using kd-tree
knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
knn.fit(X)
distances, indices = knn.kneighbors(X)

print("Nearest neighbors indices:", indices)
```

---

## **Why Other Choices Are Incorrect?**
### **1. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X).kneighbors(X)`**
- **Brute force** computes pairwise distances for **every single point**.
- **Extremely slow** for large datasets but fine for small datasets.

Python demonstration:
```python
knn_brute = NearestNeighbors(n_neighbors=5, algorithm='brute')
knn_brute.fit(X)
distances_brute, indices_brute = knn_brute.kneighbors(X)
```

### **2. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X).kneighbors(X)`**
- kd-tree is **optimal only when dimensionality is low (<20)**.
- Beyond **20+ dimensions**, performance deteriorates due to **curse of dimensionality**.

### **3. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X).kneighbors(X)`**
- Ball-tree handles **non-Euclidean metrics** better than kd-tree.
- Efficient in **high-dimensional settings**, often outperforming kd-tree.

Python example:
```python
knn_ball = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
knn_ball.fit(X)
distances_ball, indices_ball = knn_ball.kneighbors(X)
```
