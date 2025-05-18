# Data Science Study Guide - Question 04

## **Question:**
Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?

## **Answer Choices:**
1. **`scipy.stats.wasserstein_distance(x, y)`**  
2. **`numpy.linalg.norm(np.sort(x) - np.sort(y), ord=1)`**  
3. **`scipy.spatial.distance.cdist(x.reshape(-1,1), y.reshape(-1,1), metric='euclidean').min(axis=1).sum()`**  
4. **`sklearn.metrics.pairwise_distances(x.reshape(-1,1), y.reshape(-1,1), metric='manhattan').min(axis=1).mean()`**  

---

## **Correct Answer:** `scipy.stats.wasserstein_distance(x, y)`

### **Explanation:**
The **Wasserstein distance**, also known as **Earth Mover’s Distance (EMD)**, measures the minimum amount of "work" required to transform one distribution into another.  
- It calculates the cumulative distribution function (CDF) of two datasets.
- Uses integral-based formulation rather than pairwise distances.
- **`scipy.stats.wasserstein_distance(x, y)`** implements the correct approach for **1D Wasserstein distance**.

Python implementation:
```python
import numpy as np
from scipy.stats import wasserstein_distance

# Example data
x = np.array([0.1, 0.5, 0.7, 1.0, 1.5])
y = np.array([0.2, 0.6, 0.8, 1.2, 1.7])

# Compute Wasserstein distance
distance = wasserstein_distance(x, y)
print("Wasserstein Distance:", distance)
```

---

## **Why Other Choices Are Incorrect?**
### **1. `numpy.linalg.norm(np.sort(x) - np.sort(y), ord=1)`**
- This **computes L1 norm** between sorted distributions.
- L1 norm measures the absolute difference but **doesn't account for distribution mass shifting**, unlike Wasserstein distance.

Python demonstration:
```python
dist_l1 = np.linalg.norm(np.sort(x) - np.sort(y), ord=1)
print("L1 Distance:", dist_l1)
```
This **lacks probability mass transport information**, making it incorrect.

### **2. `scipy.spatial.distance.cdist(x.reshape(-1,1), y.reshape(-1,1), metric='euclidean').min(axis=1).sum()`**
- Computes **minimum Euclidean distance** for each point rather than integrating probability measures.
- **Does not satisfy Wasserstein formulation**, as pairwise Euclidean distances ignore mass transport.

### **3. `sklearn.metrics.pairwise_distances(x.reshape(-1,1), y.reshape(-1,1), metric='manhattan').min(axis=1).mean()`**
- **Uses Manhattan distance**, which is similar to Euclidean but **does not properly capture probability mass transport**.
- Wasserstein **integrates distribution movement**, which pairwise distances fail to do.

