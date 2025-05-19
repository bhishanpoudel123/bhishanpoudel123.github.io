
---

### qn_07.md
```markdown
# Question 7: Custom SVM Kernels in scikit-learn

**Category:** Modelling  
**Question:** Which approach correctly implements a custom kernel?

## Correct Answer:
**Define a function that takes two arrays and returns a kernel matrix**

## Python Implementation:
```python
from sklearn.svm import SVC
import numpy as np

def tanimoto_kernel(X, Y):
    """Tanimoto similarity kernel for binary data"""
    XY = X @ Y.T
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)
    return XY / (XX + YY.T - XY)

# Usage
model = SVC(kernel=tanimoto_kernel)
model.fit(X_binary, y)

# Precomputed kernel alternative
K_train = tanimoto_kernel(X_train, X_train)
model_precomp = SVC(kernel='precomputed')
model_precomp.fit(K_train, y_train)