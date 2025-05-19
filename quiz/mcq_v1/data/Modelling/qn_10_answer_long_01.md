
---

### qn_10.md
```markdown
# Question 10: Time-Based Cross-Validation

**Category:** Modelling  
**Question:** What's the most effective time-based split?

## Correct Answer:
**Define a custom cross-validator with expanding window and purging**

## Python Implementation:
```python
from sklearn.model_selection import BaseCrossValidator
import numpy as np

class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        
    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        for i in range(1, self.n_splits + 1):
            train_end = int(len(X) * (i/(self.n_splits+1)))
            test_start = train_end + self.purge_gap
            test_end = int(len(X) * ((i+1)/(self.n_splits+1)))
            
            yield (indices[:train_end], 
                   indices[test_start:test_end])

# Usage
cv = PurgedTimeSeriesSplit(n_splits=5)
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]