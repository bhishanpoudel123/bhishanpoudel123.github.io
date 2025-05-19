
---

### qn_09.md
```markdown
# Question 9: Ordinal Encoding Preservation

**Category:** Modelling  
**Question:** Which implementation preserves ordinal nature?

## Correct Answer:
**Custom encoding using pd.Categorical with ordered=True**

## Python Implementation:
```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories, ordered=True):
        self.categories = categories
        self.ordered = ordered
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return pd.Categorical(X, 
                            categories=self.categories,
                            ordered=self.ordered).codes

# Usage
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
encoder = OrdinalEncoder(education_levels)
X_train_encoded = encoder.fit_transform(X_train['education'])