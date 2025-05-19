### ✅ Question 24

**Q: What's the most statistically sound approach to implement feature selection for time series forecasting?**

**Correct Answer:**  
`Implement feature importance from tree-based models with purged cross-validation`

---

### 🧠 Key Concepts

- **Temporal Dependency:** In time series, future data should never be used to inform past predictions (avoiding lookahead bias).
- **Feature Importance (Tree-Based Models):** Algorithms like Random Forest or LightGBM can provide insight into which features are important for prediction.
- **Purged Cross-Validation:** Prevents leakage by ensuring there's a time gap (purge window) between training and test sets during CV splits.

---

### 🧪 Why Tree-Based + Purged CV?

- Tree-based models handle feature interactions and nonlinearities.
- Feature importance is calculated based on actual contribution to reduction in error.
- When combined with **purged CV**, it ensures validation mimics real-world temporal splits without contamination.

---

### 🔧 Purged CV in Python

We’ll use a custom purged time series splitter and `LightGBM` for feature importance.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
import lightgbm as lgb
import matplotlib.pyplot as plt

class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size

            train_end = max(0, test_start - self.purge_gap)
            yield (
                np.arange(train_end),  # training indices
                np.arange(test_start, test_end)  # test indices
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Simulated time series data
n = 500
X = pd.DataFrame({
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n) * 0.5,
    'feature3': np.random.randn(n) * 2
})
y = X['feature1'] * 0.5 + X['feature2'] * (-1.2) + np.random.randn(n) * 0.1

# Train LightGBM with purged time-series CV
cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=10)
feature_importances = np.zeros(X.shape[1])

for train_idx, test_idx in cv.split(X):
    train_data = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y[test_idx], reference=train_data)
    
    model = lgb.train({'objective': 'regression', 'verbosity': -1},
                      train_data,
                      num_boost_round=100,
                      valid_sets=[val_data],
                      early_stopping_rounds=10,
                      verbose_eval=False)
    
    feature_importances += model.feature_importance()

# Average importances
feature_importances /= cv.get_n_splits()

# Plot
pd.Series(feature_importances, index=X.columns).sort_values().plot(kind='barh')
plt.title('Feature Importance (Purged CV)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
````

---

### 📌 Benefits of This Approach

* Prevents **data leakage** using temporal gap.
* Handles **nonlinear interactions**.
* Provides **interpretable feature ranking**.
* Suitable for real-world **forecasting pipelines**.

---

### 📚 References

* Marcos López de Prado, *Advances in Financial Machine Learning*
* [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)


