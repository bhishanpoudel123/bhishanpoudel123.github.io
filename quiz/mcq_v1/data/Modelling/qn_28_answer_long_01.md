### ✅ Question 28

**Q: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?**

**Correct Answer:**  
`Use quantile regression with multiple target quantiles`

---

### 🧠 Why This is Correct

Quantile regression with gradient boosting allows you to predict **conditional quantiles** (e.g., 5th, 50th, 95th percentile) rather than just the mean. This provides a **prediction interval** for each input, offering a **distributional view of uncertainty**.

- **Non-parametric**: Makes no assumptions about the error distribution.
- **Heteroscedasticity-aware**: Can handle changing variance across feature space.
- **Rigorous**: Each quantile is modeled directly using custom loss functions (pinball loss).

---

### 🔍 Explanation of All Choices

#### ✅ Option A: `Use quantile regression with multiple target quantiles`
- **Best choice** for modeling **distributional uncertainty**.
- Predicts intervals like 5th–95th percentile directly.
- Works well even with skewed or heteroscedastic errors.

#### ❌ Option B: `Implement Monte Carlo dropout in gradient boosting`
- Monte Carlo dropout is a **Bayesian approximation technique** suited for neural networks, not gradient boosting.
- Gradient boosting lacks a dropout mechanism — applying this idea here is not straightforward or natively supported.

#### ❌ Option C: `Apply jackknife resampling to estimate prediction variance`
- Jackknife estimates **variance** by resampling, but:
  - Computationally **intensive** for large datasets.
  - Doesn't offer **prediction intervals**.
  - Not well-suited for fast, scalable uncertainty estimation in GBDTs.

#### ❌ Option D: `Use Lower Upper Bound Estimation (LUBE) with pinball loss`
- LUBE methods are typically used in neural networks and **require bounding functions**.
- Although they also use pinball loss, they are less interpretable and more **complex to implement** in tree-based models.

---

### 🧪 Example: Quantile Regression with LightGBM

```python
# Install: pip install lightgbm

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define model for lower, median, upper quantiles
def train_quantile_model(alpha):
    return lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=100)

models = {
    "lower": train_quantile_model(0.05),
    "median": train_quantile_model(0.5),
    "upper": train_quantile_model(0.95)
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Predict
preds = {name: model.predict(X_test) for name, model in models.items()}

# Plot prediction intervals
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, label="True", alpha=0.3)
plt.plot(preds["median"], label="Predicted Median", color="blue")
plt.fill_between(
    range(len(y_test)),
    preds["lower"],
    preds["upper"],
    color="gray",
    alpha=0.4,
    label="90% Prediction Interval"
)
plt.title("Quantile Regression with LightGBM")
plt.legend()
plt.tight_layout()
plt.show()
````

---

### 📌 Advantages of Quantile Regression for Uncertainty

| Feature                       | Supported by Quantile GBDT      |
| ----------------------------- | ------------------------------- |
| Captures prediction intervals | ✅                               |
| Models asymmetric uncertainty | ✅                               |
| Easy to implement             | ✅ (native in LightGBM, XGBoost) |
| Non-parametric                | ✅                               |
| Robust to outliers            | ✅                               |

---

### 📚 References

* Meinshausen, N. (2006). *"Quantile Regression Forests"*
* LightGBM Documentation: [Quantile Regression](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective)
* Koenker, R. (2005). *"Quantile Regression"*

