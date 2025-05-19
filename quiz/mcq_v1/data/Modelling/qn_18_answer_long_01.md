# qn_18.md

## Question 18: Handling Target Outliers in Regression

**Question:**  
What's the most rigorous approach to handle outliers in the target variable for regression problems?

**Correct Answer:**  
Use Huber or Quantile regression with robust loss functions

### Python Implementation

```python
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.datasets import make_regression
import numpy as np

# Create data with outliers
X, y = make_regression(n_samples=200, n_features=5, noise=1, random_state=42)
y[:10] += 50  # Add outliers

# Huber Regression (smooth loss)
huber = HuberRegressor(epsilon=1.35)  # Default epsilon
huber.fit(X, y)

# Quantile Regression (median-focused)
quantile = QuantileRegressor(quantile=0.5, solver='highs')
quantile.fit(X, y)

# Compare with OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X, y)

# Evaluation on clean test set
X_test, y_test = make_regression(n_samples=50, n_features=5, noise=1, random_state=24)
print(f"OLS MAE: {np.mean(np.abs(ols.predict(X_test) - y_test)):.2f}")
print(f"Huber MAE: {np.mean(np.abs(huber.predict(X_test) - y_test)):.2f}")
print(f"Quantile MAE: {np.mean(np.abs(quantile.predict(X_test) - y_test)):.2f}")

# Diagnostic plot
plt.figure(figsize=(10, 6))
plt.scatter(ols.predict(X), y, label='OLS', alpha=0.5)
plt.scatter(huber.predict(X), y, label='Huber', alpha=0.5)
plt.scatter(quantile.predict(X), y, label='Quantile', alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--')
plt.legend()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Key Considerations:
1. **Huber Loss**: Smooth transition between squared and absolute loss
2. **Quantile Regression**: Directly models specific quantiles
3. **Outlier Resistance**: Neither approach requires manual outlier removal

### Advanced Usage:
```python
# Tuning Huber's epsilon parameter
epsilons = [1.01, 1.35, 1.5, 2.0]
for eps in epsilons:
    model = HuberRegressor(epsilon=eps).fit(X, y)
    score = np.mean(np.abs(model.predict(X_test) - y_test))
    print(f"Epsilon: {eps:.2f}, MAE: {score:.2f}")

# Multiple quantiles for uncertainty estimation
quantiles = [0.1, 0.5, 0.9]
models = {q: QuantileRegressor(quantile=q) for q in quantiles}
for q, model in models.items():
    model.fit(X, y)
    print(f"Quantile {q} coefficients:", model.coef_)
```

[Continuing to final questions in next message...]
```

The remaining questions (19-20) will follow the same structure with:
1. Problem-specific implementations
2. Comparative analysis
3. Diagnostic visualizations
4. Parameter tuning guidance
5. Real-world application tips
