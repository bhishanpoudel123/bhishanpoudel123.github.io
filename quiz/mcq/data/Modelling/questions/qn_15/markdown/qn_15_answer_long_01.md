# qn_15.md

## Question 15: Feature Selection with Heteroscedastic Errors

**Question:**  
Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?

**Correct Answer:**  
Implement weighted LASSO with weight inversely proportional to error variance

### Python Implementation

```python
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
import numpy as np

# Generate data with heteroscedastic errors
X, y, coef = make_regression(n_samples=1000, n_features=20, 
                            n_informative=5, noise=0.5,
                            coef=True, random_state=42)
# Create heteroscedastic noise
noise = 0.5 * np.abs(X[:, 0]) * np.random.normal(size=1000)
y += noise

# Estimate variance function
from sklearn.ensemble import RandomForestRegressor
var_model = RandomForestRegressor()
var_model.fit(X, (y - y.mean())**2)
pred_var = var_model.predict(X)
weights = 1.0 / np.sqrt(pred_var + 1e-6)  # Stabilize division

# Weighted LASSO
lasso = LassoCV(cv=5, n_alphas=50)
lasso.fit(X, y, sample_weight=weights)

# Compare with standard LASSO
std_lasso = LassoCV(cv=5)
std_lasso.fit(X, y)

print("True features:", np.where(coef != 0)[0])
print("Weighted LASSO selected:", np.where(lasso.coef_ != 0)[0])
print("Standard LASSO selected:", np.where(std_lasso.coef_ != 0)[0])
```

### Key Advantages:
1. Accounts for non-constant error variance
2. More reliable feature selection
3. Maintains convex optimization properties

### Diagnostic Visualization:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data')
plt.scatter(X[:, 0], lasso.predict(X), color='red', 
           label='Weighted LASSO')
plt.scatter(X[:, 0], std_lasso.predict(X), color='green',
           label='Standard LASSO')
plt.xlabel('Feature 0')
plt.ylabel('Target')
plt.legend()
plt.show()
```

[Continued in next message due to length...]
```

I'll continue with the remaining questions in the following messages. Would you like me to make any adjustments to the format or content depth? Each subsequent question will maintain:
1. Detailed conceptual background
2. Production-ready Python code
3. Diagnostic visualizations
4. Alternative approaches comparison
5. Practical implementation tips
