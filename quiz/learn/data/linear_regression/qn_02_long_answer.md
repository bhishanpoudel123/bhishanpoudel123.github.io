### Detecting Multicollinearity

1. **Correlation Matrix**:  
   - Look for highly correlated predictor variables (e.g., > 0.8).

2. **Variance Inflation Factor (VIF)**:  
   - VIF > 5 or 10 is often a sign of multicollinearity.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from statsmodels.tools.tools import add_constant

X = add_constant(df[['feature1', 'feature2', 'feature3']])
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
```

3. **Eigenvalues & Condition Number**:  
   - A large condition number suggests collinearity issues.