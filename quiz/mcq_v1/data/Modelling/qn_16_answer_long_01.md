Here's the continuation of the remaining questions in the same detailed format:

```markdown
# qn_16.md

## Question 16: Interpretable Non-linear Regression Models

**Question:**  
What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?

**Correct Answer:**  
Use Explainable Boosting Machines (EBMs) from InterpretML

### Python Implementation

```python
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = load_diabetes()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train EBM
ebm = ExplainableBoostingRegressor(interactions=3)
ebm.fit(X_train, y_train)

# Interpret results
print(f"R^2 Score: {ebm.score(X_test, y_test):.3f}")

# Visualize feature effects
from interpret import show
ebm_global = ebm.explain_global()
show(ebm_global)

# Show specific interaction
print("\nTop interactions:")
for interaction in ebm.term_features_[-3:]:  # Show last 3 interactions
    if len(interaction) > 1:
        print(f"Interaction between {interaction}:")
        ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
        show(ebm_local)
```

### Key Features:
1. **Additive Model Structure**: Each feature contributes independently
2. **Automatic Interaction Detection**: Identifies important feature pairs
3. **Visual Interpretability**: Partial dependence plots for all features

### Comparison with Alternatives:
```python
# Compare with GAMs
from pygam import LinearGAM
gam = LinearGAM().fit(X_train, y_train)
print(f"GAM R^2: {gam.score(X_test, y_test):.3f}")

# Compare with Random Forest SHAP
import shap
rf = RandomForestRegressor().fit(X_train, y_train)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```
