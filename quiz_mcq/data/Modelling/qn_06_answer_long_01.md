
---

### qn_06.md
```markdown
# Question 6: Monotonic Constraints in Gradient Boosting

**Category:** Modelling  
**Question:** What's the most statistically sound approach to implement monotonic constraints?

## Correct Answer:
**Using XGBoost's monotone_constraints parameter**

## Python Implementation:
```python
import xgboost as xgb

# Assuming 'age' is feature 0 and we want positive monotonicity
constraints = [1, 0, -1]  # +1: inc, 0: no constraint, -1: dec

model = xgb.XGBClassifier(
    monotone_constraints=constraints,
    tree_method='hist',
    n_estimators=100
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10)

# Verify monotonicity
from xgboost import plot_tree
plot_tree(model, num_trees=0)