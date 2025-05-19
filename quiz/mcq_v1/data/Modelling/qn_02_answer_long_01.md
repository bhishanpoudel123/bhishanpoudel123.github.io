
---

### qn_02.md
```markdown
# Question 2: Calibrating Gradient Boosting Probabilities

**Category:** Modelling  
**Question:** What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?

## Correct Answer:
**Apply sklearn's CalibratedClassifierCV with isotonic regression**

## Python Implementation:
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# Original model
gb = GradientBoostingClassifier()

# Calibrated model (prefit=False for proper cross-validated calibration)
calibrated_gb = CalibratedClassifierCV(gb, method='isotonic', cv=5)
calibrated_gb.fit(X_train, y_train)

# Compare calibration
from sklearn.calibration import calibration_curve
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, gb.predict_proba(X_test)[:, 1], n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_gb.predict_proba(X_test)[:, 1], n_bins=10)