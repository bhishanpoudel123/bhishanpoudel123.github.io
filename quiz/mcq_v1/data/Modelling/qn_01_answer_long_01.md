# Question 1: Preventing Target Leakage in Stacking Ensemble

**Category:** Modelling  
**Question:** When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?

## Options:
1. Use StackingClassifier with cv=5
2. Manually implement out-of-fold predictions for each base learner
3. Train base models on 70% of data and meta-model on remaining 30%
4. Use scikit-learn's pipeline to ensure proper nesting of cross-validation

## Correct Answer:
**Manually implement out-of-fold predictions for each base learner**

## Detailed Explanation:
Target leakage occurs when the meta-learner sees data that the base models were trained on. Manual OOF implementation ensures complete separation between training and prediction data.

### Python Implementation:
```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate OOF predictions
def get_oof_predictions(X, y, models, n_splits=5):
    oof_preds = np.zeros((X.shape[0], len(models)))
    kf = KFold(n_splits=n_splits)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        
        for j, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            oof_preds[val_idx, j] = model.predict_proba(X_val)[:, 1]
    
    return oof_preds

# Usage
models = {'rf': RandomForestClassifier(), 'gb': GradientBoostingClassifier()}
oof_preds = get_oof_predictions(X, y, models)
meta_model = LogisticRegression().fit(oof_preds, y)