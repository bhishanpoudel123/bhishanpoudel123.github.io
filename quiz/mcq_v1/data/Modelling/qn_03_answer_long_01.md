
---

### qn_03.md
```markdown
# Question 3: Nested Cross-Validation

**Category:** Modelling  
**Question:** Which approach correctly implements proper nested cross-validation?

## Correct Answer:
**Nested loops of KFold.split(), with inner loop for hyperparameter tuning**

## Python Implementation:
```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

outer_cv = KFold(n_splits=5)
inner_cv = KFold(n_splits=3)

outer_scores = []
for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner CV for parameter tuning
    gs = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid={'max_depth': [3, 5, 10]},
                     cv=inner_cv)
    gs.fit(X_train, y_train)
    
    # Evaluate with best params
    best_model = gs.best_estimator_
    outer_scores.append(accuracy_score(y_test, best_model.predict(X_test)))

print(f"Final performance: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")