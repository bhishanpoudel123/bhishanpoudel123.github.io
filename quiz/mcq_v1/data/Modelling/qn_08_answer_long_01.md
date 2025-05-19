
---

### qn_08.md
```markdown
# Question 8: Feature Selection with Correlated Features

**Category:** Modelling  
**Question:** What's the most rigorous approach for correlated features?

## Correct Answer:
**Elastic Net regularization with randomized hyperparameter search**

## Python Implementation:
```python
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with standardization
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('en', ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        cv=5,
        n_alphas=100,
        random_state=42
    ))
])

pipe.fit(X_train, y_train)

# Extract selected features
coef = pipe.named_steps['en'].coef_
selected = np.where(coef != 0)[0]