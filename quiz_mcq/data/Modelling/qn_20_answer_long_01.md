# qn_20.md

## Question 20: Early Stopping in Gradient Boosting

**Question:**  
What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?

**Correct Answer:**  
Use early_stopping_rounds with a validation set in XGBoost/LightGBM

### Python Implementation

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prepare data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# XGBoost with early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=10,
    evals_result=evals_result
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(evals_result['train']['error'], label='Train')
plt.plot(evals_result['eval']['error'], label='Validation')
plt.axvline(model.best_iteration, color='gray', linestyle='--')
plt.title('Early Stopping at Iteration %d' % model.best_iteration)
plt.xlabel('Boosting Rounds')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

# LightGBM implementation
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'max_depth': 5,
    'learning_rate': 0.1,
    'metric': 'binary_error'
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(10)
    ]
)
```

### Key Features:
1. **Automatic Stopping**: Halts when validation metric doesn't improve
2. **Best Model Retention**: Returns model from best iteration
3. **Visual Feedback**: Track training/validation metrics

### Advanced Usage:
```python
# Custom early stopping callback
def custom_early_stop(stopping_rounds, metric_name, maximize=False):
    def callback(env):
        if maximize:
            best = max(env.evaluation_result_list[-stopping_rounds:], 
                      key=lambda x: x[1][metric_name])
            current = env.evaluation_result_list[-1][1][metric_name]
            if current < best[1][metric_name]:
                env.model.stop_training = True
        else:
            best = min(env.evaluation_result_list[-stopping_rounds:], 
                     key=lambda x: x[1][metric_name])
            current = env.evaluation_result_list[-1][1][metric_name]
            if current > best[1][metric_name]:
                env.model.stop_training = True
    return callback

# Feature importance analysis
xgb.plot_importance(model, max_num_features=10)
plt.show()
```

### Comparison of Methods:
| Method | Pros | Cons |
|--------|------|------|
| Fixed n_estimators | Simple | Risk of under/overfitting |
| CV-tuned rounds | Robust | Computationally expensive |
| **Early stopping** | **Efficient** | **Requires validation set** |
```
