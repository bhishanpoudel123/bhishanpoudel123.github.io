```markdown
# qn_14.md

## Question 14: Custom Scoring Function for RandomizedSearchCV

**Question:**  
What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?

**Options:**
1. Use make_scorer with a function that combines multiple metrics
2. Implement a custom Scorer class with a custom __call__ method
3. Use multiple evaluation metrics with refit parameter specifying the primary metric
4. Create a pipeline with a custom transformer that adds a penalty term based on complexity

**Correct Answer:**  
Use make_scorer with a function that combines multiple metrics

### Detailed Explanation

`make_scorer` provides the most flexible way to:
1. Combine multiple evaluation criteria into a single score
2. Maintain compatibility with sklearn's CV infrastructure
3. Handle both maximization and minimization cases
4. Add custom weights to different components

### Python Implementation

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# Custom scoring function
def complexity_penalized_score(y_true, y_pred, estimator, 
                             accuracy_weight=1.0, 
                             complexity_weight=0.1):
    """
    Combines accuracy with model complexity penalty
    Complexity measured by number of features used (for feature selection)
    or number of nodes (for tree-based models)
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Measure complexity - different approaches for different models
    if hasattr(estimator, 'n_features_in_'):
        complexity = estimator.n_features_in_
    elif hasattr(estimator, 'tree_') and hasattr(estimator.tree_, 'node_count'):
        complexity = estimator.tree_.node_count
    else:
        complexity = 0
    
    # Combine metrics (higher is better)
    return accuracy_weight * accuracy - complexity_weight * complexity

# Create custom scorer
custom_scorer = make_scorer(complexity_penalized_score, 
                           needs_proba=False,
                           needs_threshold=False,
                           estimator=True,
                           accuracy_weight=1.0,
                           complexity_weight=0.01)

# Example usage with RandomizedSearchCV
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_dist, n_iter=20,
                          scoring=custom_scorer,
                          n_jobs=-1, cv=5, random_state=42)
search.fit(X, y)

# Display results
results = pd.DataFrame(search.cv_results_)
print(results[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score'))
```

### Key Features:
1. **Flexible Metric Combination**: Weight accuracy vs complexity as needed
2. **Model-Aware Complexity**: Different measures for different model types
3. **Seamless Integration**: Works with all sklearn search methods

### Advanced Usage:
For more sophisticated complexity measures:
```python
def tree_complexity(estimator):
    """Calculate total nodes across all trees in ensemble"""
    if hasattr(estimator, 'estimators_'):
        return sum(t.tree_.node_count for t in estimator.estimators_)
    elif hasattr(estimator, 'tree_'):
        return estimator.tree_.node_count
    return 0
```