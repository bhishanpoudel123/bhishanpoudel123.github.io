### **Question 17**

**Q:** When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?

**Options:**

* A. Random search with early stopping
* B. Genetic algorithms with tournament selection
* C. Bayesian optimization with Gaussian processes
* D. Hyperband with successive halving

**✅ Correct Answer:** Bayesian optimization with Gaussian processes

---

### 🧠 Explanation:

#### ✅ C. **Bayesian optimization with Gaussian processes**

Bayesian optimization is ideal for **expensive black-box functions** like complex ML model tuning. It builds a **surrogate probabilistic model** (usually a Gaussian Process) of the objective function and chooses hyperparameters **intelligently** based on prior observations.

* ✅ Fewer evaluations required than grid/random search
* ✅ Efficient for **high-dimensional** and **expensive** search spaces
* ✅ Can balance **exploration vs. exploitation**

```python
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20)

search_space = {
    'n_estimators': (50, 300),
    'max_depth': (3, 20),
    'min_samples_split': (2, 10)
}

opt = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=search_space,
    n_iter=32,
    cv=3,
    scoring='f1_macro',
    random_state=42
)
opt.fit(X, y)
```

✔️ **Best for complex, expensive models.**

---

### ❌ Other Options:

#### A. **Random search with early stopping**

Random search samples hyperparameters randomly and is better than grid search for high-dimensional spaces. With **early stopping**, it reduces computation, but:

* ❌ Still inefficient — many configurations are **wasted**
* ❌ Doesn’t leverage **previous results**
* ✅ Good baseline for smaller problems

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
}

rs = RandomizedSearchCV(GradientBoostingClassifier(), param_dist, n_iter=10)
rs.fit(X, y)
```

🟡 **Basic but not optimal for complex models.**

---

#### B. **Genetic algorithms with tournament selection**

Genetic Algorithms (GAs) are **evolutionary approaches** that mimic natural selection. They can explore large search spaces and escape local optima, but:

* ❌ High **computational cost**
* ❌ Many **hyperparameters of their own**
* ✅ Flexible for unusual search spaces (e.g., categorical/mixed)

```python
# With DEAP or TPOT (example only — complex setup)
# GAs are experimental and hard to tune themselves
```

🔴 **Creative, but less practical in standard workflows.**

---

#### D. **Hyperband with successive halving**

Hyperband is a **bandit-based** method that tries many configurations quickly using small budgets and allocates more resources to promising ones.

* ✅ Very efficient for models that support **partial training**
* ❌ Less effective when **training is non-incremental** (e.g., SVMs)
* ❌ Doesn’t model performance surface

```python
# Supported via Ray Tune or Optuna
# Good for neural networks where epoch-based training allows budget control
```

🟡 Good for deep learning, but not always best for general ML models.

---

### 📚 Summary

| Method                    | Models Performance Surface? | Adaptive? | Good for High Dim. Spaces? | Verdict          |
| ------------------------- | --------------------------- | --------- | -------------------------- | ---------------- |
| Random Search             | ❌ No                        | ❌ No      | ✅ Yes                      | 🟡               |
| Genetic Algorithms        | ❌ No                        | ✅ Yes     | ✅ Yes                      | 🔴               |
| **Bayesian Optimization** | ✅ Yes                       | ✅ Yes     | ✅ Yes                      | ✅ **Best**       |
| Hyperband                 | ❌ No                        | ✅ Yes     | ✅ Yes                      | 🟡 (good for NN) |

---
