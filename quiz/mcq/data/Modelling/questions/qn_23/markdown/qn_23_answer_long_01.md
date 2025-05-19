### ✅ **Question 23**

**Q: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?**

**Correct Answer:**
`Implement multi-fidelity optimization with Hyperband`

---

### 🧠 Key Concepts

* **Multi-fidelity Optimization:** Evaluates model configurations using partial training (e.g., fewer epochs) to save time and resources.
* **Hyperband:** A bandit-based algorithm that efficiently allocates resources by:

  * Randomly sampling configurations.
  * Early-stopping poor performers.
  * Allocating more resources to promising candidates.
* **Advantage:** Drastically reduces time-to-tune while maintaining near-optimal performance.

---

### 🔧 How Hyperband Works

1. Randomly samples a large number of hyperparameter configurations.
2. Trains each configuration with limited resources (e.g., few epochs or small data subset).
3. Evaluates and retains top performers.
4. Increases resource allocation (fidelity) for survivors.
5. Repeats until optimal resource use.

---

### 🧪 Example in Python (Using `scikit-optimize` and `ray[tune]`)

```python
# Install: pip install ray[tune] scikit-learn xgboost

from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def train_xgb(config):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    results = {}
    xgb.train(
        config,
        dtrain,
        num_boost_round=100,
        evals=[(dval, "eval")],
        early_stopping_rounds=10,
        evals_result=results,
        verbose_eval=False,
    )

    tune.report(eval_logloss=results["eval"]["logloss"][-1])

# Define hyperparameter space
search_space = {
    "objective": "binary:logistic",
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "max_depth": tune.randint(3, 10),
    "subsample": tune.uniform(0.5, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
}

scheduler = HyperBandScheduler(metric="eval_logloss", mode="min")

analysis = tune.run(
    train_xgb,
    resources_per_trial={"cpu": 1},
    config=search_space,
    num_samples=20,
    scheduler=scheduler,
)

print("Best config: ", analysis.get_best_config(metric="eval_logloss", mode="min"))
```

---

### 🚀 Benefits

* **Fast convergence:** Quickly eliminates poor configurations.
* **Resource-aware:** Makes the best use of limited time or compute.
* **Parallelizable:** Can be distributed across multiple CPUs/GPUs.

---

### 📚 References

* Li et al., 2017: *"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"*
* [Ray Tune Hyperband Docs](https://docs.ray.io/en/latest/tune/api/schedulers.html#hyperbandscheduler)

---

