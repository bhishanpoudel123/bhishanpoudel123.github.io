### ✅ Question 27

**Q: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?**

**Correct Answer:**  
`Individual Conditional Expectation (ICE) plots with centered PDP`

---

### 🧠 Why This is Correct

**ICE (Individual Conditional Expectation)** plots visualize how a feature affects model predictions for individual data points. This is crucial when feature effects are **non-linear** and **heterogeneous** (i.e., the effect differs across instances).

- **Centered ICE plots** show deviations from a baseline (usually the prediction at the feature’s median), which helps detect **interaction effects** and **non-linear patterns**.
- Unlike PDPs (Partial Dependence Plots) that average across all instances, ICE shows *per-instance sensitivity*.

---

### 🔍 Explanation of All Choices

#### ✅ Option D: `Individual Conditional Expectation (ICE) plots with centered PDP`  
- Best for capturing **individual-level effects**.
- Can uncover **non-linear** and **non-monotonic** behaviors missed by global summaries.
- Centering emphasizes how prediction changes as a function of a feature, independent of baseline prediction.

#### ❌ Option A: `Partial dependence plots with contour plots for interactions`  
- PDPs average effects across all data points, which can **mask non-linearities** or **interactions**.
- Contour plots (for 2D interactions) are useful but still represent **global averages**, not individual behaviors.
- Cannot capture **heterogeneous effects** as effectively as ICE.

#### ❌ Option B: `Accumulated Local Effects (ALE) plots with bootstrap confidence intervals`  
- ALE plots improve over PDP by avoiding unrealistic extrapolations.
- More robust to correlated features.
- However, they still reflect **global trends**, not individual-level effects. Great for summarization, not for detailed detection.

#### ❌ Option C: `SHAP interaction values with dependency plots`  
- SHAP interaction values can identify and quantify **pairwise interactions**.
- SHAP plots are excellent for **feature importance** but:
  - Can be **computationally expensive**.
  - Do **not visualize full functional forms** across feature values like ICE.
- Less intuitive than ICE for interpreting smooth or complex feature effects.

---

### 🧪 Python Example: ICE vs PDP

```python
# Install: pip install scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence, PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Fit model
model = GradientBoostingRegressor().fit(X, y)

# Feature to inspect
feature = "AveRooms"
feature_index = X.columns.get_loc(feature)

# Plot ICE with centered PDP
display = PartialDependenceDisplay.from_estimator(
    model,
    X,
    features=[feature_index],
    kind="individual",  # ICE
    centered=True,
    subsample=100,
    grid_resolution=50,
    random_state=42
)
display.figure_.suptitle("Centered ICE Plot for AveRooms")
plt.tight_layout()
plt.show()
````

---

### 🔬 Summary of Techniques

| Technique | Captures Individual Effects | Captures Interactions  | Non-linear Friendly | Comments                         |
| --------- | --------------------------- | ---------------------- | ------------------- | -------------------------------- |
| PDP       | ✖️                          | ✔️ (limited, global)   | ✖️                  | Averages out individual effects  |
| ICE       | ✔️                          | ✔️ (if centered)       | ✔️                  | Best for detecting heterogeneity |
| ALE       | ✖️                          | ✔️ (local gradients)   | ✔️                  | Robust to correlation            |
| SHAP      | ✔️ (via values)             | ✔️ (interaction terms) | ✔️                  | High computational cost          |

---

### 📚 References

* Goldstein et al. (2015). *"Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation"*
* Molnar, C. (2022). *Interpretable Machine Learning*, Chapter: ICE and PDP
* scikit-learn documentation: [`PartialDependenceDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html)

