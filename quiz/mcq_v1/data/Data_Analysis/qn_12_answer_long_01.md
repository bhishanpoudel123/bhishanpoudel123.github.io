### **Question 12**

**Q:** What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?

**Options:**

* A. Propensity score matching with sensitivity analysis
* B. Instrumental variable analysis with validity tests
* C. Causal graphical models with do-calculus
* D. Difference-in-differences with parallel trends validation

**✅ Correct Answer:** Causal graphical models with do-calculus

---

### 🧠 Explanation:

#### ✅ C. **Causal graphical models with do-calculus**

Causal graphical models allow for formal expression of **causal assumptions** using **directed acyclic graphs (DAGs)**. **Do-calculus**, introduced by Judea Pearl, provides a rigorous symbolic framework for reasoning about interventions (using the `do(X)` operator) in the presence of confounding variables.

These models enable identification of **causal effects** under specific assumptions and can mathematically verify **identifiability** of causal queries even in complex systems.

```python
# Using `dowhy` for causal inference
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'X': np.random.binomial(1, 0.5, 1000),
    'Z': np.random.normal(0, 1, 1000),
    'Y': np.random.binomial(1, 0.5, 1000)
})

# Define causal model
model = CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    common_causes=['Z']
)

model.view_model()

# Identify the causal effect using backdoor criterion
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
```

✔️ **Best choice** when mathematical rigor and identifiability are required.

---

### ❌ Other Options:

#### A. **Propensity score matching with sensitivity analysis**

This method estimates causal effects by balancing observed covariates. However, it **only adjusts for observed confounders** and is **sensitive to unobserved variables**. While it improves balance, it cannot guarantee identification of the causal effect without strong assumptions.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(data[['Z']], data['X'])
data['propensity_score'] = ps_model.predict_proba(data[['Z']])[:, 1]

# Matching
treated = data[data['X'] == 1]
control = data[data['X'] == 0]
matcher = NearestNeighbors(n_neighbors=1).fit(control[['propensity_score']])
distances, indices = matcher.kneighbors(treated[['propensity_score']])
```

🔴 **Limited by unmeasured confounding.**

---

#### B. **Instrumental variable (IV) analysis with validity tests**

IVs can help in the presence of unmeasured confounding **if a valid instrument is available**. However, finding a **valid IV** (relevant, exogenous, and only affecting the outcome through the treatment) is difficult. IV analysis requires strong assumptions and does not always guarantee identifiability.

```python
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Simulate data with instrument W
data['W'] = np.random.binomial(1, 0.5, 1000)

# 2-stage least squares
iv_model = IV2SLS.from_formula('Y ~ 1 + [X ~ W]', data=data)
iv_result = iv_model.fit()
```

🟡 **Useful if valid instruments exist, but harder to validate.**

---

#### D. **Difference-in-differences (DiD) with parallel trends validation**

DiD compares changes in outcomes over time between treated and control groups. It assumes that, in the absence of treatment, both groups would have **followed the same trend** ("parallel trends"). This assumption is often untestable and may not hold.

```python
import statsmodels.formula.api as smf

# Simulated pre/post data
data['time'] = np.random.choice([0, 1], size=1000)
data['treated'] = data['X']
data['interaction'] = data['time'] * data['treated']

# DiD regression
model = smf.ols('Y ~ time + treated + interaction', data=data).fit()
```

🔴 **Less general and relies heavily on untestable assumptions.**

---

### 📚 Summary

| Method                      | Handles Unobserved Confounding? | Requires Strong Assumptions? | Causal Guarantees? | Verdict    |
| --------------------------- | ------------------------------- | ---------------------------- | ------------------ | ---------- |
| Propensity Score Matching   | ❌ No                            | ✅ Yes                        | ❌                  | ❌          |
| Instrumental Variable (IV)  | ✅ Yes (if valid IV)             | ✅ Yes                        | ✅ if valid IV      | 🟡         |
| Causal Graphs + Do-calculus | ✅ Yes                           | ✅ Explicit but testable      | ✅                  | ✅ **Best** |
| Difference-in-Differences   | ❌ No                            | ✅ Yes (parallel trends)      | ❌                  | ❌          |

