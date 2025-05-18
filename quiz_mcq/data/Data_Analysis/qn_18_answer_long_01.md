### **Question 18**

**Q:** What's the most statistically sound approach to handle heteroscedasticity in a regression model?

**Options:**

* A. Visual inspection of residuals vs. fitted values plot
* B. `Breusch-Pagan` test for constant variance
* C. `White's test` for homoscedasticity
* D. Both B and C, with different null hypotheses

**✅ Correct Answer:** Both B and C, with different null hypotheses

---

### 🧠 Explanation:

#### ✅ D. **Both B and C, with different null hypotheses**

To rigorously assess **heteroscedasticity** (non-constant variance of errors) in regression models, we rely on **statistical tests**:

* **Breusch-Pagan Test**:

  * Null Hypothesis: Residuals have **constant variance** (homoscedasticity).
  * Assumes that heteroscedasticity is a **linear function of predictors**.
* **White’s Test**:

  * Null Hypothesis: Residuals have **constant variance**.
  * Does **not** assume any specific functional form; robust to **model misspecification**.

Using **both** gives a more complete assessment since they test under **different assumptions**.

```python
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np

# Simulated data
np.random.seed(0)
X = np.random.normal(0, 1, 100)
y = 3 * X + np.random.normal(0, 1 + X**2, 100)  # Heteroscedastic errors

df = pd.DataFrame({'X': X, 'y': y})
model = ols('y ~ X', data=df).fit()

# Breusch-Pagan test
bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
print("Breusch-Pagan p-value:", bp_test[1])

# White test
white_test = sms.het_white(model.resid, model.model.exog)
print("White test p-value:", white_test[1])
```

✔️ Using both helps **validate assumptions** and guide next steps (e.g., robust SEs or transformation).

---

### ❌ Other Options:

#### A. **Visual inspection of residuals vs. fitted values**

This is a **diagnostic** tool, not a statistical test.

* ✅ Helps detect patterns (e.g., funnel shape)
* ❌ Subjective and non-quantitative
* ❌ Can't confirm heteroscedasticity statistically

```python
import matplotlib.pyplot as plt

plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
```

🟡 **Useful for EDA**, but not sufficient for inference.

---

#### B. **Breusch-Pagan test for constant variance**

A good test, but:

* ❌ Assumes **linear relationship** between residual variance and predictors.
* ✅ More **powerful** if that assumption holds.

```python
bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
```

🟡 **Correct but limited** in flexibility.

---

#### C. **White's test for homoscedasticity**

More general than Breusch-Pagan:

* ✅ No assumption about the form of heteroscedasticity
* ❌ Slightly **less powerful** if the form is known

```python
white_test = sms.het_white(model.resid, model.model.exog)
```

🟡 **Flexible and robust**, but should be used alongside BP test.

---

### 📚 Summary

| Method           | Statistical Test? | Assumes Linear Relation? | Robust to Misspecification? | Verdict    |
| ---------------- | ----------------- | ------------------------ | --------------------------- | ---------- |
| Residual Plot    | ❌ No              | ❓ N/A                    | ❌ No                        | 🔴         |
| Breusch-Pagan    | ✅ Yes             | ✅ Yes                    | ❌ No                        | 🟡         |
| White’s Test     | ✅ Yes             | ❌ No                     | ✅ Yes                       | 🟡         |
| **Both B and C** | ✅ Yes             | Mixed                    | ✅ Comprehensive             | ✅ **Best** |

