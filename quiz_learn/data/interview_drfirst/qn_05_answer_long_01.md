
# A/B Testing

### 🧪 **A/B Testing – PowerBI Dashboard Optimization (AmerisourceBergen)**

#### 🎯 **Goal:**

Compare two PowerBI dashboard designs to identify which one drives better user engagement.

#### ⚙️ **Steps:**

* **Experimental Design:**
  Created two dashboard versions (A & B), randomly assigned users to each.

* **Power Analysis (Sample Size):**
  Used `statsmodels` to estimate required sample size with:

  * Alpha = 0.05, Power = 0.8
  * Assumed 10% expected improvement
  * Calculated using `NormalIndPower().solve_power(...)`

* **Metrics Tracked:**

  * Click-through rates
  * Time on dashboard
  * Drill-down usage
  * Daily active users

* **Statistical Tests:**

  * T-test for means
  * Z-test for proportions
  * Fisher’s Exact Test & Permutation Test for robustness

* **Outcome:**
  Version B improved engagement by **13%** (statistically significant). Rolled out as the new standard.

#### 🧰 **Tools:**

Python (`statsmodels`, `scipy`, `pandas`), PowerBI, Databricks, seaborn


### ✅ **A/B Testing for Report Variation (PowerBI Dashboards)**

**Objective:**
To determine which variation of a PowerBI dashboard (e.g., layout, chart type, or feature placement) led to better user engagement and decision-making efficiency for business stakeholders.



### 📊 **What I Did**

#### 1. **Designing the Experiment**

* Created **two versions** of the dashboard:

  * Version A: Original layout.
  * Version B: New design based on UX feedback.
* Randomly assigned users (e.g., business analysts, managers) to either group to ensure unbiased comparison.

#### 2. **Power Analysis (Sample Size Estimation)**

* Before launching the experiment, I conducted **power analysis** to determine the **minimum number of users needed per group** to detect a meaningful difference.
* Used assumptions such as:

  * Baseline engagement rate (e.g., 45%)
  * Minimum detectable lift (e.g., +10%)
  * Alpha = 0.05 (significance level)
  * Power = 0.8 (probability of detecting a true effect)

📌 *Tool:* Python (`statsmodels.stats.power`) to calculate required sample size:

```python
from statsmodels.stats.power import NormalIndPower

effect_size = 0.2  # Cohen's h for small to medium effect
analysis = NormalIndPower()
sample_size = analysis.solve_power(effect_size, power=0.8, alpha=0.05)
```



#### 3. **Data Collection**

* Tracked key metrics:

  * Click-through rates
  * Time spent on dashboard
  * Frequency of use
  * Drill-down interaction count
* Collected using PowerBI usage logs, exported into a centralized analytics pipeline (Databricks + Pandas).



#### 4. **Statistical Testing**

* Applied **two-proportion z-test** and **T-tests** for numerical metrics.
* For categorical comparisons (e.g., button clicks), also used:

  * **Fisher’s Exact Test** (for small samples)
  * **Permutation Test** (non-parametric validation)



#### 5. **Results & Outcome**

* Found that **Version B** increased key interaction metrics by **13%**, with statistically significant p-values (p < 0.05).
* Presented findings to stakeholders with visualizations (PowerBI and seaborn).
* Rolled out Version B across the org as the new default.



### 📦 Tools & Technologies

* `statsmodels`, `scipy`, `pandas`, `PowerBI`
* Databricks for data handling
* Visualized uplift with `matplotlib` & `seaborn`
