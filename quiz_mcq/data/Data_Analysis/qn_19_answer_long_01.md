### **Question 19**

**Q:** Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?

**Options:**

* A. Bottom-up approach: forecast at lowest level and aggregate upwards
* B. Top-down approach: forecast at highest level and disaggregate proportionally
* C. Middle-out approach: forecast at a middle level and propagate in both directions
* D. Reconciliation approach: forecast at all levels independently then reconcile with constraints

**✅ Correct Answer:** Reconciliation approach: forecast at all levels independently then reconcile with constraints

---

### 🧠 Explanation:

#### ✅ D. **Reconciliation approach**

The **reconciliation approach** solves the problem of inconsistent forecasts across hierarchical levels (e.g., region, country, continent) by:

* Forecasting each node **independently**
* Applying a **reconciliation algorithm** to ensure that forecasts respect the **hierarchy’s aggregation constraints**
* Using methods like **Minimum Trace (MinT)** or **OLS-based reconciliation**

This is the most statistically sound and flexible method.

```python
from hts import HierarchicalTimeSeries
from hts.model import HoltWintersModel
from hts.revision import TopDown, BottomUp, MinTrace
from hts.hierarchy import HierarchyTree

# Example (simplified, synthetic)
# Assume df contains time series at multiple hierarchical levels
# Structure defined in a JSON or dictionary tree

hts_model = HierarchicalTimeSeries(
    model='prophet',  # or 'holt_winters', 'auto_arima'
    revision_method='mint',  # Reconciliation method
    hierarchy=HierarchyTree.from_nodes(nodes),  # tree of nodes
    n_jobs=4
)

hts_model.fit(df)
forecast = hts_model.predict(steps_ahead=12)
```

✔️ Produces **coherent forecasts** across all levels of the hierarchy.

---

### ❌ Other Options:

#### A. **Bottom-up approach**

This method:

* Forecasts at the **lowest level** (e.g., individual stores)
* Aggregates to get higher levels (e.g., regional or national sales)

Issues:

* ❌ Low-level data is often **noisy**
* ❌ Doesn’t leverage potentially better **top-level signal**

```python
# Simple bottom-up: sum forecasts
bottom_level_forecasts = ...
total_forecast = bottom_level_forecasts.sum(axis=0)
```

🟡 Simple but can be **inaccurate** at higher levels.

---

#### B. **Top-down approach**

This method:

* Forecasts at the **top level** (e.g., total sales)
* Disaggregates to lower levels using proportions

Drawbacks:

* ❌ Assumes **historical proportions** are stable
* ❌ Ignores bottom-level dynamics

```python
# Disaggregate based on historical shares
proportions = lower_level_series.sum() / total_series.sum()
lower_forecasts = total_forecast * proportions
```

🔴 Oversimplifies hierarchical dynamics.

---

#### C. **Middle-out approach**

Forecasting at a **middle level**, then:

* Aggregates upwards
* Disaggregates downwards

Limitations:

* ❌ Hybrid of two flawed methods
* ❌ Still doesn’t guarantee **hierarchical consistency**

🟡 Used in retail and practical settings, but not **statistically optimal**.

---

### 📚 Summary

| Method             | Coherent? | Uses All Levels?       | Statistically Rigorous? | Verdict    |
| ------------------ | --------- | ---------------------- | ----------------------- | ---------- |
| Bottom-up          | ✅ Yes     | ❌ No (only low level)  | ❌ No                    | 🟡         |
| Top-down           | ✅ Yes     | ❌ No (only high level) | ❌ No                    | 🔴         |
| Middle-out         | ❌ No      | ❌ No                   | ❌ No                    | 🔴         |
| **Reconciliation** | ✅ Yes     | ✅ Yes                  | ✅ Yes                   | ✅ **Best** |

