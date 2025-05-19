
---

### qn_05.md
```markdown
# Question 5: Competing Risks Survival Analysis

**Category:** Modelling  
**Question:** When dealing with competing risks, which implementation correctly handles the problem?

## Correct Answer:
**Fine-Gray subdistribution hazard model from pysurvival**

## Python Implementation:
```python
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.display import compare_to_actual

# Prepare data (T=time, E=event_type)
model = RandomSurvivalForestModel(num_trees=100)
model.fit(X_train, T_train, E_train, 
          event_of_interest=1)  # Specify main event type

# Predict cumulative incidence
cif = model.predict_cumulative_hazard(X_test)

# Validate with actual events
results = compare_to_actual(model, X_test, T_test, E_test,
                           events_of_interest=[1], 
                           figuresize=(16,6))