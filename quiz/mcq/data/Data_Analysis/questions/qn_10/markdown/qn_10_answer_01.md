# Data Science Study Guide - Question 10

## **Question:**
Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?

## **Answer Choices:**
1. **`MultiOutputRegressor(GradientBoostingRegressor())`**  
2. **`GradientBoostingRegressor` with `multioutput='raw_values'`**  
3. **`RegressorChain(GradientBoostingRegressor())` with StandardScaler for each target**  
4. **Separate scaled `GradientBoostingRegressor` for each target in a Pipeline**  

---

## **Correct Answer:** `MultiOutputRegressor(GradientBoostingRegressor())`

### **Explanation:**
The **MultiOutputRegressor** wrapper in scikit-learn allows a **separate GradientBoostingRegressor** to be trained for each target variable, ensuring independent optimization when dealing with multiple continuous targets.  
- Each regressor can adjust hyperparameters **independently**, improving predictive accuracy.
- This method works best when target variables **do not share feature dependencies**.

Python implementation:
```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Simulated dataset with multiple target variables
X = np.random.rand(100, 5)
y = np.random.rand(100, 2)  # Two target variables

# Multi-output regression model
model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(X, y)

predictions = model.predict(X)
print(predictions)
```

---

## **Why Other Choices Are Incorrect?**
### **1. `GradientBoostingRegressor` with `multioutput='raw_values'`**
- **No such parameter (`multioutput='raw_values'`)** exists in `GradientBoostingRegressor`.
- Gradient Boosting does **not inherently support multi-output regression**, requiring wrappers like `MultiOutputRegressor`.

### **2. `RegressorChain(GradientBoostingRegressor())` with StandardScaler**
- **`RegressorChain` forces sequential predictions**, where one target depends on another.
- This works for **correlated targets**, but **not when they are independent**.
- Scaling targets individually is useful, but this method is **not optimal for general multi-output regression**.

Python demonstration:
```python
from sklearn.multioutput import RegressorChain

chain_model = RegressorChain(GradientBoostingRegressor())
chain_model.fit(X, y)

print(chain_model.predict(X))  # Predictions for multiple targets
```
This **forces dependencies** between targets unnecessarily.

### **3. Separate scaled `GradientBoostingRegressor` for each target in a Pipeline**
- **Manually managing separate models** increases complexity.
- `MultiOutputRegressor` automatically handles separate models, making this approach **redundant**.

