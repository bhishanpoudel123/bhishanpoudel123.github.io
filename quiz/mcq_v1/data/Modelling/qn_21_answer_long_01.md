### ✅ **Question 21**

**Q: Which approach correctly implements a counterfactual explanation method for a black-box classifier?**

**Correct Answer:**
`Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals`

**Explanation:**
DiCE (Diverse Counterfactual Explanations) is a model-agnostic method designed to generate multiple, diverse counterfactual examples that can help answer "what needs to change for a different outcome?" for black-box models. It is particularly useful in high-stakes domains like healthcare and finance for model transparency and actionable insights.

Unlike LIME or SHAP which provide feature importance or local sensitivity, DiCE focuses on *changing the outcome*, which is ideal for explainability under intervention scenarios.

---

#### 🧠 Key Concepts

* **Counterfactual Explanation:** Suggests minimal feature changes required to change a model's prediction.
* **Diversity Constraint:** Encourages generating multiple unique and feasible examples.
* **Model Agnosticism:** DiCE works with any classifier, including black-box models like XGBoost, neural nets, etc.

---

#### 🧪 Example in Python

You can use the `dice-ml` library for implementing DiCE:

```python
# Install: pip install dice-ml

import dice_ml
from dice_ml.utils import helpers
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# For simplicity, we turn this into a binary classification
X = X[y != 2]
y = y[y != 2]

# Train a classifier
clf = RandomForestClassifier().fit(X, y)

# Wrap model using DiCE
data_dice = dice_ml.Data(dataframe=pd.concat([X, pd.Series(y, name='target')], axis=1),
                         continuous_features=data.feature_names,
                         outcome_name='target')

model_dice = dice_ml.Model(model=clf, backend="sklearn")

# Create DiCE explainer
explainer = dice_ml.Dice(data_dice, model_dice, method="random")

# Select an instance to explain
query_instance = X.iloc[0:1]

# Generate counterfactuals
dice_exp = explainer.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")

# View results
dice_exp.visualize_as_dataframe()
```

---

#### 📚 References

* [DiCE GitHub Repo](https://github.com/interpretml/DiCE)
* Mothilal, D. R., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations.

---
