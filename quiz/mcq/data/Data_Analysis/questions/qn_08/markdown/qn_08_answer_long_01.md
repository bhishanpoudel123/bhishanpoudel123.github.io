# Data Science Study Guide - Question 08

## **Question:**
What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?

## **Answer Choices:**
1. **Oversampling minority classes using SMOTE**  
2. **Undersampling majority classes using NearMiss**  
3. **Cost-sensitive learning with class weights inversely proportional to frequencies**  
4. **Ensemble of balanced subsets with `META` learning**  

---

## **Correct Answer:** `Ensemble of balanced subsets with META learning`

### **Explanation:**
Severe multiclass imbalance often requires **ensemble-based techniques** to balance class distributions while maintaining model performance.  
- **META (Minority Ethnicity and Threshold Adjustment) learning** trains multiple models on **balanced subsets**.
- **Combining multiple models** prevents information loss from undersampling while avoiding the artificial patterns introduced by synthetic oversampling.
- Preserves **true data distribution** while reducing bias in underrepresented classes.

Python implementation (META learning ensemble):
```python
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Simulated imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, weights=[0.01, 0.05, 0.94], n_classes=3)

# META-based ensemble classifier
meta_model = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10, sampling_strategy='auto')

meta_model.fit(X, y)
print("META model trained successfully")
```

---

## **Why Other Choices Are Incorrect?**
### **1. Oversampling minority classes using SMOTE**
- SMOTE generates synthetic samples but **can introduce noise**, leading to **overfitting**.
- Works well for binary classification but **less effective for highly imbalanced multiclass scenarios**.

Python demonstration:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE applied, but may introduce artifacts")
```

### **2. Undersampling majority classes using NearMiss**
- **Drops many majority class samples**, **reducing overall dataset size**, potentially losing valuable information.
- Works better when majority classes **overpower minority ones**, but not optimal for multiclass.

### **3. Cost-sensitive learning with class weights**
- Helps models recognize **underrepresented classes**, but can struggle in **severely imbalanced cases**.
- Doesn't prevent **biased decision boundaries**, leading to suboptimal results.

Python example:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X, y)
print("Class-weighted model applied")
```

