### ✅ Question 30

**Q: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?**

**Correct Answer:**  
`Apply precision-recall curves with prevalence-corrected metrics`

---

### 🧠 Why This is Correct

In **multi-class imbalanced classification**, standard accuracy or even ROC AUC can be **misleading** because:

- Majority class performance dominates the overall score.
- Minority class errors are underrepresented.

**Precision-Recall (PR) curves**, especially when weighted by class prevalence, focus on **minority class performance**, which is crucial in imbalanced datasets (e.g., medical diagnosis, fraud detection).

- **Precision** = TP / (TP + FP): How many predicted positives are truly positive?
- **Recall** = TP / (TP + FN): How many actual positives are captured?

Prevalence correction ensures that class-specific performance is not biased by class imbalance.

---

### 🔍 Explanation of All Choices

#### ✅ Option D: `Apply precision-recall curves with prevalence-corrected metrics`
- Best choice for **imbalanced classification**.
- PR curves highlight the **tradeoff between precision and recall**, especially for minority classes.
- Class prevalence weighting ensures **fair evaluation across all classes**.

#### ❌ Option A: `Use macro-averaged precision, recall, and F1 score`
- **Macro-average** treats all classes equally, which is good, but:
  - It does **not account for class imbalance**.
  - Can be misleading if performance on rare classes is poor.

#### ❌ Option B: `Implement balanced accuracy and Cohen's kappa statistic`
- **Balanced accuracy** averages recall per class, but ignores **precision**.
- **Cohen’s kappa** adjusts for chance agreement, useful for inter-rater reliability, but:
  - Not specialized for evaluating **minority class detection**.
  - Lacks class-level granularity.

#### ❌ Option C: `Use ROC AUC with one-vs-rest approach and weighted averaging`
- ROC AUC often looks good even on **imbalanced datasets**.
- High AUC does not guarantee high **precision** for minority classes.
- ROC curves can be **overly optimistic** when negatives dominate.

---

### 🧪 Example: Multi-Class PR Curves with Class Weights

```python
# Install: pip install scikit-learn matplotlib

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Simulate imbalanced multi-class dataset
X, y = make_classification(n_samples=1000, n_classes=3, weights=[0.7, 0.2, 0.1], n_informative=5, n_clusters_per_class=1)
y_bin = label_binarize(y, classes=[0, 1, 2])

# Fit classifier
clf = RandomForestClassifier().fit(X, y)
y_score = clf.predict_proba(X)

# Plot PR curves for each class
plt.figure(figsize=(8, 5))
for i in range(3):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves by Class")
plt.legend()
plt.tight_layout()
plt.show()

# Also print classification report
print(classification_report(y, clf.predict(X)))
````

---

### 📌 Why PR Curves Are Better Than ROC for Imbalanced Data

| Metric              | Sensitive to Class Imbalance | Highlights Minority Classes | Useful in Multi-Class? |
| ------------------- | ---------------------------- | --------------------------- | ---------------------- |
| Accuracy            | ❌                            | ❌                           | ❌                      |
| ROC AUC             | ❌                            | ❌                           | ✅ (but biased)         |
| PR Curve            | ✅                            | ✅                           | ✅                      |
| F1 (macro/weighted) | ✅                            | ✅                           | ✅                      |

---

### 📚 References

* Saito & Rehmsmeier (2015). *"The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"*
* scikit-learn: [`precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
* scikit-learn: [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

