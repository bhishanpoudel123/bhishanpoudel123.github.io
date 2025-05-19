Here are the final two questions (19-20) in the same comprehensive format:

```markdown
# qn_19.md

## Question 19: Curse of Dimensionality in Nearest Neighbor Models

**Question:**  
Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?

**Correct Answer:**  
Implement distance metric learning with NCA or LMNN

### Python Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from metric_learn import NCA, LMNN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# High-dimensional data
X, y = make_classification(n_samples=1000, n_features=100, 
                         n_informative=10, n_redundant=5,
                         n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Baseline KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"Baseline KNN Accuracy: {knn.score(X_test, y_test):.2f}")

# Neighborhood Components Analysis (NCA)
nca = NCA(max_iter=1000, random_state=42)
nca_pipe = make_pipeline(nca, KNeighborsClassifier())
nca_pipe.fit(X_train, y_train)
print(f"NCA+KNN Accuracy: {nca_pipe.score(X_test, y_test):.2f}")

# Large Margin Nearest Neighbor (LMNN)
lmnn = LMNN(k=5, learn_rate=1e-5, max_iter=1000)
lmnn_pipe = make_pipeline(lmnn, KNeighborsClassifier())
lmnn_pipe.fit(X_train, y_train)
print(f"LMNN+KNN Accuracy: {lmnn_pipe.score(X_test, y_test):.2f}")

# Visualization of transformed space
def plot_embedding(transformer, title):
    X_embedded = transformer.transform(X_test)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, alpha=0.7)
    plt.title(title)
    plt.colorbar()

plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_embedding(PCA(n_components=2), "PCA")
plt.subplot(132)
plot_embedding(nca, "NCA")
plt.subplot(133)
plot_embedding(lmnn, "LMNN")
plt.tight_layout()
plt.show()
```

### Key Advantages:
1. **Learned Metrics**: Adapts distance function to the data
2. **Class Awareness**: Optimizes for classification boundaries
3. **Dimension Reduction**: Implicitly focuses on discriminative features

### Performance Comparison:
```python
from sklearn.model_selection import cross_val_score

methods = {
    "Raw KNN": KNeighborsClassifier(),
    "PCA+KNN": make_pipeline(PCA(n_components=10), KNeighborsClassifier()),
    "NCA+KNN": make_pipeline(NCA(n_components=10), KNeighborsClassifier()),
    "LMNN+KNN": make_pipeline(LMNN(n_components=10), KNeighborsClassifier())
}

for name, model in methods.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2f} ± {scores.std():.2f}")
```

