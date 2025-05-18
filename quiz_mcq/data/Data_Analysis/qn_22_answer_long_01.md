# Study Guide: Interpretable Anomaly Detection in High-Dimensional Data

## Question 22
**Which method is most appropriate for interpretable anomaly detection in high-dimensional data?**

### Correct Answer
**`Isolation Forest` with LIME explanations**

#### Explanation
This combination provides:
1. **Efficient detection**: Isolation Forest handles high dimensions well with random partitioning
2. **Interpretability**: LIME explains individual anomalies by showing feature contributions
3. **Scalability**: Both methods work well with large datasets

```python
from sklearn.ensemble import IsolationForest
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Generate sample high-dimensional data
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
X[-10:] += 5  # Add 10 anomalies

# Train Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)
anomaly_scores = clf.decision_function(X)
anomalies = X[anomaly_scores < np.quantile(anomaly_scores, 0.01)]

# Set up LIME explainer
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=[f'feature_{i}' for i in range(X.shape[1])],
    mode='classification'
)

# Explain an anomaly
exp = explainer.explain_instance(
    anomalies[0],
    clf.decision_function,
    num_features=5
)
exp.show_in_notebook()
```

### Alternative Options Analysis

#### Option 1: Autoencoders with attention mechanisms
**Pros:**
- Can capture complex patterns
- Attention provides some interpretability

**Cons:**
- Computationally expensive
- Requires large amounts of data
- Attention may not clearly explain anomalies

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Layer
import tensorflow as tf

class AttentionLayer(Layer):
    def call(self, inputs):
        attention = tf.nn.softmax(inputs)
        return Multiply()([inputs, attention])

# Build autoencoder with attention
inputs = Input(shape=(20,))
encoded = Dense(10, activation='relu')(inputs)
attention = AttentionLayer()(encoded)
decoded = Dense(20, activation='linear')(attention)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=32)

# Get reconstruction errors as anomaly scores
reconstructions = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=1)
```

#### Option 2: SHAP values on `One-Class SVM` predictions
**Pros:**
- SHAP provides global interpretability
- One-Class SVM is effective for novelty detection

**Cons:**
- Doesn't scale well to high dimensions
- SHAP computations are expensive
- May miss local feature interactions

```python
from sklearn.svm import OneClassSVM
import shap

# Train One-Class SVM
ocsvm = OneClassSVM(gamma='auto', nu=0.01)
ocsvm.fit(X)

# Compute SHAP values (warning: slow in high dimensions)
explainer = shap.KernelExplainer(ocsvm.decision_function, X[:100])
shap_values = explainer.shap_values(X[0:1])

shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    feature_names=[f'feature_{i}' for i in range(20)]
)
```

#### Option 3: Supervised anomaly detection with feature importance
**Pros:**
- Clear feature importance from models
- Can be very accurate if labels are reliable

**Cons:**
- Requires labeled anomaly data
- May not generalize to new anomaly types
- Feature importance can be misleading for interactions

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Create synthetic labels (1=normal, 0=anomaly)
y = np.ones(len(X))
y[-10:] = 0

# Train classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Get feature importances
result = permutation_importance(clf, X, y, n_repeats=10)
importances = result.importances_mean
```

### Why the Correct Answer is Best
1. **Computational Efficiency**: Isolation Forest has linear time complexity
2. **Interpretability**: LIME provides intuitive local explanations
3. **No Label Requirement**: Works in unsupervised setting
4. **Feature Relevance**: Clearly shows which features contributed to each anomaly

### Key Concepts
- **Isolation Forest**: Anomaly detection by randomly partitioning feature space
- **LIME (Local Interpretable Model-agnostic Explanations)**: Creates local linear approximations
- **Anomaly Interpretation**: Understanding why a point is flagged as anomalous
- **High-Dimensional Challenges**: Curse of dimensionality, feature interactions
