### **Question 11**

**Q:** When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?

**Options:**

* A. `Isolation Forest` with sliding windows
* B. `One-class SVM` on feature vectors
* C. `LSTM Autoencoder` with reconstruction error thresholding
* D. `ARIMA` with Mahalanobis distance on residuals

**✅ Correct Answer:** `LSTM Autoencoder` with reconstruction error thresholding

---

### 🧠 Explanation:

#### ✅ C. `LSTM Autoencoder` with reconstruction error thresholding

This method is ideal for detecting **contextual anomalies** in **multivariate time series** data. LSTM (Long Short-Term Memory) networks can capture complex **temporal dependencies**, and autoencoders help in reconstructing the input. Anomalies are flagged when the **reconstruction error** exceeds a predefined threshold.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

# Example data: shape (samples, timesteps, features)
X_train = np.random.random((1000, 10, 5))

# Build LSTM Autoencoder
inputs = Input(shape=(10, 5))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(10)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(5))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=10, batch_size=32)

# Detecting anomalies
X_pred = model.predict(X_train)
mse = np.mean(np.power(X_train - X_pred, 2), axis=(1, 2))
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
```

---

### ❌ Other Options:

#### A. `Isolation Forest` with sliding windows

While Isolation Forest is efficient for **global anomaly detection**, it is not tailored for **contextual anomalies** in time series. Applying a sliding window over time series converts it to a tabular structure, but it loses temporal dependency.

```python
from sklearn.ensemble import IsolationForest

# Sliding window reshape
window_size = 10
X_windows = [X_train[i:i+window_size].flatten() for i in range(len(X_train)-window_size)]
model = IsolationForest()
model.fit(X_windows)
```

🔴 **Not ideal for contextual anomalies.**

---

#### B. `One-class SVM` on feature vectors

One-class SVM is another global anomaly detector and works well for static feature vectors. However, like Isolation Forest, it does not inherently capture **temporal dependencies** or context from time series.

```python
from sklearn.svm import OneClassSVM

X_vectorized = X_train.reshape((X_train.shape[0], -1))
model = OneClassSVM(gamma='auto').fit(X_vectorized)
```

🔴 **Fails to utilize temporal context.**

---

#### D. `ARIMA` with Mahalanobis distance on residuals

ARIMA is a **univariate** time series model. While you can extend it and use **Mahalanobis distance** on the residuals to detect anomalies, it is not well-suited for **high-dimensional or multivariate** contextual anomaly detection.

```python
from statsmodels.tsa.arima.model import ARIMA
from scipy.spatial.distance import mahalanobis

# Univariate example for one feature
model = ARIMA(X_train[:, :, 0].flatten(), order=(5,1,0)).fit()
residuals = model.resid
# Mahalanobis requires multivariate residuals
```

🔴 **Limited to univariate; weak for multivariate contextual anomalies.**

---

### 📚 Summary

| Method              | Contextual? | Temporal Dependencies? | Multivariate?     | Verdict    |
| ------------------- | ----------- | ---------------------- | ----------------- | ---------- |
| Isolation Forest    | ❌ No        | ❌ No                   | ✅ Yes (w/ tricks) | ❌          |
| One-Class SVM       | ❌ No        | ❌ No                   | ✅ Yes             | ❌          |
| LSTM Autoencoder    | ✅ Yes       | ✅ Yes                  | ✅ Yes             | ✅ **Best** |
| ARIMA + Mahalanobis | ❌ No        | ✅ Yes (univariate)     | ❌ No              | ❌          |

