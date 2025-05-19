# Study Guide: Handling Concept Drift in Production ML Systems

## Question 21
**What's the most robust approach to handle concept drift in a production machine learning system?**

### Correct Answer
**Implement drift detection algorithms with adaptive learning techniques**

#### Explanation
This approach combines statistical drift detection (e.g., ADWIN, DDM, or KSWIN) with adaptive learning methods that can continuously update models or model weights as new patterns emerge. The key advantages are:

1. **Proactive detection**: Identifies drift before it significantly impacts performance
2. **Continuous adaptation**: Models adjust incrementally without full retraining
3. **Resource efficiency**: Only triggers major updates when necessary

```python
# Example using River library for adaptive learning with drift detection
from river import drift, linear_model, metrics, preprocessing
from river import stream

# Initialize drift detector and adaptive model
drift_detector = drift.ADWIN()
model = linear_model.LogisticRegression()
scaler = preprocessing.StandardScaler()
metric = metrics.Accuracy()

for x, y in stream.iter_csv('data_stream.csv'):
    # Scale features
    x_scaled = scaler.learn_one(x).transform_one(x)
    
    # Check for drift before prediction
    y_pred = model.predict_one(x_scaled)
    if drift_detector.update(y == y_pred):  # Drift detected
        print(f"Drift detected at step {drift_detector.n}")
        model = linear_model.LogisticRegression()  # Reset model
    
    # Update model and metric
    model.learn_one(x_scaled, y)
    metric.update(y, y_pred)
```

### Alternative Options Analysis

#### Option 1: Implement automatic model retraining when performance degrades below a threshold
**Pros:**
- Simple to implement
- Guarantees fresh model when triggered

**Cons:**
- Reactive rather than proactive (damage may already be done)
- Requires setting arbitrary thresholds
- Doesn't distinguish between temporary anomalies and true concept drift

```python
# Basic implementation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

threshold = 0.85  # Accuracy threshold
window_size = 1000
X_buffer, y_buffer = [], []
model = LogisticRegression()

for x_new, y_new in data_stream:
    X_buffer.append(x_new)
    y_buffer.append(y_new)
    
    if len(X_buffer) >= window_size:
        y_pred = model.predict(X_buffer[-window_size:])
        current_acc = accuracy_score(y_buffer[-window_size:], y_pred)
        
        if current_acc < threshold:
            model.fit(X_buffer, y_buffer)  # Full retraining
            X_buffer, y_buffer = [], []  # Reset buffer
```

#### Option 2: Use an ensemble of models with different time windows
**Pros:**
- Captures different temporal patterns
- Naturally handles gradual drift

**Cons:**
- Resource intensive
- Doesn't explicitly detect drift
- May delay response to abrupt changes

```python
from sklearn.ensemble import VotingClassifier
from river import compose, linear_model, naive_bayes

# Ensemble with different learning rates
fast_model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)
slow_model = compose.Pipeline(
    preprocessing.StandardScaler(),
    naive_bayes.GaussianNB()
)

ensemble = VotingClassifier([('fast', fast_model), ('slow', slow_model)])

for x, y in stream.iter_csv('data_stream.csv'):
    y_pred = ensemble.predict_one(x)
    ensemble.learn_one(x, y)
```

#### Option 3: Deploy a champion-challenger framework with continuous evaluation
**Pros:**
- Allows safe testing of new models
- Provides clear rollback mechanism

**Cons:**
- Requires significant infrastructure
- Slow response to sudden drift
- Doesn't automatically adapt to changes

```python
# Champion-challenger framework sketch
class ChampionChallenger:
    def __init__(self, champion_model, challenger_models):
        self.champion = champion_model
        self.challengers = challenger_models
        self.metrics = {name: metrics.Accuracy() for name in challenger_models}
    
    def update(self, x, y):
        # Update champion
        y_pred = self.champion.predict_one(x)
        self.champion.learn_one(x, y)
        
        # Evaluate challengers
        for name, model in self.challengers.items():
            y_pred_challenger = model.predict_one(x)
            self.metrics[name].update(y, y_pred_challenger)
            model.learn_one(x, y)
        
        # Check for promotion (e.g., after 10k samples)
        if self.total_samples % 10000 == 0:
            self.evaluate_promotion()
```

### Why the Correct Answer is Best
The recommended approach of combining drift detection with adaptive learning provides:
1. **Statistical rigor**: Formal detection of distributional changes
2. **Adaptability**: Models evolve with the data stream
3. **Efficiency**: Minimal computational overhead
4. **Generality**: Works across different types of drift (sudden, gradual, recurring)

### Key Concepts
- **Concept Drift**: Change in the relationship between input data and target variable over time
- **ADWIN (Adaptive Windowing)**: Adjusts window size automatically to detect changes
- **DDM (Drift Detection Method)**: Monitors error rate for significant changes
- **KSWIN**: Kolmogorov-Smirnov test for windowed data comparison
