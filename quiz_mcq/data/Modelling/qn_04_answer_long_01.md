# Question 4: Incremental Learning for Large Datasets

**Category:** Modelling  
**Question:** What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?

## Correct Answer:
**Use SGDClassifier with partial_fit on data chunks**

## Python Implementation:
```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

# Initialize model and feature extractor
model = SGDClassifier(loss='log_loss', warm_start=True)
vectorizer = HashingVectorizer(n_features=2**18)

# Process data in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    X = vectorizer.transform(chunk['text'])
    y = chunk['label']
    
    # Partial fit with classes for multilabel
    if not hasattr(model, 'classes_'):
        model.partial_fit(X, y, classes=np.unique(y))
    else:
        model.partial_fit(X, y)

# Final model can now predict
probs = model.predict_proba(X_new)