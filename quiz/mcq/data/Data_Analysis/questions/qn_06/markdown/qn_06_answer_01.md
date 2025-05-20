# Data Science Study Guide - Question 06

## **Question:**
When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?

## **Answer Choices:**
1. **`df.set_index('timestamp').asfreq('1H').interpolate(method='time')`**  
2. **`df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`**  
3. **`df.set_index('timestamp').resample('1H').mean().interpolate(method='time')`**  
4. **`df.groupby(pd.Grouper(key='timestamp', freq='1H')).apply(lambda x: x.mean() if not x.empty else pd.Series(np.nan, index=df.columns))`**  

---

## **Correct Answer:** `df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`

### **Explanation:**
Time series data often has **irregular timestamps**, requiring **resampling** to enforce consistency before analysis.  
- **`.resample('1H').asfreq()`** correctly converts irregular timestamps to a regular hourly frequency.  
- **`.interpolate(method='time')`** ensures missing values are filled **based on time-based interpolation**, making it **statistically sound** while respecting time intervals.

Python implementation:
```python
import pandas as pd
import numpy as np

# Example dataset with irregular timestamps
data = {'timestamp': pd.to_datetime(['2023-04-01 01:05', '2023-04-01 02:15', '2023-04-01 03:45', '2023-04-01 06:30']),
        'value': [10, 15, 20, 30]}

df = pd.DataFrame(data).set_index('timestamp')

# Resample to 1-hour intervals with interpolation
df_resampled = df.resample('1H').asfreq().interpolate(method='time')

print(df_resampled)
```

---

## **Why Other Choices Are Incorrect?**
### **1. `df.set_index('timestamp').asfreq('1H').interpolate(method='time')`**
- **Missing `.resample('1H')`**, causing an **incorrect frequency conversion**.

### **2. `df.set_index('timestamp').resample('1H').mean().interpolate(method='time')`**
- Using `.mean()` aggregates values instead of preserving individual observations.

### **3. `df.groupby(pd.Grouper(key='timestamp', freq='1H'))...`**
- **Too complex and inefficient**, especially for large datasets.
