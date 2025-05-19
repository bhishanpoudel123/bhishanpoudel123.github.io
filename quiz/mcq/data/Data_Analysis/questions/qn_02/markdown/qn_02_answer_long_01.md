# Efficient Grouped Sampling with Replacement in Pandas

## Question 2
**What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?**

- A. `df.groupby('group').apply(lambda x: x.sample(n=len(x), replace=True))`
- B. `pd.concat([df[df['group']==g].sample(n=sum(df['group']==g), replace=True) for g in df['group'].unique()])`
- C. `df.set_index('group').sample(frac=1, replace=True).reset_index()`
- D. `df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`

## Detailed Explanation

This question tests your understanding of efficient grouped sampling operations in pandas, which is a common requirement for bootstrapping, data augmentation, and cross-validation techniques.

Let's examine each option with example code and explanations:

### A. `df.groupby('group').apply(lambda x: x.sample(n=len(x), replace=True))` - INCORRECT

```python
import pandas as pd
import numpy as np
import time

# Create a sample dataframe
np.random.seed(42)
df = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], [10000, 5000, 15000]),
    'value': np.random.randn(30000)
})

# Measure time for option A
start_time = time.time()
result_a = df.groupby('group').apply(lambda x: x.sample(n=len(x), replace=True))
end_time = time.time()
print(f"Option A time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result_a.shape}")
```

**Why it's incorrect:**
- This approach works correctly for the task but is inefficient
- It uses pandas' `sample()` function, which has additional overhead for each group
- The `apply()` operation creates intermediate DataFrames for each group
- It requires copying data multiple times during the operation
- The operation has high computational complexity for large datasets with many groups

### B. `pd.concat([df[df['group']==g].sample(n=sum(df['group']==g), replace=True) for g in df['group'].unique()])` - INCORRECT

```python
# Measure time for option B
start_time = time.time()
result_b = pd.concat([df[df['group']==g].sample(n=sum(df['group']==g), replace=True) 
                      for g in df['group'].unique()])
end_time = time.time()
print(f"Option B time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result_b.shape}")
```

**Why it's incorrect:**
- This approach requires filtering the original DataFrame for each group (`df[df['group']==g]`)
- It recalculates the group size for each group (`sum(df['group']==g)`)
- It creates multiple intermediate DataFrames before concatenation
- List comprehension and concatenation add overhead
- The filtering operation is inefficient compared to using `groupby()`
- For large datasets with many groups, this approach is very slow

### C. `df.set_index('group').sample(frac=1, replace=True).reset_index()` - INCORRECT

```python
# Measure time for option C
start_time = time.time()
result_c = df.set_index('group').sample(frac=1, replace=True).reset_index()
end_time = time.time()
print(f"Option C time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result_c.shape}")
```

**Why it's incorrect:**
- This doesn't perform grouped sampling at all
- It samples from the entire dataset with replacement, ignoring group boundaries
- The resulting dataset won't maintain the original group sizes
- It only resamples the rows without respect to their groups
- The `set_index()` operation doesn't change the sampling behavior in this context

### D. `df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])` - CORRECT

```python
# Measure time for option D
start_time = time.time()
result_d = df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])
end_time = time.time()
print(f"Option D time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result_d.shape}")

# Verify group sizes are maintained
print("Original group sizes:", df.groupby('group').size())
print("Result group sizes:", result_d.groupby('group').size())
```

**Why it's correct:**
- This approach uses NumPy's highly optimized `random.choice()` function directly on indices
- It avoids the overhead of pandas' `sample()` function
- It efficiently uses integer indexing with `iloc[]`
- It properly maintains original group sizes
- The operation is performed on indices rather than copying data multiple times
- NumPy's vectorized operations are faster than pandas' sampling for this specific task

## Performance Comparison

We can run a benchmark to compare the performance of these approaches:

```python
import pandas as pd
import numpy as np
import time

# Create a larger dataframe for more meaningful benchmark
np.random.seed(42)
df = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C', 'D', 'E'], [50000, 30000, 20000, 40000, 60000]),
    'value1': np.random.randn(200000),
    'value2': np.random.randn(200000)
})

# Function to measure execution time
def time_execution(func, name):
    start_time = time.time()
    result = func()
    end_time = time.time()
    print(f"{name} time: {end_time - start_time:.4f} seconds")
    return result

# Option A
result_a = time_execution(
    lambda: df.groupby('group').apply(lambda x: x.sample(n=len(x), replace=True)),
    "Option A"
)

# Option B
result_b = time_execution(
    lambda: pd.concat([df[df['group']==g].sample(n=sum(df['group']==g), replace=True) 
                      for g in df['group'].unique()]),
    "Option B"
)

# Option C
result_c = time_execution(
    lambda: df.set_index('group').sample(frac=1, replace=True).reset_index(),
    "Option C"
)

# Option D
result_d = time_execution(
    lambda: df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)]),
    "Option D"
)

# Verify group sizes
original_sizes = df.groupby('group').size()
result_d_sizes = result_d.groupby('group').size()

print("\nOriginal group sizes:")
print(original_sizes)
print("\nOption D result group sizes:")
print(result_d_sizes)
print("\nMatching group sizes:", (original_sizes == result_d_sizes).all())
```

Expected results would show that Option D is significantly faster than Options A and B, while Option C doesn't perform the correct task at all.

## Practical Applications

This technique is commonly used for:

1. **Bootstrapping**: Generating bootstrap samples for statistical inference
2. **Cross-validation**: Creating multiple training sets with the same distribution
3. **Data augmentation**: Generating synthetic samples while maintaining class balance
4. **Ensemble methods**: Creating diverse training sets for ensemble models

## Implementation Example

Here's a complete example showing how to use this technique for bootstrap confidence intervals:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], [100, 150, 200]),
    'value': np.concatenate([
        np.random.normal(10, 2, 100),  # Group A
        np.random.normal(12, 3, 150),  # Group B
        np.random.normal(8, 1, 200)    # Group C
    ])
})

# Function to generate bootstrap samples and calculate means
def bootstrap_means(data, n_bootstraps=1000):
    bootstrap_samples = []
    
    for _ in range(n_bootstraps):
        # Use the efficient grouped sampling method
        bootstrap_sample = data.groupby('group').apply(
            lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)]
        )
        group_means = bootstrap_sample.groupby('group')['value'].mean()
        bootstrap_samples.append(group_means)
    
    return pd.DataFrame(bootstrap_samples)

# Generate bootstrap samples
bootstrap_results = bootstrap_means(df)

# Calculate 95% confidence intervals
confidence_intervals = bootstrap_results.apply(
    lambda x: pd.Series([x.quantile(0.025), x.quantile(0.975)], index=['lower', 'upper']),
    axis=0
).T

# Plot original means with confidence intervals
plt.figure(figsize=(10, 6))
original_means = df.groupby('group')['value'].mean()

for i, group in enumerate(original_means.index):
    plt.scatter(i, original_means[group], s=100, color='blue', label='Mean' if i==0 else '')
    plt.errorbar(i, original_means[group], 
                 yerr=[[original_means[group]-confidence_intervals.loc[group, 'lower']], 
                       [confidence_intervals.loc[group, 'upper']-original_means[group]]],
                 capsize=10, color='red', label='95% CI' if i==0 else '')

plt.xticks(range(len(original_means)), original_means.index)
plt.title('Group Means with Bootstrap 95% Confidence Intervals')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Summary

Option D (`df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`) is the most efficient approach because:

1. It uses NumPy's highly optimized sampling directly on indices
2. It avoids the overhead of pandas' sampling functions
3. It maintains original group sizes correctly
4. It minimizes data copying and intermediate object creation
5. It leverages vectorized operations for better performance

This approach is particularly valuable for large datasets where performance matters, especially in resampling-intensive techniques like bootstrapping and cross-validation.