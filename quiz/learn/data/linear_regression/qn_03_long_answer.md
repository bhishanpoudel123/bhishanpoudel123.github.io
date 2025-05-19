### Testing Normality of a Feature

To check if a variable (or model residuals) is normally distributed, you can use.

![](/assets/images/normality_test_plot.jpeg)


### Test: Use image using html syntax

<div style="text-align: center;">
    <img src="/assets/images/normality_test_plot.jpeg" 
         alt="Normality Test Plot"
         style="max-width: 100%; height: auto; border: 1px solid #ddd; margin: 1em 0;">
    <p style="font-size: 0.9em; color: #666;">Click image to enlarge</p>
</div>

#### 1. **Statistical Tests**:

- **Shapiro-Wilk Test**  
```python
from scipy.stats import shapiro
stat, p = shapiro(data)
print(f'Statistic={stat:.3f}, p={p:.3f}')
```

- **Anderson-Darling Test**  
```python
from scipy.stats import anderson
result = anderson(data)
print(f'Statistic: {result.statistic}')
for i in range(len(result.critical_values)):
    print(f'At {result.significance_level[i]}%: {result.critical_values[i]}')
```

#### 2. **Visual Techniques**:

- **Histogram**
- **Boxplot**
- **QQ-Plot**  
```python
import matplotlib.pyplot as plt
import scipy.stats as stats

stats.probplot(data, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()
```

Refer to the image: `assets/images/normality_test_plot.png`.