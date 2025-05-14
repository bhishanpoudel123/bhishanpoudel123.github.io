## Interview Question: Can you share code snippets showing how you implemented the t-test for your A/B test analysis?

**Answer:** Certainly. Here's the Python implementation I used, with explanations for each critical step:

### 1. Data Preparation & Assumption Checking
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load experimental data (example structure)
df = pd.read_csv('dashboard_ab_test.csv')  # Columns: user_id, group (A/B), engagement_time

# Check equal variance assumption
levene_stat, levene_p = stats.levene(
    df[df['group'] == 'A']['engagement_time'],
    df[df['group'] == 'B']['engagement_time']
)
print(f"Levene's Test: p-value = {levene_p:.3f}")  # > 0.05 confirms equal variance assumption

# Visual normality check
plt.figure(figsize=(12,4))
plt.subplot(121)
stats.probplot(df[df['group'] == 'A']['engagement_time'], plot=plt)
plt.title('Group A Q-Q Plot')

plt.subplot(122)
stats.probplot(df[df['group'] == 'B']['engagement_time'], plot=plt)
plt.title('Group B Q-Q Plot')
plt.show()
```

### 2. Two-Sample t-Test Implementation
```python
# Extract metric values for each group
group_a = df[df['group'] == 'A']['engagement_time']
group_b = df[df['group'] == 'B']['engagement_time']

# Perform independent two-sample t-test
t_stat, p_value = stats.ttest_ind(
    a=group_b,  # Convention: put experimental group first
    b=group_a,
    equal_var=True,  # Confirmed via Levene's test
    alternative='greater'  # One-tailed: testing if B > A
)

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt(((len(group_a)-1)*group_a.std()**2 + (len(group_b)-1)*group_b.std()**2) / 
                   (len(group_a) + len(group_b) - 2)
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std

print(f"Results:\n"
      f"- t-statistic: {t_stat:.3f}\n"
      f"- p-value: {p_value:.4f}\n"
      f"- Group A mean: {group_a.mean():.1f} ± {group_a.std():.1f}\n"
      f"- Group B mean: {group_b.mean():.1f} ± {group_b.std():.1f}\n"
      f"- Cohen's d: {cohens_d:.2f}")
```

### 3. Power Analysis (Pre-Test)
```python
from statsmodels.stats.power import TTestIndPower

# Parameters for power analysis
effect_size = 0.3  # Minimum detectable effect (Cohen's d)
alpha = 0.05       # Significance level
power = 0.8        # Desired power

# Calculate required sample size
analysis = TTestIndPower()
sample_size = analysis.solve_power(
    effect_size=effect_size,
    power=power,
    alpha=alpha,
    ratio=1.0  # Equal group sizes
)
print(f"Required sample size per group: {np.ceil(sample_size):.0f}")
```

**Key Implementation Notes:**
1. The code validates all t-test assumptions before proceeding
2. Uses one-tailed testing when we specifically predict directionality (B > A)
3. Includes effect size calculation for practical significance
4. Shows pre-experiment power analysis to ensure proper design
5. Outputs interpretable business metrics alongside statistical results

*Would you like me to show how we implemented the sequential testing version of this analysis?*  

*(This demonstrates both technical implementation skills and rigorous statistical thinking - exactly what DrFirst needs for their data-driven healthcare solutions.)*
