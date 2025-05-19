# Interview Question: Aside from sample size, what other conditions must be satisfied to properly use a t-test?

**Answer:** When selecting a t-test, I rigorously evaluate these five key assumptions beyond just sample size:

### 1. **Scale of Measurement**
```python
# Check if data is continuous
is_continuous = np.issubdtype(df['engagement_time'].dtype, np.number)
print(f"Data is continuous: {is_continuous}")
```
*"The t-test requires interval or ratio data. For ordinal data like Likert scales, I'd use Mann-Whitney U instead."*

### 2. **Random Sampling & Independence**
```python
# Check for between-subjects design
assert df.groupby('user_id').size().max() == 1, "Violated independence - same users in both groups"
```
*"Each observation must be independent - violated in pre/post designs (use paired t-test) or repeated measures (use ANOVA)."*

### 3. **Normality Distribution**
```python
from scipy.stats import shapiro

_, p_a = shapiro(group_a)
_, p_b = shapiro(group_b)
print(f"Group A normality p-value: {p_a:.3f}\nGroup B normality p-value: {p_b:.3f}")
```
*"With n < 50 per group, I use Shapiro-Wilk. For larger samples (n > 50), the Central Limit Theorem makes the t-test robust to mild non-normality."*

### 4. **Homogeneity of Variance**
```python
# Brown-Forsythe test is more robust than Levene's for non-normal data
stat, p_var = stats.levene(group_a, group_b, center='median')
print(f"Equal variances p-value: {p_var:.3f}")
```
*"If variances are unequal (p < 0.05), I switch to Welch's t-test (`scipy.stats.ttest_ind(equal_var=False`)."*

### 5. **Absence of Outliers**
```python
# Tukey's fences method
q1, q3 = np.percentile(group_a, [25, 75])
iqr = q3 - q1
upper_bound = q3 + 3*iqr
outliers = sum(group_a > upper_bound)
print(f"Number of extreme outliers: {outliers}")
```
*"For >3 outliers per group, I either transform data or use non-parametric tests. Winsorization is another option for preserving sample size."*

**Healthcare-Specific Considerations:**
- For skewed clinical data (e.g., hospital stay duration), I log-transform first
- When comparing proportions (e.g., conversion rates), I use Fisher's exact test
- For survival data, Kaplan-Meier with log-rank test is more appropriate

