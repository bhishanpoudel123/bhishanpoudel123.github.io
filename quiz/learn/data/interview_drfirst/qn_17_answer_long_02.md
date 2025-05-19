
### ✅ **Assumptions Required for a Two-Sample t-test** (especially independent t-test):

1. **Independence**

   * Observations in each group should be independent of each other.
   * In A/B testing: Ensure that users in group A are not also in group B.

2. **Normality**

   * The **data in each group** should be approximately normally distributed.
   * This is **most important when sample size is small**. For large sample sizes (thanks to the **Central Limit Theorem**), this assumption can be relaxed.

3. **Homogeneity of Variance** (also called **equal variances**)

   * The variances in the two groups should be similar.
   * You can test this using **Levene’s Test** or **F-test**.
   * If this assumption is violated, use **Welch's t-test** instead of Student’s t-test.

4. **Continuous Data**

   * The dependent variable should be **interval or ratio scale** (e.g., revenue, time on site).
   * Not suitable for categorical outcomes — use proportions/Z-tests for that.

5. **No Significant Outliers**

   * Outliers can skew the mean and inflate variances, which affects the test.
   * Check boxplots or use methods like the IQR rule to identify outliers.



### ❌ **Assumptions Not Required from Linear Regression**

* **Linearity** (linear relationship between X and Y) → **Not needed** for t-test.
* **No multicollinearity** → Not relevant here.
* **Homoscedasticity of residuals** → In t-test, we only care about **equal variances** across groups, not residuals from a regression line.
* **Independence of residuals** → Again, not applicable here directly.



### ✅ Summary for A/B Testing:

| Assumption              | Needed for t-test? | Notes                                              |
| ----------------------- | ------------------ | -------------------------------------------------- |
| Data is continuous      | ✅ Yes              | Outcome should be numeric, interval or ratio scale |
| Normal distribution     | ✅ Yes              | Mainly matters if n < 30 per group                 |
| Homogeneity of variance | ✅ Yes              | Use Welch’s t-test if variances are unequal        |
| Absence of outliers     | ✅ Yes              | Outliers can distort results                       |
| Linearity               | ❌ No               | Only required for linear regression                |
| No multicollinearity    | ❌ No               | Not relevant in t-tests                            |

