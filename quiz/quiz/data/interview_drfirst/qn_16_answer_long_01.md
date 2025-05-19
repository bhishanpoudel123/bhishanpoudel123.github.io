# T-test in A/B Test

## Interview Question: In your A/B testing project, you mentioned using a t-test. Why did you choose a t-test over other statistical tests?

**Answer:** Excellent question. For our PowerBI dashboard A/B test at AmerisourceBergen, I selected the two-sample t-test after carefully evaluating several alternatives. Here's my reasoning:

1. **Data Characteristics Fit**  
   *"We were comparing mean engagement times (continuous, normally distributed metrics) between two independent user groups. The t-test is specifically designed for this scenario where we want to compare means of normally distributed data with unknown population variances."*

2. **Sample Size Considerations**  
   *"With our sample sizes (n=450 per group), the t-test was more appropriate than a z-test (which requires known population variances) and more robust than non-parametric alternatives like Mann-Whitney U. The central limit theorem ensured our sample means were normally distributed despite moderate sample sizes."*

3. **Comparison to Other Options**  
   *"I considered but ruled out:  
   - Welch's t-test (unequal variances weren't significant per Levene's test)  
   - ANOVA (only two groups)  
   - Non-parametric tests (would lose power since our data met parametric assumptions)  
   - Bayesian methods (stakeholders preferred frequentist p-values for this application)"*

4. **Practical Implementation**  
   *"Using scipy.stats.ttest_ind, I verified:  
   - Equal variance assumption (p=0.32 in Levene's test)  
   - Normality via Q-Q plots  
   - Effect size (Cohen's d=0.41) to ensure practical significance beyond p<0.05"*

**Key Insight:** The t-test gave us both statistical rigor (controlling Type I error) and clear business interpretation ("Dashboard B increased engagement by 13% with p=0.008"). For non-normal metrics like click-through rates, I would have used Fisher's exact test instead.

*Would you like me to elaborate on how we determined sample size via power analysis beforehand?*  

*(This demonstrates both statistical proficiency and practical decision-making - crucial for DrFirst's data-driven culture.)*
