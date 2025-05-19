# qn_11.md

## Question 11: Interpretable Binary Classification with Uncertainty Quantification

**Question:**  
Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?

**Options:**
1. Random Forest with prediction intervals based on quantiles of tree predictions
2. Gradient Boosting with NGBoost for natural gradient boosting
3. Bayesian Logistic Regression with MCMC sampling for posterior distribution
4. Bootstrapped ensemble of decision trees with variance estimation

**Correct Answer:**  
Bayesian Logistic Regression with MCMC sampling for posterior distribution

### Detailed Explanation

Bayesian Logistic Regression provides both interpretability and principled uncertainty quantification through:
- Clear interpretation of coefficients (like standard logistic regression)
- Full posterior distributions of parameters (capturing epistemic uncertainty)
- Probabilistic predictions (capturing aleatoric uncertainty)

#### Key Advantages:
1. **Interpretability:** Coefficients represent log-odds ratios
2. **Uncertainty Quantification:** Credible intervals for parameters and predictions
3. **Flexibility:** Can incorporate prior knowledge

### Python Implementation

```python
import numpy as np
import pymc3 as pm
import arviz as az
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with pm.Model() as bayesian_logreg:
    # Priors for coefficients
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coefficients = pm.Normal('coefficients', mu=0, sigma=10, shape=X_train.shape[1])
    
    # Logistic function
    logit_p = intercept + pm.math.dot(X_train, coefficients)
    p = pm.Deterministic('p', 1 / (1 + pm.math.exp(-logit_p)))
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
    
    # Sample from posterior
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)

# Analyze results
print(az.summary(trace, var_names=['intercept', 'coefficients']))

# Make probabilistic predictions
with bayesian_logreg:
    pm.set_data({'coefficients': X_test})  # For new data
    post_pred = pm.sample_posterior_predictive(trace, var_names=['p'])

# Get prediction intervals
pred_mean = post_pred['p'].mean(axis=0)
pred_std = post_pred['p'].std(axis=0)
pred_95ci = np.percentile(post_pred['p'], [2.5, 97.5], axis=0)

print(f"Prediction mean: {pred_mean[:5]}")
print(f"Prediction 95% CI: {pred_95ci[:, :5]}")
```

#### Key Points:
- Uses PyMC3 for Bayesian modeling
- Provides full posterior distributions for all parameters
- Generates probabilistic predictions with uncertainty intervals
- Maintains interpretability of logistic regression coefficients

#### When to Use:
- When you need both interpretability and uncertainty quantification
- For problems where understanding feature importance is crucial
- When you want to incorporate prior knowledge into your model

#### Alternatives Considered:
- Random Forest: Provides uncertainty via tree variance but less interpretable
- NGBoost: Good for uncertainty but less interpretable than logistic regression
- Bootstrapped Trees: Captures variance but lacks Bayesian interpretation
