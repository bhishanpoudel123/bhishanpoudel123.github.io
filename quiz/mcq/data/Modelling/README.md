# Modelling Quiz

## Table of Contents
- [Qn 01: When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?](#1)
- [Qn 02: What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?](#2)
- [Qn 03: Which approach correctly implements proper nested cross-validation for model selection and evaluation?](#3)
- [Qn 04: What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?](#4)
- [Qn 05: When dealing with competing risks in survival analysis, which implementation correctly handles the problem?](#5)
- [Qn 06: What's the most statistically sound approach to implement monotonic constraints in gradient boosting?](#6)
- [Qn 07: Which approach correctly implements a custom kernel for SVM in scikit-learn?](#7)
- [Qn 08: What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?](#8)
- [Qn 09: Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?](#9)
- [Qn 10: What's the most effective way to implement a time-based split for cross-validation in time series forecasting?](#10)
- [Qn 11: Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?](#11)
- [Qn 12: What's the most robust approach to handling class imbalance in a multi-class classification problem?](#12)
- [Qn 13: Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?](#13)
- [Qn 14: What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?](#14)
- [Qn 15: Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?](#15)
- [Qn 16: What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?](#16)
- [Qn 17: Which approach correctly implements quantile regression forests for prediction intervals?](#17)
- [Qn 18: What's the most rigorous approach to handle outliers in the target variable for regression problems?](#18)
- [Qn 19: Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?](#19)
- [Qn 20: What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?](#20)
- [Qn 21: Which approach correctly implements a counterfactual explanation method for a black-box classifier?](#21)
- [Qn 22: What's the most effective approach to implement online learning for a regression task with concept drift?](#22)
- [Qn 23: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?](#23)
- [Qn 24: What's the most statistically sound approach to implement feature selection for time series forecasting?](#24)
- [Qn 25: Which approach correctly addresses Simpson's paradox in a predictive modeling context?](#25)
- [Qn 26: What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?](#26)
- [Qn 27: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?](#27)
- [Qn 28: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?](#28)
- [Qn 29: What's the most appropriate technique for automated feature engineering in time series forecasting?](#29)
- [Qn 30: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?](#30)

---

### 1. Qn 01: When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Manually implement out-of-fold predictions for each base learner

**Explanation:** Manually generating out-of-fold predictions ensures the meta-learner only sees predictions made on data that base models weren't trained on, fully preventing leakage while utilizing all data. This approach is more flexible than StackingClassifier and can incorporate diverse base models while maintaining proper validation boundaries.

**Learning Resources:**
- [qn_01_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_01/markdown/qn_01_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Apply sklearn's CalibratedClassifierCV with isotonic regression

**Explanation:** Isotonic regression via CalibratedClassifierCV is non-parametric and can correct any monotonic distortion in probability estimates, making it more flexible than Platt scaling, particularly for gradient boosting models which often produce well-ranked but not well-calibrated probabilities.

**Learning Resources:**
- [qn_02_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_02/markdown/qn_02_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: Which approach correctly implements proper nested cross-validation for model selection and evaluation?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Nested loops of KFold.split(), with inner loop for hyperparameter tuning

**Explanation:** Proper nested cross-validation requires an outer loop for performance estimation and an inner loop for hyperparameter tuning, completely separating the data used for model selection from the data used for model evaluation, avoiding optimistic bias.

**Learning Resources:**
- [qn_03_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_03/markdown/qn_03_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use SGDClassifier with partial_fit on data chunks

**Explanation:** SGDClassifier with partial_fit allows true incremental learning, processing data in chunks without storing the entire dataset in memory, updating model parameters with each batch and converging to the same solution as batch processing would with sufficient iterations.

**Learning Resources:**
- [qn_04_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_04/markdown/qn_04_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: When dealing with competing risks in survival analysis, which implementation correctly handles the problem?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Fine-Gray subdistribution hazard model from pysurvival

**Explanation:** The Fine-Gray model explicitly accounts for competing risks by modeling the subdistribution hazard, allowing for valid inference about the probability of an event in the presence of competing events, unlike standard Cox models or Kaplan-Meier which can produce biased estimates under competing risks.

**Learning Resources:**
- [qn_05_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_05/markdown/qn_05_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: What's the most statistically sound approach to implement monotonic constraints in gradient boosting?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Using XGBoost's monotone_constraints parameter

**Explanation:** XGBoost's native monotone_constraints parameter enforces monotonicity during tree building by constraining only monotonic splits, resulting in a fully monotonic model without sacrificing performance—unlike post-processing which can degrade model accuracy or pre-processing which doesn't guarantee model monotonicity.

**Learning Resources:**
- [qn_06_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_06/markdown/qn_06_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: Which approach correctly implements a custom kernel for SVM in scikit-learn?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Define a function that takes two arrays and returns a kernel matrix

**Explanation:** For custom kernels in scikit-learn SVMs, one must define a function K(X, Y) that calculates the kernel matrix between arrays X and Y, then pass this function as the 'kernel' parameter to SVC. This approach allows full flexibility in kernel design while maintaining compatibility with scikit-learn's implementation.

**Learning Resources:**
- [qn_07_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_07/markdown/qn_07_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Elastic Net regularization with randomized hyperparameter search

**Explanation:** Elastic Net combines L1 and L2 penalties, effectively handling correlated features by either selecting one from a correlated group (via L1) or assigning similar coefficients to correlated features (via L2), with the optimal balance determined through randomized hyperparameter search across different alpha and l1_ratio values.

**Learning Resources:**
- [qn_08_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_08/markdown/qn_08_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Custom encoding using pd.Categorical with ordered=True

**Explanation:** Using pandas Categorical with ordered=True preserves the ordinal relationship and allows for appropriate distance calculations between categories, which is essential for models that consider feature relationships (unlike OrdinalEncoder which assigns arbitrary numeric values without preserving distances).

**Learning Resources:**
- [qn_09_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_09/markdown/qn_09_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: What's the most effective way to implement a time-based split for cross-validation in time series forecasting?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Define a custom cross-validator with expanding window and purging

**Explanation:** A custom cross-validator with expanding windows (increasing training set) and purging (gap between train and test to prevent leakage) most accurately simulates real-world forecasting scenarios while handling temporal dependencies and avoiding lookahead bias.

**Learning Resources:**
- [qn_10_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_10/markdown/qn_10_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 11. Qn 11: Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Bayesian Logistic Regression with MCMC sampling for posterior distribution

**Explanation:** Bayesian Logistic Regression provides both interpretability (coefficients have clear meanings) and principled uncertainty quantification through the posterior distribution of parameters, capturing both aleatoric and epistemic uncertainty while maintaining model transparency.

**Learning Resources:**
- [qn_11_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_11/markdown/qn_11_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 12. Qn 12: What's the most robust approach to handling class imbalance in a multi-class classification problem?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use ensemble methods with resampling strategies specific to each classifier

**Explanation:** Ensemble methods with class-specific resampling strategies (e.g., EasyEnsemble or SMOTEBoost) combine the diversity of multiple classifiers with targeted handling of class imbalance, outperforming both global resampling and simple class weighting, especially for multi-class problems with varying degrees of imbalance.

**Learning Resources:**
- [qn_12_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_12/markdown/qn_12_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 13. Qn 13: Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement H-statistic from Friedman and Popescu

**Explanation:** The H-statistic specifically quantifies interaction strength between features by comparing the variation in predictions when features are varied together versus independently, providing a statistical measure of interactions that can't be captured by standard importance metrics or partial dependence alone.

**Learning Resources:**
- [qn_13_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_13/markdown/qn_13_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 14. Qn 14: What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use make_scorer with a function that combines multiple metrics

**Explanation:** make_scorer allows creating a custom scoring function that can combine predictive performance (e.g., AUC) with penalties for model complexity (e.g., number of features or model parameters), providing a single metric for optimization that balances performance and parsimony.

**Learning Resources:**
- [qn_14_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_14/markdown/qn_14_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 15. Qn 15: Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement weighted LASSO with weight inversely proportional to error variance

**Explanation:** Weighted LASSO that downweights observations with high error variance accounts for heteroscedasticity in the selection process, ensuring that features aren't selected or rejected due to non-constant error variance, resulting in more reliable feature selection.

**Learning Resources:**
- [qn_15_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_15/markdown/qn_15_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 16. Qn 16: What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use Explainable Boosting Machines (EBMs) from InterpretML

**Explanation:** EBMs combine the interpretability of GAMs with the predictive power of boosting, learning feature functions and pairwise interactions in an additive structure while remaining highly interpretable, offering better performance than standard GAMs while maintaining transparency.

**Learning Resources:**
- [qn_16_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_16/markdown/qn_16_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 17. Qn 17: Which approach correctly implements quantile regression forests for prediction intervals?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement a custom version of RandomForestRegressor that stores all leaf node samples

**Explanation:** Quantile regression forests require storing the empirical distribution of training samples in each leaf node (not just their mean), requiring a custom implementation that extends standard random forests to compute conditional quantiles from these stored distributions.

**Learning Resources:**
- [qn_17_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_17/markdown/qn_17_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 18. Qn 18: What's the most rigorous approach to handle outliers in the target variable for regression problems?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use Huber or Quantile regression with robust loss functions

**Explanation:** Robust regression methods like Huber or Quantile regression use loss functions that inherently reduce the influence of outliers during model training, addressing the issue without removing potentially valuable data points or distorting the target distribution through transformations.

**Learning Resources:**
- [qn_18_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_18/markdown/qn_18_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 19. Qn 19: Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement distance metric learning with NCA or LMNN

**Explanation:** Distance metric learning adaptively learns a transformation of the feature space that emphasizes discriminative dimensions, effectively addressing the curse of dimensionality by creating a more semantically meaningful distance metric, unlike fixed trees or general dimensionality reduction.

**Learning Resources:**
- [qn_19_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_19/markdown/qn_19_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 20. Qn 20: What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use early_stopping_rounds with a validation set in XGBoost/LightGBM

**Explanation:** Using early_stopping_rounds with a separate validation set stops training when performance on the validation set stops improving for a specified number of rounds, efficiently determining the optimal number of trees in a single training run without requiring multiple cross-validation runs.

**Learning Resources:**
- [qn_20_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_20/markdown/qn_20_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 21. Qn 21: Which approach correctly implements a counterfactual explanation method for a black-box classifier?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals

**Explanation:** DiCE specifically generates diverse counterfactual explanations that show how an instance's features would need to change to receive a different classification, addressing the 'what-if' question directly rather than just explaining the current prediction.

**Learning Resources:**
- [qn_21_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_21/markdown/qn_21_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 22. Qn 22: What's the most effective approach to implement online learning for a regression task with concept drift?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use incremental learning with drift detection algorithms to trigger model updates

**Explanation:** Combining incremental learning with explicit drift detection (e.g., ADWIN, DDM) allows the model to adapt continuously to new data while only performing major updates when the data distribution actually changes, balancing computational efficiency with adaptation to concept drift.

**Learning Resources:**
- [qn_22_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_22/markdown/qn_22_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 23. Qn 23: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement multi-fidelity optimization with Hyperband

**Explanation:** Hyperband uses a bandit-based approach to allocate resources efficiently, quickly discarding poor configurations and allocating more compute to promising ones, making it particularly effective when training time is limited and early performance is indicative of final performance.

**Learning Resources:**
- [qn_23_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_23/markdown/qn_23_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 24. Qn 24: What's the most statistically sound approach to implement feature selection for time series forecasting?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement feature importance from tree-based models with purged cross-validation

**Explanation:** Tree-based feature importance combined with purged cross-validation (which leaves gaps between train and test sets) correctly handles temporal dependence in the data, preventing information leakage while identifying features that have genuine predictive power for future time points.

**Learning Resources:**
- [qn_24_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_24/markdown/qn_24_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 25. Qn 25: Which approach correctly addresses Simpson's paradox in a predictive modeling context?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use causal graphical models to identify proper conditioning sets

**Explanation:** Causal graphical models (e.g., DAGs) allow identifying which variables should or should not be conditioned on to avoid Simpson's paradox, ensuring that the model captures the true causal relationship rather than spurious associations that reverse with conditioning.

**Learning Resources:**
- [qn_25_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_25/markdown/qn_25_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 26. Qn 26: What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Apply multi-objective Bayesian optimization to balance diversity and performance

**Explanation:** Multi-objective Bayesian optimization can simultaneously optimize for both individual model performance and ensemble diversity, finding an optimal set of hyperparameters for each model type while ensuring the ensemble as a whole performs well through complementary strengths.

**Learning Resources:**
- [qn_26_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_26/markdown/qn_26_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 27. Qn 27: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Individual Conditional Expectation (ICE) plots with centered PDP

**Explanation:** ICE plots show how predictions change for individual instances across the range of a feature, while centering them helps visualize heterogeneous effects that would be masked by averaging in standard partial dependence plots, making them ideal for detecting complex non-linear relationships.

**Learning Resources:**
- [qn_27_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_27/markdown/qn_27_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 28. Qn 28: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use quantile regression with multiple target quantiles

**Explanation:** Training multiple gradient boosting models with quantile loss functions at different quantiles (e.g., 5%, 50%, 95%) directly models the conditional distribution of the target variable, providing a rigorous non-parametric approach to uncertainty quantification that captures heteroscedasticity.

**Learning Resources:**
- [qn_28_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_28/markdown/qn_28_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 29. Qn 29: What's the most appropriate technique for automated feature engineering in time series forecasting?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use tsfresh with appropriate feature filtering based on p-values

**Explanation:** tsfresh automatically extracts and selects relevant time series features (over 700 features) while controlling for multiple hypothesis testing, specifically designed for time series data unlike general feature engineering tools, making it ideal for time series forecasting tasks.

**Learning Resources:**
- [qn_29_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_29/markdown/qn_29_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 30. Qn 30: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Apply precision-recall curves with prevalence-corrected metrics

**Explanation:** For imbalanced multi-class problems, precision-recall curves with prevalence correction (e.g., weighted by actual class frequencies) provide more informative evaluation than accuracy or ROC-based metrics, focusing on relevant performance for minority classes while accounting for class distribution.

**Learning Resources:**
- [qn_30_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Modelling/questions/qn_30/markdown/qn_30_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
