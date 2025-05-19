# Modelling Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?](#q01)  
- [Qn 02: What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?](#q02)  
- [Qn 03: Which approach correctly implements proper nested cross-validation for model selection and evaluation?](#q03)  
- [Qn 04: What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?](#q04)  
- [Qn 05: When dealing with competing risks in survival analysis, which implementation correctly handles the problem?](#q05)  
- [Qn 06: What's the most statistically sound approach to implement monotonic constraints in gradient boosting?](#q06)  
- [Qn 07: Which approach correctly implements a custom kernel for SVM in scikit-learn?](#q07)  
- [Qn 08: What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?](#q08)  
- [Qn 09: Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?](#q09)  
- [Qn 10: What's the most effective way to implement a time-based split for cross-validation in time series forecasting?](#q10)  
- [Qn 11: Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?](#q11)  
- [Qn 12: What's the most robust approach to handling class imbalance in a multi-class classification problem?](#q12)  
- [Qn 13: Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?](#q13)  
- [Qn 14: What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?](#q14)  
- [Qn 15: Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?](#q15)  
- [Qn 16: What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?](#q16)  
- [Qn 17: Which approach correctly implements quantile regression forests for prediction intervals?](#q17)  
- [Qn 18: What's the most rigorous approach to handle outliers in the target variable for regression problems?](#q18)  
- [Qn 19: Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?](#q19)  
- [Qn 20: What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?](#q20)  
- [Qn 21: Which approach correctly implements a counterfactual explanation method for a black-box classifier?](#q21)  
- [Qn 22: What's the most effective approach to implement online learning for a regression task with concept drift?](#q22)  
- [Qn 23: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?](#q23)  
- [Qn 24: What's the most statistically sound approach to implement feature selection for time series forecasting?](#q24)  
- [Qn 25: Which approach correctly addresses Simpson's paradox in a predictive modeling context?](#q25)  
- [Qn 26: What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?](#q26)  
- [Qn 27: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?](#q27)  
- [Qn 28: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?](#q28)  
- [Qn 29: What's the most appropriate technique for automated feature engineering in time series forecasting?](#q29)  
- [Qn 30: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?](#q30)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?

**Options**  

1. Use StackingClassifier with cv=5  
2. Manually implement out-of-fold predictions for each base learner  
3. Train base models on 70% of data and meta-model on remaining 30%  
4. Use scikit-learn's pipeline to ensure proper nesting of cross-validation  

**Answer**  
Manually implement out-of-fold predictions for each base learner

**Explanation**  
Manually generating out-of-fold predictions ensures the meta-learner only sees
  predictions made on data that base models weren't trained on, fully preventing
  leakage while utilizing all data. This approach is more flexible than
  StackingClassifier and can incorporate diverse base models while maintaining
  proper validation boundaries.

**Detailed Explanation**  
See detailed documentation: [qn_01_answer_long_01.md](data/Modelling/qn_01_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q02"></a> Qn 02

**Question**  
What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?

**Options**  

1. Use XGBoost's built-in calibration with scale_pos_weight parameter  
2. Apply sklearn's CalibratedClassifierCV with isotonic regression  
3. Implement custom Platt scaling with holdout validation  
4. Use quantile regression forests instead of standard gradient boosting  

**Answer**  
Apply sklearn's CalibratedClassifierCV with isotonic regression

**Explanation**  
Isotonic regression via CalibratedClassifierCV is non-parametric and can correct
  any monotonic distortion in probability estimates, making it more flexible
  than Platt scaling, particularly for gradient boosting models which often
  produce well-ranked but not well-calibrated probabilities.

**Detailed Explanation**  
See detailed documentation: [qn_02_answer_long_01.md](data/Modelling/qn_02_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q03"></a> Qn 03

**Question**  
Which approach correctly implements proper nested cross-validation for model selection and evaluation?

**Options**  

1. GridSearchCV inside a for loop of train_test_split iterations  
2. Nested loops of KFold.split(), with inner loop for hyperparameter tuning  
3. Pipeline with GridSearchCV followed by cross_val_score  
4. Custom implementation with an outer cross-validation and inner RandomizedSearchCV  

**Answer**  
Nested loops of KFold.split(), with inner loop for hyperparameter tuning

**Explanation**  
Proper nested cross-validation requires an outer loop for performance estimation
  and an inner loop for hyperparameter tuning, completely separating the data
  used for model selection from the data used for model evaluation, avoiding
  optimistic bias.

**Detailed Explanation**  
See detailed documentation: [qn_03_answer_long_01.md](data/Modelling/qn_03_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q04"></a> Qn 04

**Question**  
What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?

**Options**  

1. Use SGDClassifier with partial_fit on data chunks  
2. Use MiniBatchKMeans for unsupervised feature extraction followed by classification  
3. Implement dask-ml for distributed model training  
4. Use HistGradientBoostingClassifier with max_bins parameter tuning  

**Answer**  
Use SGDClassifier with partial_fit on data chunks

**Explanation**  
SGDClassifier with partial_fit allows true incremental learning, processing data
  in chunks without storing the entire dataset in memory, updating model
  parameters with each batch and converging to the same solution as batch
  processing would with sufficient iterations.

**Detailed Explanation**  
See detailed documentation: [qn_04_answer_long_01.md](data/Modelling/qn_04_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q05"></a> Qn 05

**Question**  
When dealing with competing risks in survival analysis, which implementation correctly handles the problem?

**Options**  

1. Cox Proportional Hazards model with stratification by risk type  
2. Kaplan-Meier estimator with censoring of competing events  
3. Fine-Gray subdistribution hazard model from pysurvival  
4. Random Survival Forests with cause-specific cumulative incidence function  

**Answer**  
Fine-Gray subdistribution hazard model from pysurvival

**Explanation**  
The Fine-Gray model explicitly accounts for competing risks by modeling the
  subdistribution hazard, allowing for valid inference about the probability of
  an event in the presence of competing events, unlike standard Cox models or
  Kaplan-Meier which can produce biased estimates under competing risks.

**Detailed Explanation**  
See detailed documentation: [qn_05_answer_long_01.md](data/Modelling/qn_05_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q06"></a> Qn 06

**Question**  
What's the most statistically sound approach to implement monotonic constraints in gradient boosting?

**Options**  

1. Post-processing model predictions to enforce monotonicity  
2. Using XGBoost's monotone_constraints parameter  
3. Transforming features with isotonic regression before modeling  
4. Implementing a custom callback for LightGBM that penalizes non-monotonic splits  

**Answer**  
Using XGBoost's monotone_constraints parameter

**Explanation**  
XGBoost's native monotone_constraints parameter enforces monotonicity during
  tree building by constraining only monotonic splits, resulting in a fully
  monotonic model without sacrificing performance—unlike post-processing which
  can degrade model accuracy or pre-processing which doesn't guarantee model
  monotonicity.

**Detailed Explanation**  
See detailed documentation: [qn_06_answer_long_01.md](data/Modelling/qn_06_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q07"></a> Qn 07

**Question**  
Which approach correctly implements a custom kernel for SVM in scikit-learn?

**Options**  

1. Subclass sklearn.svm.SVC and override the _compute_kernel method  
2. Define a function that takes two arrays and returns a kernel matrix  
3. Use sklearn.metrics.pairwise.pairwise_kernels with a custom metric  
4. Define a custom kernel using sklearn.gaussian_process.kernels.Kernel  

**Answer**  
Define a function that takes two arrays and returns a kernel matrix

**Explanation**  
For custom kernels in scikit-learn SVMs, one must define a function K(X, Y) that
  calculates the kernel matrix between arrays X and Y, then pass this function
  as the 'kernel' parameter to SVC. This approach allows full flexibility in
  kernel design while maintaining compatibility with scikit-learn's
  implementation.

**Detailed Explanation**  
See detailed documentation: [qn_07_answer_long_01.md](data/Modelling/qn_07_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q08"></a> Qn 08

**Question**  
What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?

**Options**  

1. Sequential feature selection with tolerance for multicollinearity  
2. Recursive feature elimination with cross-validation (RFECV)  
3. Elastic Net regularization with randomized hyperparameter search  
4. Use variance inflation factor (VIF) with backward elimination  

**Answer**  
Elastic Net regularization with randomized hyperparameter search

**Explanation**  
Elastic Net combines L1 and L2 penalties, effectively handling correlated
  features by either selecting one from a correlated group (via L1) or assigning
  similar coefficients to correlated features (via L2), with the optimal balance
  determined through randomized hyperparameter search across different alpha and
  l1_ratio values.

**Detailed Explanation**  
See detailed documentation: [qn_08_answer_long_01.md](data/Modelling/qn_08_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q09"></a> Qn 09

**Question**  
Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?

**Options**  

1. sklearn.preprocessing.OrdinalEncoder  
2. Custom encoding using pd.Categorical with ordered=True  
3. sklearn.preprocessing.PolynomialFeatures with degree=1  
4. Target-guided ordinal encoding based on response variable  

**Answer**  
Custom encoding using pd.Categorical with ordered=True

**Explanation**  
Using pandas Categorical with ordered=True preserves the ordinal relationship
  and allows for appropriate distance calculations between categories, which is
  essential for models that consider feature relationships (unlike
  OrdinalEncoder which assigns arbitrary numeric values without preserving
  distances).

**Detailed Explanation**  
See detailed documentation: [qn_09_answer_long_01.md](data/Modelling/qn_09_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q10"></a> Qn 10

**Question**  
What's the most effective way to implement a time-based split for cross-validation in time series forecasting?

**Options**  

1. Use sklearn's TimeSeriesSplit with appropriate gap  
2. Implement a sliding window validation with fixed lookback period  
3. Use BlockingTimeSeriesSplit from sktime with custom test window growth  
4. Define a custom cross-validator with expanding window and purging  

**Answer**  
Define a custom cross-validator with expanding window and purging

**Explanation**  
A custom cross-validator with expanding windows (increasing training set) and
  purging (gap between train and test to prevent leakage) most accurately
  simulates real-world forecasting scenarios while handling temporal
  dependencies and avoiding lookahead bias.

**Detailed Explanation**  
See detailed documentation: [qn_10_answer_long_01.md](data/Modelling/qn_10_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q11"></a> Qn 11

**Question**  
Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?

**Options**  

1. Random Forest with prediction intervals based on quantiles of tree predictions  
2. Gradient Boosting with NGBoost for natural gradient boosting  
3. Bayesian Logistic Regression with MCMC sampling for posterior distribution  
4. Bootstrapped ensemble of decision trees with variance estimation  

**Answer**  
Bayesian Logistic Regression with MCMC sampling for posterior distribution

**Explanation**  
Bayesian Logistic Regression provides both interpretability (coefficients have
  clear meanings) and principled uncertainty quantification through the
  posterior distribution of parameters, capturing both aleatoric and epistemic
  uncertainty while maintaining model transparency.

**Detailed Explanation**  
See detailed documentation: [qn_11_answer_long_01.md](data/Modelling/qn_11_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q12"></a> Qn 12

**Question**  
What's the most robust approach to handling class imbalance in a multi-class classification problem?

**Options**  

1. Use class_weight='balanced' in sklearn classifiers  
2. Apply SMOTE for oversampling minority classes  
3. Implement a cost-sensitive learning approach with custom loss function  
4. Use ensemble methods with resampling strategies specific to each classifier  

**Answer**  
Use ensemble methods with resampling strategies specific to each classifier

**Explanation**  
Ensemble methods with class-specific resampling strategies (e.g., EasyEnsemble
  or SMOTEBoost) combine the diversity of multiple classifiers with targeted
  handling of class imbalance, outperforming both global resampling and simple
  class weighting, especially for multi-class problems with varying degrees of
  imbalance.

**Detailed Explanation**  
See detailed documentation: [qn_12_answer_long_01.md](data/Modelling/qn_12_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q13"></a> Qn 13

**Question**  
Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?

**Options**  

1. Use feature_importances_ attribute and partial dependence plots  
2. Implement H-statistic from Friedman and Popescu  
3. Extract and analyze individual decision paths from trees  
4. Use permutation importance with pairwise feature shuffling  

**Answer**  
Implement H-statistic from Friedman and Popescu

**Explanation**  
The H-statistic specifically quantifies interaction strength between features by
  comparing the variation in predictions when features are varied together
  versus independently, providing a statistical measure of interactions that
  can't be captured by standard importance metrics or partial dependence alone.

**Detailed Explanation**  
See detailed documentation: [qn_13_answer_long_01.md](data/Modelling/qn_13_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q14"></a> Qn 14

**Question**  
What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?

**Options**  

1. Use make_scorer with a function that combines multiple metrics  
2. Implement a custom Scorer class with a custom __call__ method  
3. Use multiple evaluation metrics with refit parameter specifying the primary metric  
4. Create a pipeline with a custom transformer that adds a penalty term based on complexity  

**Answer**  
Use make_scorer with a function that combines multiple metrics

**Explanation**  
make_scorer allows creating a custom scoring function that can combine
  predictive performance (e.g., AUC) with penalties for model complexity (e.g.,
  number of features or model parameters), providing a single metric for
  optimization that balances performance and parsimony.

**Detailed Explanation**  
See detailed documentation: [qn_14_answer_long_01.md](data/Modelling/qn_14_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q15"></a> Qn 15

**Question**  
Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?

**Options**  

1. Use sklearn's SelectFromModel with LassoCV  
2. Implement weighted LASSO with weight inversely proportional to error variance  
3. Apply robust feature selection using Huber regression  
4. Implement stability selection with bootstrapped samples  

**Answer**  
Implement weighted LASSO with weight inversely proportional to error variance

**Explanation**  
Weighted LASSO that downweights observations with high error variance accounts
  for heteroscedasticity in the selection process, ensuring that features aren't
  selected or rejected due to non-constant error variance, resulting in more
  reliable feature selection.

**Detailed Explanation**  
See detailed documentation: [qn_15_answer_long_01.md](data/Modelling/qn_15_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q16"></a> Qn 16

**Question**  
What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?

**Options**  

1. Use a Random Forest with post-hoc SHAP explanations  
2. Implement Generalized Additive Models (GAMs) with shape constraints  
3. Use Explainable Boosting Machines (EBMs) from InterpretML  
4. Linear model with carefully engineered non-linear features  

**Answer**  
Use Explainable Boosting Machines (EBMs) from InterpretML

**Explanation**  
EBMs combine the interpretability of GAMs with the predictive power of boosting,
  learning feature functions and pairwise interactions in an additive structure
  while remaining highly interpretable, offering better performance than
  standard GAMs while maintaining transparency.

**Detailed Explanation**  
See detailed documentation: [qn_16_answer_long_01.md](data/Modelling/qn_16_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q17"></a> Qn 17

**Question**  
Which approach correctly implements quantile regression forests for prediction intervals?

**Options**  

1. Use sklearn's RandomForestRegressor with bootstrap=True and calculate empirical quantiles of tree predictions  
2. Use the forestci package to compute jackknife-based prediction intervals  
3. Use GradientBoostingRegressor with loss='quantile' and train separate models for each quantile  
4. Implement a custom version of RandomForestRegressor that stores all leaf node samples  

**Answer**  
Implement a custom version of RandomForestRegressor that stores all leaf node samples

**Explanation**  
Quantile regression forests require storing the empirical distribution of
  training samples in each leaf node (not just their mean), requiring a custom
  implementation that extends standard random forests to compute conditional
  quantiles from these stored distributions.

**Detailed Explanation**  
See detailed documentation: [qn_17_answer_long_01.md](data/Modelling/qn_17_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q18"></a> Qn 18

**Question**  
What's the most rigorous approach to handle outliers in the target variable for regression problems?

**Options**  

1. Winsorize the target variable at specific quantiles  
2. Use Huber or Quantile regression with robust loss functions  
3. Remove observations with Cook's distance > 4/n  
4. Apply a power transform to the target before modeling  

**Answer**  
Use Huber or Quantile regression with robust loss functions

**Explanation**  
Robust regression methods like Huber or Quantile regression use loss functions
  that inherently reduce the influence of outliers during model training,
  addressing the issue without removing potentially valuable data points or
  distorting the target distribution through transformations.

**Detailed Explanation**  
See detailed documentation: [qn_18_answer_long_01.md](data/Modelling/qn_18_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q19"></a> Qn 19

**Question**  
Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?

**Options**  

1. Use KNeighborsClassifier with algorithm='kd_tree'  
2. Apply dimensionality reduction like PCA before KNN  
3. Use approximate nearest neighbors with Annoy or FAISS  
4. Implement distance metric learning with NCA or LMNN  

**Answer**  
Implement distance metric learning with NCA or LMNN

**Explanation**  
Distance metric learning adaptively learns a transformation of the feature space
  that emphasizes discriminative dimensions, effectively addressing the curse of
  dimensionality by creating a more semantically meaningful distance metric,
  unlike fixed trees or general dimensionality reduction.

**Detailed Explanation**  
See detailed documentation: [qn_19_answer_long_01.md](data/Modelling/qn_19_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q20"></a> Qn 20

**Question**  
What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?

**Options**  

1. Set max_depth and n_estimators conservatively based on cross-validation  
2. Use early_stopping_rounds with a validation set in XGBoost/LightGBM  
3. Implement a custom callback function that monitors training metrics  
4. Use cross-validation to find optimal number of boosting rounds then retrain  

**Answer**  
Use early_stopping_rounds with a validation set in XGBoost/LightGBM

**Explanation**  
Using early_stopping_rounds with a separate validation set stops training when
  performance on the validation set stops improving for a specified number of
  rounds, efficiently determining the optimal number of trees in a single
  training run without requiring multiple cross-validation runs.

**Detailed Explanation**  
See detailed documentation: [qn_20_answer_long_01.md](data/Modelling/qn_20_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q21"></a> Qn 21

**Question**  
Which approach correctly implements a counterfactual explanation method for a black-box classifier?

**Options**  

1. Use LIME to generate local explanations around the instance  
2. Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals  
3. Apply SHAP values to identify feature importance for the prediction  
4. Use a surrogate explainable model to approximate the black-box decision boundary  

**Answer**  
Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals

**Explanation**  
DiCE specifically generates diverse counterfactual explanations that show how an
  instance's features would need to change to receive a different
  classification, addressing the 'what-if' question directly rather than just
  explaining the current prediction.

**Detailed Explanation**  
See detailed documentation: [qn_21_answer_long_01.md](data/Modelling/qn_21_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q22"></a> Qn 22

**Question**  
What's the most effective approach to implement online learning for a regression task with concept drift?

**Options**  

1. Use SGDRegressor with warm_start=True and smaller alpha as more data arrives  
2. Implement a sliding window approach that retrains on recent data periodically  
3. Use incremental learning with drift detection algorithms to trigger model updates  
4. Maintain an ensemble of models trained on different time windows  

**Answer**  
Use incremental learning with drift detection algorithms to trigger model updates

**Explanation**  
Combining incremental learning with explicit drift detection (e.g., ADWIN, DDM)
  allows the model to adapt continuously to new data while only performing major
  updates when the data distribution actually changes, balancing computational
  efficiency with adaptation to concept drift.

**Detailed Explanation**  
See detailed documentation: [qn_22_answer_long_01.md](data/Modelling/qn_22_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q23"></a> Qn 23

**Question**  
Which method is most appropriate for tuning hyperparameters when training time is extremely limited?

**Options**  

1. Use model-based optimization with Gaussian Processes  
2. Implement multi-fidelity optimization with Hyperband  
3. Apply Optuna with pruning functionality  
4. Use meta-learning from similar tasks to warm-start optimization  

**Answer**  
Implement multi-fidelity optimization with Hyperband

**Explanation**  
Hyperband uses a bandit-based approach to allocate resources efficiently,
  quickly discarding poor configurations and allocating more compute to
  promising ones, making it particularly effective when training time is limited
  and early performance is indicative of final performance.

**Detailed Explanation**  
See detailed documentation: [qn_23_answer_long_01.md](data/Modelling/qn_23_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q24"></a> Qn 24

**Question**  
What's the most statistically sound approach to implement feature selection for time series forecasting?

**Options**  

1. Apply recursive feature elimination with time series cross-validation  
2. Use LASSO regression with temporal blocking of folds  
3. Implement feature importance from tree-based models with purged cross-validation  
4. Use filter methods based on mutual information with time-lagged target variable  

**Answer**  
Implement feature importance from tree-based models with purged cross-validation

**Explanation**  
Tree-based feature importance combined with purged cross-validation (which
  leaves gaps between train and test sets) correctly handles temporal dependence
  in the data, preventing information leakage while identifying features that
  have genuine predictive power for future time points.

**Detailed Explanation**  
See detailed documentation: [qn_24_answer_long_01.md](data/Modelling/qn_24_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q25"></a> Qn 25

**Question**  
Which approach correctly addresses Simpson's paradox in a predictive modeling context?

**Options**  

1. Include interaction terms between potentially confounding variables  
2. Use causal graphical models to identify proper conditioning sets  
3. Apply hierarchical/multilevel modeling to account for grouping  
4. Use propensity score matching before building predictive models  

**Answer**  
Use causal graphical models to identify proper conditioning sets

**Explanation**  
Causal graphical models (e.g., DAGs) allow identifying which variables should or
  should not be conditioned on to avoid Simpson's paradox, ensuring that the
  model captures the true causal relationship rather than spurious associations
  that reverse with conditioning.

**Detailed Explanation**  
See detailed documentation: [qn_25_answer_long_01.md](data/Modelling/qn_25_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q26"></a> Qn 26

**Question**  
What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?

**Options**  

1. Use separate GridSearchCV for each model type and combine best models  
2. Implement nested hyperparameter optimization with DEAP genetic algorithm  
3. Use FLAML for automated and efficient hyperparameter tuning  
4. Apply multi-objective Bayesian optimization to balance diversity and performance  

**Answer**  
Apply multi-objective Bayesian optimization to balance diversity and performance

**Explanation**  
Multi-objective Bayesian optimization can simultaneously optimize for both
  individual model performance and ensemble diversity, finding an optimal set of
  hyperparameters for each model type while ensuring the ensemble as a whole
  performs well through complementary strengths.

**Detailed Explanation**  
See detailed documentation: [qn_26_answer_long_01.md](data/Modelling/qn_26_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q27"></a> Qn 27

**Question**  
Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?

**Options**  

1. Partial dependence plots with contour plots for interactions  
2. Accumulated Local Effects (ALE) plots with bootstrap confidence intervals  
3. SHAP interaction values with dependency plots  
4. Individual Conditional Expectation (ICE) plots with centered PDP  

**Answer**  
Individual Conditional Expectation (ICE) plots with centered PDP

**Explanation**  
ICE plots show how predictions change for individual instances across the range
  of a feature, while centering them helps visualize heterogeneous effects that
  would be masked by averaging in standard partial dependence plots, making them
  ideal for detecting complex non-linear relationships.

**Detailed Explanation**  
See detailed documentation: [qn_27_answer_long_01.md](data/Modelling/qn_27_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q28"></a> Qn 28

**Question**  
What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?

**Options**  

1. Use quantile regression with multiple target quantiles  
2. Implement Monte Carlo dropout in gradient boosting  
3. Apply jackknife resampling to estimate prediction variance  
4. Use Lower Upper Bound Estimation (LUBE) with pinball loss  

**Answer**  
Use quantile regression with multiple target quantiles

**Explanation**  
Training multiple gradient boosting models with quantile loss functions at
  different quantiles (e.g., 5%, 50%, 95%) directly models the conditional
  distribution of the target variable, providing a rigorous non-parametric
  approach to uncertainty quantification that captures heteroscedasticity.

**Detailed Explanation**  
See detailed documentation: [qn_28_answer_long_01.md](data/Modelling/qn_28_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q29"></a> Qn 29

**Question**  
What's the most appropriate technique for automated feature engineering in time series forecasting?

**Options**  

1. Use tsfresh with appropriate feature filtering based on p-values  
2. Implement custom feature extractors with domain-specific transformations  
3. Apply featuretools with time-aware aggregation primitives  
4. Use automatic feature engineering with symbolic transformations and genetic programming  

**Answer**  
Use tsfresh with appropriate feature filtering based on p-values

**Explanation**  
tsfresh automatically extracts and selects relevant time series features (over
  700 features) while controlling for multiple hypothesis testing, specifically
  designed for time series data unlike general feature engineering tools, making
  it ideal for time series forecasting tasks.

**Detailed Explanation**  
See detailed documentation: [qn_29_answer_long_01.md](data/Modelling/qn_29_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q30"></a> Qn 30

**Question**  
Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?

**Options**  

1. Use macro-averaged precision, recall, and F1 score  
2. Implement balanced accuracy and Cohen's kappa statistic  
3. Use ROC AUC with one-vs-rest approach and weighted averaging  
4. Apply precision-recall curves with prevalence-corrected metrics  

**Answer**  
Apply precision-recall curves with prevalence-corrected metrics

**Explanation**  
For imbalanced multi-class problems, precision-recall curves with prevalence
  correction (e.g., weighted by actual class frequencies) provide more
  informative evaluation than accuracy or ROC-based metrics, focusing on
  relevant performance for minority classes while accounting for class
  distribution.

**Detailed Explanation**  
See detailed documentation: [qn_30_answer_long_01.md](data/Modelling/qn_30_answer_long_01.md)

[↑ Go to TOC](#toc)


---

*Automatically generated from [modelling_questions.json](modelling_questions.json)*
*Updated: 2025-05-18 13:57*
