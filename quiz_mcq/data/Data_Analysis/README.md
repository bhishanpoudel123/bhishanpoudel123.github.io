# Data Analysis Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: What technique would you use to handle high-dimensional sparse data when performing PCA?](#q01)  
- [Qn 02: What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?](#q02)  
- [Qn 03: When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?](#q03)  
- [Qn 04: Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?](#q04)  
- [Qn 05: What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?](#q05)  
- [Qn 06: When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?](#q06)  
- [Qn 07: Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?](#q07)  
- [Qn 08: What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?](#q08)  
- [Qn 09: What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?](#q09)  
- [Qn 10: Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?](#q10)  
- [Qn 11: When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?](#q11)  
- [Qn 12: What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?](#q12)  
- [Qn 13: Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?](#q13)  
- [Qn 14: What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?](#q14)  
- [Qn 15: What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?](#q15)  
- [Qn 16: Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?](#q16)  
- [Qn 17: When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?](#q17)  
- [Qn 18: What's the most statistically sound approach to handle heteroscedasticity in a regression model?](#q18)  
- [Qn 19: Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?](#q19)  
- [Qn 20: What technique is most appropriate for analyzing complex network data with community structures?](#q20)  
- [Qn 21: What's the most robust approach to handle concept drift in a production machine learning system?](#q21)  
- [Qn 22: Which method is most appropriate for interpretable anomaly detection in high-dimensional data?](#q22)  
- [Qn 23: When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?](#q23)  
- [Qn 24: What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?](#q24)  
- [Qn 25: Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?](#q25)  
- [Qn 26: What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?](#q26)  
- [Qn 27: Which approach correctly implements online learning for a classification task with a non-stationary data distribution?](#q27)  
- [Qn 28: What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?](#q28)  
- [Qn 29: Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?](#q29)  
- [Qn 30: What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?](#q30)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
What technique would you use to handle high-dimensional sparse data when performing PCA?

**Options**  
1. Standard PCA with normalization  
2. `Truncated SVD` (also known as LSA)  
3. `Kernel PCA` with RBF kernel  
4. `Factor Analysis`  

**Answer**  
`Truncated SVD` (also known as LSA)

**Explanation**  
Truncated SVD is specifically designed for sparse matrices and doesn't center
  the data (which would destroy sparsity), making it more memory-efficient and
  appropriate for high-dimensional sparse datasets.

[↑ Go to TOC](#toc)

  

### <a id="q02"></a> Qn 02

**Question**  
What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?

**Options**  
1. `df.groupby('group').apply(lambda x: x.sample(n=len(x), replace=True))`  
2. `pd.concat([df[df['group']==g].sample(n=sum(df['group']==g), replace=True) for g in df['group'].unique()])`  
3. `df.set_index('group').sample(frac=1, replace=True).reset_index()`  
4. `df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`  

**Answer**  
`df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`

**Explanation**  
This approach uses numpy's efficient random sampling directly on indices,
  avoiding the overhead of pandas' sample function while maintaining group sizes
  and allowing replacement.

[↑ Go to TOC](#toc)

  

### <a id="q03"></a> Qn 03

**Question**  
When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?

**Options**  
1. Use sklearn's `StratifiedKFold` with the most common label for each instance  
2. Create an iterative partitioning algorithm that balances all label combinations across folds  
3. Use sklearn's `MultilabelStratifiedKFold` from the iterative-stratification package  
4. Convert to a multi-class problem using label powerset and then apply standard stratification  

**Answer**  
Use sklearn's `MultilabelStratifiedKFold` from the iterative-stratification package

**Explanation**  
MultilabelStratifiedKFold implements iterative stratification, which preserves
  the distribution of all labels across folds, addressing the key challenge in
  multi-label stratification that normal StratifiedKFold cannot handle.

[↑ Go to TOC](#toc)

  

### <a id="q04"></a> Qn 04

**Question**  
Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?

**Options**  
1. `scipy.stats.wasserstein_distance(x, y)`  
2. `numpy.linalg.norm(np.sort(x) - np.sort(y), ord=1)`  
3. `scipy.spatial.distance.cdist(x.reshape(-1,1), y.reshape(-1,1), metric='euclidean').min(axis=1).sum()`  
4. `sklearn.metrics.pairwise_distances(x.reshape(-1,1), y.reshape(-1,1), metric='manhattan').min(axis=1).mean()`  

**Answer**  
`scipy.stats.wasserstein_distance(x, y)`

**Explanation**  
`scipy.stats.wasserstein_distance` correctly implements the 1D Wasserstein
  distance between empirical distributions, which measures the minimum 'work'
  required to transform one distribution into another.

[↑ Go to TOC](#toc)

  

### <a id="q05"></a> Qn 05

**Question**  
What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?

**Options**  
1. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X).kneighbors(X)`  
2. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X).kneighbors(X)`  
3. `sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X).kneighbors(X)`  
4. Depends on data dimensionality, size, and structure  

**Answer**  
Depends on data dimensionality, size, and structure

**Explanation**  
The most efficient algorithm depends on the dataset characteristics: brute force
  works well for small datasets and high dimensions, kd_tree excels in low
  dimensions (<20), and ball_tree performs better in higher dimensions or with
  non-Euclidean metrics.

[↑ Go to TOC](#toc)

  

### <a id="q06"></a> Qn 06

**Question**  
When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?

**Options**  
1. `df.set_index('timestamp').asfreq('1H').interpolate(method='time')`  
2. `df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`  
3. `df.set_index('timestamp').resample('1H').mean().interpolate(method='time')`  
4. `df.groupby(pd.Grouper(key='timestamp', freq='1H')).apply(lambda x: x.mean() if not x.empty else pd.Series(np.nan, index=df.columns))`  

**Answer**  
`df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`

**Explanation**  
This approach correctly converts irregular timestamps to a regular frequency
  with .resample('1H').asfreq(), then intelligently fills missing values using
  time-based interpolation which respects the actual timing of observations.

[↑ Go to TOC](#toc)

  

### <a id="q07"></a> Qn 07

**Question**  
Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?

**Options**  
1. Pearson correlation matrix with hierarchical clustering  
2. Distance correlation matrix with MDS visualization  
3. `MINE` statistics (Maximal Information-based Nonparametric Exploration)  
4. Random Forest feature importance with partial dependence plots  

**Answer**  
`MINE` statistics (Maximal Information-based Nonparametric Exploration)

**Explanation**  
MINE statistics, particularly the Maximal Information Coefficient (MIC), detect
  both linear and non-linear associations without assuming a specific functional
  form, outperforming traditional correlation measures for complex
  relationships.

[↑ Go to TOC](#toc)

  

### <a id="q08"></a> Qn 08

**Question**  
What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?

**Options**  
1. Oversampling minority classes using SMOTE  
2. Undersampling majority classes using NearMiss  
3. Cost-sensitive learning with class weights inversely proportional to frequencies  
4. Ensemble of balanced subsets with `META` learning  

**Answer**  
Ensemble of balanced subsets with `META` learning

**Explanation**  
META (Minority Ethnicity and Threshold Adjustment) learning with ensembling
  addresses severe multiclass imbalance by training multiple models on balanced
  subsets and combining them, avoiding information loss from undersampling while
  preventing the artificial patterns that can be introduced by synthetic
  oversampling.

[↑ Go to TOC](#toc)

  

### <a id="q09"></a> Qn 09

**Question**  
What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?

**Options**  
1. `pd.get_dummies(df, sparse=True)`  
2. `pd.Categorical(df['col']).codes` in combination with sklearn's `OneHotEncoder(sparse=True)`  
3. Use `pd.factorize()` on all categorical columns followed by scipy's sparse matrices  
4. Convert to category dtype then use `df['col'].cat.codes` with sklearn's `OneHotEncoder(sparse=True)`  

**Answer**  
Convert to category dtype then use `df['col'].cat.codes` with sklearn's `OneHotEncoder(sparse=True)`

**Explanation**  
Converting to pandas' memory-efficient category dtype first, then using
  cat.codes with a sparse OneHotEncoder creates a memory-efficient pipeline that
  preserves category labels and works well with scikit-learn while minimizing
  memory usage.

[↑ Go to TOC](#toc)

  

### <a id="q10"></a> Qn 10

**Question**  
Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?

**Options**  
1. `MultiOutputRegressor(GradientBoostingRegressor())`  
2. `GradientBoostingRegressor` with `multioutput='raw_values'`  
3. `RegressorChain(GradientBoostingRegressor())` with StandardScaler for each target  
4. Separate scaled `GradientBoostingRegressor` for each target in a Pipeline  

**Answer**  
`MultiOutputRegressor(GradientBoostingRegressor())`

**Explanation**  
MultiOutputRegressor fits a separate GradientBoostingRegressor for each target,
  allowing each model to optimize independently, which is crucial when targets
  have different scales and relationships with features.

[↑ Go to TOC](#toc)

  

### <a id="q11"></a> Qn 11

**Question**  
When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?

**Options**  
1. `Isolation Forest` with sliding windows  
2. `One-class SVM` on feature vectors  
3. `LSTM Autoencoder` with reconstruction error thresholding  
4. `ARIMA` with Mahalanobis distance on residuals  

**Answer**  
`LSTM Autoencoder` with reconstruction error thresholding

**Explanation**  
LSTM Autoencoders can capture complex temporal dependencies in multivariate time
  series data, making them ideal for detecting contextual anomalies where data
  points are abnormal specifically in their context rather than globally.

[↑ Go to TOC](#toc)

  

### <a id="q12"></a> Qn 12

**Question**  
What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?

**Options**  
1. Propensity score matching with sensitivity analysis  
2. Instrumental variable analysis with validity tests  
3. Causal graphical models with do-calculus  
4. Difference-in-differences with parallel trends validation  

**Answer**  
Causal graphical models with do-calculus

**Explanation**  
Causal graphical models using do-calculus provide a comprehensive mathematical
  framework for identifying causal effects from observational data, allowing
  researchers to formally express causal assumptions and determine whether
  causal quantities are identifiable from available data.

[↑ Go to TOC](#toc)

  

### <a id="q13"></a> Qn 13

**Question**  
Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?

**Options**  
1. `Mini-batch K-means` with dimensionality reduction  
2. `HDBSCAN` with feature selection  
3. `Birch` (Balanced Iterative Reducing and Clustering using Hierarchies)  
4. `Spectral clustering` with Nyström approximation  

**Answer**  
`Birch` (Balanced Iterative Reducing and Clustering using Hierarchies)

**Explanation**  
Birch is specifically designed for very large datasets as it builds a tree
  structure in a single pass through the data, has linear time complexity,
  limited memory requirements, and can handle outliers effectively, making it
  ideal for clustering massive high-dimensional datasets.

[↑ Go to TOC](#toc)

  

### <a id="q14"></a> Qn 14

**Question**  
What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?

**Options**  
1. Elbow method with distortion scores  
2. Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC)  
3. Cross-validation with log-likelihood scoring  
4. Variational Bayesian inference with automatic relevance determination  

**Answer**  
Variational Bayesian inference with automatic relevance determination

**Explanation**  
Variational Bayesian inference with automatic relevance determination
  (implemented in sklearn as GaussianMixture(n_components=n,
  weight_concentration_prior_type='dirichlet_process')) can automatically prune
  unnecessary components, effectively determining the optimal number without
  requiring multiple model fits and comparisons.

[↑ Go to TOC](#toc)

  

### <a id="q15"></a> Qn 15

**Question**  
What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?

**Options**  
1. `sklearn.metrics.make_scorer(custom_metric, greater_is_better=True)`  
2. `sklearn.metrics.make_scorer(custom_metric, needs_proba=True, greater_is_better=True)`  
3. Create a scorer class that implements __call__(self, estimator, X, y) and gets_score() methods  
4. A and B are both correct depending on the custom_metric function  

**Answer**  
A and B are both correct depending on the custom_metric function

**Explanation**  
make_scorer() is the correct approach, but the parameters depend on the specific
  metric: needs_proba=True for metrics requiring probability estimates (like
  AUC), and needs_threshold=True for metrics requiring decision thresholds; the
  appropriate configuration varies based on the specific imbalance-handling
  metric.

[↑ Go to TOC](#toc)

  

### <a id="q16"></a> Qn 16

**Question**  
Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?

**Options**  
1. Use pandas with `low_memory=True` and `chunksize` parameter  
2. Implement `dask.dataframe` with lazy evaluation and out-of-core computation  
3. Use pandas-on-spark (formerly Koalas) with distributed processing  
4. Implement `vaex` for memory-mapping and out-of-core dataframes  

**Answer**  
Implement `dask.dataframe` with lazy evaluation and out-of-core computation

**Explanation**  
dask.dataframe provides a pandas-like API with lazy evaluation, parallel
  execution, and out-of-core computation, allowing for scalable data processing
  beyond available RAM while maintaining familiar pandas operations and
  requiring minimal code changes.

[↑ Go to TOC](#toc)

  

### <a id="q17"></a> Qn 17

**Question**  
When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?

**Options**  
1. Random search with early stopping  
2. Genetic algorithms with tournament selection  
3. Bayesian optimization with Gaussian processes  
4. Hyperband with successive halving  

**Answer**  
Bayesian optimization with Gaussian processes

**Explanation**  
Bayesian optimization with Gaussian processes builds a probabilistic model of
  the objective function to intelligently select the most promising
  hyperparameter configurations based on previous evaluations, making it more
  efficient than random or grid search for exploring high-dimensional parameter
  spaces.

[↑ Go to TOC](#toc)

  

### <a id="q18"></a> Qn 18

**Question**  
What's the most statistically sound approach to handle heteroscedasticity in a regression model?

**Options**  
1. Visual inspection of residuals vs. fitted values plot  
2. `Breusch-Pagan` test for constant variance  
3. `White's test` for homoscedasticity  
4. Both B and C, with different null hypotheses  

**Answer**  
Both B and C, with different null hypotheses

**Explanation**  
Both tests detect heteroscedasticity but with different assumptions: Breusch-
  Pagan assumes that heteroscedasticity is a linear function of the independent
  variables, while White's test is more general and doesn't make this
  assumption, making them complementary approaches.

[↑ Go to TOC](#toc)

  

### <a id="q19"></a> Qn 19

**Question**  
Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?

**Options**  
1. Bottom-up approach: forecast at lowest level and aggregate upwards  
2. Top-down approach: forecast at highest level and disaggregate proportionally  
3. Middle-out approach: forecast at a middle level and propagate in both directions  
4. Reconciliation approach: forecast at all levels independently then reconcile with constraints  

**Answer**  
Reconciliation approach: forecast at all levels independently then reconcile with constraints

**Explanation**  
The reconciliation approach (optimal combination) generates forecasts at all
  levels independently, then applies a mathematical reconciliation procedure
  that minimizes revisions while ensuring hierarchical consistency, typically
  outperforming both bottom-up and top-down approaches.

[↑ Go to TOC](#toc)

  

### <a id="q20"></a> Qn 20

**Question**  
What technique is most appropriate for analyzing complex network data with community structures?

**Options**  
1. K-means clustering on the adjacency matrix  
2. Spectral clustering with normalized Laplacian  
3. `Louvain` algorithm for community detection  
4. DBSCAN on node2vec embeddings  

**Answer**  
`Louvain` algorithm for community detection

**Explanation**  
The Louvain algorithm specifically optimizes modularity to detect communities in
  networks, automatically finding the appropriate number of communities and
  handling multi-scale resolution, making it ideal for complex networks with
  hierarchical community structures.

[↑ Go to TOC](#toc)

  

### <a id="q21"></a> Qn 21

**Question**  
What's the most robust approach to handle concept drift in a production machine learning system?

**Options**  
1. Implement automatic model retraining when performance degrades below a threshold  
2. Use an ensemble of models with different time windows  
3. Implement drift detection algorithms with adaptive learning techniques  
4. Deploy a champion-challenger framework with continuous evaluation  

**Answer**  
Implement drift detection algorithms with adaptive learning techniques

**Explanation**  
This approach combines statistical drift detection (e.g., ADWIN, DDM, or KSWIN)
  with adaptive learning methods that can continuously update models or model
  weights as new patterns emerge, allowing for immediate adaptation to changing
  data distributions.

[↑ Go to TOC](#toc)

  

### <a id="q22"></a> Qn 22

**Question**  
Which method is most appropriate for interpretable anomaly detection in high-dimensional data?

**Options**  
1. `Isolation Forest` with LIME explanations  
2. Autoencoders with attention mechanisms  
3. SHAP values on `One-Class SVM` predictions  
4. Supervised anomaly detection with feature importance  

**Answer**  
`Isolation Forest` with LIME explanations

**Explanation**  
Isolation Forest efficiently detects anomalies in high dimensions by isolating
  observations, while LIME provides local interpretable explanations for each
  anomaly, showing which features contributed most to its identification, making
  the detection both efficient and explainable.

[↑ Go to TOC](#toc)

  

### <a id="q23"></a> Qn 23

**Question**  
When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?

**Options**  
1. Epsilon-greedy with annealing schedule  
2. `Upper Confidence Bound` (UCB) algorithm  
3. `Thompson Sampling` with prior distribution updates  
4. Contextual bandits with linear payoffs  

**Answer**  
`Thompson Sampling` with prior distribution updates

**Explanation**  
Thompson Sampling with Bayesian updates to prior distributions maintains
  explicit uncertainty estimates and naturally balances
  exploration/exploitation, with theoretical guarantees of optimality and
  empirically better performance than UCB and epsilon-greedy methods in many
  applications.

[↑ Go to TOC](#toc)

  

### <a id="q24"></a> Qn 24

**Question**  
What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?

**Options**  
1. `numpy.linalg.norm` with broadcasting  
2. `scipy.spatial.distance.pdist` with `squareform`  
3. `sklearn.metrics.pairwise_distances` with `n_jobs=-1`  
4. Custom `numba`-accelerated function with parallel processing  

**Answer**  
`scipy.spatial.distance.pdist` with `squareform`

**Explanation**  
pdist computes distances using an optimized implementation that avoids redundant
  calculations (since distance matrices are symmetric), and squareform can
  convert to a square matrix if needed; this approach is significantly more
  memory-efficient than computing the full distance matrix directly.

[↑ Go to TOC](#toc)

  

### <a id="q25"></a> Qn 25

**Question**  
Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?

**Options**  
1. Z-scores on each dimension independently  
2. `Mahalanobis distance` with robust covariance estimation  
3. `Local Outlier Factor` with appropriate neighborhood size  
4. `Isolation Forest` with random projection  

**Answer**  
`Mahalanobis distance` with robust covariance estimation

**Explanation**  
Mahalanobis distance accounts for the covariance structure of the data, and
  using robust covariance estimation (e.g., Minimum Covariance Determinant)
  prevents outliers from influencing the distance metric itself, making it ideal
  for identifying multivariate outliers.

[↑ Go to TOC](#toc)

  

### <a id="q26"></a> Qn 26

**Question**  
What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?

**Options**  
1. Forward stepwise selection with VIF thresholding  
2. `Elastic Net` regularization with cross-validation  
3. Principal Component Regression (PCR)  
4. Recursive Feature Elimination with stability selection  

**Answer**  
`Elastic Net` regularization with cross-validation

**Explanation**  
Elastic Net combines L1 and L2 penalties, handling multicollinearity by grouping
  correlated features while still performing feature selection, with the optimal
  balance determined through cross-validation—making it more effective than
  methods that either eliminate or transform features.

[↑ Go to TOC](#toc)

  

### <a id="q27"></a> Qn 27

**Question**  
Which approach correctly implements online learning for a classification task with a non-stationary data distribution?

**Options**  
1. `SGDClassifier` with `partial_fit` and appropriate `class_weight` adjustments  
2. `River's HoeffdingTreeClassifier` with drift detection  
3. Custom implementation using incremental learning and time-based feature weighting  
4. Ensemble of incremental learners with dynamic weighting based on recent performance  

**Answer**  
Ensemble of incremental learners with dynamic weighting based on recent performance

**Explanation**  
This ensemble approach maintains multiple incremental models updated with new
  data, dynamically adjusting their weights based on recent performance,
  allowing the system to adapt to concept drift by giving more influence to
  models that perform well on recent data.

[↑ Go to TOC](#toc)

  

### <a id="q28"></a> Qn 28

**Question**  
What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?

**Options**  
1. Multiple imputation by chained equations (MICE) with auxiliary variables  
2. Pattern mixture models with sensitivity analysis  
3. Joint modeling of missingness and outcomes  
4. Inverse probability weighting with doubly robust estimation  

**Answer**  
Joint modeling of missingness and outcomes

**Explanation**  
Joint modeling directly incorporates the missingness mechanism into the analysis
  model, allowing for valid inference under non-random missingness (MNAR)
  scenarios by explicitly modeling the relationship between the missing data
  process and the outcomes of interest.

[↑ Go to TOC](#toc)

  

### <a id="q29"></a> Qn 29

**Question**  
Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?

**Options**  
1. Generalized Additive Models with tensor product smooths  
2. `Random Forest` with partial dependence plots and ICE curves  
3. `Neural networks` with feature crossing and attention mechanisms  
4. `Gradient Boosting` with SHAP interaction values  

**Answer**  
`Gradient Boosting` with SHAP interaction values

**Explanation**  
Gradient Boosting effectively captures complex non-linear relationships, while
  SHAP interaction values specifically quantify how much of the prediction is
  attributable to interactions between features, providing a rigorous
  statistical framework for analyzing and visualizing interactions.

[↑ Go to TOC](#toc)

  

### <a id="q30"></a> Qn 30

**Question**  
What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?

**Options**  
1. `Mutual information`-based selection with permutation testing  
2. `LASSO` regression with stability selection  
3. `Random Forest` with Boruta algorithm  
4. `Generalized Additive Models` with significance testing of smooth terms  

**Answer**  
`Mutual information`-based selection with permutation testing

**Explanation**  
Mutual information captures both linear and non-linear dependencies between
  variables without assuming functional form, while permutation testing provides
  a statistically rigorous way to assess the significance of these dependencies,
  controlling for multiple testing issues.

[↑ Go to TOC](#toc)



---

*Automatically generated from [data_analysis_questions.json](data_analysis_questions.json)*  
*Updated: 2025-05-16 15:26*
