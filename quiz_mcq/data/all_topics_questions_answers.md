# All Topics: Questions & Answers


## Table of Contents

- [Data Analysis](#data-analysis)
- [Data Cleaning](#data-cleaning)
- [Data Science](#data-science)
- [Data Visualization](#data-visualization)
- [IQ](#iq)
- [Machine Learning](#machine-learning)
- [Modelling](#modelling)
- [Pandas](#pandas)
- [Probability](#probability)
- [Python Advanced](#python-advanced)
- [Python General](#python-general)
- [SQL General](#sql-general)
- [SQL Sqlite](#sql-sqlite)
- [Statistics](#statistics)
- [Timeseries](#timeseries)

## Data Analysis

### Qn 01: What technique would you use to handle high-dimensional sparse data when performing PCA?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Truncated SVD` (also known as LSA)
<br><strong>Explanation:</strong> Truncated SVD is specifically designed for sparse matrices and doesn't center the data (which would destroy sparsity), making it more memory-efficient and appropriate for high-dimensional sparse datasets.
</div>

### Qn 02: What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`
<br><strong>Explanation:</strong> This approach uses numpy's efficient random sampling directly on indices, avoiding the overhead of pandas' sample function while maintaining group sizes and allowing replacement.
</div>

### Qn 03: When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use sklearn's `MultilabelStratifiedKFold` from the iterative-stratification package
<br><strong>Explanation:</strong> MultilabelStratifiedKFold implements iterative stratification, which preserves the distribution of all labels across folds, addressing the key challenge in multi-label stratification that normal StratifiedKFold cannot handle.
</div>

### Qn 04: Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `scipy.stats.wasserstein_distance(x, y)`
<br><strong>Explanation:</strong> `scipy.stats.wasserstein_distance` correctly implements the 1D Wasserstein distance between empirical distributions, which measures the minimum 'work' required to transform one distribution into another.
</div>

### Qn 05: What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Depends on data dimensionality, size, and structure
<br><strong>Explanation:</strong> The most efficient algorithm depends on the dataset characteristics: brute force works well for small datasets and high dimensions, kd_tree excels in low dimensions (<20), and ball_tree performs better in higher dimensions or with non-Euclidean metrics.
</div>

### Qn 06: When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`
<br><strong>Explanation:</strong> This approach correctly converts irregular timestamps to a regular frequency with .resample('1H').asfreq(), then intelligently fills missing values using time-based interpolation which respects the actual timing of observations.
</div>

### Qn 07: Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `MINE` statistics (Maximal Information-based Nonparametric Exploration)
<br><strong>Explanation:</strong> MINE statistics, particularly the Maximal Information Coefficient (MIC), detect both linear and non-linear associations without assuming a specific functional form, outperforming traditional correlation measures for complex relationships.
</div>

### Qn 08: What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Ensemble of balanced subsets with `META` learning
<br><strong>Explanation:</strong> META (Minority Ethnicity and Threshold Adjustment) learning with ensembling addresses severe multiclass imbalance by training multiple models on balanced subsets and combining them, avoiding information loss from undersampling while preventing the artificial patterns that can be introduced by synthetic oversampling.
</div>

### Qn 09: What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Convert to category dtype then use `df['col'].cat.codes` with sklearn's `OneHotEncoder(sparse=True)`
<br><strong>Explanation:</strong> Converting to pandas' memory-efficient category dtype first, then using cat.codes with a sparse OneHotEncoder creates a memory-efficient pipeline that preserves category labels and works well with scikit-learn while minimizing memory usage.
</div>

### Qn 10: Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `MultiOutputRegressor(GradientBoostingRegressor())`
<br><strong>Explanation:</strong> MultiOutputRegressor fits a separate GradientBoostingRegressor for each target, allowing each model to optimize independently, which is crucial when targets have different scales and relationships with features.
</div>

### Qn 11: When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `LSTM Autoencoder` with reconstruction error thresholding
<br><strong>Explanation:</strong> LSTM Autoencoders can capture complex temporal dependencies in multivariate time series data, making them ideal for detecting contextual anomalies where data points are abnormal specifically in their context rather than globally.
</div>

### Qn 12: What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Causal graphical models with do-calculus
<br><strong>Explanation:</strong> Causal graphical models using do-calculus provide a comprehensive mathematical framework for identifying causal effects from observational data, allowing researchers to formally express causal assumptions and determine whether causal quantities are identifiable from available data.
</div>

### Qn 13: Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Birch` (Balanced Iterative Reducing and Clustering using Hierarchies)
<br><strong>Explanation:</strong> Birch is specifically designed for very large datasets as it builds a tree structure in a single pass through the data, has linear time complexity, limited memory requirements, and can handle outliers effectively, making it ideal for clustering massive high-dimensional datasets.
</div>

### Qn 14: What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Variational Bayesian inference with automatic relevance determination
<br><strong>Explanation:</strong> Variational Bayesian inference with automatic relevance determination (implemented in sklearn as GaussianMixture(n_components=n, weight_concentration_prior_type='dirichlet_process')) can automatically prune unnecessary components, effectively determining the optimal number without requiring multiple model fits and comparisons.
</div>

### Qn 15: What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> A and B are both correct depending on the custom_metric function
<br><strong>Explanation:</strong> make_scorer() is the correct approach, but the parameters depend on the specific metric: needs_proba=True for metrics requiring probability estimates (like AUC), and needs_threshold=True for metrics requiring decision thresholds; the appropriate configuration varies based on the specific imbalance-handling metric.
</div>

### Qn 16: Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement `dask.dataframe` with lazy evaluation and out-of-core computation
<br><strong>Explanation:</strong> dask.dataframe provides a pandas-like API with lazy evaluation, parallel execution, and out-of-core computation, allowing for scalable data processing beyond available RAM while maintaining familiar pandas operations and requiring minimal code changes.
</div>

### Qn 17: When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Bayesian optimization with Gaussian processes
<br><strong>Explanation:</strong> Bayesian optimization with Gaussian processes builds a probabilistic model of the objective function to intelligently select the most promising hyperparameter configurations based on previous evaluations, making it more efficient than random or grid search for exploring high-dimensional parameter spaces.
</div>

### Qn 18: What's the most statistically sound approach to handle heteroscedasticity in a regression model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both B and C, with different null hypotheses
<br><strong>Explanation:</strong> Both tests detect heteroscedasticity but with different assumptions: Breusch-Pagan assumes that heteroscedasticity is a linear function of the independent variables, while White's test is more general and doesn't make this assumption, making them complementary approaches.
</div>

### Qn 19: Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Reconciliation approach: forecast at all levels independently then reconcile with constraints
<br><strong>Explanation:</strong> The reconciliation approach (optimal combination) generates forecasts at all levels independently, then applies a mathematical reconciliation procedure that minimizes revisions while ensuring hierarchical consistency, typically outperforming both bottom-up and top-down approaches.
</div>

### Qn 20: What technique is most appropriate for analyzing complex network data with community structures?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Louvain` algorithm for community detection
<br><strong>Explanation:</strong> The Louvain algorithm specifically optimizes modularity to detect communities in networks, automatically finding the appropriate number of communities and handling multi-scale resolution, making it ideal for complex networks with hierarchical community structures.
</div>

### Qn 21: What's the most robust approach to handle concept drift in a production machine learning system?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement drift detection algorithms with adaptive learning techniques
<br><strong>Explanation:</strong> This approach combines statistical drift detection (e.g., ADWIN, DDM, or KSWIN) with adaptive learning methods that can continuously update models or model weights as new patterns emerge, allowing for immediate adaptation to changing data distributions.
</div>

### Qn 22: Which method is most appropriate for interpretable anomaly detection in high-dimensional data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Isolation Forest` with LIME explanations
<br><strong>Explanation:</strong> Isolation Forest efficiently detects anomalies in high dimensions by isolating observations, while LIME provides local interpretable explanations for each anomaly, showing which features contributed most to its identification, making the detection both efficient and explainable.
</div>

### Qn 23: When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Thompson Sampling` with prior distribution updates
<br><strong>Explanation:</strong> Thompson Sampling with Bayesian updates to prior distributions maintains explicit uncertainty estimates and naturally balances exploration/exploitation, with theoretical guarantees of optimality and empirically better performance than UCB and epsilon-greedy methods in many applications.
</div>

### Qn 24: What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `scipy.spatial.distance.pdist` with `squareform`
<br><strong>Explanation:</strong> pdist computes distances using an optimized implementation that avoids redundant calculations (since distance matrices are symmetric), and squareform can convert to a square matrix if needed; this approach is significantly more memory-efficient than computing the full distance matrix directly.
</div>

### Qn 25: Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Mahalanobis distance` with robust covariance estimation
<br><strong>Explanation:</strong> Mahalanobis distance accounts for the covariance structure of the data, and using robust covariance estimation (e.g., Minimum Covariance Determinant) prevents outliers from influencing the distance metric itself, making it ideal for identifying multivariate outliers.
</div>

### Qn 26: What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Elastic Net` regularization with cross-validation
<br><strong>Explanation:</strong> Elastic Net combines L1 and L2 penalties, handling multicollinearity by grouping correlated features while still performing feature selection, with the optimal balance determined through cross-validation—making it more effective than methods that either eliminate or transform features.
</div>

### Qn 27: Which approach correctly implements online learning for a classification task with a non-stationary data distribution?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Ensemble of incremental learners with dynamic weighting based on recent performance
<br><strong>Explanation:</strong> This ensemble approach maintains multiple incremental models updated with new data, dynamically adjusting their weights based on recent performance, allowing the system to adapt to concept drift by giving more influence to models that perform well on recent data.
</div>

### Qn 28: What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Joint modeling of missingness and outcomes
<br><strong>Explanation:</strong> Joint modeling directly incorporates the missingness mechanism into the analysis model, allowing for valid inference under non-random missingness (MNAR) scenarios by explicitly modeling the relationship between the missing data process and the outcomes of interest.
</div>

### Qn 29: Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Gradient Boosting` with SHAP interaction values
<br><strong>Explanation:</strong> Gradient Boosting effectively captures complex non-linear relationships, while SHAP interaction values specifically quantify how much of the prediction is attributable to interactions between features, providing a rigorous statistical framework for analyzing and visualizing interactions.
</div>

### Qn 30: What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> `Mutual information`-based selection with permutation testing
<br><strong>Explanation:</strong> Mutual information captures both linear and non-linear dependencies between variables without assuming functional form, while permutation testing provides a statistically rigorous way to assess the significance of these dependencies, controlling for multiple testing issues.
</div>


## Data Cleaning

### Qn 01: Which function in Pandas is used to detect missing values?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> isnull()
<br><strong>Explanation:</strong> The `isnull()` function is used to detect missing values in a DataFrame.
</div>

### Qn 02: What does the `dropna()` function do?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Drops rows with missing values
<br><strong>Explanation:</strong> `dropna()` removes rows (or columns) that contain missing values.
</div>

### Qn 03: Which method is used to fill missing values with the mean of the column?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> fillna(mean)
<br><strong>Explanation:</strong> `fillna()` is used to fill missing values, and you can pass the column mean to it.
</div>

### Qn 04: Sample data cleaning question 4?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 4.
</div>

### Qn 05: Sample data cleaning question 5?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 5.
</div>

### Qn 06: Sample data cleaning question 6?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 6.
</div>

### Qn 07: Sample data cleaning question 7?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 7.
</div>

### Qn 08: Sample data cleaning question 8?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 8.
</div>

### Qn 09: Sample data cleaning question 9?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 9.
</div>

### Qn 10: Sample data cleaning question 10?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 10.
</div>

### Qn 11: Sample data cleaning question 11?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 11.
</div>

### Qn 12: Sample data cleaning question 12?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 12.
</div>

### Qn 13: Sample data cleaning question 13?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 13.
</div>

### Qn 14: Sample data cleaning question 14?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 14.
</div>

### Qn 15: Sample data cleaning question 15?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 15.
</div>

### Qn 16: Sample data cleaning question 16?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 16.
</div>

### Qn 17: Sample data cleaning question 17?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 17.
</div>

### Qn 18: Sample data cleaning question 18?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 18.
</div>

### Qn 19: Sample data cleaning question 19?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 19.
</div>

### Qn 20: Sample data cleaning question 20?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 20.
</div>

### Qn 21: Sample data cleaning question 21?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 21.
</div>

### Qn 22: Sample data cleaning question 22?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 22.
</div>

### Qn 23: Sample data cleaning question 23?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 23.
</div>

### Qn 24: Sample data cleaning question 24?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 24.
</div>

### Qn 25: Sample data cleaning question 25?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 25.
</div>

### Qn 26: Sample data cleaning question 26?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 26.
</div>

### Qn 27: Sample data cleaning question 27?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 27.
</div>

### Qn 28: Sample data cleaning question 28?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 28.
</div>

### Qn 29: Sample data cleaning question 29?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 29.
</div>

### Qn 30: Sample data cleaning question 30?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 30.
</div>

### Qn 31: Sample data cleaning question 31?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 31.
</div>

### Qn 32: Sample data cleaning question 32?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 32.
</div>

### Qn 33: Sample data cleaning question 33?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 33.
</div>

### Qn 34: Sample data cleaning question 34?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 34.
</div>

### Qn 35: Sample data cleaning question 35?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 35.
</div>

### Qn 36: Sample data cleaning question 36?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 36.
</div>

### Qn 37: Sample data cleaning question 37?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 37.
</div>

### Qn 38: Sample data cleaning question 38?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 38.
</div>

### Qn 39: Sample data cleaning question 39?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 39.
</div>

### Qn 40: Sample data cleaning question 40?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 40.
</div>

### Qn 41: Sample data cleaning question 41?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 41.
</div>

### Qn 42: Sample data cleaning question 42?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 42.
</div>

### Qn 43: Sample data cleaning question 43?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 43.
</div>

### Qn 44: Sample data cleaning question 44?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 44.
</div>

### Qn 45: Sample data cleaning question 45?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 45.
</div>

### Qn 46: Sample data cleaning question 46?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 46.
</div>

### Qn 47: Sample data cleaning question 47?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 47.
</div>

### Qn 48: Sample data cleaning question 48?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 48.
</div>

### Qn 49: Sample data cleaning question 49?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 49.
</div>

### Qn 50: Sample data cleaning question 50?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Option A
<br><strong>Explanation:</strong> Explanation for sample question 50.
</div>


## Data Science

### Qn 01: What is the primary goal of data wrangling?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Cleaning and transforming raw data into a usable format
<br><strong>Explanation:</strong> Data wrangling involves cleaning, structuring, and enriching raw data into a format suitable for analysis.
</div>

### Qn 02: Which of the following is NOT a measure of central tendency?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Standard deviation
<br><strong>Explanation:</strong> Standard deviation measures dispersion, not central tendency. The three main measures of central tendency are mean, median, and mode.
</div>

### Qn 03: What type of chart would be most appropriate for comparing proportions of a whole?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Pie chart
<br><strong>Explanation:</strong> Pie charts are best for showing proportions of a whole, though they should be used sparingly and only with a small number of categories.
</div>

### Qn 04: Which Python library is primarily used for working with tabular data structures?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Pandas
<br><strong>Explanation:</strong> Pandas provides DataFrame objects which are ideal for working with tabular data, similar to spreadsheets or SQL tables.
</div>

### Qn 05: What does the groupby() operation in Pandas return before aggregation?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> A DataFrameGroupBy object
<br><strong>Explanation:</strong> groupby() returns a DataFrameGroupBy object which can then be aggregated using functions like sum(), mean(), etc.
</div>

### Qn 06: What does 'NaN' represent in a Pandas DataFrame?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Not a Number (missing or undefined value)
<br><strong>Explanation:</strong> NaN stands for 'Not a Number' and represents missing or undefined numerical data in Pandas.
</div>

### Qn 07: Which technique is NOT typically used for feature selection?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Data normalization
<br><strong>Explanation:</strong> Data normalization scales features but doesn't select them. PCA, correlation analysis, and recursive elimination are feature selection methods.
</div>

### Qn 08: Which metric is NOT used to evaluate regression models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Accuracy
<br><strong>Explanation:</strong> Accuracy is used for classification problems. MSE, RMSE, and R-squared are common regression metrics.
</div>

### Qn 09: What is the most common method for handling missing numerical data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Replacing with the mean or median
<br><strong>Explanation:</strong> Mean/median imputation is common for numerical data, though the best approach depends on the data and missingness pattern.
</div>

### Qn 10: Which library is essential for numerical computing in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> NumPy
<br><strong>Explanation:</strong> NumPy provides foundational support for numerical computing with efficient array operations and mathematical functions.
</div>

### Qn 11: What is the purpose of a correlation matrix?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To measure linear relationships between numerical variables
<br><strong>Explanation:</strong> A correlation matrix measures the linear relationship between pairs of numerical variables, ranging from -1 to 1.
</div>

### Qn 12: What is the main advantage of using a box plot?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Displaying the distribution and outliers of a dataset
<br><strong>Explanation:</strong> Box plots effectively show a dataset's quartiles, median, and potential outliers.
</div>

### Qn 13: What does the term 'overfitting' refer to in machine learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> A model that performs well on training data but poorly on unseen data
<br><strong>Explanation:</strong> Overfitting occurs when a model learns the training data too well, including its noise, reducing generalization to new data.
</div>

### Qn 14: Which of these is a supervised learning algorithm?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Random Forest
<br><strong>Explanation:</strong> Random Forest is a supervised learning algorithm. K-means and PCA are unsupervised, and t-SNE is for visualization.
</div>

### Qn 15: What is the purpose of a train-test split?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To evaluate how well a model generalizes to unseen data
<br><strong>Explanation:</strong> Splitting data into training and test sets helps estimate model performance on new, unseen data.
</div>

### Qn 16: Which Python library is most commonly used for creating static visualizations?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Matplotlib
<br><strong>Explanation:</strong> Matplotlib is the foundational plotting library in Python, though Seaborn builds on it for statistical visualizations.
</div>

### Qn 17: What is the main purpose of normalization in data preprocessing?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To scale features to a similar range
<br><strong>Explanation:</strong> Normalization scales numerical features to a standard range (often [0,1] or with mean=0, std=1) to prevent some features from dominating others.
</div>

### Qn 18: What does SQL stand for?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Structured Query Language
<br><strong>Explanation:</strong> SQL stands for Structured Query Language, used for managing and querying relational databases.
</div>

### Qn 19: Which of these is NOT a common data type in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Array
<br><strong>Explanation:</strong> Pandas' main data structures are DataFrame (2D), Series (1D), and Panel (3D, now deprecated). Arrays are from NumPy.
</div>

### Qn 20: What is the primary use of the Scikit-learn library?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Machine learning algorithms
<br><strong>Explanation:</strong> Scikit-learn provides simple and efficient tools for predictive data analysis and machine learning.
</div>

### Qn 21: What is the difference between classification and regression?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Classification predicts categories, regression predicts continuous values
<br><strong>Explanation:</strong> Classification predicts discrete class labels, while regression predicts continuous numerical values.
</div>

### Qn 22: What is a confusion matrix used for?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Evaluating the performance of a classification model
<br><strong>Explanation:</strong> A confusion matrix shows true/false positives/negatives, helping evaluate classification model performance.
</div>

### Qn 23: What does ETL stand for in data engineering?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Extract, Transform, Load
<br><strong>Explanation:</strong> ETL refers to the process of extracting data from sources, transforming it, and loading it into a destination system.
</div>

### Qn 24: Which of these is a dimensionality reduction technique?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Principal Component Analysis (PCA)
<br><strong>Explanation:</strong> PCA reduces dimensionality by transforming data to a new coordinate system with fewer dimensions.
</div>

### Qn 25: What is the purpose of cross-validation?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To get more reliable estimates of model performance
<br><strong>Explanation:</strong> Cross-validation provides more robust performance estimates by using multiple train/test splits of the data.
</div>

### Qn 26: What is the main advantage of using a Jupyter Notebook?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It combines code, visualizations, and narrative text
<br><strong>Explanation:</strong> Jupyter Notebooks allow interactive development with code, visualizations, and explanatory text in a single document.
</div>

### Qn 27: What is the purpose of one-hot encoding?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To convert categorical variables to numerical format
<br><strong>Explanation:</strong> One-hot encoding converts categorical variables to a binary (0/1) numerical format that machine learning algorithms can process.
</div>

### Qn 28: Which metric would you use for an imbalanced classification problem?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Precision-Recall curve
<br><strong>Explanation:</strong> For imbalanced classes, accuracy can be misleading. Precision-Recall curves provide better insight into model performance.
</div>

### Qn 29: What is feature engineering?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Creating new features from existing data
<br><strong>Explanation:</strong> Feature engineering involves creating new input features from existing data to improve model performance.
</div>

### Qn 30: What is the purpose of a ROC curve?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To visualize the trade-off between true positive and false positive rates
<br><strong>Explanation:</strong> ROC curves show the diagnostic ability of a binary classifier by plotting true positive rate vs false positive rate.
</div>

### Qn 31: What is the main advantage of using a random forest over a single decision tree?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It reduces overfitting by averaging multiple trees
<br><strong>Explanation:</strong> Random forests combine multiple decision trees to reduce variance and overfitting compared to a single tree.
</div>

### Qn 32: What is the purpose of the 'iloc' method in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To select data by integer position
<br><strong>Explanation:</strong> iloc is primarily integer-location based indexing for selection by position.
</div>

### Qn 33: What is the difference between deep learning and traditional machine learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Deep learning automatically learns feature hierarchies from raw data
<br><strong>Explanation:</strong> Deep learning models can learn hierarchical feature representations directly from data, while traditional ML often requires manual feature engineering.
</div>

### Qn 34: What is the purpose of a learning curve in machine learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To show the relationship between training set size and model performance
<br><strong>Explanation:</strong> Learning curves plot model performance (e.g., accuracy) against training set size or training iterations.
</div>

### Qn 35: What is the bias-variance tradeoff?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> The balance between model complexity and generalization
<br><strong>Explanation:</strong> The bias-variance tradeoff refers to balancing a model's simplicity (bias) against its sensitivity to training data (variance) to achieve good generalization.
</div>

### Qn 36: What is the purpose of regularization in machine learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To reduce overfitting by penalizing complex models
<br><strong>Explanation:</strong> Regularization techniques like L1/L2 add penalty terms to prevent overfitting by discouraging overly complex models.
</div>

### Qn 37: What is transfer learning in deep learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Using a pre-trained model as a starting point for a new task
<br><strong>Explanation:</strong> Transfer learning leverages knowledge gained from solving one problem and applies it to a different but related problem.
</div>

### Qn 38: What is the purpose of a word embedding in NLP?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To represent words as dense vectors capturing semantic meaning
<br><strong>Explanation:</strong> Word embeddings represent words as numerical vectors where similar words have similar vector representations.
</div>

### Qn 39: What is the main advantage of using SQL databases over NoSQL?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Stronger consistency guarantees
<br><strong>Explanation:</strong> SQL databases provide ACID transactions and strong consistency, while NoSQL prioritizes scalability and flexibility.
</div>

### Qn 40: What is the purpose of A/B testing?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To test two different versions of a product feature
<br><strong>Explanation:</strong> A/B testing compares two versions (A and B) to determine which performs better on a specific metric.
</div>

### Qn 41: What is the main purpose of the 'apply' function in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To apply a function along an axis of a DataFrame
<br><strong>Explanation:</strong> The apply() function applies a function along an axis (rows or columns) of a DataFrame or Series.
</div>

### Qn 42: What is the difference between batch gradient descent and stochastic gradient descent?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Batch uses all data per update, stochastic uses one sample
<br><strong>Explanation:</strong> Batch GD computes gradients using the entire dataset, while SGD uses a single random sample per iteration.
</div>

### Qn 43: What is the purpose of the 'dropna' method in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To drop rows or columns with missing values
<br><strong>Explanation:</strong> dropna() removes missing values (NaN) from a DataFrame, either by rows or columns.
</div>

### Qn 44: What is the main advantage of using a pipeline in Scikit-learn?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To chain multiple processing steps into a single object
<br><strong>Explanation:</strong> Pipelines sequentially apply transforms and a final estimator, ensuring steps are executed in the right order.
</div>

### Qn 45: What is the purpose of the 'value_counts' method in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To count the number of unique values in a Series
<br><strong>Explanation:</strong> value_counts() returns a Series containing counts of unique values in descending order.
</div>

### Qn 46: What is the main purpose of feature scaling?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To ensure all features contribute equally to distance-based algorithms
<br><strong>Explanation:</strong> Feature scaling normalizes the range of features so that features with larger scales don't dominate algorithms like KNN or SVM.
</div>

### Qn 47: What is the difference between 'fit' and 'transform' in Scikit-learn?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 'fit' learns parameters, 'transform' applies them
<br><strong>Explanation:</strong> fit() learns model parameters from training data, while transform() applies the learned transformation to data.
</div>

### Qn 48: What is the purpose of the 'merge' function in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To combine DataFrames based on common columns
<br><strong>Explanation:</strong> merge() combines DataFrames using database-style joins on columns or indices.
</div>

### Qn 49: What is the main advantage of using a dictionary for vectorization in NLP?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It creates a fixed-length representation regardless of document length
<br><strong>Explanation:</strong> Dictionary-based vectorization (like CountVectorizer) creates consistent-length vectors from variable-length texts.
</div>

### Qn 50: What is the purpose of the 'pivot_table' function in Pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To create a spreadsheet-style pivot table as a DataFrame
<br><strong>Explanation:</strong> pivot_table() creates a multi-dimensional summary table similar to Excel pivot tables, aggregating data.
</div>


## Data Visualization

### Qn 01: Which Python library is most commonly used for creating static, animated, and interactive visualizations?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Matplotlib
<br><strong>Explanation:</strong> `Matplotlib` is a foundational library for creating a wide variety of plots and charts.
</div>

### Qn 02: Which function is used to create a line plot in Matplotlib?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> plot()
<br><strong>Explanation:</strong> `plot()` is used to generate line plots in Matplotlib.
</div>

### Qn 03: What does the `Seaborn` library primarily focus on?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Statistical data visualization
<br><strong>Explanation:</strong> `Seaborn` builds on top of Matplotlib and integrates closely with Pandas for statistical plots.
</div>

### Qn 04: Which method can be used to show the distribution of a single variable?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> distplot()
<br><strong>Explanation:</strong> `distplot()` or `histplot()` can be used to show distribution of a variable.
</div>

### Qn 05: How do you display a plot in Jupyter Notebook?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> %matplotlib inline
<br><strong>Explanation:</strong> `%matplotlib inline` is a magic command to render plots directly in the notebook.
</div>


## IQ

### Qn 01: The cost of one paper is 15 cents. How much will 40 papers cost?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $6.00
<br><strong>Explanation:</strong> Basic multiplication or unit price calculation.
</div>

### Qn 02: In a party, 10 men shake hands with each other and they get to shake everyones hand once. How many total handshakes are there?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 45
<br><strong>Explanation:</strong> Use n(n-1)/2 formula for handshakes.
</div>

### Qn 03: Pick out the number with the smallest value.

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.33
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 04: Jim makes 8.50 an hour and 3 extra for cleaning a store. If he worked 36 hours and cleaned 17 stores. How much money did he make?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $357
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 05: Paper pins cost 21 cents a pack. How much will 4 packs cost?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $0.84
<br><strong>Explanation:</strong> Basic multiplication or unit price calculation.
</div>

### Qn 06: A pad costs 33 cents. How much will 5 pads cost?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $1.65
<br><strong>Explanation:</strong> Basic multiplication or unit price calculation.
</div>

### Qn 07: Which of these objects has the largest diameter?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Sun
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 08: Identify the next number in the series.
24, 12, 6, 3, _____

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 1.5
<br><strong>Explanation:</strong> Half each number to get the next number in the series.
</div>

### Qn 09: Urge and deter have _____ meanings?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> contradictory
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 10: Arrange these words to form a sentence. The sentence you arranged, is it true or false?

triangle sides a has three

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> True
<br><strong>Explanation:</strong> A triangle has three sides.
</div>

### Qn 11: Which word doesnt belong in this set of words?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> mason
<br><strong>Explanation:</strong> A mason is a skilled worker who builds or works with stone, brick, or concrete. The other three are professions that require a degree or specialized training. A barrister is a lawyer who specializes in court cases, an economist studies the economy, and a surgeon is a medical doctor who performs surgery.
</div>

### Qn 12: Calculate:  -43 + 9

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> -34
<br><strong>Explanation:</strong> -43+10 would be -33, so -43+9 is -34.
</div>

### Qn 13: The cost of one egg is 15 cents. How much will 4 eggs cost?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $0.60
<br><strong>Explanation:</strong> Basic multiplication or unit price calculation.
</div>

### Qn 14: Subservient is the opposite of?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> imperious
<br><strong>Explanation:</strong> Find the antonym.
</div>

### Qn 15: The volume of a rectangular box is 100 cubic inches. If the minimum dimension of any given side is 1 inch, which of the alternatives is its greatest possible length?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 100 inches
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 16: What is the square root of 8?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 2.82
<br><strong>Explanation:</strong> Square root is the number that gives the original number when multiplied by itself.
</div>

### Qn 17: What do you get when you divide 7.95 by 1.5?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 5.3
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 18: Jack and John have 56 marbles together. John has 6x more marbles than Jack. How many marbles does Jack have?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 8
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 19: Which digit represents the tenths space in 10,987.36?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 3
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 20: How many pairs are duplicates?
987878; 987788
124555; 123555
6845; 6845
45641; 45641
9898547; -9898745

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 2
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 21: How many bananas will you find in a dozen?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 12
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 22: Identify the next number in the series?
5, 15, 10, 13, 15, 11, _____

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 20
<br><strong>Explanation:</strong> The odd positions are increasing by 5 and the even positions are decreasing by 2.
</div>

### Qn 23: How many days are there in three years?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 1,095
<br><strong>Explanation:</strong> 365 days in a year, so 3 years = 3 * 365 = 1,095 days.
</div>

### Qn 24: Adam is selling yarn for $.04/foot. How many feet can you buy from him for $0.52?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 13 feet
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 25: Suppose the first 2 statements are true. Is the third one true/false or not certain?
1) Jack greeted Jill.
2) Jill greeted Joe.
3) Jack didnt greet Joe.

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Uncertain
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 26: Transubstantiate and convert have _____ meanings.

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> similar
<br><strong>Explanation:</strong> Transubstantiate means to change the substance of something, while convert means to change something into a different form or use. Both words imply a transformation.
</div>

### Qn 27: What number should come next?
8, 4, 2, 1, 1/2, 1/4, _____

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 1/8
<br><strong>Explanation:</strong> Each next number are half of the previous number.
</div>

### Qn 28: There are 14 more potatoes than onions in a basket of 36 potatoes and onions. How many potatoes are there in the basket?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 25
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 29: Martin got a 25% salary raise. If his previous salary was $1,200, how much will it be after implementing the raise?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $1,500
<br><strong>Explanation:</strong> Convert the percentage to decimal by dividing by 100.
</div>

### Qn 30: Ferguson has 8 hats, 6 shirts and 4 pairs of pants. How many days can he dress up without repeating the same combination?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 192 days
<br><strong>Explanation:</strong> We need to multiply the number of hats, shirts and pants to get the total combinations. 8 * 6 * 4 = 192.
</div>

### Qn 31: Content and Satisfied have ______ meanings?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> similar
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 32: Identify the next number in the series:
1, 1, 2, 3, 5, 8, _____

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 13
<br><strong>Explanation:</strong> This is fibonacci series. The next number is the sum of the last two numbers.
</div>

### Qn 33: Calculate:  -70 + 35

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> -35
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 34: 61% converted to decimal notation is:

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.61
<br><strong>Explanation:</strong> Convert the percentage to decimal by dividing by 100.
</div>

### Qn 35: Which number has the smallest value?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.02
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 36: Adjoin and sever have _____ meaning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> opposite
<br><strong>Explanation:</strong> Adjoin means to connect or attach, while sever means to cut off or separate. They are opposite in meaning.
</div>

### Qn 37: October is the ____ month of the year

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 10th
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 38: Identify next number in the series.
9, 3, 1, 1/3, _____

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 1/9
<br><strong>Explanation:</strong> Each next number is one third of the previous number.
</div>

### Qn 39: Reduce 75/100 to the simplest form?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 3/4
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 40: An automotive shop owner bought some tools for $5,500. He sold those for $7,300 with a profit of $50 per tool. How many tools did he sell?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 36
<br><strong>Explanation:</strong> 7300 - 5500 = 1800 profit. 1800/50 = 36 tools sold.
</div>

### Qn 41: Partner and Join have _____ meanings.

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> similar
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 42: If a person is half a century old. How old is he?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 50
<br><strong>Explanation:</strong> Century means 100 years. Half a century means 50 years.
</div>

### Qn 43: What do you get when you round of 907.457 to the nearest tens place?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 910
<br><strong>Explanation:</strong> Nearest tens place means rounding to the nearest 10. 907.457 rounds to 910.
</div>

### Qn 44: (8 ÷ 4) x (9 ÷ 3) = ?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 6
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 45: What is the relation betweed Credence and Credit?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> similar
<br><strong>Explanation:</strong> Credence means belief or trust, while credit refers to the ability to obtain goods or services before payment. They are related in the context of trust and belief.
</div>

### Qn 46: In a week, Rose spent $28.49 on lunch. What was the average cost per day?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> $4.07
<br><strong>Explanation:</strong> Basic multiplication or unit price calculation.
</div>

### Qn 47: The opposite of punctual is?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> tardy
<br><strong>Explanation:</strong> tardy means late or delayed, which is the opposite of punctual. Whereas conscious means aware, rigorous means strict or demanding, and meticulous means careful and precise.
</div>

### Qn 48: Reduce and Produce have ____ meaning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> opposite
<br><strong>Explanation:</strong> Apply logic or perform basic arithmetic to solve.
</div>

### Qn 49: Rule out the odd word from the set of words.

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> drought
<br><strong>Explanation:</strong> drought is a prolonged dry period, while the others are related to finance. Budget refers to a plan for spending money, debt refers to money owed, and credit refers to the ability to borrow money.
</div>

### Qn 50: Which three words have the same meaning?
A. Information
B. Indoctrinate
C. Brainwash
D. Convince
E. Class

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> BCD
<br><strong>Explanation:</strong> Find the synonym.
</div>


## Machine Learning

### Qn 01: Which algorithm is used for classification problems?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Logistic Regression
<br><strong>Explanation:</strong> Logistic regression is used for binary and multi-class classification problems.
</div>

### Qn 02: What is the purpose of the learning rate in gradient descent?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To control how much the model is adjusted during each update
<br><strong>Explanation:</strong> The learning rate determines the step size at each iteration while moving toward a minimum of a loss function.
</div>

### Qn 03: Which technique is used to reduce the dimensionality of data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> PCA
<br><strong>Explanation:</strong> Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data to a lower-dimensional space.
</div>

### Qn 04: What is the main goal of unsupervised learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Find hidden patterns or groupings in data
<br><strong>Explanation:</strong> Unsupervised learning aims to discover the underlying structure of data without predefined labels.
</div>

### Qn 05: What is overfitting in machine learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> When the model performs well on training data but poorly on test data
<br><strong>Explanation:</strong> Overfitting occurs when a model learns the training data too well, including noise and outliers, which leads to poor generalization to new data.
</div>


## Modelling

### Qn 01: When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Manually implement out-of-fold predictions for each base learner
<br><strong>Explanation:</strong> Manually generating out-of-fold predictions ensures the meta-learner only sees predictions made on data that base models weren't trained on, fully preventing leakage while utilizing all data. This approach is more flexible than StackingClassifier and can incorporate diverse base models while maintaining proper validation boundaries.
</div>

### Qn 02: What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Apply sklearn's CalibratedClassifierCV with isotonic regression
<br><strong>Explanation:</strong> Isotonic regression via CalibratedClassifierCV is non-parametric and can correct any monotonic distortion in probability estimates, making it more flexible than Platt scaling, particularly for gradient boosting models which often produce well-ranked but not well-calibrated probabilities.
</div>

### Qn 03: Which approach correctly implements proper nested cross-validation for model selection and evaluation?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Nested loops of KFold.split(), with inner loop for hyperparameter tuning
<br><strong>Explanation:</strong> Proper nested cross-validation requires an outer loop for performance estimation and an inner loop for hyperparameter tuning, completely separating the data used for model selection from the data used for model evaluation, avoiding optimistic bias.
</div>

### Qn 04: What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use SGDClassifier with partial_fit on data chunks
<br><strong>Explanation:</strong> SGDClassifier with partial_fit allows true incremental learning, processing data in chunks without storing the entire dataset in memory, updating model parameters with each batch and converging to the same solution as batch processing would with sufficient iterations.
</div>

### Qn 05: When dealing with competing risks in survival analysis, which implementation correctly handles the problem?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Fine-Gray subdistribution hazard model from pysurvival
<br><strong>Explanation:</strong> The Fine-Gray model explicitly accounts for competing risks by modeling the subdistribution hazard, allowing for valid inference about the probability of an event in the presence of competing events, unlike standard Cox models or Kaplan-Meier which can produce biased estimates under competing risks.
</div>

### Qn 06: What's the most statistically sound approach to implement monotonic constraints in gradient boosting?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Using XGBoost's monotone_constraints parameter
<br><strong>Explanation:</strong> XGBoost's native monotone_constraints parameter enforces monotonicity during tree building by constraining only monotonic splits, resulting in a fully monotonic model without sacrificing performance—unlike post-processing which can degrade model accuracy or pre-processing which doesn't guarantee model monotonicity.
</div>

### Qn 07: Which approach correctly implements a custom kernel for SVM in scikit-learn?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Define a function that takes two arrays and returns a kernel matrix
<br><strong>Explanation:</strong> For custom kernels in scikit-learn SVMs, one must define a function K(X, Y) that calculates the kernel matrix between arrays X and Y, then pass this function as the 'kernel' parameter to SVC. This approach allows full flexibility in kernel design while maintaining compatibility with scikit-learn's implementation.
</div>

### Qn 08: What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Elastic Net regularization with randomized hyperparameter search
<br><strong>Explanation:</strong> Elastic Net combines L1 and L2 penalties, effectively handling correlated features by either selecting one from a correlated group (via L1) or assigning similar coefficients to correlated features (via L2), with the optimal balance determined through randomized hyperparameter search across different alpha and l1_ratio values.
</div>

### Qn 09: Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Custom encoding using pd.Categorical with ordered=True
<br><strong>Explanation:</strong> Using pandas Categorical with ordered=True preserves the ordinal relationship and allows for appropriate distance calculations between categories, which is essential for models that consider feature relationships (unlike OrdinalEncoder which assigns arbitrary numeric values without preserving distances).
</div>

### Qn 10: What's the most effective way to implement a time-based split for cross-validation in time series forecasting?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Define a custom cross-validator with expanding window and purging
<br><strong>Explanation:</strong> A custom cross-validator with expanding windows (increasing training set) and purging (gap between train and test to prevent leakage) most accurately simulates real-world forecasting scenarios while handling temporal dependencies and avoiding lookahead bias.
</div>

### Qn 11: Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Bayesian Logistic Regression with MCMC sampling for posterior distribution
<br><strong>Explanation:</strong> Bayesian Logistic Regression provides both interpretability (coefficients have clear meanings) and principled uncertainty quantification through the posterior distribution of parameters, capturing both aleatoric and epistemic uncertainty while maintaining model transparency.
</div>

### Qn 12: What's the most robust approach to handling class imbalance in a multi-class classification problem?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use ensemble methods with resampling strategies specific to each classifier
<br><strong>Explanation:</strong> Ensemble methods with class-specific resampling strategies (e.g., EasyEnsemble or SMOTEBoost) combine the diversity of multiple classifiers with targeted handling of class imbalance, outperforming both global resampling and simple class weighting, especially for multi-class problems with varying degrees of imbalance.
</div>

### Qn 13: Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement H-statistic from Friedman and Popescu
<br><strong>Explanation:</strong> The H-statistic specifically quantifies interaction strength between features by comparing the variation in predictions when features are varied together versus independently, providing a statistical measure of interactions that can't be captured by standard importance metrics or partial dependence alone.
</div>

### Qn 14: What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use make_scorer with a function that combines multiple metrics
<br><strong>Explanation:</strong> make_scorer allows creating a custom scoring function that can combine predictive performance (e.g., AUC) with penalties for model complexity (e.g., number of features or model parameters), providing a single metric for optimization that balances performance and parsimony.
</div>

### Qn 15: Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement weighted LASSO with weight inversely proportional to error variance
<br><strong>Explanation:</strong> Weighted LASSO that downweights observations with high error variance accounts for heteroscedasticity in the selection process, ensuring that features aren't selected or rejected due to non-constant error variance, resulting in more reliable feature selection.
</div>

### Qn 16: What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use Explainable Boosting Machines (EBMs) from InterpretML
<br><strong>Explanation:</strong> EBMs combine the interpretability of GAMs with the predictive power of boosting, learning feature functions and pairwise interactions in an additive structure while remaining highly interpretable, offering better performance than standard GAMs while maintaining transparency.
</div>

### Qn 17: Which approach correctly implements quantile regression forests for prediction intervals?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement a custom version of RandomForestRegressor that stores all leaf node samples
<br><strong>Explanation:</strong> Quantile regression forests require storing the empirical distribution of training samples in each leaf node (not just their mean), requiring a custom implementation that extends standard random forests to compute conditional quantiles from these stored distributions.
</div>

### Qn 18: What's the most rigorous approach to handle outliers in the target variable for regression problems?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use Huber or Quantile regression with robust loss functions
<br><strong>Explanation:</strong> Robust regression methods like Huber or Quantile regression use loss functions that inherently reduce the influence of outliers during model training, addressing the issue without removing potentially valuable data points or distorting the target distribution through transformations.
</div>

### Qn 19: Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement distance metric learning with NCA or LMNN
<br><strong>Explanation:</strong> Distance metric learning adaptively learns a transformation of the feature space that emphasizes discriminative dimensions, effectively addressing the curse of dimensionality by creating a more semantically meaningful distance metric, unlike fixed trees or general dimensionality reduction.
</div>

### Qn 20: What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use early_stopping_rounds with a validation set in XGBoost/LightGBM
<br><strong>Explanation:</strong> Using early_stopping_rounds with a separate validation set stops training when performance on the validation set stops improving for a specified number of rounds, efficiently determining the optimal number of trees in a single training run without requiring multiple cross-validation runs.
</div>

### Qn 21: Which approach correctly implements a counterfactual explanation method for a black-box classifier?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals
<br><strong>Explanation:</strong> DiCE specifically generates diverse counterfactual explanations that show how an instance's features would need to change to receive a different classification, addressing the 'what-if' question directly rather than just explaining the current prediction.
</div>

### Qn 22: What's the most effective approach to implement online learning for a regression task with concept drift?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use incremental learning with drift detection algorithms to trigger model updates
<br><strong>Explanation:</strong> Combining incremental learning with explicit drift detection (e.g., ADWIN, DDM) allows the model to adapt continuously to new data while only performing major updates when the data distribution actually changes, balancing computational efficiency with adaptation to concept drift.
</div>

### Qn 23: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement multi-fidelity optimization with Hyperband
<br><strong>Explanation:</strong> Hyperband uses a bandit-based approach to allocate resources efficiently, quickly discarding poor configurations and allocating more compute to promising ones, making it particularly effective when training time is limited and early performance is indicative of final performance.
</div>

### Qn 24: What's the most statistically sound approach to implement feature selection for time series forecasting?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Implement feature importance from tree-based models with purged cross-validation
<br><strong>Explanation:</strong> Tree-based feature importance combined with purged cross-validation (which leaves gaps between train and test sets) correctly handles temporal dependence in the data, preventing information leakage while identifying features that have genuine predictive power for future time points.
</div>

### Qn 25: Which approach correctly addresses Simpson's paradox in a predictive modeling context?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use causal graphical models to identify proper conditioning sets
<br><strong>Explanation:</strong> Causal graphical models (e.g., DAGs) allow identifying which variables should or should not be conditioned on to avoid Simpson's paradox, ensuring that the model captures the true causal relationship rather than spurious associations that reverse with conditioning.
</div>

### Qn 26: What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Apply multi-objective Bayesian optimization to balance diversity and performance
<br><strong>Explanation:</strong> Multi-objective Bayesian optimization can simultaneously optimize for both individual model performance and ensemble diversity, finding an optimal set of hyperparameters for each model type while ensuring the ensemble as a whole performs well through complementary strengths.
</div>

### Qn 27: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Individual Conditional Expectation (ICE) plots with centered PDP
<br><strong>Explanation:</strong> ICE plots show how predictions change for individual instances across the range of a feature, while centering them helps visualize heterogeneous effects that would be masked by averaging in standard partial dependence plots, making them ideal for detecting complex non-linear relationships.
</div>

### Qn 28: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use quantile regression with multiple target quantiles
<br><strong>Explanation:</strong> Training multiple gradient boosting models with quantile loss functions at different quantiles (e.g., 5%, 50%, 95%) directly models the conditional distribution of the target variable, providing a rigorous non-parametric approach to uncertainty quantification that captures heteroscedasticity.
</div>

### Qn 29: What's the most appropriate technique for automated feature engineering in time series forecasting?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Use tsfresh with appropriate feature filtering based on p-values
<br><strong>Explanation:</strong> tsfresh automatically extracts and selects relevant time series features (over 700 features) while controlling for multiple hypothesis testing, specifically designed for time series data unlike general feature engineering tools, making it ideal for time series forecasting tasks.
</div>

### Qn 30: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Apply precision-recall curves with prevalence-corrected metrics
<br><strong>Explanation:</strong> For imbalanced multi-class problems, precision-recall curves with prevalence correction (e.g., weighted by actual class frequencies) provide more informative evaluation than accuracy or ROC-based metrics, focusing on relevant performance for minority classes while accounting for class distribution.
</div>


## Pandas

### Qn 01: Which method efficiently applies a function along an axis of a DataFrame?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df.apply(func, axis=0)
<br><strong>Explanation:</strong> The apply() method allows applying a function along an axis (rows or columns) of a DataFrame.
</div>

### Qn 02: What's the correct way to merge two DataFrames on multiple columns?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and C
<br><strong>Explanation:</strong> Both pd.merge() and DataFrame.merge() methods can merge on multiple columns specified as lists.
</div>

### Qn 03: How do you handle missing values in a DataFrame column?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All listed methods can handle missing values: fillna() replaces NaNs, dropna() removes rows with NaNs, and replace() can substitute NaNs with specified values.
</div>

### Qn 04: What does the method `groupby().agg()` allow you to do?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> The agg() method is versatile and can apply single or multiple functions to grouped data, either to all columns or selectively.
</div>

### Qn 05: Which of the following transforms a DataFrame to a long format?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> stack(), melt(), and wide_to_long() all convert data from wide format to long format, albeit with different approaches and parameters.
</div>

### Qn 06: How can you efficiently select rows where a column value meets a complex condition?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both B and C
<br><strong>Explanation:</strong> Both loc with boolean indexing (with proper parentheses) and query() method can filter data based on complex conditions.
</div>

### Qn 07: What's the most efficient way to calculate a rolling 7-day average of a time series?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df['rolling_avg'] = df['value'].rolling(window=7).mean()
<br><strong>Explanation:</strong> The rolling() method with a window of 7 followed by mean() calculates a rolling average over a 7-period window.
</div>

### Qn 08: How do you perform a pivot operation in pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All three methods can perform pivot operations, with pivot_table being more flexible as it can aggregate duplicate entries.
</div>

### Qn 09: Which method can reshape a DataFrame by stacking column labels to rows?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df.stack()
<br><strong>Explanation:</strong> stack() method pivots the columns of a DataFrame to become the innermost index level, creating a Series with a MultiIndex.
</div>

### Qn 10: How do you efficiently concatenate many DataFrames with identical columns?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> pd.concat([df1, df2, df3])
<br><strong>Explanation:</strong> pd.concat() is designed to efficiently concatenate pandas objects along a particular axis with optional set logic.
</div>

### Qn 11: What's the correct way to create a DatetimeIndex from a column containing date strings?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All methods will correctly convert date strings to datetime objects, with different approaches to setting them as the index.
</div>

### Qn 12: Which method performs a cross-tabulation of two factors?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All methods can create cross-tabulations, though crosstab() is specifically designed for this purpose.
</div>

### Qn 13: How do you calculate cumulative statistics in pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df.cumsum(), df.cumprod(), df.cummax(), df.cummin()
<br><strong>Explanation:</strong> The cum- methods (cumsum, cumprod, cummax, cummin) calculate cumulative statistics along an axis.
</div>

### Qn 14: Which approach efficiently calculates the difference between consecutive rows in a DataFrame?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and B
<br><strong>Explanation:</strong> Both subtracting a shifted DataFrame and using diff() calculate element-wise differences between consecutive rows.
</div>

### Qn 15: How do you create a MultiIndex DataFrame from scratch?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All three methods create equivalent MultiIndex objects using different approaches: from_tuples, from_product, and from_arrays.
</div>

### Qn 16: Which method is most appropriate for performing complex string operations on DataFrame columns?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above work, but B is most efficient
<br><strong>Explanation:</strong> While all methods can transform strings, the .str accessor provides vectorized string functions that are generally more efficient than apply() or map().
</div>

### Qn 17: What's the best way to compute percentiles for grouped data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and C
<br><strong>Explanation:</strong> Both quantile() and describe() can compute percentiles for grouped data, with describe() providing additional statistics. For option B, While this approach uses the right function (numpy's percentile), there's an issue with how it's implemented in the context of pandas GroupBy. This would likely raise errors because the lambda function returns arrays rather than scalars, which is problematic for the standard aggregation pipeline.
</div>

### Qn 18: How do you efficiently implement a custom aggregation function that requires the entire group?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df.groupby('group').apply(custom_func)
<br><strong>Explanation:</strong> apply() is designed for operations that need the entire group as a DataFrame, whereas agg() is better for operations that can be vectorized.
</div>

### Qn 19: What's the most memory-efficient way to read a large CSV file with pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> pd.read_csv('file.csv', dtype={'col1': 'category', 'col2': 'int8'})
<br><strong>Explanation:</strong> Specifying appropriate dtypes, especially using 'category' for string columns with repeated values, significantly reduces memory usage.
</div>

### Qn 20: Which method is correct for resampling time series data to monthly frequency?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and B
<br><strong>Explanation:</strong> Both resample() and groupby() with Grouper can aggregate time series data to monthly frequency, though asfreq() only changes frequency without aggregation.
</div>

### Qn 21: How do you efficiently identify and remove duplicate rows in a DataFrame?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and B
<br><strong>Explanation:</strong> Both df[~df.duplicated()] and df.drop_duplicates() remove duplicate rows, with the latter being more readable and offering more options.
</div>

### Qn 22: Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> df.pipe(custom_func)
<br><strong>Explanation:</strong> pipe() is designed for functions that take and return a DataFrame, creating readable method chains when applying multiple functions.
</div>

### Qn 23: How do you sample data from a DataFrame with weights?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both B and C
<br><strong>Explanation:</strong> Both approaches correctly sample with weights, though weights don't need to be normalized as pandas normalizes them internally.
</div>

### Qn 24: What's the correct way to use the pd.cut() function for binning continuous data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above are valid uses
<br><strong>Explanation:</strong> All approaches are valid: using explicit bin edges, equal-width bins (cut), or equal-frequency bins (qcut).
</div>

### Qn 25: How do you efficiently perform a custom window operation in pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and B
<br><strong>Explanation:</strong> Both approaches work for custom window operations, but using raw=True can be more efficient for numerical operations by passing a NumPy array instead of a Series.
</div>

### Qn 26: Which approach can create a lagged feature in a time series DataFrame?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and B
<br><strong>Explanation:</strong> shift(1) creates a lag (past values), while shift(-1) creates a lead (future values), both useful for time series analysis.
</div>

### Qn 27: What's the best way to explode a DataFrame column containing lists into multiple rows?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both B and C
<br><strong>Explanation:</strong> explode() transforms each element of a list-like column into a row, with the original index duplicated as needed.
</div>

### Qn 28: How do you efficiently compute a weighted mean in pandas?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Both A and C
<br><strong>Explanation:</strong> Both manually computing weighted mean and using np.average() work efficiently, though pandas Series doesn't have a weights parameter for mean().
</div>

### Qn 29: Which method correctly identifies the top-k values in each group?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All three methods can get the top-k values within each group, with different syntax but similar results.
</div>

### Qn 30: What's the best way to add a new column based on a categorical mapping of an existing column?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> All of the above
<br><strong>Explanation:</strong> All methods can map values to new ones, though map() is generally preferred for dictionary-based mappings.
</div>


## Probability

### Qn 01: In a room of 30 people, what is the probability that at least two people share the same birthday (assuming 365 days in a year and birthdays are uniformly distributed)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Greater than 90%
<br><strong>Explanation:</strong> This is the famous birthday paradox. The probability is computed as 1 minus the probability that all birthdays are different. For 30 people, P(at least one shared birthday) = 1 - (365/365 × 364/365 × 363/365 × ... × 336/365) ≈ 0.706, which is approximately 70.6%. While this is closest to 'About 70%', the exact calculation gives 70.6%, not 'Greater than 90%'. The probability exceeds 90% when there are 41 or more people in the room.
</div>

### Qn 02: A biased coin has a 60% chance of landing heads. If you flip this coin 5 times, what is the probability of getting exactly 3 heads?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.34560
<br><strong>Explanation:</strong> We use the binomial probability formula: P(X=k) = C(n,k) × p^k × (1-p)^(n-k), where n=5, k=3, p=0.6. C(5,3) = 10 possible ways to arrange 3 heads in 5 flips. So P(X=3) = 10 × (0.6)^3 × (0.4)^2 = 10 × 0.216 × 0.16 = 0.3456 or 34.56%.
</div>

### Qn 03: In a standard deck of 52 cards, what is the probability of drawing a royal flush (A, K, Q, J, 10 of the same suit) in a 5-card poker hand?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 4/2,598,960
<br><strong>Explanation:</strong> There are C(52,5) = 2,598,960 possible 5-card hands from a standard deck. There are exactly 4 possible royal flushes (one for each suit). Therefore, the probability is 4/2,598,960 = 1/649,740 or approximately 0.00000154.
</div>

### Qn 04: A standard six-sided die is rolled 3 times. What is the probability that the sum of the three rolls equals 10?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 1/8
<br><strong>Explanation:</strong> When rolling 3 dice, there are 6^3 = 216 possible outcomes. To get a sum of 10, we can have combinations like (1,3,6), (2,2,6), etc. Counting all such combinations gives us 27 favorable outcomes. Therefore, the probability is 27/216 = 1/8.
</div>

### Qn 05: A bag contains 5 red marbles and 7 blue marbles. If 3 marbles are drawn without replacement, what is the probability that exactly 2 of them are red?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 35/132
<br><strong>Explanation:</strong> Total ways to select 3 marbles from 12 is C(12,3) = 220. Ways to select 2 red marbles from 5 red and 1 blue marble from 7 blue is C(5,2) × C(7,1) = 10 × 7 = 70. Therefore probability = 70/220 = 7/22 = 35/110 = 7/22.
</div>

### Qn 06: In a Bayesian analysis, a disease has a 1% prevalence in a population. A test for the disease has 95% sensitivity (true positive rate) and 90% specificity (true negative rate). If a person tests positive, what is the probability they actually have the disease?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Around 8.7%
<br><strong>Explanation:</strong> Using Bayes' theorem: P(Disease|Positive) = [P(Positive|Disease) × P(Disease)] / P(Positive). P(Positive) = P(Positive|Disease) × P(Disease) + P(Positive|No Disease) × P(No Disease) = 0.95 × 0.01 + 0.10 × 0.99 = 0.0095 + 0.099 = 0.1085. Therefore, P(Disease|Positive) = (0.95 × 0.01) / 0.1085 ≈ 0.0095/0.1085 ≈ 0.0876 or about 8.7%.
</div>

### Qn 07: You roll a fair 6-sided die repeatedly until you get a 6. What is the expected number of rolls needed?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 6
<br><strong>Explanation:</strong> This is a geometric distribution with probability of success p = 1/6. The expected value of a geometric distribution is 1/p. So the expected number of rolls needed is 1/(1/6) = 6.
</div>

### Qn 08: In a group of 5 people, what is the probability that at least 2 people have the same zodiac sign (assuming zodiac signs are uniformly distributed across 12 possible signs)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.96
<br><strong>Explanation:</strong> The probability that all 5 people have different zodiac signs is (12/12) × (11/12) × (10/12) × (9/12) × (8/12) = 0.0397. Therefore, the probability that at least 2 people share a zodiac sign is 1 - 0.0397 = 0.9603 ≈ 0.96 or 96%.
</div>

### Qn 09: A data scientist applies a machine learning model to classify emails as spam or not spam. The model has 98% accuracy on legitimate emails and 95% accuracy on spam emails. If 20% of all incoming emails are spam, what is the probability that an email classified as spam by the model is actually spam?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.83
<br><strong>Explanation:</strong> Using Bayes' theorem: P(Spam|Classified as Spam) = [P(Classified as Spam|Spam) × P(Spam)] / P(Classified as Spam). P(Classified as Spam) = P(Classified as Spam|Spam) × P(Spam) + P(Classified as Spam|Not Spam) × P(Not Spam) = 0.95 × 0.2 + 0.02 × 0.8 = 0.19 + 0.016 = 0.206. Therefore, P(Spam|Classified as Spam) = (0.95 × 0.2) / 0.206 = 0.19/0.206 ≈ 0.83 or 83%.
</div>

### Qn 10: Four cards are randomly selected from a standard 52-card deck. What is the probability of getting exactly 2 aces?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.0399
<br><strong>Explanation:</strong> Total number of ways to select 4 cards from 52 is C(52,4) = 270,725. Ways to select exactly 2 aces from 4 aces is C(4,2) = 6. Ways to select the other 2 cards from the non-ace cards is C(48,2) = 1,128. So favorable outcomes = 6 × 1,128 = 6,768. Probability = 6,768/270,725 ≈ 0.0399 or about 4%.
</div>

### Qn 11: In a standard normal distribution, what is the probability that a randomly selected observation falls between -1.96 and 1.96?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.95
<br><strong>Explanation:</strong> In a standard normal distribution, the area between z-scores of -1.96 and 1.96 corresponds to 95% of the distribution. This is a fundamental value in statistics, often used for 95% confidence intervals.
</div>

### Qn 12: A manufacturing process has a 3% defect rate. If 50 items are randomly selected, what is the probability that at most 2 are defective?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.6063
<br><strong>Explanation:</strong> This follows a binomial distribution with n=50 and p=0.03. P(X ≤ 2) = P(X=0) + P(X=1) + P(X=2) = C(50,0) × (0.03)^0 × (0.97)^50 + C(50,1) × (0.03)^1 × (0.97)^49 + C(50,2) × (0.03)^2 × (0.97)^48 ≈ 0.2231 + 0.3453 + 0.2379 = 0.6063 or about 60.63%.
</div>

### Qn 13: In the Monty Hall problem, you're on a game show with three doors. Behind one door is a car; behind the others are goats. You pick a door. The host, who knows what's behind each door, opens one of the other doors to reveal a goat. He then offers you the chance to switch your choice to the remaining unopened door. What is the probability of winning the car if you switch?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 2/3
<br><strong>Explanation:</strong> Initially, you have a 1/3 probability of choosing the car and a 2/3 probability of choosing a goat. If you initially chose the car (probability 1/3), switching will always make you lose. If you initially chose a goat (probability 2/3), the host will reveal the other goat, and switching will always make you win. Therefore, the probability of winning by switching is 2/3.
</div>

### Qn 14: A researcher is testing a new drug. In reality, the drug has no effect, but the researcher will conclude it works if the p-value is less than 0.05. What is the probability that the researcher incorrectly concludes the drug works?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.05
<br><strong>Explanation:</strong> The p-value is the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true. Here, the null hypothesis is that the drug has no effect (which is actually true). By definition, the probability of getting a p-value below 0.05 when the null hypothesis is true (Type I error) is 0.05 or 5%.
</div>

### Qn 15: A database has 1,000,000 records, and a data scientist estimates that 50 records are corrupted. If the data scientist randomly samples 100 records for manual inspection, what is the probability of finding at least one corrupted record?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.4988
<br><strong>Explanation:</strong> The probability of selecting a corrupted record is 50/1,000,000 = 0.00005. The probability of not finding any corrupted records in 100 samples is (1 - 0.00005)^100 ≈ 0.9950. Therefore, the probability of finding at least one corrupted record is 1 - 0.9950 = 0.0050 or 0.5%.
</div>

### Qn 16: In a certain city, 60% of days are sunny, 30% are cloudy, and 10% are rainy. The probability of a traffic jam is 0.1 on sunny days, 0.3 on cloudy days, and 0.5 on rainy days. If there is a traffic jam today, what is the probability that it is a sunny day?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.33
<br><strong>Explanation:</strong> Using Bayes' theorem: P(Sunny|Traffic Jam) = [P(Traffic Jam|Sunny) × P(Sunny)] / P(Traffic Jam). P(Traffic Jam) = 0.1 × 0.6 + 0.3 × 0.3 + 0.5 × 0.1 = 0.06 + 0.09 + 0.05 = 0.2. Therefore, P(Sunny|Traffic Jam) = (0.1 × 0.6) / 0.2 = 0.06/0.2 = 0.3 or 30%.
</div>

### Qn 17: Five fair six-sided dice are rolled. What is the probability that all five dice show different numbers?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.0926
<br><strong>Explanation:</strong> Total number of possible outcomes when rolling 5 dice is 6^5 = 7,776. For all dice to show different numbers, we can arrange 5 different numbers from the set {1,2,3,4,5,6} in 6!/1! = 720 ways. Therefore, the probability is 720/7,776 = 0.0926 or about 9.26%.
</div>

### Qn 18: A data center has 5 servers, each with a 1% probability of failing in a given day, independently of the others. What is the probability that at least one server fails today?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.0490
<br><strong>Explanation:</strong> The probability that a specific server doesn't fail is 0.99. The probability that all servers don't fail is (0.99)^5 ≈ 0.9510. Therefore, the probability that at least one server fails is 1 - 0.9510 = 0.0490 or about 4.9%.
</div>

### Qn 19: In a random sample of 20 people, what is the probability that at least 2 people were born in the same month of the year (assuming uniform distribution of birth months)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.9139
<br><strong>Explanation:</strong> The probability that all 20 people were born in different months is 0 since there are only 12 months. The probability that no two people share the same birth month in a sample of 12 or fewer is calculated using the birthday problem formula for 12 months: 1 - P(no matching months) = 1 - (12!/12^n × (12-n)!) for n=12. For n>12, the probability of at least one match is 1.
</div>

### Qn 20: A biased coin has an unknown probability p of landing heads. After 10 flips, you observe 7 heads. Using a uniform prior distribution for p, what is the expected value of p according to Bayesian analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.636
<br><strong>Explanation:</strong> With a uniform prior distribution (Beta(1,1)) and 7 heads out of 10 flips, the posterior distribution is Beta(1+7, 1+3) = Beta(8, 4). The expected value of a Beta(α, β) distribution is α/(α+β). So the expected value of p is 8/(8+4) = 8/12 = 2/3 ≈ 0.667, which is closest to 0.636.
</div>

### Qn 21: In a multiple-choice test with 5 questions, each question has 4 options with exactly one correct answer. If a student guesses randomly on all questions, what is the probability of getting at least 3 questions correct?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.0537
<br><strong>Explanation:</strong> The probability of getting a single question correct by random guessing is 1/4 = 0.25. Using the binomial distribution with n=5 and p=0.25: P(X ≥ 3) = P(X=3) + P(X=4) + P(X=5) = C(5,3) × (0.25)^3 × (0.75)^2 + C(5,4) × (0.25)^4 × (0.75)^1 + C(5,5) × (0.25)^5 × (0.75)^0 ≈ 0.0439 + 0.0073 + 0.0010 = 0.0522 or about 5.22%.
</div>

### Qn 22: In a hypergeometric distribution scenario, a shipment of 100 electronic components contains 8 defective parts. If 10 components are randomly selected without replacement for inspection, what is the probability of finding exactly 1 defective component?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.3816
<br><strong>Explanation:</strong> Using the hypergeometric probability mass function: P(X=1) = [C(8,1) × C(92,9)] / C(100,10) = [8 × 1,742,281,695] / 17,310,309,728 = 13,938,253,560 / 17,310,309,728 ≈ 0.3816 or about 38.16%.
</div>

### Qn 23: A fair six-sided die is rolled 10 times. What is the probability of getting exactly 2 sixes?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.2907
<br><strong>Explanation:</strong> This follows a binomial distribution with n=10 and p=1/6. P(X=2) = C(10,2) × (1/6)^2 × (5/6)^8 = 45 × (1/36) × (1,679,616/1,679,616) = 45/36 × 0.2323 ≈ 1.25 × 0.2323 = 0.2904 or about 29.04%.
</div>

### Qn 24: In a large city, 45% of residents prefer public transportation, 35% prefer driving, and 20% prefer cycling. If three residents are randomly selected, what is the probability that at least one of them prefers cycling?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.488
<br><strong>Explanation:</strong> The probability that a selected resident does not prefer cycling is 1 - 0.2 = 0.8. The probability that none of the three selected residents prefers cycling is (0.8)^3 = 0.512. Therefore, the probability that at least one prefers cycling is 1 - 0.512 = 0.488 or 48.8%.
</div>

### Qn 25: A genetics researcher is studying a trait that is determined by two alleles. The dominant allele A occurs with probability 0.7 and the recessive allele a with probability 0.3. Assuming Hardy-Weinberg equilibrium, what is the probability of a randomly selected individual having the genotype Aa?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.42
<br><strong>Explanation:</strong> Under Hardy-Weinberg equilibrium, the probability of genotype Aa is 2pq, where p is the probability of allele A and q is the probability of allele a. So P(Aa) = 2 × 0.7 × 0.3 = 0.42 or 42%.
</div>

### Qn 26: In a Poisson process where events occur at an average rate of 3 per hour, what is the probability that exactly 2 events occur in a 1-hour period?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.224
<br><strong>Explanation:</strong> For a Poisson distribution with parameter λ=3, the probability mass function gives P(X=2) = e^(-λ) × λ^2 / 2! = e^(-3) × 3^2 / 2 = e^(-3) × 9 / 2 = 0.0498 × 4.5 ≈ 0.224 or about 22.4%.
</div>

### Qn 27: A data scientist is analyzing user engagement on a website. If the probability distribution of the number of pages viewed by a visitor follows a geometric distribution with p=0.2, what is the probability that a visitor views exactly 5 pages before leaving the site?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.082
<br><strong>Explanation:</strong> For a geometric distribution with parameter p=0.2, the probability mass function gives P(X=5) = p(1-p)^(k-1) = 0.2 × (0.8)^4 = 0.2 × 0.4096 = 0.08192 ≈ 0.082 or about 8.2%.
</div>

### Qn 28: A medical test for a disease has sensitivity (true positive rate) of 90% and specificity (true negative rate) of 95%. In a population where 2% of people have the disease, what is the positive predictive value (probability that a person with a positive test result actually has the disease)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.27
<br><strong>Explanation:</strong> Using Bayes' theorem: PPV = P(Disease|Positive test) = [P(Positive test|Disease) × P(Disease)] / P(Positive test). P(Positive test) = P(Positive test|Disease) × P(Disease) + P(Positive test|No Disease) × P(No Disease) = 0.9 × 0.02 + 0.05 × 0.98 = 0.018 + 0.049 = 0.067. Therefore, PPV = (0.9 × 0.02) / 0.067 = 0.018/0.067 ≈ 0.27 or 27%.
</div>

### Qn 29: In a lottery where 5 numbers are drawn from 1 to 49 without replacement, what is the probability of matching exactly 3 numbers on a single ticket?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.015
<br><strong>Explanation:</strong> Total number of possible 5-number combinations is C(49,5) = 1,906,884. Ways to match exactly 3 numbers out of 5: You must match 3 of the winning numbers [C(5,3) = 10] and 2 of the non-winning numbers [C(44,2) = 946]. So favorable outcomes = 10 × 946 = 9,460. Probability = 9,460/1,906,884 ≈ 0.00496 or about 0.5%.
</div>

### Qn 30: In a random sample from a normal distribution with mean 100 and standard deviation 15, what is the probability that a single observation exceeds 125?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.0478
<br><strong>Explanation:</strong> Standardizing, z = (125 - 100)/15 = 1.67. The probability P(X > 125) = P(Z > 1.67) ≈ 0.0475 or about 4.75%, which is closest to 0.0478.
</div>

### Qn 31: In a randomized controlled trial, patients are randomly assigned to either treatment or control groups with equal probability. If 10 patients are enrolled, what is the probability that exactly 5 are assigned to the treatment group?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.246
<br><strong>Explanation:</strong> This follows a binomial distribution with n=10 and p=0.5. P(X=5) = C(10,5) × (0.5)^5 × (0.5)^5 = 252 × (0.5)^10 = 252/1024 = 0.2461 or about 24.6%.
</div>

### Qn 32: A data scientist runs 20 independent A/B tests, each with a 5% false positive rate (Type I error). What is the probability of observing at least one false positive result across all tests if none of the tested hypotheses are actually true?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 0.64
<br><strong>Explanation:</strong> The probability of not observing a false positive in a single test is 1 - 0.05 = 0.95. The probability of not observing any false positives in 20 independent tests is (0.95)^20 ≈ 0.358. Therefore, the probability of observing at least one false positive is 1 - 0.358 = 0.642 or about 64.2%.
</div>

### Qn 33: Two fair six-sided dice are rolled. Given that the sum of the dice is greater than 7, what is the probability that at least one die shows a 6?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 7/12
<br><strong>Explanation:</strong> The possible outcomes for sum > 7 are: (2,6), (3,5), (3,6), (4,4), (4,5), (4,6), (5,3), (5,4), (5,5), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6) - a total of 15 outcomes. Of these, 11 outcomes include at least one 6: (2,6), (3,6), (4,6), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6). Therefore, the probability is 11/15 = 11/15 or about 73.3%.
</div>


## Python Advanced

### Qn 01: What is the time complexity of inserting an element at the beginning of a Python list?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> O(n)
<br><strong>Explanation:</strong> Inserting at the beginning of a Python list requires shifting all elements, hence O(n).
</div>

### Qn 02: Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> NumPy arrays
<br><strong>Explanation:</strong> NumPy arrays are memory efficient and optimized for numerical operations.
</div>

### Qn 03: Which Python library provides decorators and context managers to handle retries with exponential backoff?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> tenacity
<br><strong>Explanation:</strong> Tenacity provides powerful retry strategies including exponential backoff.
</div>

### Qn 04: What is a key difference between multiprocessing and threading in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Processes can utilize multiple CPUs
<br><strong>Explanation:</strong> Due to the GIL, threads are limited; multiprocessing uses separate memory space and cores.
</div>

### Qn 05: What is the purpose of Python's `__slots__` declaration?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Reduce memory usage by preventing dynamic attribute creation
<br><strong>Explanation:</strong> `__slots__` limits attribute assignment and avoids `__dict__` overhead.
</div>

### Qn 06: What will `functools.lru_cache` do?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Cache function output to speed up subsequent calls
<br><strong>Explanation:</strong> `lru_cache` stores results of expensive function calls for reuse.
</div>

### Qn 07: What does the `@staticmethod` decorator do in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Defines a method that takes no self or cls argument
<br><strong>Explanation:</strong> `@staticmethod` defines a method that does not receive an implicit first argument.
</div>

### Qn 08: How can you profile memory usage in a Python function?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Using tracemalloc
<br><strong>Explanation:</strong> `tracemalloc` tracks memory allocations in Python.
</div>

### Qn 09: Which built-in function returns the identity of an object?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> id()
<br><strong>Explanation:</strong> `id()` returns the identity (memory address) of an object.
</div>

### Qn 10: What happens when you use the `is` operator between two equal strings in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It compares object identity
<br><strong>Explanation:</strong> `is` checks whether two variables point to the same object, not if their values are equal.
</div>


## Python General

### Qn 01: What is the output of `len('Python')`?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 6
<br><strong>Explanation:</strong> The string 'Python' has 6 characters.
</div>

### Qn 02: What is the output of `type([])` in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> <class 'list'>
<br><strong>Explanation:</strong> `[]` represents an empty list in Python.
</div>

### Qn 03: Which data structure allows duplicate elements?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> List
<br><strong>Explanation:</strong> Lists in Python can contain duplicate elements.
</div>

### Qn 04: What is the result of `5 // 2`?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 2
<br><strong>Explanation:</strong> `//` is floor division; it returns the largest whole number less than or equal to the result.
</div>

### Qn 05: Which of the following is a mutable data type?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> list
<br><strong>Explanation:</strong> Lists are mutable in Python, allowing modifications.
</div>

### Qn 06: How do you insert a comment in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> # comment
<br><strong>Explanation:</strong> Python uses the `#` symbol to indicate a comment.
</div>

### Qn 07: Which of the following is used to handle exceptions in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> try-except
<br><strong>Explanation:</strong> Python uses `try-except` blocks to handle exceptions.
</div>

### Qn 08: What keyword is used to define a function in Python?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> def
<br><strong>Explanation:</strong> The `def` keyword is used to define functions in Python.
</div>

### Qn 09: What does `range(3)` return?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> [0, 1, 2]
<br><strong>Explanation:</strong> `range(3)` generates numbers starting from 0 up to (but not including) 3.
</div>


## SQL General

### Qn 01: Which SQL statement is used to extract data from a database?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> SELECT
<br><strong>Explanation:</strong> The SELECT statement is used to extract data from a database table.
</div>

### Qn 02: Which SQL clause is used to filter records?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> WHERE
<br><strong>Explanation:</strong> The WHERE clause is used to filter records based on specific conditions.
</div>

### Qn 03: What does the COUNT() function do in SQL?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Counts non-NULL rows
<br><strong>Explanation:</strong> COUNT() returns the number of non-NULL values in a specified column.
</div>

### Qn 04: Which SQL keyword is used to sort the result-set?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> ORDER BY
<br><strong>Explanation:</strong> ORDER BY is used to sort the results of a SELECT query.
</div>

### Qn 05: Which command is used to remove all records from a table in SQL without deleting the table?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> TRUNCATE
<br><strong>Explanation:</strong> TRUNCATE removes all records from a table but retains the table structure.
</div>

### Qn 06: Which SQL clause is used with aggregate functions to group result-set by one or more columns?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> GROUP BY
<br><strong>Explanation:</strong> GROUP BY groups rows that have the same values into summary rows.
</div>

### Qn 07: Which SQL keyword is used to retrieve only distinct values?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> DISTINCT
<br><strong>Explanation:</strong> DISTINCT is used to return only different (distinct) values.
</div>

### Qn 08: Which of the following is a DDL command?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> CREATE
<br><strong>Explanation:</strong> CREATE is a DDL (Data Definition Language) command used to create a new table or database.
</div>

### Qn 09: What does the SQL INNER JOIN keyword do?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Returns rows when there is a match in both tables
<br><strong>Explanation:</strong> INNER JOIN selects records that have matching values in both tables.
</div>

### Qn 10: What will the result of the query 'SELECT * FROM employees WHERE department IS NULL;' be?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It selects employees with NULL department
<br><strong>Explanation:</strong> IS NULL checks for columns that contain NULL values.
</div>


## SQL Sqlite

### Qn 01: What is the purpose of the `WITHOUT ROWID` clause in SQLite?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To create a table without the implicit ROWID column
<br><strong>Explanation:</strong> `WITHOUT ROWID` creates a table without the implicit `ROWID`, useful for certain optimizations.
</div>

### Qn 02: Which function would you use in SQLite to get the current timestamp?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> CURRENT_TIMESTAMP
<br><strong>Explanation:</strong> `CURRENT_TIMESTAMP` returns the current date and time in SQLite.
</div>

### Qn 03: What is the default data type of a column in SQLite if not specified?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> NONE
<br><strong>Explanation:</strong> If no type is specified, SQLite assigns it an affinity of NONE.
</div>

### Qn 04: How are boolean values stored in SQLite?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> As 1 and 0 integers
<br><strong>Explanation:</strong> SQLite does not have a separate BOOLEAN type; it uses integers 1 (true) and 0 (false).
</div>

### Qn 05: Which of the following is true about SQLite's `VACUUM` command?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> It compacts the database file
<br><strong>Explanation:</strong> `VACUUM` rebuilds the database file to defragment it and reduce its size.
</div>

### Qn 06: Which SQLite command lists all tables in the database?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> SELECT * FROM sqlite_master WHERE type='table'
<br><strong>Explanation:</strong> SQLite uses the `sqlite_master` table to store metadata about the database, including table names.
</div>

### Qn 07: Which SQLite command allows you to see the schema of a table?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> .schema
<br><strong>Explanation:</strong> `.schema` is a command in the SQLite shell that shows the schema for tables.
</div>

### Qn 08: How does SQLite handle foreign key constraints by default?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> They are off by default and must be enabled
<br><strong>Explanation:</strong> SQLite supports foreign keys, but enforcement must be enabled with `PRAGMA foreign_keys = ON`.
</div>

### Qn 09: How does SQLite implement AUTOINCREMENT?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Using INTEGER PRIMARY KEY
<br><strong>Explanation:</strong> SQLite uses `INTEGER PRIMARY KEY AUTOINCREMENT` to create an auto-incrementing ID.
</div>

### Qn 10: What pragma statement turns on write-ahead logging in SQLite?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> PRAGMA journal_mode = WAL
<br><strong>Explanation:</strong> `PRAGMA journal_mode = WAL` enables write-ahead logging in SQLite.
</div>


## Statistics

### Qn 01: What does the p-value represent in hypothesis testing?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Probability of obtaining test results at least as extreme as the results actually observed
<br><strong>Explanation:</strong> The p-value quantifies the evidence against the null hypothesis. A small p-value suggests the observed data is unlikely under the null hypothesis.
</div>

### Qn 02: What is the main difference between population and sample in statistics?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Sample is a subset of population
<br><strong>Explanation:</strong> A population includes all elements from a set of data, while a sample consists of one or more observations drawn from the population.
</div>

### Qn 03: What does standard deviation measure?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Spread of data
<br><strong>Explanation:</strong> Standard deviation measures the amount of variation or dispersion of a set of values.
</div>

### Qn 04: In a normal distribution, what percentage of data lies within one standard deviation of the mean?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> 68%
<br><strong>Explanation:</strong> In a normal distribution, approximately 68% of the data falls within one standard deviation of the mean.
</div>

### Qn 05: Which of the following measures of central tendency is not affected by extreme values?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Median
<br><strong>Explanation:</strong> Median is the middle value and is not affected by extremely large or small values, unlike the mean.
</div>


## Timeseries

### Qn 01: What does the 'AR' component in ARIMA represent, and how does it capture patterns in time series data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Autoregressive - using past values to predict future values
<br><strong>Explanation:</strong> The 'AR' in ARIMA stands for Autoregressive, which means the model uses past values of the time series to predict future values. Specifically, an AR(p) model uses p previous time steps as predictors. For example, in an AR(2) model, the current value is predicted using a linear combination of the previous two values, plus an error term. This component is particularly useful for capturing momentum or inertia in time series where recent values influence future values.
</div>

### Qn 02: What does the 'I' component in ARIMA represent, and why is it necessary?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Integrated - differencing to achieve stationarity
<br><strong>Explanation:</strong> The 'I' in ARIMA stands for Integrated, which refers to differencing the time series to achieve stationarity. Many time series have trends or seasonal patterns that make them non-stationary. The 'd' parameter in ARIMA(p,d,q) indicates how many times the data needs to be differenced to achieve stationarity. For example, if d=1, we take the difference between consecutive observations. This transformation is necessary because ARIMA models assume the underlying process is stationary, meaning its statistical properties do not change over time.
</div>

### Qn 03: What does the 'MA' component in ARIMA represent, and how does it differ from AR?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Moving Average - using past forecast errors in the model
<br><strong>Explanation:</strong> The 'MA' in ARIMA stands for Moving Average, which incorporates past forecast errors (residuals) into the model rather than past values of the time series itself. An MA(q) model uses the previous q forecast errors as predictors. This differs fundamentally from AR, which uses the actual past values. MA components capture the short-term reactions to past shocks or random events in the system. For example, an MA(1) model would use the forecast error from the previous time step to adjust the current prediction.
</div>

### Qn 04: How do you interpret the parameters p, d, and q in ARIMA(p,d,q)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> p = AR order, d = differencing order, q = MA order
<br><strong>Explanation:</strong> In ARIMA(p,d,q), p represents the order of the autoregressive (AR) component, indicating how many lagged values of the series are included in the model. A higher p means more past values are used for prediction. The parameter d represents the degree of differencing required to make the series stationary, with d=1 meaning first difference, d=2 meaning second difference, etc. Finally, q is the order of the moving average (MA) component, indicating how many lagged forecast errors are included in the model. Together, these parameters define the structure of the ARIMA model and must be carefully selected based on the characteristics of the time series.
</div>

### Qn 05: What is the key assumption that must be satisfied before applying ARIMA models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> The time series must be stationary
<br><strong>Explanation:</strong> The fundamental assumption for ARIMA models is that the time series is stationary or can be made stationary through differencing. A stationary time series has constant mean, variance, and autocorrelation structure over time. Without stationarity, the model cannot reliably learn patterns from the data. This is why the 'I' (Integrated) component exists in ARIMA - to transform non-stationary data through differencing. Analysts typically use statistical tests like the Augmented Dickey-Fuller (ADF) test to check for stationarity before applying ARIMA models.
</div>

### Qn 06: How can you determine the appropriate values for p and q in an ARIMA(p,d,q) model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> By examining ACF and PACF plots
<br><strong>Explanation:</strong> The appropriate values for p and q in an ARIMA model can be determined by examining the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of the stationary time series. For identifying the AR order (p), look for significant spikes in the PACF that cut off after lag p. For the MA order (q), look for significant spikes in the ACF that cut off after lag q. Additionally, information criteria like AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) can be used to compare different model specifications and select the best combination of parameters.
</div>

### Qn 07: What is the purpose of the Augmented Dickey-Fuller (ADF) test in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To test for stationarity
<br><strong>Explanation:</strong> The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary or not. The null hypothesis of the test is that the time series contains a unit root, implying it is non-stationary. If the p-value from the test is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the series is stationary. This test is crucial before applying ARIMA models because stationarity is a key assumption. The test includes lags of the differenced series to account for serial correlation, making it more robust than the simple Dickey-Fuller test.
</div>

### Qn 08: What is the difference between ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> ACF measures correlation between series and lagged values accounting for intermediate lags, PACF removes indirect correlation effects
<br><strong>Explanation:</strong> The ACF (Autocorrelation Function) measures the correlation between a time series and its lagged values, including both direct and indirect effects. It shows the correlation at each lag without controlling for correlations at shorter lags. In contrast, the PACF (Partial Autocorrelation Function) measures the correlation between a time series and its lagged values while controlling for the values of the time series at all shorter lags. This effectively removes the indirect correlation effects, showing only the direct relationship between observations separated by a specific lag. ACF helps identify MA(q) order, while PACF helps identify AR(p) order in ARIMA modeling.
</div>

### Qn 09: What additional component does SARIMAX add compared to ARIMA?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Seasonal components and exogenous variables
<br><strong>Explanation:</strong> SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) extends ARIMA by adding two important capabilities. First, it incorporates seasonal components, allowing the model to capture repeating patterns that occur at fixed intervals (like daily, weekly, or yearly seasonality). The seasonal component is specified with parameters (P,D,Q)m, where m is the seasonal period. Second, SARIMAX allows for exogenous variables (the 'X' part), which are external factors that can influence the time series but are not part of the series itself. These could include variables like temperature affecting energy consumption, or promotions affecting sales. This makes SARIMAX much more versatile than standard ARIMA for real-world applications with seasonal patterns and external influences.
</div>

### Qn 10: How are seasonality parameters represented in a SARIMA model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> As (P,D,Q)m where m is the seasonal period
<br><strong>Explanation:</strong> In a SARIMA (Seasonal ARIMA) model, seasonality parameters are represented as (P,D,Q)m, where P is the seasonal autoregressive order, D is the seasonal differencing order, Q is the seasonal moving average order, and m is the number of periods in each season (the seasonal period). For example, in monthly data with yearly seasonality, m would be 12. In a SARIMA(1,1,1)(1,1,1)12 model, the non-seasonal components are (1,1,1) and the seasonal components are (1,1,1)12. The seasonal components operate at lag m, 2m, etc., capturing patterns that repeat every m periods. Seasonal differencing (D) involves subtracting the value from m periods ago, helping to remove seasonal non-stationarity.
</div>

### Qn 11: What does it mean when we say a time series exhibits 'stationarity'?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Its statistical properties remain constant over time
<br><strong>Explanation:</strong> A stationary time series has statistical properties that remain constant over time. Specifically, it has a constant mean, constant variance, and a constant autocorrelation structure. This means the process generating the time series is in statistical equilibrium. Stationarity is a crucial assumption for many time series models, including ARIMA, because it ensures that patterns learned from historical data will continue to be valid in the future. Non-stationary series might have trends (changing mean) or heteroscedasticity (changing variance), which can lead to unreliable forecasts if not properly addressed through transformations like differencing or variance stabilization.
</div>

### Qn 12: What is the purpose of differencing in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To remove trends and achieve stationarity
<br><strong>Explanation:</strong> Differencing in time series analysis involves computing the differences between consecutive observations. The primary purpose is to remove trends and achieve stationarity, which is a key requirement for ARIMA modeling. First-order differencing (d=1) can eliminate linear trends by calculating Yt - Yt-1. If the series still shows non-stationarity after first differencing, second-order differencing (d=2) can be applied to remove quadratic trends. However, over-differencing can introduce unnecessary complexity and artificial patterns, so it's important to use statistical tests like the ADF test to determine the appropriate level of differencing needed.
</div>

### Qn 13: What is seasonal differencing and when should it be applied?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Calculating differences between observations separated by the seasonal period, applied when there's seasonal non-stationarity
<br><strong>Explanation:</strong> Seasonal differencing involves calculating differences between observations separated by the seasonal period (e.g., 12 months for monthly data with yearly seasonality). It's represented by the D parameter in SARIMA models and is applied when the time series exhibits seasonal non-stationarity, meaning the seasonal pattern changes over time. For example, with monthly data, seasonal differencing would compute Yt - Yt-12. This helps remove repeating seasonal patterns just as regular differencing removes trends. You should apply seasonal differencing when visual inspection shows persistent seasonal patterns after regular differencing, or when seasonal unit root tests indicate seasonal non-stationarity.
</div>

### Qn 14: What are residuals in the context of ARIMA modeling, and why are they important?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> The differences between observed and predicted values, important for diagnostic checking
<br><strong>Explanation:</strong> In ARIMA modeling, residuals are the differences between the observed values and the values predicted by the model. They represent the part of the data that the model couldn't explain. Residuals are crucial for diagnostic checking because a well-fitted ARIMA model should have residuals that resemble white noise - they should be uncorrelated, have zero mean, constant variance, and follow a normal distribution. If patterns remain in the residuals, it suggests the model hasn't captured all the systematic information in the time series. Common residual diagnostics include ACF/PACF plots of residuals, the Ljung-Box test for autocorrelation, and Q-Q plots for normality checking.
</div>

### Qn 15: What does the Ljung-Box test evaluate in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Whether residuals exhibit autocorrelation
<br><strong>Explanation:</strong> The Ljung-Box test is a statistical test used to evaluate whether residuals from a time series model exhibit autocorrelation. The null hypothesis is that the residuals are independently distributed (i.e., no autocorrelation). If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the residuals contain significant autocorrelation, suggesting the model hasn't captured all the patterns in the data. The test examines multiple lags simultaneously, making it more comprehensive than just looking at individual autocorrelation values. A good ARIMA model should have residuals that pass the Ljung-Box test, indicating they approximate white noise.
</div>

### Qn 16: What is the primary difference between ARMA and ARIMA models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> ARIMA includes differencing for non-stationary data, while ARMA requires stationary data
<br><strong>Explanation:</strong> The primary difference between ARMA (AutoRegressive Moving Average) and ARIMA (AutoRegressive Integrated Moving Average) models is that ARIMA includes a differencing step (the 'I' component) to handle non-stationary data. ARMA models combine autoregressive (AR) and moving average (MA) components but assume that the time series is already stationary. ARIMA extends this by first differencing the data d times to achieve stationarity before applying the ARMA model. This makes ARIMA more versatile for real-world time series that often contain trends. Essentially, an ARIMA(p,d,q) model is equivalent to applying an ARMA(p,q) model to a time series after differencing it d times.
</div>

### Qn 17: What is meant by the 'order of integration' in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> The number of times a series needs to be differenced to achieve stationarity
<br><strong>Explanation:</strong> The 'order of integration' refers to the number of times a time series needs to be differenced to achieve stationarity. It's represented by the parameter d in ARIMA(p,d,q) models. A series that requires differencing once (d=1) to become stationary is said to be integrated of order 1, or I(1). Similarly, a series requiring two differences is I(2). A naturally stationary series is I(0). The concept is important because it quantifies how persistent trends are in the data. Most economic and business time series are I(1), meaning they have stochastic trends that can be removed with first differencing. The order of integration can be determined using unit root tests like the Augmented Dickey-Fuller test.
</div>

### Qn 18: What is the purpose of the Box-Jenkins methodology in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> A systematic approach to identify, estimate, and validate ARIMA models
<br><strong>Explanation:</strong> The Box-Jenkins methodology is a systematic approach to identify, estimate, and validate ARIMA models for time series forecasting. It consists of three main stages: identification, estimation, and diagnostic checking. In the identification stage, you determine appropriate values for p, d, and q by analyzing ACF/PACF plots and using stationarity tests. In the estimation stage, you fit the selected ARIMA model to the data and estimate its parameters. In the diagnostic checking stage, you analyze residuals to ensure they resemble white noise and refine the model if needed. Box-Jenkins emphasizes iterative model building, where you cycle through these stages until you find an adequate model. This methodical approach helps ensure that the final model captures the data's patterns efficiently.
</div>

### Qn 19: What information criterion is commonly used to select between different ARIMA models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> AIC (Akaike Information Criterion)
<br><strong>Explanation:</strong> The AIC (Akaike Information Criterion) is commonly used to select between different ARIMA models. It balances model fit against complexity by penalizing models with more parameters. The formula is AIC = -2log(L) + 2k, where L is the likelihood of the model and k is the number of parameters. A lower AIC value indicates a better model. When comparing ARIMA models with different p, d, and q values, analysts typically choose the model with the lowest AIC. Other similar criteria include BIC (Bayesian Information Criterion), which penalizes model complexity more heavily. These criteria help prevent overfitting by ensuring that additional parameters are only included if they substantially improve the model's fit to the data.
</div>

### Qn 20: In the context of ARIMA residual analysis, what should a Q-Q plot ideally show?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Points falling approximately along a straight line, indicating normally distributed residuals
<br><strong>Explanation:</strong> In ARIMA residual analysis, a Q-Q (Quantile-Quantile) plot should ideally show points falling approximately along a straight line. This indicates that the residuals follow a normal distribution, which is an assumption for valid statistical inference in ARIMA modeling. The Q-Q plot compares the quantiles of the residuals against the quantiles of a theoretical normal distribution. Deviations from the straight line suggest non-normality: a sigmoidal pattern indicates skewness, while an S-shaped curve suggests heavy or light tails compared to a normal distribution. Serious deviations might indicate model misspecification or the presence of outliers that could affect the reliability of confidence intervals and hypothesis tests for the model parameters.
</div>

### Qn 21: What is the meaning of the 'exogenous variables' in the context of SARIMAX models?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> External predictor variables that influence the time series but are not influenced by it
<br><strong>Explanation:</strong> In SARIMAX models, exogenous variables (the 'X' part) are external predictor variables that influence the time series being modeled but are not influenced by it. These are independent variables that provide additional information beyond what's contained in the past values of the time series itself. For example, when forecasting electricity demand, temperature might be an exogenous variable since it affects demand but isn't affected by it. Unlike the autoregressive components that use the series' own past values, exogenous variables inject outside information into the model. This can significantly improve forecast accuracy when the time series is known to be affected by measurable external factors. Mathematically, exogenous variables enter the SARIMAX equation as a regression component.
</div>

### Qn 22: Why might you perform a Box-Cox transformation before applying an ARIMA model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To stabilize variance and make the data more normally distributed
<br><strong>Explanation:</strong> A Box-Cox transformation is often performed before applying an ARIMA model to stabilize variance and make the data more normally distributed. Many time series exhibit heteroscedasticity (changing variance over time) or skewness, which can violate ARIMA assumptions. The Box-Cox transformation is a family of power transformations defined by the parameter λ: when λ=0, it's equivalent to a log transformation; when λ=1, it's essentially the original data (with a shift). The optimal λ value can be determined by maximizing the log-likelihood function. This transformation helps make the time series' variance more constant across time and its distribution more symmetric, leading to more reliable parameter estimates and prediction intervals in the ARIMA model.
</div>

### Qn 23: What does it mean when an ARIMA model is said to be 'invertible'?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> The MA component can be rewritten as an infinite AR process
<br><strong>Explanation:</strong> In time series analysis, when an ARIMA model is said to be 'invertible,' it means that its Moving Average (MA) component can be rewritten as an infinite Autoregressive (AR) process. This property ensures that the MA coefficients decrease in impact as we go further back in time, allowing the process to be approximated by a finite AR model. Invertibility is a mathematical property that ensures a unique MA representation and stable forecasting. Technically, for invertibility, the roots of the MA polynomial must lie outside the unit circle. Without invertibility, different MA models could produce identical autocorrelation patterns, making identification problematic. Invertibility is analogous to stationarity for AR processes and is checked during the model estimation phase.
</div>

### Qn 24: What is the difference between strong and weak stationarity in time series?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Weak stationarity requires constant mean and variance and time-invariant autocorrelation; strong stationarity requires the entire distribution to be time-invariant
<br><strong>Explanation:</strong> The distinction between strong (strict) and weak stationarity lies in how much of the data's statistical properties must remain constant over time. Weak stationarity, which is usually sufficient for ARIMA modeling, requires only that the mean and variance remain constant and that the autocorrelation function depends only on the lag between points, not their absolute position in time. In contrast, strong stationarity is more demanding, requiring that the entire joint probability distribution of the process remains unchanged when shifted in time. This means all higher moments (not just the first two) must be constant, and all multivariate distributions (not just bivariate correlations) must be time-invariant. In practice, analysts typically work with weak stationarity because it's easier to test for and sufficient for many applications.
</div>

### Qn 25: What is the primary purpose of the KPSS test in time series analysis?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> To test for stationarity with a null hypothesis of stationarity
<br><strong>Explanation:</strong> The KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test is used to test for stationarity in time series analysis, but unlike the ADF test, its null hypothesis is that the series is stationary. This reversal makes it a complementary test to ADF, which has a null hypothesis of non-stationarity. Using both tests together provides stronger evidence: if the ADF test rejects its null and the KPSS fails to reject its null, you have consistent evidence of stationarity. The KPSS test specifically tests whether the series can be described as stationary around a deterministic trend or has a unit root. A low p-value leads to rejecting the null, suggesting non-stationarity. This test is particularly useful for distinguishing between trend-stationary processes and difference-stationary processes.
</div>

### Qn 26: What does Facebook Prophet use to model seasonality in time series data?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Fourier series for multiple seasonal periods
<br><strong>Explanation:</strong> Facebook Prophet uses Fourier series to model seasonality in time series data. This approach represents seasonal patterns as a sum of sine and cosine terms of different frequencies, allowing for flexible modeling of complex seasonal patterns. Prophet can simultaneously model multiple seasonal periods (e.g., daily, weekly, and yearly seasonality) by using different Fourier series for each. The number of terms in each Fourier series (specified by the 'order' parameter) controls the flexibility of the seasonal component - higher orders capture more complex patterns but risk overfitting. This approach is particularly powerful because it can handle irregular time series and missing data better than traditional seasonal ARIMA models, which require regular time intervals.
</div>

### Qn 27: What are the three main components of a Facebook Prophet model?

<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
<strong>Answer:</strong> Trend, seasonality, and holidays/events
<br><strong>Explanation:</strong> Facebook Prophet decomposes time series into three main components: trend, seasonality, and holidays/events. The trend component captures non-periodic changes, and can be modeled as either linear or logistic growth with automatic changepoint detection to accommodate trend changes. The seasonality component captures periodic patterns using Fourier series, and can simultaneously model multiple seasonal patterns (e.g., daily, weekly, annual). The holidays/events component accounts for irregular schedules and events that affect the time series but don't follow a seasonal pattern. Users can provide a custom list of holidays or events with their dates. By modeling these components separately and then adding them together, Prophet creates an interpretable forecast that can be easily understood and adjusted by analysts.
</div>
