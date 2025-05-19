# Data_Analysis Quiz

## Table of Contents
- [Qn 01: What technique would you use to handle high-dimensional sparse data when performing PCA?](#1)
- [Qn 02: What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?](#2)
- [Qn 03: When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?](#3)
- [Qn 04: Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?](#4)
- [Qn 05: What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?](#5)
- [Qn 06: When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?](#6)
- [Qn 07: Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?](#7)
- [Qn 08: What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?](#8)
- [Qn 09: What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?](#9)
- [Qn 10: Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?](#10)
- [Qn 11: When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?](#11)
- [Qn 12: What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?](#12)
- [Qn 13: Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?](#13)
- [Qn 14: What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?](#14)
- [Qn 15: What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?](#15)
- [Qn 16: Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?](#16)
- [Qn 17: When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?](#17)
- [Qn 18: What's the most statistically sound approach to handle heteroscedasticity in a regression model?](#18)
- [Qn 19: Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?](#19)
- [Qn 20: What technique is most appropriate for analyzing complex network data with community structures?](#20)
- [Qn 21: What's the most robust approach to handle concept drift in a production machine learning system?](#21)
- [Qn 22: Which method is most appropriate for interpretable anomaly detection in high-dimensional data?](#22)
- [Qn 23: When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?](#23)
- [Qn 24: What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?](#24)
- [Qn 25: Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?](#25)
- [Qn 26: What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?](#26)
- [Qn 27: Which approach correctly implements online learning for a classification task with a non-stationary data distribution?](#27)
- [Qn 28: What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?](#28)
- [Qn 29: Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?](#29)
- [Qn 30: What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?](#30)

---

### 1. Qn 01: What technique would you use to handle high-dimensional sparse data when performing PCA?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Truncated SVD` (also known as LSA)

**Explanation:** Truncated SVD is specifically designed for sparse matrices and doesn't center the data (which would destroy sparsity), making it more memory-efficient and appropriate for high-dimensional sparse datasets.

**Learning Resources:**
- [qn_01_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_01/markdown/qn_01_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])`

**Explanation:** This approach uses numpy's efficient random sampling directly on indices, avoiding the overhead of pandas' sample function while maintaining group sizes and allowing replacement.

**Learning Resources:**
- [qn_02_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_02/markdown/qn_02_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Use sklearn's `MultilabelStratifiedKFold` from the iterative-stratification package

**Explanation:** MultilabelStratifiedKFold implements iterative stratification, which preserves the distribution of all labels across folds, addressing the key challenge in multi-label stratification that normal StratifiedKFold cannot handle.

**Learning Resources:**
- [qn_03_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_03/markdown/qn_03_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `scipy.stats.wasserstein_distance(x, y)`

**Explanation:** `scipy.stats.wasserstein_distance` correctly implements the 1D Wasserstein distance between empirical distributions, which measures the minimum 'work' required to transform one distribution into another.

**Learning Resources:**
- [qn_04_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_04/markdown/qn_04_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Depends on data dimensionality, size, and structure

**Explanation:** The most efficient algorithm depends on the dataset characteristics: brute force works well for small datasets and high dimensions, kd_tree excels in low dimensions (<20), and ball_tree performs better in higher dimensions or with non-Euclidean metrics.

**Learning Resources:**
- [qn_05_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_05/markdown/qn_05_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')`

**Explanation:** This approach correctly converts irregular timestamps to a regular frequency with .resample('1H').asfreq(), then intelligently fills missing values using time-based interpolation which respects the actual timing of observations.

**Learning Resources:**
- [qn_06_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_06/markdown/qn_06_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `MINE` statistics (Maximal Information-based Nonparametric Exploration)

**Explanation:** MINE statistics, particularly the Maximal Information Coefficient (MIC), detect both linear and non-linear associations without assuming a specific functional form, outperforming traditional correlation measures for complex relationships.

**Learning Resources:**
- [qn_07_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_07/markdown/qn_07_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Ensemble of balanced subsets with `META` learning

**Explanation:** META (Minority Ethnicity and Threshold Adjustment) learning with ensembling addresses severe multiclass imbalance by training multiple models on balanced subsets and combining them, avoiding information loss from undersampling while preventing the artificial patterns that can be introduced by synthetic oversampling.

**Learning Resources:**
- [qn_08_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_08/markdown/qn_08_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Convert to category dtype then use `df['col'].cat.codes` with sklearn's `OneHotEncoder(sparse=True)`

**Explanation:** Converting to pandas' memory-efficient category dtype first, then using cat.codes with a sparse OneHotEncoder creates a memory-efficient pipeline that preserves category labels and works well with scikit-learn while minimizing memory usage.

**Learning Resources:**
- [qn_09_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_09/markdown/qn_09_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `MultiOutputRegressor(GradientBoostingRegressor())`

**Explanation:** MultiOutputRegressor fits a separate GradientBoostingRegressor for each target, allowing each model to optimize independently, which is crucial when targets have different scales and relationships with features.

**Learning Resources:**
- [qn_10_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_10/markdown/qn_10_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 11. Qn 11: When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `LSTM Autoencoder` with reconstruction error thresholding

**Explanation:** LSTM Autoencoders can capture complex temporal dependencies in multivariate time series data, making them ideal for detecting contextual anomalies where data points are abnormal specifically in their context rather than globally.

**Learning Resources:**
- [qn_11_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_11/markdown/qn_11_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 12. Qn 12: What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Causal graphical models with do-calculus

**Explanation:** Causal graphical models using do-calculus provide a comprehensive mathematical framework for identifying causal effects from observational data, allowing researchers to formally express causal assumptions and determine whether causal quantities are identifiable from available data.

**Learning Resources:**
- [qn_12_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_12/markdown/qn_12_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 13. Qn 13: Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Birch` (Balanced Iterative Reducing and Clustering using Hierarchies)

**Explanation:** Birch is specifically designed for very large datasets as it builds a tree structure in a single pass through the data, has linear time complexity, limited memory requirements, and can handle outliers effectively, making it ideal for clustering massive high-dimensional datasets.

**Learning Resources:**
- [qn_13_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_13/markdown/qn_13_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 14. Qn 14: What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Variational Bayesian inference with automatic relevance determination

**Explanation:** Variational Bayesian inference with automatic relevance determination (implemented in sklearn as GaussianMixture(n_components=n, weight_concentration_prior_type='dirichlet_process')) can automatically prune unnecessary components, effectively determining the optimal number without requiring multiple model fits and comparisons.

**Learning Resources:**
- [qn_14_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_14/markdown/qn_14_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 15. Qn 15: What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** A and B are both correct depending on the custom_metric function

**Explanation:** make_scorer() is the correct approach, but the parameters depend on the specific metric: needs_proba=True for metrics requiring probability estimates (like AUC), and needs_threshold=True for metrics requiring decision thresholds; the appropriate configuration varies based on the specific imbalance-handling metric.

**Learning Resources:**
- [qn_15_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_15/markdown/qn_15_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 16. Qn 16: Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement `dask.dataframe` with lazy evaluation and out-of-core computation

**Explanation:** dask.dataframe provides a pandas-like API with lazy evaluation, parallel execution, and out-of-core computation, allowing for scalable data processing beyond available RAM while maintaining familiar pandas operations and requiring minimal code changes.

**Learning Resources:**
- [qn_16_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_16/markdown/qn_16_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 17. Qn 17: When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Bayesian optimization with Gaussian processes

**Explanation:** Bayesian optimization with Gaussian processes builds a probabilistic model of the objective function to intelligently select the most promising hyperparameter configurations based on previous evaluations, making it more efficient than random or grid search for exploring high-dimensional parameter spaces.

**Learning Resources:**
- [qn_17_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_17/markdown/qn_17_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 18. Qn 18: What's the most statistically sound approach to handle heteroscedasticity in a regression model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both B and C, with different null hypotheses

**Explanation:** Both tests detect heteroscedasticity but with different assumptions: Breusch-Pagan assumes that heteroscedasticity is a linear function of the independent variables, while White's test is more general and doesn't make this assumption, making them complementary approaches.

**Learning Resources:**
- [qn_18_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_18/markdown/qn_18_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 19. Qn 19: Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Reconciliation approach: forecast at all levels independently then reconcile with constraints

**Explanation:** The reconciliation approach (optimal combination) generates forecasts at all levels independently, then applies a mathematical reconciliation procedure that minimizes revisions while ensuring hierarchical consistency, typically outperforming both bottom-up and top-down approaches.

**Learning Resources:**
- [qn_19_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_19/markdown/qn_19_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 20. Qn 20: What technique is most appropriate for analyzing complex network data with community structures?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Louvain` algorithm for community detection

**Explanation:** The Louvain algorithm specifically optimizes modularity to detect communities in networks, automatically finding the appropriate number of communities and handling multi-scale resolution, making it ideal for complex networks with hierarchical community structures.

**Learning Resources:**
- [qn_20_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_20/markdown/qn_20_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 21. Qn 21: What's the most robust approach to handle concept drift in a production machine learning system?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Implement drift detection algorithms with adaptive learning techniques

**Explanation:** This approach combines statistical drift detection (e.g., ADWIN, DDM, or KSWIN) with adaptive learning methods that can continuously update models or model weights as new patterns emerge, allowing for immediate adaptation to changing data distributions.

**Learning Resources:**
- [qn_21_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_21/markdown/qn_21_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 22. Qn 22: Which method is most appropriate for interpretable anomaly detection in high-dimensional data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Isolation Forest` with LIME explanations

**Explanation:** Isolation Forest efficiently detects anomalies in high dimensions by isolating observations, while LIME provides local interpretable explanations for each anomaly, showing which features contributed most to its identification, making the detection both efficient and explainable.

**Learning Resources:**
- [qn_22_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_22/markdown/qn_22_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 23. Qn 23: When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Thompson Sampling` with prior distribution updates

**Explanation:** Thompson Sampling with Bayesian updates to prior distributions maintains explicit uncertainty estimates and naturally balances exploration/exploitation, with theoretical guarantees of optimality and empirically better performance than UCB and epsilon-greedy methods in many applications.

**Learning Resources:**
- [qn_23_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_23/markdown/qn_23_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 24. Qn 24: What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `scipy.spatial.distance.pdist` with `squareform`

**Explanation:** pdist computes distances using an optimized implementation that avoids redundant calculations (since distance matrices are symmetric), and squareform can convert to a square matrix if needed; this approach is significantly more memory-efficient than computing the full distance matrix directly.

**Learning Resources:**
- [qn_24_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_24/markdown/qn_24_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 25. Qn 25: Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Mahalanobis distance` with robust covariance estimation

**Explanation:** Mahalanobis distance accounts for the covariance structure of the data, and using robust covariance estimation (e.g., Minimum Covariance Determinant) prevents outliers from influencing the distance metric itself, making it ideal for identifying multivariate outliers.

**Learning Resources:**
- [qn_25_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_25/markdown/qn_25_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 26. Qn 26: What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Elastic Net` regularization with cross-validation

**Explanation:** Elastic Net combines L1 and L2 penalties, handling multicollinearity by grouping correlated features while still performing feature selection, with the optimal balance determined through cross-validationâ€”making it more effective than methods that either eliminate or transform features.

**Learning Resources:**
- [qn_26_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_26/markdown/qn_26_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 27. Qn 27: Which approach correctly implements online learning for a classification task with a non-stationary data distribution?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Ensemble of incremental learners with dynamic weighting based on recent performance

**Explanation:** This ensemble approach maintains multiple incremental models updated with new data, dynamically adjusting their weights based on recent performance, allowing the system to adapt to concept drift by giving more influence to models that perform well on recent data.

**Learning Resources:**
- [qn_27_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_27/markdown/qn_27_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 28. Qn 28: What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Joint modeling of missingness and outcomes

**Explanation:** Joint modeling directly incorporates the missingness mechanism into the analysis model, allowing for valid inference under non-random missingness (MNAR) scenarios by explicitly modeling the relationship between the missing data process and the outcomes of interest.

**Learning Resources:**
- [qn_28_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_28/markdown/qn_28_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 29. Qn 29: Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Gradient Boosting` with SHAP interaction values

**Explanation:** Gradient Boosting effectively captures complex non-linear relationships, while SHAP interaction values specifically quantify how much of the prediction is attributable to interactions between features, providing a rigorous statistical framework for analyzing and visualizing interactions.

**Learning Resources:**
- [qn_29_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_29/markdown/qn_29_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 30. Qn 30: What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** `Mutual information`-based selection with permutation testing

**Explanation:** Mutual information captures both linear and non-linear dependencies between variables without assuming functional form, while permutation testing provides a statistically rigorous way to assess the significance of these dependencies, controlling for multiple testing issues.

**Learning Resources:**
- [qn_30_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Data_Analysis/questions/qn_30/markdown/qn_30_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
