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

**Answer:** Truncated SVD (also known as LSA)

**Explanation:** Truncated SVD is specifically designed for sparse matrices and doesn't center the data (which would destroy sparsity), making it more memory-efficient and appropriate for high-dimensional sparse datasets.

### Qn 02: What's the most efficient way to perform grouped sampling with replacement in pandas, ensuring each group maintains its original size?

**Answer:** df.groupby('group').apply(lambda x: x.iloc[np.random.choice(len(x), size=len(x), replace=True)])

**Explanation:** This approach uses numpy's efficient random sampling directly on indices, avoiding the overhead of pandas' sample function while maintaining group sizes and allowing replacement.

### Qn 03: When implementing stratified k-fold cross-validation for a multi-label classification problem, which approach is most statistically sound?

**Answer:** Use sklearn's MultilabelStratifiedKFold from the iterative-stratification package

**Explanation:** MultilabelStratifiedKFold implements iterative stratification, which preserves the distribution of all labels across folds, addressing the key challenge in multi-label stratification that normal StratifiedKFold cannot handle.

### Qn 04: Which approach correctly calculates the Wasserstein distance (Earth Mover's Distance) between two empirical distributions in Python?

**Answer:** scipy.stats.wasserstein_distance(x, y)

**Explanation:** scipy.stats.wasserstein_distance correctly implements the 1D Wasserstein distance between empirical distributions, which measures the minimum 'work' required to transform one distribution into another.

### Qn 05: What's the most computationally efficient way to find the k-nearest neighbors for each point in a large dataset using scikit-learn?

**Answer:** Depends on data dimensionality, size, and structure

**Explanation:** The most efficient algorithm depends on the dataset characteristics: brute force works well for small datasets and high dimensions, kd_tree excels in low dimensions (â‰¤20), and ball_tree performs better in higher dimensions or with non-Euclidean metrics.

### Qn 06: When dealing with millions of rows of time series data with irregular timestamps, which method is most efficient for resampling to regular intervals with proper handling of missing values?

**Answer:** df.set_index('timestamp').resample('1H').asfreq().interpolate(method='time')

**Explanation:** This approach correctly converts irregular timestamps to a regular frequency with .resample('1H').asfreq(), then intelligently fills missing values using time-based interpolation which respects the actual timing of observations.

### Qn 07: Which technique is most appropriate for identifying non-linear relationships between variables in a high-dimensional dataset?

**Answer:** MINE statistics (Maximal Information-based Nonparametric Exploration)

**Explanation:** MINE statistics, particularly the Maximal Information Coefficient (MIC), detect both linear and non-linear associations without assuming a specific functional form, outperforming traditional correlation measures for complex relationships.

### Qn 08: What's the most statistically sound approach to handle imbalanced multiclass classification with severe class imbalance?

**Answer:** Ensemble of balanced subsets with META learning

**Explanation:** META (Minority Ethnicity and Threshold Adjustment) learning with ensembling addresses severe multiclass imbalance by training multiple models on balanced subsets and combining them, avoiding information loss from undersampling while preventing the artificial patterns that can be introduced by synthetic oversampling.

### Qn 09: What's the correct approach to implement a memory-efficient pipeline for one-hot encoding categorical variables with high cardinality in pandas?

**Answer:** Convert to category dtype then use df['col'].cat.codes with sklearn's OneHotEncoder(sparse=True)

**Explanation:** Converting to pandas' memory-efficient category dtype first, then using cat.codes with a sparse OneHotEncoder creates a memory-efficient pipeline that preserves category labels and works well with scikit-learn while minimizing memory usage.

### Qn 10: Which approach correctly implements a multi-output Gradient Boosting Regressor for simultaneously predicting multiple continuous targets with different scales?

**Answer:** MultiOutputRegressor(GradientBoostingRegressor())

**Explanation:** MultiOutputRegressor fits a separate GradientBoostingRegressor for each target, allowing each model to optimize independently, which is crucial when targets have different scales and relationships with features.

### Qn 11: When performing anomaly detection in a multivariate time series, which technique is most appropriate for detecting contextual anomalies?

**Answer:** LSTM Autoencoder with reconstruction error thresholding

**Explanation:** LSTM Autoencoders can capture complex temporal dependencies in multivariate time series data, making them ideal for detecting contextual anomalies where data points are abnormal specifically in their context rather than globally.

### Qn 12: What's the most rigorous approach to perform causal inference from observational data when randomized experiments aren't possible?

**Answer:** Causal graphical models with do-calculus

**Explanation:** Causal graphical models using do-calculus provide a comprehensive mathematical framework for identifying causal effects from observational data, allowing researchers to formally express causal assumptions and determine whether causal quantities are identifiable from available data.

### Qn 13: Which technique is most appropriate for efficiently clustering a dataset with millions of data points and hundreds of features?

**Answer:** Birch (Balanced Iterative Reducing and Clustering using Hierarchies)

**Explanation:** Birch is specifically designed for very large datasets as it builds a tree structure in a single pass through the data, has linear time complexity, limited memory requirements, and can handle outliers effectively, making it ideal for clustering massive high-dimensional datasets.

### Qn 14: What's the most rigorous method for selecting the optimal number of components in a Gaussian Mixture Model?

**Answer:** Variational Bayesian inference with automatic relevance determination

**Explanation:** Variational Bayesian inference with automatic relevance determination (implemented in sklearn as GaussianMixture(n_components=n, weight_concentration_prior_type='dirichlet_process')) can automatically prune unnecessary components, effectively determining the optimal number without requiring multiple model fits and comparisons.

### Qn 15: What's the correct approach to implement a custom scoring function for model evaluation in scikit-learn that handles class imbalance better than accuracy?

**Answer:** A and B are both correct depending on the custom_metric function

**Explanation:** make_scorer() is the correct approach, but the parameters depend on the specific metric: needs_proba=True for metrics requiring probability estimates (like AUC), and needs_threshold=True for metrics requiring decision thresholds; the appropriate configuration varies based on the specific imbalance-handling metric.

### Qn 16: Which approach correctly implements a memory-efficient data pipeline for processing and analyzing a dataset too large to fit in memory?

**Answer:** Implement dask.dataframe with lazy evaluation and out-of-core computation

**Explanation:** dask.dataframe provides a pandas-like API with lazy evaluation, parallel execution, and out-of-core computation, allowing for scalable data processing beyond available RAM while maintaining familiar pandas operations and requiring minimal code changes.

### Qn 17: When performing hyperparameter tuning for a complex model with many parameters, which advanced optimization technique is most efficient?

**Answer:** Bayesian optimization with Gaussian processes

**Explanation:** Bayesian optimization with Gaussian processes builds a probabilistic model of the objective function to intelligently select the most promising hyperparameter configurations based on previous evaluations, making it more efficient than random or grid search for exploring high-dimensional parameter spaces.

### Qn 18: What's the most appropriate method for detecting and quantifying heteroscedasticity in a regression model?

**Answer:** Both B and C, with different null hypotheses

**Explanation:** Both tests detect heteroscedasticity but with different assumptions: Breusch-Pagan assumes that heteroscedasticity is a linear function of the independent variables, while White's test is more general and doesn't make this assumption, making them complementary approaches.

### Qn 19: Which approach correctly implements a hierarchical time series forecasting model that respects aggregation constraints?

**Answer:** Reconciliation approach: forecast at all levels independently then reconcile with constraints

**Explanation:** The reconciliation approach (optimal combination) generates forecasts at all levels independently, then applies a mathematical reconciliation procedure that minimizes revisions while ensuring hierarchical consistency, typically outperforming both bottom-up and top-down approaches.

### Qn 20: What technique is most appropriate for analyzing complex network data with community structures?

**Answer:** Louvain algorithm for community detection

**Explanation:** The Louvain algorithm specifically optimizes modularity to detect communities in networks, automatically finding the appropriate number of communities and handling multi-scale resolution, making it ideal for complex networks with hierarchical community structures.

### Qn 21: What's the most robust approach to handle concept drift in a production machine learning system?

**Answer:** Implement drift detection algorithms with adaptive learning techniques

**Explanation:** This approach combines statistical drift detection (e.g., ADWIN, DDM, or KSWIN) with adaptive learning methods that can continuously update models or model weights as new patterns emerge, allowing for immediate adaptation to changing data distributions.

### Qn 22: Which method is most appropriate for interpretable anomaly detection in high-dimensional data?

**Answer:** Isolation Forest with LIME explanations

**Explanation:** Isolation Forest efficiently detects anomalies in high dimensions by isolating observations, while LIME provides local interpretable explanations for each anomaly, showing which features contributed most to its identification, making the detection both efficient and explainable.

### Qn 23: When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?

**Answer:** Thompson Sampling with prior distribution updates

**Explanation:** Thompson Sampling with Bayesian updates to prior distributions maintains explicit uncertainty estimates and naturally balances exploration/exploitation, with theoretical guarantees of optimality and empirically better performance than UCB and epsilon-greedy methods in many applications.

### Qn 24: What's the most efficient technique for calculating pairwise distances between all points in a very large dataset?

**Answer:** scipy.spatial.distance.pdist with squareform

**Explanation:** pdist computes distances using an optimized implementation that avoids redundant calculations (since distance matrices are symmetric), and squareform can convert to a square matrix if needed; this approach is significantly more memory-efficient than computing the full distance matrix directly.

### Qn 25: Which method is most appropriate for detecting and handling multivariate outliers in high-dimensional data?

**Answer:** Mahalanobis distance with robust covariance estimation

**Explanation:** Mahalanobis distance accounts for the covariance structure of the data, and using robust covariance estimation (e.g., Minimum Covariance Determinant) prevents outliers from influencing the distance metric itself, making it ideal for identifying multivariate outliers.

### Qn 26: What's the most appropriate technique for feature selection when dealing with multicollinearity in a regression context?

**Answer:** Elastic Net regularization with cross-validation

**Explanation:** Elastic Net combines L1 and L2 penalties, handling multicollinearity by grouping correlated features while still performing feature selection, with the optimal balance determined through cross-validationâ€”making it more effective than methods that either eliminate or transform features.

### Qn 27: Which approach correctly implements online learning for a classification task with a non-stationary data distribution?

**Answer:** Ensemble of incremental learners with dynamic weighting based on recent performance

**Explanation:** This ensemble approach maintains multiple incremental models updated with new data, dynamically adjusting their weights based on recent performance, allowing the system to adapt to concept drift by giving more influence to models that perform well on recent data.

### Qn 28: What's the most rigorous approach to handle missing data in a longitudinal study with potential non-random missingness?

**Answer:** Joint modeling of missingness and outcomes

**Explanation:** Joint modeling directly incorporates the missingness mechanism into the analysis model, allowing for valid inference under non-random missingness (MNAR) scenarios by explicitly modeling the relationship between the missing data process and the outcomes of interest.

### Qn 29: Which technique is most appropriate for analyzing complex interactions between variables in a predictive modeling context?

**Answer:** Gradient Boosting with SHAP interaction values

**Explanation:** Gradient Boosting effectively captures complex non-linear relationships, while SHAP interaction values specifically quantify how much of the prediction is attributable to interactions between features, providing a rigorous statistical framework for analyzing and visualizing interactions.

### Qn 30: What's the most statistically sound approach to perform feature selection for a regression task with potential non-linear relationships?

**Answer:** Mutual information-based selection with permutation testing

**Explanation:** Mutual information captures both linear and non-linear dependencies between variables without assuming functional form, while permutation testing provides a statistically rigorous way to assess the significance of these dependencies, controlling for multiple testing issues.


## Data Cleaning

### Qn 01: Which function in Pandas is used to detect missing values?

**Answer:** isnull()

**Explanation:** The `isnull()` function is used to detect missing values in a DataFrame.

### Qn 02: What does the `dropna()` function do?

**Answer:** Drops rows with missing values

**Explanation:** `dropna()` removes rows (or columns) that contain missing values.

### Qn 03: Which method is used to fill missing values with the mean of the column?

**Answer:** fillna(mean)

**Explanation:** `fillna()` is used to fill missing values, and you can pass the column mean to it.

### Qn 04: Sample data cleaning question 4?

**Answer:** Option A

**Explanation:** Explanation for sample question 4.

### Qn 05: Sample data cleaning question 5?

**Answer:** Option A

**Explanation:** Explanation for sample question 5.

### Qn 06: Sample data cleaning question 6?

**Answer:** Option A

**Explanation:** Explanation for sample question 6.

### Qn 07: Sample data cleaning question 7?

**Answer:** Option A

**Explanation:** Explanation for sample question 7.

### Qn 08: Sample data cleaning question 8?

**Answer:** Option A

**Explanation:** Explanation for sample question 8.

### Qn 09: Sample data cleaning question 9?

**Answer:** Option A

**Explanation:** Explanation for sample question 9.

### Qn 10: Sample data cleaning question 10?

**Answer:** Option A

**Explanation:** Explanation for sample question 10.

### Qn 11: Sample data cleaning question 11?

**Answer:** Option A

**Explanation:** Explanation for sample question 11.

### Qn 12: Sample data cleaning question 12?

**Answer:** Option A

**Explanation:** Explanation for sample question 12.

### Qn 13: Sample data cleaning question 13?

**Answer:** Option A

**Explanation:** Explanation for sample question 13.

### Qn 14: Sample data cleaning question 14?

**Answer:** Option A

**Explanation:** Explanation for sample question 14.

### Qn 15: Sample data cleaning question 15?

**Answer:** Option A

**Explanation:** Explanation for sample question 15.

### Qn 16: Sample data cleaning question 16?

**Answer:** Option A

**Explanation:** Explanation for sample question 16.

### Qn 17: Sample data cleaning question 17?

**Answer:** Option A

**Explanation:** Explanation for sample question 17.

### Qn 18: Sample data cleaning question 18?

**Answer:** Option A

**Explanation:** Explanation for sample question 18.

### Qn 19: Sample data cleaning question 19?

**Answer:** Option A

**Explanation:** Explanation for sample question 19.

### Qn 20: Sample data cleaning question 20?

**Answer:** Option A

**Explanation:** Explanation for sample question 20.

### Qn 21: Sample data cleaning question 21?

**Answer:** Option A

**Explanation:** Explanation for sample question 21.

### Qn 22: Sample data cleaning question 22?

**Answer:** Option A

**Explanation:** Explanation for sample question 22.

### Qn 23: Sample data cleaning question 23?

**Answer:** Option A

**Explanation:** Explanation for sample question 23.

### Qn 24: Sample data cleaning question 24?

**Answer:** Option A

**Explanation:** Explanation for sample question 24.

### Qn 25: Sample data cleaning question 25?

**Answer:** Option A

**Explanation:** Explanation for sample question 25.

### Qn 26: Sample data cleaning question 26?

**Answer:** Option A

**Explanation:** Explanation for sample question 26.

### Qn 27: Sample data cleaning question 27?

**Answer:** Option A

**Explanation:** Explanation for sample question 27.

### Qn 28: Sample data cleaning question 28?

**Answer:** Option A

**Explanation:** Explanation for sample question 28.

### Qn 29: Sample data cleaning question 29?

**Answer:** Option A

**Explanation:** Explanation for sample question 29.

### Qn 30: Sample data cleaning question 30?

**Answer:** Option A

**Explanation:** Explanation for sample question 30.

### Qn 31: Sample data cleaning question 31?

**Answer:** Option A

**Explanation:** Explanation for sample question 31.

### Qn 32: Sample data cleaning question 32?

**Answer:** Option A

**Explanation:** Explanation for sample question 32.

### Qn 33: Sample data cleaning question 33?

**Answer:** Option A

**Explanation:** Explanation for sample question 33.

### Qn 34: Sample data cleaning question 34?

**Answer:** Option A

**Explanation:** Explanation for sample question 34.

### Qn 35: Sample data cleaning question 35?

**Answer:** Option A

**Explanation:** Explanation for sample question 35.

### Qn 36: Sample data cleaning question 36?

**Answer:** Option A

**Explanation:** Explanation for sample question 36.

### Qn 37: Sample data cleaning question 37?

**Answer:** Option A

**Explanation:** Explanation for sample question 37.

### Qn 38: Sample data cleaning question 38?

**Answer:** Option A

**Explanation:** Explanation for sample question 38.

### Qn 39: Sample data cleaning question 39?

**Answer:** Option A

**Explanation:** Explanation for sample question 39.

### Qn 40: Sample data cleaning question 40?

**Answer:** Option A

**Explanation:** Explanation for sample question 40.

### Qn 41: Sample data cleaning question 41?

**Answer:** Option A

**Explanation:** Explanation for sample question 41.

### Qn 42: Sample data cleaning question 42?

**Answer:** Option A

**Explanation:** Explanation for sample question 42.

### Qn 43: Sample data cleaning question 43?

**Answer:** Option A

**Explanation:** Explanation for sample question 43.

### Qn 44: Sample data cleaning question 44?

**Answer:** Option A

**Explanation:** Explanation for sample question 44.

### Qn 45: Sample data cleaning question 45?

**Answer:** Option A

**Explanation:** Explanation for sample question 45.

### Qn 46: Sample data cleaning question 46?

**Answer:** Option A

**Explanation:** Explanation for sample question 46.

### Qn 47: Sample data cleaning question 47?

**Answer:** Option A

**Explanation:** Explanation for sample question 47.

### Qn 48: Sample data cleaning question 48?

**Answer:** Option A

**Explanation:** Explanation for sample question 48.

### Qn 49: Sample data cleaning question 49?

**Answer:** Option A

**Explanation:** Explanation for sample question 49.

### Qn 50: Sample data cleaning question 50?

**Answer:** Option A

**Explanation:** Explanation for sample question 50.


## Data Science

### Qn 01: What is the primary goal of data wrangling?

**Answer:** Cleaning and transforming raw data into a usable format

**Explanation:** Data wrangling involves cleaning, structuring, and enriching raw data into a format suitable for analysis.

### Qn 02: Which of the following is NOT a measure of central tendency?

**Answer:** Standard deviation

**Explanation:** Standard deviation measures dispersion, not central tendency. The three main measures of central tendency are mean, median, and mode.

### Qn 03: What type of chart would be most appropriate for comparing proportions of a whole?

**Answer:** Pie chart

**Explanation:** Pie charts are best for showing proportions of a whole, though they should be used sparingly and only with a small number of categories.

### Qn 04: Which Python library is primarily used for working with tabular data structures?

**Answer:** Pandas

**Explanation:** Pandas provides DataFrame objects which are ideal for working with tabular data, similar to spreadsheets or SQL tables.

### Qn 05: What does the groupby() operation in Pandas return before aggregation?

**Answer:** A DataFrameGroupBy object

**Explanation:** groupby() returns a DataFrameGroupBy object which can then be aggregated using functions like sum(), mean(), etc.

### Qn 06: What does 'NaN' represent in a Pandas DataFrame?

**Answer:** Not a Number (missing or undefined value)

**Explanation:** NaN stands for 'Not a Number' and represents missing or undefined numerical data in Pandas.

### Qn 07: Which technique is NOT typically used for feature selection?

**Answer:** Data normalization

**Explanation:** Data normalization scales features but doesn't select them. PCA, correlation analysis, and recursive elimination are feature selection methods.

### Qn 08: Which metric is NOT used to evaluate regression models?

**Answer:** Accuracy

**Explanation:** Accuracy is used for classification problems. MSE, RMSE, and R-squared are common regression metrics.

### Qn 09: What is the most common method for handling missing numerical data?

**Answer:** Replacing with the mean or median

**Explanation:** Mean/median imputation is common for numerical data, though the best approach depends on the data and missingness pattern.

### Qn 10: Which library is essential for numerical computing in Python?

**Answer:** NumPy

**Explanation:** NumPy provides foundational support for numerical computing with efficient array operations and mathematical functions.

### Qn 11: What is the purpose of a correlation matrix?

**Answer:** To measure linear relationships between numerical variables

**Explanation:** A correlation matrix measures the linear relationship between pairs of numerical variables, ranging from -1 to 1.

### Qn 12: What is the main advantage of using a box plot?

**Answer:** Displaying the distribution and outliers of a dataset

**Explanation:** Box plots effectively show a dataset's quartiles, median, and potential outliers.

### Qn 13: What does the term 'overfitting' refer to in machine learning?

**Answer:** A model that performs well on training data but poorly on unseen data

**Explanation:** Overfitting occurs when a model learns the training data too well, including its noise, reducing generalization to new data.

### Qn 14: Which of these is a supervised learning algorithm?

**Answer:** Random Forest

**Explanation:** Random Forest is a supervised learning algorithm. K-means and PCA are unsupervised, and t-SNE is for visualization.

### Qn 15: What is the purpose of a train-test split?

**Answer:** To evaluate how well a model generalizes to unseen data

**Explanation:** Splitting data into training and test sets helps estimate model performance on new, unseen data.

### Qn 16: Which Python library is most commonly used for creating static visualizations?

**Answer:** Matplotlib

**Explanation:** Matplotlib is the foundational plotting library in Python, though Seaborn builds on it for statistical visualizations.

### Qn 17: What is the main purpose of normalization in data preprocessing?

**Answer:** To scale features to a similar range

**Explanation:** Normalization scales numerical features to a standard range (often [0,1] or with mean=0, std=1) to prevent some features from dominating others.

### Qn 18: What does SQL stand for?

**Answer:** Structured Query Language

**Explanation:** SQL stands for Structured Query Language, used for managing and querying relational databases.

### Qn 19: Which of these is NOT a common data type in Pandas?

**Answer:** Array

**Explanation:** Pandas' main data structures are DataFrame (2D), Series (1D), and Panel (3D, now deprecated). Arrays are from NumPy.

### Qn 20: What is the primary use of the Scikit-learn library?

**Answer:** Machine learning algorithms

**Explanation:** Scikit-learn provides simple and efficient tools for predictive data analysis and machine learning.

### Qn 21: What is the difference between classification and regression?

**Answer:** Classification predicts categories, regression predicts continuous values

**Explanation:** Classification predicts discrete class labels, while regression predicts continuous numerical values.

### Qn 22: What is a confusion matrix used for?

**Answer:** Evaluating the performance of a classification model

**Explanation:** A confusion matrix shows true/false positives/negatives, helping evaluate classification model performance.

### Qn 23: What does ETL stand for in data engineering?

**Answer:** Extract, Transform, Load

**Explanation:** ETL refers to the process of extracting data from sources, transforming it, and loading it into a destination system.

### Qn 24: Which of these is a dimensionality reduction technique?

**Answer:** Principal Component Analysis (PCA)

**Explanation:** PCA reduces dimensionality by transforming data to a new coordinate system with fewer dimensions.

### Qn 25: What is the purpose of cross-validation?

**Answer:** To get more reliable estimates of model performance

**Explanation:** Cross-validation provides more robust performance estimates by using multiple train/test splits of the data.

### Qn 26: What is the main advantage of using a Jupyter Notebook?

**Answer:** It combines code, visualizations, and narrative text

**Explanation:** Jupyter Notebooks allow interactive development with code, visualizations, and explanatory text in a single document.

### Qn 27: What is the purpose of one-hot encoding?

**Answer:** To convert categorical variables to numerical format

**Explanation:** One-hot encoding converts categorical variables to a binary (0/1) numerical format that machine learning algorithms can process.

### Qn 28: Which metric would you use for an imbalanced classification problem?

**Answer:** Precision-Recall curve

**Explanation:** For imbalanced classes, accuracy can be misleading. Precision-Recall curves provide better insight into model performance.

### Qn 29: What is feature engineering?

**Answer:** Creating new features from existing data

**Explanation:** Feature engineering involves creating new input features from existing data to improve model performance.

### Qn 30: What is the purpose of a ROC curve?

**Answer:** To visualize the trade-off between true positive and false positive rates

**Explanation:** ROC curves show the diagnostic ability of a binary classifier by plotting true positive rate vs false positive rate.

### Qn 31: What is the main advantage of using a random forest over a single decision tree?

**Answer:** It reduces overfitting by averaging multiple trees

**Explanation:** Random forests combine multiple decision trees to reduce variance and overfitting compared to a single tree.

### Qn 32: What is the purpose of the 'iloc' method in Pandas?

**Answer:** To select data by integer position

**Explanation:** iloc is primarily integer-location based indexing for selection by position.

### Qn 33: What is the difference between deep learning and traditional machine learning?

**Answer:** Deep learning automatically learns feature hierarchies from raw data

**Explanation:** Deep learning models can learn hierarchical feature representations directly from data, while traditional ML often requires manual feature engineering.

### Qn 34: What is the purpose of a learning curve in machine learning?

**Answer:** To show the relationship between training set size and model performance

**Explanation:** Learning curves plot model performance (e.g., accuracy) against training set size or training iterations.

### Qn 35: What is the bias-variance tradeoff?

**Answer:** The balance between model complexity and generalization

**Explanation:** The bias-variance tradeoff refers to balancing a model's simplicity (bias) against its sensitivity to training data (variance) to achieve good generalization.

### Qn 36: What is the purpose of regularization in machine learning?

**Answer:** To reduce overfitting by penalizing complex models

**Explanation:** Regularization techniques like L1/L2 add penalty terms to prevent overfitting by discouraging overly complex models.

### Qn 37: What is transfer learning in deep learning?

**Answer:** Using a pre-trained model as a starting point for a new task

**Explanation:** Transfer learning leverages knowledge gained from solving one problem and applies it to a different but related problem.

### Qn 38: What is the purpose of a word embedding in NLP?

**Answer:** To represent words as dense vectors capturing semantic meaning

**Explanation:** Word embeddings represent words as numerical vectors where similar words have similar vector representations.

### Qn 39: What is the main advantage of using SQL databases over NoSQL?

**Answer:** Stronger consistency guarantees

**Explanation:** SQL databases provide ACID transactions and strong consistency, while NoSQL prioritizes scalability and flexibility.

### Qn 40: What is the purpose of A/B testing?

**Answer:** To test two different versions of a product feature

**Explanation:** A/B testing compares two versions (A and B) to determine which performs better on a specific metric.

### Qn 41: What is the main purpose of the 'apply' function in Pandas?

**Answer:** To apply a function along an axis of a DataFrame

**Explanation:** The apply() function applies a function along an axis (rows or columns) of a DataFrame or Series.

### Qn 42: What is the difference between batch gradient descent and stochastic gradient descent?

**Answer:** Batch uses all data per update, stochastic uses one sample

**Explanation:** Batch GD computes gradients using the entire dataset, while SGD uses a single random sample per iteration.

### Qn 43: What is the purpose of the 'dropna' method in Pandas?

**Answer:** To drop rows or columns with missing values

**Explanation:** dropna() removes missing values (NaN) from a DataFrame, either by rows or columns.

### Qn 44: What is the main advantage of using a pipeline in Scikit-learn?

**Answer:** To chain multiple processing steps into a single object

**Explanation:** Pipelines sequentially apply transforms and a final estimator, ensuring steps are executed in the right order.

### Qn 45: What is the purpose of the 'value_counts' method in Pandas?

**Answer:** To count the number of unique values in a Series

**Explanation:** value_counts() returns a Series containing counts of unique values in descending order.

### Qn 46: What is the main purpose of feature scaling?

**Answer:** To ensure all features contribute equally to distance-based algorithms

**Explanation:** Feature scaling normalizes the range of features so that features with larger scales don't dominate algorithms like KNN or SVM.

### Qn 47: What is the difference between 'fit' and 'transform' in Scikit-learn?

**Answer:** 'fit' learns parameters, 'transform' applies them

**Explanation:** fit() learns model parameters from training data, while transform() applies the learned transformation to data.

### Qn 48: What is the purpose of the 'merge' function in Pandas?

**Answer:** To combine DataFrames based on common columns

**Explanation:** merge() combines DataFrames using database-style joins on columns or indices.

### Qn 49: What is the main advantage of using a dictionary for vectorization in NLP?

**Answer:** It creates a fixed-length representation regardless of document length

**Explanation:** Dictionary-based vectorization (like CountVectorizer) creates consistent-length vectors from variable-length texts.

### Qn 50: What is the purpose of the 'pivot_table' function in Pandas?

**Answer:** To create a spreadsheet-style pivot table as a DataFrame

**Explanation:** pivot_table() creates a multi-dimensional summary table similar to Excel pivot tables, aggregating data.


## Data Visualization

### Qn 01: Which Python library is most commonly used for creating static, animated, and interactive visualizations?

**Answer:** Matplotlib

**Explanation:** `Matplotlib` is a foundational library for creating a wide variety of plots and charts.

### Qn 02: Which function is used to create a line plot in Matplotlib?

**Answer:** plot()

**Explanation:** `plot()` is used to generate line plots in Matplotlib.

### Qn 03: What does the `Seaborn` library primarily focus on?

**Answer:** Statistical data visualization

**Explanation:** `Seaborn` builds on top of Matplotlib and integrates closely with Pandas for statistical plots.

### Qn 04: Which method can be used to show the distribution of a single variable?

**Answer:** distplot()

**Explanation:** `distplot()` or `histplot()` can be used to show distribution of a variable.

### Qn 05: How do you display a plot in Jupyter Notebook?

**Answer:** %matplotlib inline

**Explanation:** `%matplotlib inline` is a magic command to render plots directly in the notebook.


## IQ

### Qn 01: The cost of one paper is 15 cents. How much will 40 papers cost?

**Answer:** $6.00

**Explanation:** Basic multiplication or unit price calculation.

### Qn 02: In a party, 10 men shake hands with each other and they get to shake everyones hand once. How many total handshakes are there?

**Answer:** 45

**Explanation:** Use n(n-1)/2 formula for handshakes.

### Qn 03: Pick out the number with the smallest value.

**Answer:** 0.33

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 04: Jim makes 8.50 an hour and 3 extra for cleaning a store. If he worked 36 hours and cleaned 17 stores. How much money did he make?

**Answer:** $357

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 05: Paper pins cost 21 cents a pack. How much will 4 packs cost?

**Answer:** $0.84

**Explanation:** Basic multiplication or unit price calculation.

### Qn 06: A pad costs 33 cents. How much will 5 pads cost?

**Answer:** $1.65

**Explanation:** Basic multiplication or unit price calculation.

### Qn 07: Which of these objects has the largest diameter?

**Answer:** Sun

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 08: Identify the next number in the series.
24, 12, 6, 3, _____

**Answer:** 1.5

**Explanation:** Half each number to get the next number in the series.

### Qn 09: Urge and deter have _____ meanings?

**Answer:** contradictory

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 10: Arrange these words to form a sentence. The sentence you arranged, is it true or false?

triangle sides a has three

**Answer:** True

**Explanation:** A triangle has three sides.

### Qn 11: Which word doesnt belong in this set of words?

**Answer:** mason

**Explanation:** A mason is a skilled worker who builds or works with stone, brick, or concrete. The other three are professions that require a degree or specialized training. A barrister is a lawyer who specializes in court cases, an economist studies the economy, and a surgeon is a medical doctor who performs surgery.

### Qn 12: Calculate:  -43 + 9

**Answer:** -34

**Explanation:** -43+10 would be -33, so -43+9 is -34.

### Qn 13: The cost of one egg is 15 cents. How much will 4 eggs cost?

**Answer:** $0.60

**Explanation:** Basic multiplication or unit price calculation.

### Qn 14: Subservient is the opposite of?

**Answer:** imperious

**Explanation:** Find the antonym.

### Qn 15: The volume of a rectangular box is 100 cubic inches. If the minimum dimension of any given side is 1 inch, which of the alternatives is its greatest possible length?

**Answer:** 100 inches

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 16: What is the square root of 8?

**Answer:** 2.82

**Explanation:** Square root is the number that gives the original number when multiplied by itself.

### Qn 17: What do you get when you divide 7.95 by 1.5?

**Answer:** 5.3

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 18: Jack and John have 56 marbles together. John has 6x more marbles than Jack. How many marbles does Jack have?

**Answer:** 8

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 19: Which digit represents the tenths space in 10,987.36?

**Answer:** 3

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 20: How many pairs are duplicates?
987878; 987788
124555; 123555
6845; 6845
45641; 45641
9898547; -9898745

**Answer:** 2

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 21: How many bananas will you find in a dozen?

**Answer:** 12

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 22: Identify the next number in the series?
5, 15, 10, 13, 15, 11, _____

**Answer:** 20

**Explanation:** The odd positions are increasing by 5 and the even positions are decreasing by 2.

### Qn 23: How many days are there in three years?

**Answer:** 1,095

**Explanation:** 365 days in a year, so 3 years = 3 * 365 = 1,095 days.

### Qn 24: Adam is selling yarn for $.04/foot. How many feet can you buy from him for $0.52?

**Answer:** 13 feet

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 25: Suppose the first 2 statements are true. Is the third one true/false or not certain?
1) Jack greeted Jill.
2) Jill greeted Joe.
3) Jack didnt greet Joe.

**Answer:** Uncertain

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 26: Transubstantiate and convert have _____ meanings.

**Answer:** similar

**Explanation:** Transubstantiate means to change the substance of something, while convert means to change something into a different form or use. Both words imply a transformation.

### Qn 27: What number should come next?
8, 4, 2, 1, 1/2, 1/4, _____

**Answer:** 1/8

**Explanation:** Each next number are half of the previous number.

### Qn 28: There are 14 more potatoes than onions in a basket of 36 potatoes and onions. How many potatoes are there in the basket?

**Answer:** 25

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 29: Martin got a 25% salary raise. If his previous salary was $1,200, how much will it be after implementing the raise?

**Answer:** $1,500

**Explanation:** Convert the percentage to decimal by dividing by 100.

### Qn 30: Ferguson has 8 hats, 6 shirts and 4 pairs of pants. How many days can he dress up without repeating the same combination?

**Answer:** 192 days

**Explanation:** We need to multiply the number of hats, shirts and pants to get the total combinations. 8 * 6 * 4 = 192.

### Qn 31: Content and Satisfied have ______ meanings?

**Answer:** similar

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 32: Identify the next number in the series:
1, 1, 2, 3, 5, 8, _____

**Answer:** 13

**Explanation:** This is fibonacci series. The next number is the sum of the last two numbers.

### Qn 33: Calculate:  -70 + 35

**Answer:** -35

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 34: 61% converted to decimal notation is:

**Answer:** 0.61

**Explanation:** Convert the percentage to decimal by dividing by 100.

### Qn 35: Which number has the smallest value?

**Answer:** 0.02

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 36: Adjoin and sever have _____ meaning?

**Answer:** opposite

**Explanation:** Adjoin means to connect or attach, while sever means to cut off or separate. They are opposite in meaning.

### Qn 37: October is the ____ month of the year

**Answer:** 10th

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 38: Identify next number in the series.
9, 3, 1, 1/3, _____

**Answer:** 1/9

**Explanation:** Each next number is one third of the previous number.

### Qn 39: Reduce 75/100 to the simplest form?

**Answer:** 3/4

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 40: An automotive shop owner bought some tools for $5,500. He sold those for $7,300 with a profit of $50 per tool. How many tools did he sell?

**Answer:** 36

**Explanation:** 7300 - 5500 = 1800 profit. 1800/50 = 36 tools sold.

### Qn 41: Partner and Join have _____ meanings.

**Answer:** similar

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 42: If a person is half a century old. How old is he?

**Answer:** 50

**Explanation:** Century means 100 years. Half a century means 50 years.

### Qn 43: What do you get when you round of 907.457 to the nearest tens place?

**Answer:** 910

**Explanation:** Nearest tens place means rounding to the nearest 10. 907.457 rounds to 910.

### Qn 44: (8 ÷ 4) x (9 ÷ 3) = ?

**Answer:** 6

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 45: What is the relation betweed Credence and Credit?

**Answer:** similar

**Explanation:** Credence means belief or trust, while credit refers to the ability to obtain goods or services before payment. They are related in the context of trust and belief.

### Qn 46: In a week, Rose spent $28.49 on lunch. What was the average cost per day?

**Answer:** $4.07

**Explanation:** Basic multiplication or unit price calculation.

### Qn 47: The opposite of punctual is?

**Answer:** tardy

**Explanation:** tardy means late or delayed, which is the opposite of punctual. Whereas conscious means aware, rigorous means strict or demanding, and meticulous means careful and precise.

### Qn 48: Reduce and Produce have ____ meaning?

**Answer:** opposite

**Explanation:** Apply logic or perform basic arithmetic to solve.

### Qn 49: Rule out the odd word from the set of words.

**Answer:** drought

**Explanation:** drought is a prolonged dry period, while the others are related to finance. Budget refers to a plan for spending money, debt refers to money owed, and credit refers to the ability to borrow money.

### Qn 50: Which three words have the same meaning?
A. Information
B. Indoctrinate
C. Brainwash
D. Convince
E. Class

**Answer:** BCD

**Explanation:** Find the synonym.


## Machine Learning

### Qn 01: Which algorithm is used for classification problems?

**Answer:** Logistic Regression

**Explanation:** Logistic regression is used for binary and multi-class classification problems.

### Qn 02: What is the purpose of the learning rate in gradient descent?

**Answer:** To control how much the model is adjusted during each update

**Explanation:** The learning rate determines the step size at each iteration while moving toward a minimum of a loss function.

### Qn 03: Which technique is used to reduce the dimensionality of data?

**Answer:** PCA

**Explanation:** Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data to a lower-dimensional space.

### Qn 04: What is the main goal of unsupervised learning?

**Answer:** Find hidden patterns or groupings in data

**Explanation:** Unsupervised learning aims to discover the underlying structure of data without predefined labels.

### Qn 05: What is overfitting in machine learning?

**Answer:** When the model performs well on training data but poorly on test data

**Explanation:** Overfitting occurs when a model learns the training data too well, including noise and outliers, which leads to poor generalization to new data.


## Modelling

### Qn 01: When implementing stacking ensemble with scikit-learn, what's the most rigorous approach to prevent target leakage in the meta-learner?

**Answer:** Manually implement out-of-fold predictions for each base learner

**Explanation:** Manually generating out-of-fold predictions ensures the meta-learner only sees predictions made on data that base models weren't trained on, fully preventing leakage while utilizing all data. This approach is more flexible than StackingClassifier and can incorporate diverse base models while maintaining proper validation boundaries.

### Qn 02: What's the most effective technique for calibrating probability estimates from a gradient boosting classifier?

**Answer:** Apply sklearn's CalibratedClassifierCV with isotonic regression

**Explanation:** Isotonic regression via CalibratedClassifierCV is non-parametric and can correct any monotonic distortion in probability estimates, making it more flexible than Platt scaling, particularly for gradient boosting models which often produce well-ranked but not well-calibrated probabilities.

### Qn 03: Which approach correctly implements proper nested cross-validation for model selection and evaluation?

**Answer:** Nested loops of KFold.split(), with inner loop for hyperparameter tuning

**Explanation:** Proper nested cross-validation requires an outer loop for performance estimation and an inner loop for hyperparameter tuning, completely separating the data used for model selection from the data used for model evaluation, avoiding optimistic bias.

### Qn 04: What's the most memory-efficient way to implement incremental learning for large datasets with scikit-learn?

**Answer:** Use SGDClassifier with partial_fit on data chunks

**Explanation:** SGDClassifier with partial_fit allows true incremental learning, processing data in chunks without storing the entire dataset in memory, updating model parameters with each batch and converging to the same solution as batch processing would with sufficient iterations.

### Qn 05: When dealing with competing risks in survival analysis, which implementation correctly handles the problem?

**Answer:** Fine-Gray subdistribution hazard model from pysurvival

**Explanation:** The Fine-Gray model explicitly accounts for competing risks by modeling the subdistribution hazard, allowing for valid inference about the probability of an event in the presence of competing events, unlike standard Cox models or Kaplan-Meier which can produce biased estimates under competing risks.

### Qn 06: What's the most statistically sound approach to implement monotonic constraints in gradient boosting?

**Answer:** Using XGBoost's monotone_constraints parameter

**Explanation:** XGBoost's native monotone_constraints parameter enforces monotonicity during tree building by constraining only monotonic splits, resulting in a fully monotonic model without sacrificing performance—unlike post-processing which can degrade model accuracy or pre-processing which doesn't guarantee model monotonicity.

### Qn 07: Which approach correctly implements a custom kernel for SVM in scikit-learn?

**Answer:** Define a function that takes two arrays and returns a kernel matrix

**Explanation:** For custom kernels in scikit-learn SVMs, one must define a function K(X, Y) that calculates the kernel matrix between arrays X and Y, then pass this function as the 'kernel' parameter to SVC. This approach allows full flexibility in kernel design while maintaining compatibility with scikit-learn's implementation.

### Qn 08: What's the most rigorous approach to handle feature selection with highly correlated features in a regression context?

**Answer:** Elastic Net regularization with randomized hyperparameter search

**Explanation:** Elastic Net combines L1 and L2 penalties, effectively handling correlated features by either selecting one from a correlated group (via L1) or assigning similar coefficients to correlated features (via L2), with the optimal balance determined through randomized hyperparameter search across different alpha and l1_ratio values.

### Qn 09: Which implementation correctly handles ordinal encoding for machine learning while preserving the ordinal nature of features?

**Answer:** Custom encoding using pd.Categorical with ordered=True

**Explanation:** Using pandas Categorical with ordered=True preserves the ordinal relationship and allows for appropriate distance calculations between categories, which is essential for models that consider feature relationships (unlike OrdinalEncoder which assigns arbitrary numeric values without preserving distances).

### Qn 10: What's the most effective way to implement a time-based split for cross-validation in time series forecasting?

**Answer:** Define a custom cross-validator with expanding window and purging

**Explanation:** A custom cross-validator with expanding windows (increasing training set) and purging (gap between train and test to prevent leakage) most accurately simulates real-world forecasting scenarios while handling temporal dependencies and avoiding lookahead bias.

### Qn 11: Which approach correctly implements an interpretable model for binary classification with uncertainty quantification?

**Answer:** Bayesian Logistic Regression with MCMC sampling for posterior distribution

**Explanation:** Bayesian Logistic Regression provides both interpretability (coefficients have clear meanings) and principled uncertainty quantification through the posterior distribution of parameters, capturing both aleatoric and epistemic uncertainty while maintaining model transparency.

### Qn 12: What's the most robust approach to handling class imbalance in a multi-class classification problem?

**Answer:** Use ensemble methods with resampling strategies specific to each classifier

**Explanation:** Ensemble methods with class-specific resampling strategies (e.g., EasyEnsemble or SMOTEBoost) combine the diversity of multiple classifiers with targeted handling of class imbalance, outperforming both global resampling and simple class weighting, especially for multi-class problems with varying degrees of imbalance.

### Qn 13: Which technique is most appropriate for detecting and quantifying the importance of interaction effects in a Random Forest model?

**Answer:** Implement H-statistic from Friedman and Popescu

**Explanation:** The H-statistic specifically quantifies interaction strength between features by comparing the variation in predictions when features are varied together versus independently, providing a statistical measure of interactions that can't be captured by standard importance metrics or partial dependence alone.

### Qn 14: What's the correct approach to implement a custom scoring function for sklearn's RandomizedSearchCV that accounts for both predictive performance and model complexity?

**Answer:** Use make_scorer with a function that combines multiple metrics

**Explanation:** make_scorer allows creating a custom scoring function that can combine predictive performance (e.g., AUC) with penalties for model complexity (e.g., number of features or model parameters), providing a single metric for optimization that balances performance and parsimony.

### Qn 15: Which is the most statistically rigorous approach to implement feature selection for a regression problem with heteroscedastic errors?

**Answer:** Implement weighted LASSO with weight inversely proportional to error variance

**Explanation:** Weighted LASSO that downweights observations with high error variance accounts for heteroscedasticity in the selection process, ensuring that features aren't selected or rejected due to non-constant error variance, resulting in more reliable feature selection.

### Qn 16: What's the most effective way to implement an interpretable yet powerful model for regression with potentially non-linear effects?

**Answer:** Use Explainable Boosting Machines (EBMs) from InterpretML

**Explanation:** EBMs combine the interpretability of GAMs with the predictive power of boosting, learning feature functions and pairwise interactions in an additive structure while remaining highly interpretable, offering better performance than standard GAMs while maintaining transparency.

### Qn 17: Which approach correctly implements quantile regression forests for prediction intervals?

**Answer:** Implement a custom version of RandomForestRegressor that stores all leaf node samples

**Explanation:** Quantile regression forests require storing the empirical distribution of training samples in each leaf node (not just their mean), requiring a custom implementation that extends standard random forests to compute conditional quantiles from these stored distributions.

### Qn 18: What's the most rigorous approach to handle outliers in the target variable for regression problems?

**Answer:** Use Huber or Quantile regression with robust loss functions

**Explanation:** Robust regression methods like Huber or Quantile regression use loss functions that inherently reduce the influence of outliers during model training, addressing the issue without removing potentially valuable data points or distorting the target distribution through transformations.

### Qn 19: Which implementation correctly addresses the curse of dimensionality in nearest neighbor models?

**Answer:** Implement distance metric learning with NCA or LMNN

**Explanation:** Distance metric learning adaptively learns a transformation of the feature space that emphasizes discriminative dimensions, effectively addressing the curse of dimensionality by creating a more semantically meaningful distance metric, unlike fixed trees or general dimensionality reduction.

### Qn 20: What's the most efficient way to implement early stopping in a gradient boosting model to prevent overfitting?

**Answer:** Use early_stopping_rounds with a validation set in XGBoost/LightGBM

**Explanation:** Using early_stopping_rounds with a separate validation set stops training when performance on the validation set stops improving for a specified number of rounds, efficiently determining the optimal number of trees in a single training run without requiring multiple cross-validation runs.

### Qn 21: Which approach correctly implements a counterfactual explanation method for a black-box classifier?

**Answer:** Implement DiCE (Diverse Counterfactual Explanations) to generate multiple feasible counterfactuals

**Explanation:** DiCE specifically generates diverse counterfactual explanations that show how an instance's features would need to change to receive a different classification, addressing the 'what-if' question directly rather than just explaining the current prediction.

### Qn 22: What's the most effective approach to implement online learning for a regression task with concept drift?

**Answer:** Use incremental learning with drift detection algorithms to trigger model updates

**Explanation:** Combining incremental learning with explicit drift detection (e.g., ADWIN, DDM) allows the model to adapt continuously to new data while only performing major updates when the data distribution actually changes, balancing computational efficiency with adaptation to concept drift.

### Qn 23: Which method is most appropriate for tuning hyperparameters when training time is extremely limited?

**Answer:** Implement multi-fidelity optimization with Hyperband

**Explanation:** Hyperband uses a bandit-based approach to allocate resources efficiently, quickly discarding poor configurations and allocating more compute to promising ones, making it particularly effective when training time is limited and early performance is indicative of final performance.

### Qn 24: What's the most statistically sound approach to implement feature selection for time series forecasting?

**Answer:** Implement feature importance from tree-based models with purged cross-validation

**Explanation:** Tree-based feature importance combined with purged cross-validation (which leaves gaps between train and test sets) correctly handles temporal dependence in the data, preventing information leakage while identifying features that have genuine predictive power for future time points.

### Qn 25: Which approach correctly addresses Simpson's paradox in a predictive modeling context?

**Answer:** Use causal graphical models to identify proper conditioning sets

**Explanation:** Causal graphical models (e.g., DAGs) allow identifying which variables should or should not be conditioned on to avoid Simpson's paradox, ensuring that the model captures the true causal relationship rather than spurious associations that reverse with conditioning.

### Qn 26: What's the most efficient way to implement hyperparameter tuning for an ensemble of diverse model types?

**Answer:** Apply multi-objective Bayesian optimization to balance diversity and performance

**Explanation:** Multi-objective Bayesian optimization can simultaneously optimize for both individual model performance and ensemble diversity, finding an optimal set of hyperparameters for each model type while ensuring the ensemble as a whole performs well through complementary strengths.

### Qn 27: Which technique is most appropriate for detecting and visualizing non-linear relationships in supervised learning?

**Answer:** Individual Conditional Expectation (ICE) plots with centered PDP

**Explanation:** ICE plots show how predictions change for individual instances across the range of a feature, while centering them helps visualize heterogeneous effects that would be masked by averaging in standard partial dependence plots, making them ideal for detecting complex non-linear relationships.

### Qn 28: What's the most rigorous approach to quantify uncertainty in predictions from a gradient boosting model?

**Answer:** Use quantile regression with multiple target quantiles

**Explanation:** Training multiple gradient boosting models with quantile loss functions at different quantiles (e.g., 5%, 50%, 95%) directly models the conditional distribution of the target variable, providing a rigorous non-parametric approach to uncertainty quantification that captures heteroscedasticity.

### Qn 29: What's the most appropriate technique for automated feature engineering in time series forecasting?

**Answer:** Use tsfresh with appropriate feature filtering based on p-values

**Explanation:** tsfresh automatically extracts and selects relevant time series features (over 700 features) while controlling for multiple hypothesis testing, specifically designed for time series data unlike general feature engineering tools, making it ideal for time series forecasting tasks.

### Qn 30: Which approach correctly implements proper evaluation metrics for a multi-class imbalanced classification problem?

**Answer:** Apply precision-recall curves with prevalence-corrected metrics

**Explanation:** For imbalanced multi-class problems, precision-recall curves with prevalence correction (e.g., weighted by actual class frequencies) provide more informative evaluation than accuracy or ROC-based metrics, focusing on relevant performance for minority classes while accounting for class distribution.


## Pandas

### Qn 01: Which method efficiently applies a function along an axis of a DataFrame?

**Answer:** df.apply(func, axis=0)

**Explanation:** The apply() method allows applying a function along an axis (rows or columns) of a DataFrame.

### Qn 02: What's the correct way to merge two DataFrames on multiple columns?

**Answer:** Both A and C

**Explanation:** Both pd.merge() and DataFrame.merge() methods can merge on multiple columns specified as lists.

### Qn 03: How do you handle missing values in a DataFrame column?

**Answer:** All of the above

**Explanation:** All listed methods can handle missing values: fillna() replaces NaNs, dropna() removes rows with NaNs, and replace() can substitute NaNs with specified values.

### Qn 04: What does the method `groupby().agg()` allow you to do?

**Answer:** All of the above

**Explanation:** The agg() method is versatile and can apply single or multiple functions to grouped data, either to all columns or selectively.

### Qn 05: Which of the following transforms a DataFrame to a long format?

**Answer:** All of the above

**Explanation:** stack(), melt(), and wide_to_long() all convert data from wide format to long format, albeit with different approaches and parameters.

### Qn 06: How can you efficiently select rows where a column value meets a complex condition?

**Answer:** Both B and C

**Explanation:** Both loc with boolean indexing (with proper parentheses) and query() method can filter data based on complex conditions.

### Qn 07: What's the most efficient way to calculate a rolling 7-day average of a time series?

**Answer:** df['rolling_avg'] = df['value'].rolling(window=7).mean()

**Explanation:** The rolling() method with a window of 7 followed by mean() calculates a rolling average over a 7-period window.

### Qn 08: How do you perform a pivot operation in pandas?

**Answer:** All of the above

**Explanation:** All three methods can perform pivot operations, with pivot_table being more flexible as it can aggregate duplicate entries.

### Qn 09: Which method can reshape a DataFrame by stacking column labels to rows?

**Answer:** df.stack()

**Explanation:** stack() method pivots the columns of a DataFrame to become the innermost index level, creating a Series with a MultiIndex.

### Qn 10: How do you efficiently concatenate many DataFrames with identical columns?

**Answer:** pd.concat([df1, df2, df3])

**Explanation:** pd.concat() is designed to efficiently concatenate pandas objects along a particular axis with optional set logic.

### Qn 11: What's the correct way to create a DatetimeIndex from a column containing date strings?

**Answer:** All of the above

**Explanation:** All methods will correctly convert date strings to datetime objects, with different approaches to setting them as the index.

### Qn 12: Which method performs a cross-tabulation of two factors?

**Answer:** All of the above

**Explanation:** All methods can create cross-tabulations, though crosstab() is specifically designed for this purpose.

### Qn 13: How do you calculate cumulative statistics in pandas?

**Answer:** df.cumsum(), df.cumprod(), df.cummax(), df.cummin()

**Explanation:** The cum- methods (cumsum, cumprod, cummax, cummin) calculate cumulative statistics along an axis.

### Qn 14: Which approach efficiently calculates the difference between consecutive rows in a DataFrame?

**Answer:** Both A and B

**Explanation:** Both subtracting a shifted DataFrame and using diff() calculate element-wise differences between consecutive rows.

### Qn 15: How do you create a MultiIndex DataFrame from scratch?

**Answer:** All of the above

**Explanation:** All three methods create equivalent MultiIndex objects using different approaches: from_tuples, from_product, and from_arrays.

### Qn 16: Which method is most appropriate for performing complex string operations on DataFrame columns?

**Answer:** All of the above work, but B is most efficient

**Explanation:** While all methods can transform strings, the .str accessor provides vectorized string functions that are generally more efficient than apply() or map().

### Qn 17: What's the best way to compute percentiles for grouped data?

**Answer:** Both A and C

**Explanation:** Both quantile() and describe() can compute percentiles for grouped data, with describe() providing additional statistics. For option B, While this approach uses the right function (numpy's percentile), there's an issue with how it's implemented in the context of pandas GroupBy. This would likely raise errors because the lambda function returns arrays rather than scalars, which is problematic for the standard aggregation pipeline.

### Qn 18: How do you efficiently implement a custom aggregation function that requires the entire group?

**Answer:** df.groupby('group').apply(custom_func)

**Explanation:** apply() is designed for operations that need the entire group as a DataFrame, whereas agg() is better for operations that can be vectorized.

### Qn 19: What's the most memory-efficient way to read a large CSV file with pandas?

**Answer:** pd.read_csv('file.csv', dtype={'col1': 'category', 'col2': 'int8'})

**Explanation:** Specifying appropriate dtypes, especially using 'category' for string columns with repeated values, significantly reduces memory usage.

### Qn 20: Which method is correct for resampling time series data to monthly frequency?

**Answer:** Both A and B

**Explanation:** Both resample() and groupby() with Grouper can aggregate time series data to monthly frequency, though asfreq() only changes frequency without aggregation.

### Qn 21: How do you efficiently identify and remove duplicate rows in a DataFrame?

**Answer:** Both A and B

**Explanation:** Both df[~df.duplicated()] and df.drop_duplicates() remove duplicate rows, with the latter being more readable and offering more options.

### Qn 22: Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?

**Answer:** df.pipe(custom_func)

**Explanation:** pipe() is designed for functions that take and return a DataFrame, creating readable method chains when applying multiple functions.

### Qn 23: How do you sample data from a DataFrame with weights?

**Answer:** Both B and C

**Explanation:** Both approaches correctly sample with weights, though weights don't need to be normalized as pandas normalizes them internally.

### Qn 24: What's the correct way to use the pd.cut() function for binning continuous data?

**Answer:** All of the above are valid uses

**Explanation:** All approaches are valid: using explicit bin edges, equal-width bins (cut), or equal-frequency bins (qcut).

### Qn 25: How do you efficiently perform a custom window operation in pandas?

**Answer:** Both A and B

**Explanation:** Both approaches work for custom window operations, but using raw=True can be more efficient for numerical operations by passing a NumPy array instead of a Series.

### Qn 26: Which approach can create a lagged feature in a time series DataFrame?

**Answer:** Both A and B

**Explanation:** shift(1) creates a lag (past values), while shift(-1) creates a lead (future values), both useful for time series analysis.

### Qn 27: What's the best way to explode a DataFrame column containing lists into multiple rows?

**Answer:** Both B and C

**Explanation:** explode() transforms each element of a list-like column into a row, with the original index duplicated as needed.

### Qn 28: How do you efficiently compute a weighted mean in pandas?

**Answer:** Both A and C

**Explanation:** Both manually computing weighted mean and using np.average() work efficiently, though pandas Series doesn't have a weights parameter for mean().

### Qn 29: Which method correctly identifies the top-k values in each group?

**Answer:** All of the above

**Explanation:** All three methods can get the top-k values within each group, with different syntax but similar results.

### Qn 30: What's the best way to add a new column based on a categorical mapping of an existing column?

**Answer:** All of the above

**Explanation:** All methods can map values to new ones, though map() is generally preferred for dictionary-based mappings.


## Probability

### Qn 01: In a room of 30 people, what is the probability that at least two people share the same birthday (assuming 365 days in a year and birthdays are uniformly distributed)?

**Answer:** Greater than 90%

**Explanation:** This is the famous birthday paradox. The probability is computed as 1 minus the probability that all birthdays are different. For 30 people, P(at least one shared birthday) = 1 - (365/365 × 364/365 × 363/365 × ... × 336/365) ≈ 0.706, which is approximately 70.6%. While this is closest to 'About 70%', the exact calculation gives 70.6%, not 'Greater than 90%'. The probability exceeds 90% when there are 41 or more people in the room.

### Qn 02: A biased coin has a 60% chance of landing heads. If you flip this coin 5 times, what is the probability of getting exactly 3 heads?

**Answer:** 0.34560

**Explanation:** We use the binomial probability formula: P(X=k) = C(n,k) × p^k × (1-p)^(n-k), where n=5, k=3, p=0.6. C(5,3) = 10 possible ways to arrange 3 heads in 5 flips. So P(X=3) = 10 × (0.6)^3 × (0.4)^2 = 10 × 0.216 × 0.16 = 0.3456 or 34.56%.

### Qn 03: In a standard deck of 52 cards, what is the probability of drawing a royal flush (A, K, Q, J, 10 of the same suit) in a 5-card poker hand?

**Answer:** 4/2,598,960

**Explanation:** There are C(52,5) = 2,598,960 possible 5-card hands from a standard deck. There are exactly 4 possible royal flushes (one for each suit). Therefore, the probability is 4/2,598,960 = 1/649,740 or approximately 0.00000154.

### Qn 04: A standard six-sided die is rolled 3 times. What is the probability that the sum of the three rolls equals 10?

**Answer:** 1/8

**Explanation:** When rolling 3 dice, there are 6^3 = 216 possible outcomes. To get a sum of 10, we can have combinations like (1,3,6), (2,2,6), etc. Counting all such combinations gives us 27 favorable outcomes. Therefore, the probability is 27/216 = 1/8.

### Qn 05: A bag contains 5 red marbles and 7 blue marbles. If 3 marbles are drawn without replacement, what is the probability that exactly 2 of them are red?

**Answer:** 35/132

**Explanation:** Total ways to select 3 marbles from 12 is C(12,3) = 220. Ways to select 2 red marbles from 5 red and 1 blue marble from 7 blue is C(5,2) × C(7,1) = 10 × 7 = 70. Therefore probability = 70/220 = 7/22 = 35/110 = 7/22.

### Qn 06: In a Bayesian analysis, a disease has a 1% prevalence in a population. A test for the disease has 95% sensitivity (true positive rate) and 90% specificity (true negative rate). If a person tests positive, what is the probability they actually have the disease?

**Answer:** Around 8.7%

**Explanation:** Using Bayes' theorem: P(Disease|Positive) = [P(Positive|Disease) × P(Disease)] / P(Positive). P(Positive) = P(Positive|Disease) × P(Disease) + P(Positive|No Disease) × P(No Disease) = 0.95 × 0.01 + 0.10 × 0.99 = 0.0095 + 0.099 = 0.1085. Therefore, P(Disease|Positive) = (0.95 × 0.01) / 0.1085 ≈ 0.0095/0.1085 ≈ 0.0876 or about 8.7%.

### Qn 07: You roll a fair 6-sided die repeatedly until you get a 6. What is the expected number of rolls needed?

**Answer:** 6

**Explanation:** This is a geometric distribution with probability of success p = 1/6. The expected value of a geometric distribution is 1/p. So the expected number of rolls needed is 1/(1/6) = 6.

### Qn 08: In a group of 5 people, what is the probability that at least 2 people have the same zodiac sign (assuming zodiac signs are uniformly distributed across 12 possible signs)?

**Answer:** 0.96

**Explanation:** The probability that all 5 people have different zodiac signs is (12/12) × (11/12) × (10/12) × (9/12) × (8/12) = 0.0397. Therefore, the probability that at least 2 people share a zodiac sign is 1 - 0.0397 = 0.9603 ≈ 0.96 or 96%.

### Qn 09: A data scientist applies a machine learning model to classify emails as spam or not spam. The model has 98% accuracy on legitimate emails and 95% accuracy on spam emails. If 20% of all incoming emails are spam, what is the probability that an email classified as spam by the model is actually spam?

**Answer:** 0.83

**Explanation:** Using Bayes' theorem: P(Spam|Classified as Spam) = [P(Classified as Spam|Spam) × P(Spam)] / P(Classified as Spam). P(Classified as Spam) = P(Classified as Spam|Spam) × P(Spam) + P(Classified as Spam|Not Spam) × P(Not Spam) = 0.95 × 0.2 + 0.02 × 0.8 = 0.19 + 0.016 = 0.206. Therefore, P(Spam|Classified as Spam) = (0.95 × 0.2) / 0.206 = 0.19/0.206 ≈ 0.83 or 83%.

### Qn 10: Four cards are randomly selected from a standard 52-card deck. What is the probability of getting exactly 2 aces?

**Answer:** 0.0399

**Explanation:** Total number of ways to select 4 cards from 52 is C(52,4) = 270,725. Ways to select exactly 2 aces from 4 aces is C(4,2) = 6. Ways to select the other 2 cards from the non-ace cards is C(48,2) = 1,128. So favorable outcomes = 6 × 1,128 = 6,768. Probability = 6,768/270,725 ≈ 0.0399 or about 4%.

### Qn 11: In a standard normal distribution, what is the probability that a randomly selected observation falls between -1.96 and 1.96?

**Answer:** 0.95

**Explanation:** In a standard normal distribution, the area between z-scores of -1.96 and 1.96 corresponds to 95% of the distribution. This is a fundamental value in statistics, often used for 95% confidence intervals.

### Qn 12: A manufacturing process has a 3% defect rate. If 50 items are randomly selected, what is the probability that at most 2 are defective?

**Answer:** 0.6063

**Explanation:** This follows a binomial distribution with n=50 and p=0.03. P(X ≤ 2) = P(X=0) + P(X=1) + P(X=2) = C(50,0) × (0.03)^0 × (0.97)^50 + C(50,1) × (0.03)^1 × (0.97)^49 + C(50,2) × (0.03)^2 × (0.97)^48 ≈ 0.2231 + 0.3453 + 0.2379 = 0.6063 or about 60.63%.

### Qn 13: In the Monty Hall problem, you're on a game show with three doors. Behind one door is a car; behind the others are goats. You pick a door. The host, who knows what's behind each door, opens one of the other doors to reveal a goat. He then offers you the chance to switch your choice to the remaining unopened door. What is the probability of winning the car if you switch?

**Answer:** 2/3

**Explanation:** Initially, you have a 1/3 probability of choosing the car and a 2/3 probability of choosing a goat. If you initially chose the car (probability 1/3), switching will always make you lose. If you initially chose a goat (probability 2/3), the host will reveal the other goat, and switching will always make you win. Therefore, the probability of winning by switching is 2/3.

### Qn 14: A researcher is testing a new drug. In reality, the drug has no effect, but the researcher will conclude it works if the p-value is less than 0.05. What is the probability that the researcher incorrectly concludes the drug works?

**Answer:** 0.05

**Explanation:** The p-value is the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true. Here, the null hypothesis is that the drug has no effect (which is actually true). By definition, the probability of getting a p-value below 0.05 when the null hypothesis is true (Type I error) is 0.05 or 5%.

### Qn 15: A database has 1,000,000 records, and a data scientist estimates that 50 records are corrupted. If the data scientist randomly samples 100 records for manual inspection, what is the probability of finding at least one corrupted record?

**Answer:** 0.4988

**Explanation:** The probability of selecting a corrupted record is 50/1,000,000 = 0.00005. The probability of not finding any corrupted records in 100 samples is (1 - 0.00005)^100 ≈ 0.9950. Therefore, the probability of finding at least one corrupted record is 1 - 0.9950 = 0.0050 or 0.5%.

### Qn 16: In a certain city, 60% of days are sunny, 30% are cloudy, and 10% are rainy. The probability of a traffic jam is 0.1 on sunny days, 0.3 on cloudy days, and 0.5 on rainy days. If there is a traffic jam today, what is the probability that it is a sunny day?

**Answer:** 0.33

**Explanation:** Using Bayes' theorem: P(Sunny|Traffic Jam) = [P(Traffic Jam|Sunny) × P(Sunny)] / P(Traffic Jam). P(Traffic Jam) = 0.1 × 0.6 + 0.3 × 0.3 + 0.5 × 0.1 = 0.06 + 0.09 + 0.05 = 0.2. Therefore, P(Sunny|Traffic Jam) = (0.1 × 0.6) / 0.2 = 0.06/0.2 = 0.3 or 30%.

### Qn 17: Five fair six-sided dice are rolled. What is the probability that all five dice show different numbers?

**Answer:** 0.0926

**Explanation:** Total number of possible outcomes when rolling 5 dice is 6^5 = 7,776. For all dice to show different numbers, we can arrange 5 different numbers from the set {1,2,3,4,5,6} in 6!/1! = 720 ways. Therefore, the probability is 720/7,776 = 0.0926 or about 9.26%.

### Qn 18: A data center has 5 servers, each with a 1% probability of failing in a given day, independently of the others. What is the probability that at least one server fails today?

**Answer:** 0.0490

**Explanation:** The probability that a specific server doesn't fail is 0.99. The probability that all servers don't fail is (0.99)^5 ≈ 0.9510. Therefore, the probability that at least one server fails is 1 - 0.9510 = 0.0490 or about 4.9%.

### Qn 19: In a random sample of 20 people, what is the probability that at least 2 people were born in the same month of the year (assuming uniform distribution of birth months)?

**Answer:** 0.9139

**Explanation:** The probability that all 20 people were born in different months is 0 since there are only 12 months. The probability that no two people share the same birth month in a sample of 12 or fewer is calculated using the birthday problem formula for 12 months: 1 - P(no matching months) = 1 - (12!/12^n × (12-n)!) for n=12. For n>12, the probability of at least one match is 1.

### Qn 20: A biased coin has an unknown probability p of landing heads. After 10 flips, you observe 7 heads. Using a uniform prior distribution for p, what is the expected value of p according to Bayesian analysis?

**Answer:** 0.636

**Explanation:** With a uniform prior distribution (Beta(1,1)) and 7 heads out of 10 flips, the posterior distribution is Beta(1+7, 1+3) = Beta(8, 4). The expected value of a Beta(α, β) distribution is α/(α+β). So the expected value of p is 8/(8+4) = 8/12 = 2/3 ≈ 0.667, which is closest to 0.636.

### Qn 21: In a multiple-choice test with 5 questions, each question has 4 options with exactly one correct answer. If a student guesses randomly on all questions, what is the probability of getting at least 3 questions correct?

**Answer:** 0.0537

**Explanation:** The probability of getting a single question correct by random guessing is 1/4 = 0.25. Using the binomial distribution with n=5 and p=0.25: P(X ≥ 3) = P(X=3) + P(X=4) + P(X=5) = C(5,3) × (0.25)^3 × (0.75)^2 + C(5,4) × (0.25)^4 × (0.75)^1 + C(5,5) × (0.25)^5 × (0.75)^0 ≈ 0.0439 + 0.0073 + 0.0010 = 0.0522 or about 5.22%.

### Qn 22: In a hypergeometric distribution scenario, a shipment of 100 electronic components contains 8 defective parts. If 10 components are randomly selected without replacement for inspection, what is the probability of finding exactly 1 defective component?

**Answer:** 0.3816

**Explanation:** Using the hypergeometric probability mass function: P(X=1) = [C(8,1) × C(92,9)] / C(100,10) = [8 × 1,742,281,695] / 17,310,309,728 = 13,938,253,560 / 17,310,309,728 ≈ 0.3816 or about 38.16%.

### Qn 23: A fair six-sided die is rolled 10 times. What is the probability of getting exactly 2 sixes?

**Answer:** 0.2907

**Explanation:** This follows a binomial distribution with n=10 and p=1/6. P(X=2) = C(10,2) × (1/6)^2 × (5/6)^8 = 45 × (1/36) × (1,679,616/1,679,616) = 45/36 × 0.2323 ≈ 1.25 × 0.2323 = 0.2904 or about 29.04%.

### Qn 24: In a large city, 45% of residents prefer public transportation, 35% prefer driving, and 20% prefer cycling. If three residents are randomly selected, what is the probability that at least one of them prefers cycling?

**Answer:** 0.488

**Explanation:** The probability that a selected resident does not prefer cycling is 1 - 0.2 = 0.8. The probability that none of the three selected residents prefers cycling is (0.8)^3 = 0.512. Therefore, the probability that at least one prefers cycling is 1 - 0.512 = 0.488 or 48.8%.

### Qn 25: A genetics researcher is studying a trait that is determined by two alleles. The dominant allele A occurs with probability 0.7 and the recessive allele a with probability 0.3. Assuming Hardy-Weinberg equilibrium, what is the probability of a randomly selected individual having the genotype Aa?

**Answer:** 0.42

**Explanation:** Under Hardy-Weinberg equilibrium, the probability of genotype Aa is 2pq, where p is the probability of allele A and q is the probability of allele a. So P(Aa) = 2 × 0.7 × 0.3 = 0.42 or 42%.

### Qn 26: In a Poisson process where events occur at an average rate of 3 per hour, what is the probability that exactly 2 events occur in a 1-hour period?

**Answer:** 0.224

**Explanation:** For a Poisson distribution with parameter λ=3, the probability mass function gives P(X=2) = e^(-λ) × λ^2 / 2! = e^(-3) × 3^2 / 2 = e^(-3) × 9 / 2 = 0.0498 × 4.5 ≈ 0.224 or about 22.4%.

### Qn 27: A data scientist is analyzing user engagement on a website. If the probability distribution of the number of pages viewed by a visitor follows a geometric distribution with p=0.2, what is the probability that a visitor views exactly 5 pages before leaving the site?

**Answer:** 0.082

**Explanation:** For a geometric distribution with parameter p=0.2, the probability mass function gives P(X=5) = p(1-p)^(k-1) = 0.2 × (0.8)^4 = 0.2 × 0.4096 = 0.08192 ≈ 0.082 or about 8.2%.

### Qn 28: A medical test for a disease has sensitivity (true positive rate) of 90% and specificity (true negative rate) of 95%. In a population where 2% of people have the disease, what is the positive predictive value (probability that a person with a positive test result actually has the disease)?

**Answer:** 0.27

**Explanation:** Using Bayes' theorem: PPV = P(Disease|Positive test) = [P(Positive test|Disease) × P(Disease)] / P(Positive test). P(Positive test) = P(Positive test|Disease) × P(Disease) + P(Positive test|No Disease) × P(No Disease) = 0.9 × 0.02 + 0.05 × 0.98 = 0.018 + 0.049 = 0.067. Therefore, PPV = (0.9 × 0.02) / 0.067 = 0.018/0.067 ≈ 0.27 or 27%.

### Qn 29: In a lottery where 5 numbers are drawn from 1 to 49 without replacement, what is the probability of matching exactly 3 numbers on a single ticket?

**Answer:** 0.015

**Explanation:** Total number of possible 5-number combinations is C(49,5) = 1,906,884. Ways to match exactly 3 numbers out of 5: You must match 3 of the winning numbers [C(5,3) = 10] and 2 of the non-winning numbers [C(44,2) = 946]. So favorable outcomes = 10 × 946 = 9,460. Probability = 9,460/1,906,884 ≈ 0.00496 or about 0.5%.

### Qn 30: In a random sample from a normal distribution with mean 100 and standard deviation 15, what is the probability that a single observation exceeds 125?

**Answer:** 0.0478

**Explanation:** Standardizing, z = (125 - 100)/15 = 1.67. The probability P(X > 125) = P(Z > 1.67) ≈ 0.0475 or about 4.75%, which is closest to 0.0478.

### Qn 31: In a randomized controlled trial, patients are randomly assigned to either treatment or control groups with equal probability. If 10 patients are enrolled, what is the probability that exactly 5 are assigned to the treatment group?

**Answer:** 0.246

**Explanation:** This follows a binomial distribution with n=10 and p=0.5. P(X=5) = C(10,5) × (0.5)^5 × (0.5)^5 = 252 × (0.5)^10 = 252/1024 = 0.2461 or about 24.6%.

### Qn 32: A data scientist runs 20 independent A/B tests, each with a 5% false positive rate (Type I error). What is the probability of observing at least one false positive result across all tests if none of the tested hypotheses are actually true?

**Answer:** 0.64

**Explanation:** The probability of not observing a false positive in a single test is 1 - 0.05 = 0.95. The probability of not observing any false positives in 20 independent tests is (0.95)^20 ≈ 0.358. Therefore, the probability of observing at least one false positive is 1 - 0.358 = 0.642 or about 64.2%.

### Qn 33: Two fair six-sided dice are rolled. Given that the sum of the dice is greater than 7, what is the probability that at least one die shows a 6?

**Answer:** 7/12

**Explanation:** The possible outcomes for sum > 7 are: (2,6), (3,5), (3,6), (4,4), (4,5), (4,6), (5,3), (5,4), (5,5), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6) - a total of 15 outcomes. Of these, 11 outcomes include at least one 6: (2,6), (3,6), (4,6), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6). Therefore, the probability is 11/15 = 11/15 or about 73.3%.


## Python Advanced

### Qn 01: What is the time complexity of inserting an element at the beginning of a Python list?

**Answer:** O(n)

**Explanation:** Inserting at the beginning of a Python list requires shifting all elements, hence O(n).

### Qn 02: Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?

**Answer:** NumPy arrays

**Explanation:** NumPy arrays are memory efficient and optimized for numerical operations.

### Qn 03: Which Python library provides decorators and context managers to handle retries with exponential backoff?

**Answer:** tenacity

**Explanation:** Tenacity provides powerful retry strategies including exponential backoff.

### Qn 04: What is a key difference between multiprocessing and threading in Python?

**Answer:** Processes can utilize multiple CPUs

**Explanation:** Due to the GIL, threads are limited; multiprocessing uses separate memory space and cores.

### Qn 05: What is the purpose of Python's `__slots__` declaration?

**Answer:** Reduce memory usage by preventing dynamic attribute creation

**Explanation:** `__slots__` limits attribute assignment and avoids `__dict__` overhead.

### Qn 06: What will `functools.lru_cache` do?

**Answer:** Cache function output to speed up subsequent calls

**Explanation:** `lru_cache` stores results of expensive function calls for reuse.

### Qn 07: What does the `@staticmethod` decorator do in Python?

**Answer:** Defines a method that takes no self or cls argument

**Explanation:** `@staticmethod` defines a method that does not receive an implicit first argument.

### Qn 08: How can you profile memory usage in a Python function?

**Answer:** Using tracemalloc

**Explanation:** `tracemalloc` tracks memory allocations in Python.

### Qn 09: Which built-in function returns the identity of an object?

**Answer:** id()

**Explanation:** `id()` returns the identity (memory address) of an object.

### Qn 10: What happens when you use the `is` operator between two equal strings in Python?

**Answer:** It compares object identity

**Explanation:** `is` checks whether two variables point to the same object, not if their values are equal.


## Python General

### Qn 01: What is the output of `len('Python')`?

**Answer:** 6

**Explanation:** The string 'Python' has 6 characters.

### Qn 02: What is the output of `type([])` in Python?

**Answer:** <class 'list'>

**Explanation:** `[]` represents an empty list in Python.

### Qn 03: Which data structure allows duplicate elements?

**Answer:** List

**Explanation:** Lists in Python can contain duplicate elements.

### Qn 04: What is the result of `5 // 2`?

**Answer:** 2

**Explanation:** `//` is floor division; it returns the largest whole number less than or equal to the result.

### Qn 05: Which of the following is a mutable data type?

**Answer:** list

**Explanation:** Lists are mutable in Python, allowing modifications.

### Qn 06: How do you insert a comment in Python?

**Answer:** # comment

**Explanation:** Python uses the `#` symbol to indicate a comment.

### Qn 07: Which of the following is used to handle exceptions in Python?

**Answer:** try-except

**Explanation:** Python uses `try-except` blocks to handle exceptions.

### Qn 08: What keyword is used to define a function in Python?

**Answer:** def

**Explanation:** The `def` keyword is used to define functions in Python.

### Qn 09: What does `range(3)` return?

**Answer:** [0, 1, 2]

**Explanation:** `range(3)` generates numbers starting from 0 up to (but not including) 3.


## SQL General

### Qn 01: Which SQL statement is used to extract data from a database?

**Answer:** SELECT

**Explanation:** The SELECT statement is used to extract data from a database table.

### Qn 02: Which SQL clause is used to filter records?

**Answer:** WHERE

**Explanation:** The WHERE clause is used to filter records based on specific conditions.

### Qn 03: What does the COUNT() function do in SQL?

**Answer:** Counts non-NULL rows

**Explanation:** COUNT() returns the number of non-NULL values in a specified column.

### Qn 04: Which SQL keyword is used to sort the result-set?

**Answer:** ORDER BY

**Explanation:** ORDER BY is used to sort the results of a SELECT query.

### Qn 05: Which command is used to remove all records from a table in SQL without deleting the table?

**Answer:** TRUNCATE

**Explanation:** TRUNCATE removes all records from a table but retains the table structure.

### Qn 06: Which SQL clause is used with aggregate functions to group result-set by one or more columns?

**Answer:** GROUP BY

**Explanation:** GROUP BY groups rows that have the same values into summary rows.

### Qn 07: Which SQL keyword is used to retrieve only distinct values?

**Answer:** DISTINCT

**Explanation:** DISTINCT is used to return only different (distinct) values.

### Qn 08: Which of the following is a DDL command?

**Answer:** CREATE

**Explanation:** CREATE is a DDL (Data Definition Language) command used to create a new table or database.

### Qn 09: What does the SQL INNER JOIN keyword do?

**Answer:** Returns rows when there is a match in both tables

**Explanation:** INNER JOIN selects records that have matching values in both tables.

### Qn 10: What will the result of the query 'SELECT * FROM employees WHERE department IS NULL;' be?

**Answer:** It selects employees with NULL department

**Explanation:** IS NULL checks for columns that contain NULL values.


## SQL Sqlite

### Qn 01: What is the purpose of the `WITHOUT ROWID` clause in SQLite?

**Answer:** To create a table without the implicit ROWID column

**Explanation:** `WITHOUT ROWID` creates a table without the implicit `ROWID`, useful for certain optimizations.

### Qn 02: Which function would you use in SQLite to get the current timestamp?

**Answer:** CURRENT_TIMESTAMP

**Explanation:** `CURRENT_TIMESTAMP` returns the current date and time in SQLite.

### Qn 03: What is the default data type of a column in SQLite if not specified?

**Answer:** NONE

**Explanation:** If no type is specified, SQLite assigns it an affinity of NONE.

### Qn 04: How are boolean values stored in SQLite?

**Answer:** As 1 and 0 integers

**Explanation:** SQLite does not have a separate BOOLEAN type; it uses integers 1 (true) and 0 (false).

### Qn 05: Which of the following is true about SQLite's `VACUUM` command?

**Answer:** It compacts the database file

**Explanation:** `VACUUM` rebuilds the database file to defragment it and reduce its size.

### Qn 06: Which SQLite command lists all tables in the database?

**Answer:** SELECT * FROM sqlite_master WHERE type='table'

**Explanation:** SQLite uses the `sqlite_master` table to store metadata about the database, including table names.

### Qn 07: Which SQLite command allows you to see the schema of a table?

**Answer:** .schema

**Explanation:** `.schema` is a command in the SQLite shell that shows the schema for tables.

### Qn 08: How does SQLite handle foreign key constraints by default?

**Answer:** They are off by default and must be enabled

**Explanation:** SQLite supports foreign keys, but enforcement must be enabled with `PRAGMA foreign_keys = ON`.

### Qn 09: How does SQLite implement AUTOINCREMENT?

**Answer:** Using INTEGER PRIMARY KEY

**Explanation:** SQLite uses `INTEGER PRIMARY KEY AUTOINCREMENT` to create an auto-incrementing ID.

### Qn 10: What pragma statement turns on write-ahead logging in SQLite?

**Answer:** PRAGMA journal_mode = WAL

**Explanation:** `PRAGMA journal_mode = WAL` enables write-ahead logging in SQLite.


## Statistics

### Qn 01: What does the p-value represent in hypothesis testing?

**Answer:** Probability of obtaining test results at least as extreme as the results actually observed

**Explanation:** The p-value quantifies the evidence against the null hypothesis. A small p-value suggests the observed data is unlikely under the null hypothesis.

### Qn 02: What is the main difference between population and sample in statistics?

**Answer:** Sample is a subset of population

**Explanation:** A population includes all elements from a set of data, while a sample consists of one or more observations drawn from the population.

### Qn 03: What does standard deviation measure?

**Answer:** Spread of data

**Explanation:** Standard deviation measures the amount of variation or dispersion of a set of values.

### Qn 04: In a normal distribution, what percentage of data lies within one standard deviation of the mean?

**Answer:** 68%

**Explanation:** In a normal distribution, approximately 68% of the data falls within one standard deviation of the mean.

### Qn 05: Which of the following measures of central tendency is not affected by extreme values?

**Answer:** Median

**Explanation:** Median is the middle value and is not affected by extremely large or small values, unlike the mean.


## Timeseries

### Qn 01: What does the 'AR' component in ARIMA represent, and how does it capture patterns in time series data?

**Answer:** Autoregressive - using past values to predict future values

**Explanation:** The 'AR' in ARIMA stands for Autoregressive, which means the model uses past values of the time series to predict future values. Specifically, an AR(p) model uses p previous time steps as predictors. For example, in an AR(2) model, the current value is predicted using a linear combination of the previous two values, plus an error term. This component is particularly useful for capturing momentum or inertia in time series where recent values influence future values.

### Qn 02: What does the 'I' component in ARIMA represent, and why is it necessary?

**Answer:** Integrated - differencing to achieve stationarity

**Explanation:** The 'I' in ARIMA stands for Integrated, which refers to differencing the time series to achieve stationarity. Many time series have trends or seasonal patterns that make them non-stationary. The 'd' parameter in ARIMA(p,d,q) indicates how many times the data needs to be differenced to achieve stationarity. For example, if d=1, we take the difference between consecutive observations. This transformation is necessary because ARIMA models assume the underlying process is stationary, meaning its statistical properties do not change over time.

### Qn 03: What does the 'MA' component in ARIMA represent, and how does it differ from AR?

**Answer:** Moving Average - using past forecast errors in the model

**Explanation:** The 'MA' in ARIMA stands for Moving Average, which incorporates past forecast errors (residuals) into the model rather than past values of the time series itself. An MA(q) model uses the previous q forecast errors as predictors. This differs fundamentally from AR, which uses the actual past values. MA components capture the short-term reactions to past shocks or random events in the system. For example, an MA(1) model would use the forecast error from the previous time step to adjust the current prediction.

### Qn 04: How do you interpret the parameters p, d, and q in ARIMA(p,d,q)?

**Answer:** p = AR order, d = differencing order, q = MA order

**Explanation:** In ARIMA(p,d,q), p represents the order of the autoregressive (AR) component, indicating how many lagged values of the series are included in the model. A higher p means more past values are used for prediction. The parameter d represents the degree of differencing required to make the series stationary, with d=1 meaning first difference, d=2 meaning second difference, etc. Finally, q is the order of the moving average (MA) component, indicating how many lagged forecast errors are included in the model. Together, these parameters define the structure of the ARIMA model and must be carefully selected based on the characteristics of the time series.

### Qn 05: What is the key assumption that must be satisfied before applying ARIMA models?

**Answer:** The time series must be stationary

**Explanation:** The fundamental assumption for ARIMA models is that the time series is stationary or can be made stationary through differencing. A stationary time series has constant mean, variance, and autocorrelation structure over time. Without stationarity, the model cannot reliably learn patterns from the data. This is why the 'I' (Integrated) component exists in ARIMA - to transform non-stationary data through differencing. Analysts typically use statistical tests like the Augmented Dickey-Fuller (ADF) test to check for stationarity before applying ARIMA models.

### Qn 06: How can you determine the appropriate values for p and q in an ARIMA(p,d,q) model?

**Answer:** By examining ACF and PACF plots

**Explanation:** The appropriate values for p and q in an ARIMA model can be determined by examining the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of the stationary time series. For identifying the AR order (p), look for significant spikes in the PACF that cut off after lag p. For the MA order (q), look for significant spikes in the ACF that cut off after lag q. Additionally, information criteria like AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) can be used to compare different model specifications and select the best combination of parameters.

### Qn 07: What is the purpose of the Augmented Dickey-Fuller (ADF) test in time series analysis?

**Answer:** To test for stationarity

**Explanation:** The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary or not. The null hypothesis of the test is that the time series contains a unit root, implying it is non-stationary. If the p-value from the test is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the series is stationary. This test is crucial before applying ARIMA models because stationarity is a key assumption. The test includes lags of the differenced series to account for serial correlation, making it more robust than the simple Dickey-Fuller test.

### Qn 08: What is the difference between ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)?

**Answer:** ACF measures correlation between series and lagged values accounting for intermediate lags, PACF removes indirect correlation effects

**Explanation:** The ACF (Autocorrelation Function) measures the correlation between a time series and its lagged values, including both direct and indirect effects. It shows the correlation at each lag without controlling for correlations at shorter lags. In contrast, the PACF (Partial Autocorrelation Function) measures the correlation between a time series and its lagged values while controlling for the values of the time series at all shorter lags. This effectively removes the indirect correlation effects, showing only the direct relationship between observations separated by a specific lag. ACF helps identify MA(q) order, while PACF helps identify AR(p) order in ARIMA modeling.

### Qn 09: What additional component does SARIMAX add compared to ARIMA?

**Answer:** Seasonal components and exogenous variables

**Explanation:** SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) extends ARIMA by adding two important capabilities. First, it incorporates seasonal components, allowing the model to capture repeating patterns that occur at fixed intervals (like daily, weekly, or yearly seasonality). The seasonal component is specified with parameters (P,D,Q)m, where m is the seasonal period. Second, SARIMAX allows for exogenous variables (the 'X' part), which are external factors that can influence the time series but are not part of the series itself. These could include variables like temperature affecting energy consumption, or promotions affecting sales. This makes SARIMAX much more versatile than standard ARIMA for real-world applications with seasonal patterns and external influences.

### Qn 10: How are seasonality parameters represented in a SARIMA model?

**Answer:** As (P,D,Q)m where m is the seasonal period

**Explanation:** In a SARIMA (Seasonal ARIMA) model, seasonality parameters are represented as (P,D,Q)m, where P is the seasonal autoregressive order, D is the seasonal differencing order, Q is the seasonal moving average order, and m is the number of periods in each season (the seasonal period). For example, in monthly data with yearly seasonality, m would be 12. In a SARIMA(1,1,1)(1,1,1)12 model, the non-seasonal components are (1,1,1) and the seasonal components are (1,1,1)12. The seasonal components operate at lag m, 2m, etc., capturing patterns that repeat every m periods. Seasonal differencing (D) involves subtracting the value from m periods ago, helping to remove seasonal non-stationarity.

### Qn 11: What does it mean when we say a time series exhibits 'stationarity'?

**Answer:** Its statistical properties remain constant over time

**Explanation:** A stationary time series has statistical properties that remain constant over time. Specifically, it has a constant mean, constant variance, and a constant autocorrelation structure. This means the process generating the time series is in statistical equilibrium. Stationarity is a crucial assumption for many time series models, including ARIMA, because it ensures that patterns learned from historical data will continue to be valid in the future. Non-stationary series might have trends (changing mean) or heteroscedasticity (changing variance), which can lead to unreliable forecasts if not properly addressed through transformations like differencing or variance stabilization.

### Qn 12: What is the purpose of differencing in time series analysis?

**Answer:** To remove trends and achieve stationarity

**Explanation:** Differencing in time series analysis involves computing the differences between consecutive observations. The primary purpose is to remove trends and achieve stationarity, which is a key requirement for ARIMA modeling. First-order differencing (d=1) can eliminate linear trends by calculating Yt - Yt-1. If the series still shows non-stationarity after first differencing, second-order differencing (d=2) can be applied to remove quadratic trends. However, over-differencing can introduce unnecessary complexity and artificial patterns, so it's important to use statistical tests like the ADF test to determine the appropriate level of differencing needed.

### Qn 13: What is seasonal differencing and when should it be applied?

**Answer:** Calculating differences between observations separated by the seasonal period, applied when there's seasonal non-stationarity

**Explanation:** Seasonal differencing involves calculating differences between observations separated by the seasonal period (e.g., 12 months for monthly data with yearly seasonality). It's represented by the D parameter in SARIMA models and is applied when the time series exhibits seasonal non-stationarity, meaning the seasonal pattern changes over time. For example, with monthly data, seasonal differencing would compute Yt - Yt-12. This helps remove repeating seasonal patterns just as regular differencing removes trends. You should apply seasonal differencing when visual inspection shows persistent seasonal patterns after regular differencing, or when seasonal unit root tests indicate seasonal non-stationarity.

### Qn 14: What are residuals in the context of ARIMA modeling, and why are they important?

**Answer:** The differences between observed and predicted values, important for diagnostic checking

**Explanation:** In ARIMA modeling, residuals are the differences between the observed values and the values predicted by the model. They represent the part of the data that the model couldn't explain. Residuals are crucial for diagnostic checking because a well-fitted ARIMA model should have residuals that resemble white noise - they should be uncorrelated, have zero mean, constant variance, and follow a normal distribution. If patterns remain in the residuals, it suggests the model hasn't captured all the systematic information in the time series. Common residual diagnostics include ACF/PACF plots of residuals, the Ljung-Box test for autocorrelation, and Q-Q plots for normality checking.

### Qn 15: What does the Ljung-Box test evaluate in time series analysis?

**Answer:** Whether residuals exhibit autocorrelation

**Explanation:** The Ljung-Box test is a statistical test used to evaluate whether residuals from a time series model exhibit autocorrelation. The null hypothesis is that the residuals are independently distributed (i.e., no autocorrelation). If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the residuals contain significant autocorrelation, suggesting the model hasn't captured all the patterns in the data. The test examines multiple lags simultaneously, making it more comprehensive than just looking at individual autocorrelation values. A good ARIMA model should have residuals that pass the Ljung-Box test, indicating they approximate white noise.

### Qn 16: What is the primary difference between ARMA and ARIMA models?

**Answer:** ARIMA includes differencing for non-stationary data, while ARMA requires stationary data

**Explanation:** The primary difference between ARMA (AutoRegressive Moving Average) and ARIMA (AutoRegressive Integrated Moving Average) models is that ARIMA includes a differencing step (the 'I' component) to handle non-stationary data. ARMA models combine autoregressive (AR) and moving average (MA) components but assume that the time series is already stationary. ARIMA extends this by first differencing the data d times to achieve stationarity before applying the ARMA model. This makes ARIMA more versatile for real-world time series that often contain trends. Essentially, an ARIMA(p,d,q) model is equivalent to applying an ARMA(p,q) model to a time series after differencing it d times.

### Qn 17: What is meant by the 'order of integration' in time series analysis?

**Answer:** The number of times a series needs to be differenced to achieve stationarity

**Explanation:** The 'order of integration' refers to the number of times a time series needs to be differenced to achieve stationarity. It's represented by the parameter d in ARIMA(p,d,q) models. A series that requires differencing once (d=1) to become stationary is said to be integrated of order 1, or I(1). Similarly, a series requiring two differences is I(2). A naturally stationary series is I(0). The concept is important because it quantifies how persistent trends are in the data. Most economic and business time series are I(1), meaning they have stochastic trends that can be removed with first differencing. The order of integration can be determined using unit root tests like the Augmented Dickey-Fuller test.

### Qn 18: What is the purpose of the Box-Jenkins methodology in time series analysis?

**Answer:** A systematic approach to identify, estimate, and validate ARIMA models

**Explanation:** The Box-Jenkins methodology is a systematic approach to identify, estimate, and validate ARIMA models for time series forecasting. It consists of three main stages: identification, estimation, and diagnostic checking. In the identification stage, you determine appropriate values for p, d, and q by analyzing ACF/PACF plots and using stationarity tests. In the estimation stage, you fit the selected ARIMA model to the data and estimate its parameters. In the diagnostic checking stage, you analyze residuals to ensure they resemble white noise and refine the model if needed. Box-Jenkins emphasizes iterative model building, where you cycle through these stages until you find an adequate model. This methodical approach helps ensure that the final model captures the data's patterns efficiently.

### Qn 19: What information criterion is commonly used to select between different ARIMA models?

**Answer:** AIC (Akaike Information Criterion)

**Explanation:** The AIC (Akaike Information Criterion) is commonly used to select between different ARIMA models. It balances model fit against complexity by penalizing models with more parameters. The formula is AIC = -2log(L) + 2k, where L is the likelihood of the model and k is the number of parameters. A lower AIC value indicates a better model. When comparing ARIMA models with different p, d, and q values, analysts typically choose the model with the lowest AIC. Other similar criteria include BIC (Bayesian Information Criterion), which penalizes model complexity more heavily. These criteria help prevent overfitting by ensuring that additional parameters are only included if they substantially improve the model's fit to the data.

### Qn 20: In the context of ARIMA residual analysis, what should a Q-Q plot ideally show?

**Answer:** Points falling approximately along a straight line, indicating normally distributed residuals

**Explanation:** In ARIMA residual analysis, a Q-Q (Quantile-Quantile) plot should ideally show points falling approximately along a straight line. This indicates that the residuals follow a normal distribution, which is an assumption for valid statistical inference in ARIMA modeling. The Q-Q plot compares the quantiles of the residuals against the quantiles of a theoretical normal distribution. Deviations from the straight line suggest non-normality: a sigmoidal pattern indicates skewness, while an S-shaped curve suggests heavy or light tails compared to a normal distribution. Serious deviations might indicate model misspecification or the presence of outliers that could affect the reliability of confidence intervals and hypothesis tests for the model parameters.

### Qn 21: What is the meaning of the 'exogenous variables' in the context of SARIMAX models?

**Answer:** External predictor variables that influence the time series but are not influenced by it

**Explanation:** In SARIMAX models, exogenous variables (the 'X' part) are external predictor variables that influence the time series being modeled but are not influenced by it. These are independent variables that provide additional information beyond what's contained in the past values of the time series itself. For example, when forecasting electricity demand, temperature might be an exogenous variable since it affects demand but isn't affected by it. Unlike the autoregressive components that use the series' own past values, exogenous variables inject outside information into the model. This can significantly improve forecast accuracy when the time series is known to be affected by measurable external factors. Mathematically, exogenous variables enter the SARIMAX equation as a regression component.

### Qn 22: Why might you perform a Box-Cox transformation before applying an ARIMA model?

**Answer:** To stabilize variance and make the data more normally distributed

**Explanation:** A Box-Cox transformation is often performed before applying an ARIMA model to stabilize variance and make the data more normally distributed. Many time series exhibit heteroscedasticity (changing variance over time) or skewness, which can violate ARIMA assumptions. The Box-Cox transformation is a family of power transformations defined by the parameter λ: when λ=0, it's equivalent to a log transformation; when λ=1, it's essentially the original data (with a shift). The optimal λ value can be determined by maximizing the log-likelihood function. This transformation helps make the time series' variance more constant across time and its distribution more symmetric, leading to more reliable parameter estimates and prediction intervals in the ARIMA model.

### Qn 23: What does it mean when an ARIMA model is said to be 'invertible'?

**Answer:** The MA component can be rewritten as an infinite AR process

**Explanation:** In time series analysis, when an ARIMA model is said to be 'invertible,' it means that its Moving Average (MA) component can be rewritten as an infinite Autoregressive (AR) process. This property ensures that the MA coefficients decrease in impact as we go further back in time, allowing the process to be approximated by a finite AR model. Invertibility is a mathematical property that ensures a unique MA representation and stable forecasting. Technically, for invertibility, the roots of the MA polynomial must lie outside the unit circle. Without invertibility, different MA models could produce identical autocorrelation patterns, making identification problematic. Invertibility is analogous to stationarity for AR processes and is checked during the model estimation phase.

### Qn 24: What is the difference between strong and weak stationarity in time series?

**Answer:** Weak stationarity requires constant mean and variance and time-invariant autocorrelation; strong stationarity requires the entire distribution to be time-invariant

**Explanation:** The distinction between strong (strict) and weak stationarity lies in how much of the data's statistical properties must remain constant over time. Weak stationarity, which is usually sufficient for ARIMA modeling, requires only that the mean and variance remain constant and that the autocorrelation function depends only on the lag between points, not their absolute position in time. In contrast, strong stationarity is more demanding, requiring that the entire joint probability distribution of the process remains unchanged when shifted in time. This means all higher moments (not just the first two) must be constant, and all multivariate distributions (not just bivariate correlations) must be time-invariant. In practice, analysts typically work with weak stationarity because it's easier to test for and sufficient for many applications.

### Qn 25: What is the primary purpose of the KPSS test in time series analysis?

**Answer:** To test for stationarity with a null hypothesis of stationarity

**Explanation:** The KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test is used to test for stationarity in time series analysis, but unlike the ADF test, its null hypothesis is that the series is stationary. This reversal makes it a complementary test to ADF, which has a null hypothesis of non-stationarity. Using both tests together provides stronger evidence: if the ADF test rejects its null and the KPSS fails to reject its null, you have consistent evidence of stationarity. The KPSS test specifically tests whether the series can be described as stationary around a deterministic trend or has a unit root. A low p-value leads to rejecting the null, suggesting non-stationarity. This test is particularly useful for distinguishing between trend-stationary processes and difference-stationary processes.

### Qn 26: What does Facebook Prophet use to model seasonality in time series data?

**Answer:** Fourier series for multiple seasonal periods

**Explanation:** Facebook Prophet uses Fourier series to model seasonality in time series data. This approach represents seasonal patterns as a sum of sine and cosine terms of different frequencies, allowing for flexible modeling of complex seasonal patterns. Prophet can simultaneously model multiple seasonal periods (e.g., daily, weekly, and yearly seasonality) by using different Fourier series for each. The number of terms in each Fourier series (specified by the 'order' parameter) controls the flexibility of the seasonal component - higher orders capture more complex patterns but risk overfitting. This approach is particularly powerful because it can handle irregular time series and missing data better than traditional seasonal ARIMA models, which require regular time intervals.

### Qn 27: What are the three main components of a Facebook Prophet model?

**Answer:** Trend, seasonality, and holidays/events

**Explanation:** Facebook Prophet decomposes time series into three main components: trend, seasonality, and holidays/events. The trend component captures non-periodic changes, and can be modeled as either linear or logistic growth with automatic changepoint detection to accommodate trend changes. The seasonality component captures periodic patterns using Fourier series, and can simultaneously model multiple seasonal patterns (e.g., daily, weekly, annual). The holidays/events component accounts for irregular schedules and events that affect the time series but don't follow a seasonal pattern. Users can provide a custom list of holidays or events with their dates. By modeling these components separately and then adding them together, Prophet creates an interpretable forecast that can be easily understood and adjusted by analysts.
