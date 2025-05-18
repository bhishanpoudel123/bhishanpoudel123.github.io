# Data Science Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: What is the primary goal of data wrangling?](#q01)  
- [Qn 02: Which of the following is NOT a measure of central tendency?](#q02)  
- [Qn 03: What type of chart would be most appropriate for comparing proportions of a whole?](#q03)  
- [Qn 04: Which Python library is primarily used for working with tabular data structures?](#q04)  
- [Qn 05: What does the groupby() operation in Pandas return before aggregation?](#q05)  
- [Qn 06: What does 'NaN' represent in a Pandas DataFrame?](#q06)  
- [Qn 07: Which technique is NOT typically used for feature selection?](#q07)  
- [Qn 08: Which metric is NOT used to evaluate regression models?](#q08)  
- [Qn 09: What is the most common method for handling missing numerical data?](#q09)  
- [Qn 10: Which library is essential for numerical computing in Python?](#q10)  
- [Qn 11: What is the purpose of a correlation matrix?](#q11)  
- [Qn 12: What is the main advantage of using a box plot?](#q12)  
- [Qn 13: What does the term 'overfitting' refer to in machine learning?](#q13)  
- [Qn 14: Which of these is a supervised learning algorithm?](#q14)  
- [Qn 15: What is the purpose of a train-test split?](#q15)  
- [Qn 16: Which Python library is most commonly used for creating static visualizations?](#q16)  
- [Qn 17: What is the main purpose of normalization in data preprocessing?](#q17)  
- [Qn 18: What does SQL stand for?](#q18)  
- [Qn 19: Which of these is NOT a common data type in Pandas?](#q19)  
- [Qn 20: What is the primary use of the Scikit-learn library?](#q20)  
- [Qn 21: What is the difference between classification and regression?](#q21)  
- [Qn 22: What is a confusion matrix used for?](#q22)  
- [Qn 23: What does ETL stand for in data engineering?](#q23)  
- [Qn 24: Which of these is a dimensionality reduction technique?](#q24)  
- [Qn 25: What is the purpose of cross-validation?](#q25)  
- [Qn 26: What is the main advantage of using a Jupyter Notebook?](#q26)  
- [Qn 27: What is the purpose of one-hot encoding?](#q27)  
- [Qn 28: Which metric would you use for an imbalanced classification problem?](#q28)  
- [Qn 29: What is feature engineering?](#q29)  
- [Qn 30: What is the purpose of a ROC curve?](#q30)  
- [Qn 31: What is the main advantage of using a random forest over a single decision tree?](#q31)  
- [Qn 32: What is the purpose of the 'iloc' method in Pandas?](#q32)  
- [Qn 33: What is the difference between deep learning and traditional machine learning?](#q33)  
- [Qn 34: What is the purpose of a learning curve in machine learning?](#q34)  
- [Qn 35: What is the bias-variance tradeoff?](#q35)  
- [Qn 36: What is the purpose of regularization in machine learning?](#q36)  
- [Qn 37: What is transfer learning in deep learning?](#q37)  
- [Qn 38: What is the purpose of a word embedding in NLP?](#q38)  
- [Qn 39: What is the main advantage of using SQL databases over NoSQL?](#q39)  
- [Qn 40: What is the purpose of A/B testing?](#q40)  
- [Qn 41: What is the main purpose of the 'apply' function in Pandas?](#q41)  
- [Qn 42: What is the difference between batch gradient descent and stochastic gradient descent?](#q42)  
- [Qn 43: What is the purpose of the 'dropna' method in Pandas?](#q43)  
- [Qn 44: What is the main advantage of using a pipeline in Scikit-learn?](#q44)  
- [Qn 45: What is the purpose of the 'value_counts' method in Pandas?](#q45)  
- [Qn 46: What is the main purpose of feature scaling?](#q46)  
- [Qn 47: What is the difference between 'fit' and 'transform' in Scikit-learn?](#q47)  
- [Qn 48: What is the purpose of the 'merge' function in Pandas?](#q48)  
- [Qn 49: What is the main advantage of using a dictionary for vectorization in NLP?](#q49)  
- [Qn 50: What is the purpose of the 'pivot_table' function in Pandas?](#q50)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
What is the primary goal of data wrangling?

**Options**  

1. Building machine learning models  
2. Cleaning and transforming raw data into a usable format  
3. Creating interactive dashboards  
4. Writing complex SQL queries  

**Answer**  
Cleaning and transforming raw data into a usable format

**Explanation**  
Data wrangling involves cleaning, structuring, and enriching raw data into a
  format suitable for analysis.

[↑ Go to TOC](#toc)


### <a id="q02"></a> Qn 02

**Question**  
Which of the following is NOT a measure of central tendency?

**Options**  

1. Mean  
2. Median  
3. Mode  
4. Standard deviation  

**Answer**  
Standard deviation

**Explanation**  
Standard deviation measures dispersion, not central tendency. The three main
  measures of central tendency are mean, median, and mode.

[↑ Go to TOC](#toc)


### <a id="q03"></a> Qn 03

**Question**  
What type of chart would be most appropriate for comparing proportions of a whole?

**Options**  

1. Scatter plot  
2. Histogram  
3. Pie chart  
4. Line chart  

**Answer**  
Pie chart

**Explanation**  
Pie charts are best for showing proportions of a whole, though they should be
  used sparingly and only with a small number of categories.

[↑ Go to TOC](#toc)


### <a id="q04"></a> Qn 04

**Question**  
Which Python library is primarily used for working with tabular data structures?

**Options**  

1. NumPy  
2. Matplotlib  
3. Pandas  
4. Scikit-learn  

**Answer**  
Pandas

**Explanation**  
Pandas provides DataFrame objects which are ideal for working with tabular data,
  similar to spreadsheets or SQL tables.

[↑ Go to TOC](#toc)


### <a id="q05"></a> Qn 05

**Question**  
What does the groupby() operation in Pandas return before aggregation?

**Options**  

1. A transformed DataFrame  
2. A list of grouped indices  
3. A DataFrameGroupBy object  
4. A Series with group labels  

**Answer**  
A DataFrameGroupBy object

**Explanation**  
groupby() returns a DataFrameGroupBy object which can then be aggregated using
  functions like sum(), mean(), etc.

[↑ Go to TOC](#toc)


### <a id="q06"></a> Qn 06

**Question**  
What does 'NaN' represent in a Pandas DataFrame?

**Options**  

1. A very large number  
2. Not a Number (missing or undefined value)  
3. Negative number  
4. Newly added node  

**Answer**  
Not a Number (missing or undefined value)

**Explanation**  
NaN stands for 'Not a Number' and represents missing or undefined numerical data
  in Pandas.

[↑ Go to TOC](#toc)


### <a id="q07"></a> Qn 07

**Question**  
Which technique is NOT typically used for feature selection?

**Options**  

1. Recursive feature elimination  
2. Principal Component Analysis (PCA)  
3. Selecting features by correlation  
4. Data normalization  

**Answer**  
Data normalization

**Explanation**  
Data normalization scales features but doesn't select them. PCA, correlation
  analysis, and recursive elimination are feature selection methods.

[↑ Go to TOC](#toc)


### <a id="q08"></a> Qn 08

**Question**  
Which metric is NOT used to evaluate regression models?

**Options**  

1. Mean Squared Error (MSE)  
2. R-squared  
3. Accuracy  
4. Root Mean Squared Error (RMSE)  

**Answer**  
Accuracy

**Explanation**  
Accuracy is used for classification problems. MSE, RMSE, and R-squared are
  common regression metrics.

[↑ Go to TOC](#toc)


### <a id="q09"></a> Qn 09

**Question**  
What is the most common method for handling missing numerical data?

**Options**  

1. Deleting all rows with missing values  
2. Replacing with the mean or median  
3. Setting all missing values to zero  
4. Using a placeholder like -999  

**Answer**  
Replacing with the mean or median

**Explanation**  
Mean/median imputation is common for numerical data, though the best approach
  depends on the data and missingness pattern.

[↑ Go to TOC](#toc)


### <a id="q10"></a> Qn 10

**Question**  
Which library is essential for numerical computing in Python?

**Options**  

1. Matplotlib  
2. Pandas  
3. NumPy  
4. Seaborn  

**Answer**  
NumPy

**Explanation**  
NumPy provides foundational support for numerical computing with efficient array
  operations and mathematical functions.

[↑ Go to TOC](#toc)


### <a id="q11"></a> Qn 11

**Question**  
What is the purpose of a correlation matrix?

**Options**  

1. To show relationships between categorical variables  
2. To measure linear relationships between numerical variables  
3. To visualize hierarchical clustering  
4. To perform dimensionality reduction  

**Answer**  
To measure linear relationships between numerical variables

**Explanation**  
A correlation matrix measures the linear relationship between pairs of numerical
  variables, ranging from -1 to 1.

[↑ Go to TOC](#toc)


### <a id="q12"></a> Qn 12

**Question**  
What is the main advantage of using a box plot?

**Options**  

1. Showing exact data points  
2. Displaying the distribution and outliers of a dataset  
3. Comparing more than 10 categories clearly  
4. Showing trends over time  

**Answer**  
Displaying the distribution and outliers of a dataset

**Explanation**  
Box plots effectively show a dataset's quartiles, median, and potential
  outliers.

[↑ Go to TOC](#toc)


### <a id="q13"></a> Qn 13

**Question**  
What does the term 'overfitting' refer to in machine learning?

**Options**  

1. A model that performs well on training data but poorly on unseen data  
2. A model that performs poorly on both training and test data  
3. A model that takes too long to train  
4. A model with too few features  

**Answer**  
A model that performs well on training data but poorly on unseen data

**Explanation**  
Overfitting occurs when a model learns the training data too well, including its
  noise, reducing generalization to new data.

[↑ Go to TOC](#toc)


### <a id="q14"></a> Qn 14

**Question**  
Which of these is a supervised learning algorithm?

**Options**  

1. K-means clustering  
2. Principal Component Analysis  
3. Random Forest  
4. t-SNE  

**Answer**  
Random Forest

**Explanation**  
Random Forest is a supervised learning algorithm. K-means and PCA are
  unsupervised, and t-SNE is for visualization.

[↑ Go to TOC](#toc)


### <a id="q15"></a> Qn 15

**Question**  
What is the purpose of a train-test split?

**Options**  

1. To reduce the size of large datasets  
2. To evaluate how well a model generalizes to unseen data  
3. To balance class distributions in classification problems  
4. To speed up model training  

**Answer**  
To evaluate how well a model generalizes to unseen data

**Explanation**  
Splitting data into training and test sets helps estimate model performance on
  new, unseen data.

[↑ Go to TOC](#toc)


### <a id="q16"></a> Qn 16

**Question**  
Which Python library is most commonly used for creating static visualizations?

**Options**  

1. Plotly  
2. Matplotlib  
3. Seaborn  
4. Bokeh  

**Answer**  
Matplotlib

**Explanation**  
Matplotlib is the foundational plotting library in Python, though Seaborn builds
  on it for statistical visualizations.

[↑ Go to TOC](#toc)


### <a id="q17"></a> Qn 17

**Question**  
What is the main purpose of normalization in data preprocessing?

**Options**  

1. To remove outliers from the data  
2. To convert categorical variables to numerical  
3. To scale features to a similar range  
4. To handle missing values  

**Answer**  
To scale features to a similar range

**Explanation**  
Normalization scales numerical features to a standard range (often [0,1] or with
  mean=0, std=1) to prevent some features from dominating others.

[↑ Go to TOC](#toc)


### <a id="q18"></a> Qn 18

**Question**  
What does SQL stand for?

**Options**  

1. Structured Question Language  
2. Standard Query Language  
3. Structured Query Language  
4. Sequential Query Language  

**Answer**  
Structured Query Language

**Explanation**  
SQL stands for Structured Query Language, used for managing and querying
  relational databases.

[↑ Go to TOC](#toc)


### <a id="q19"></a> Qn 19

**Question**  
Which of these is NOT a common data type in Pandas?

**Options**  

1. DataFrame  
2. Series  
3. Array  
4. Panel  

**Answer**  
Array

**Explanation**  
Pandas' main data structures are DataFrame (2D), Series (1D), and Panel (3D, now
  deprecated). Arrays are from NumPy.

[↑ Go to TOC](#toc)


### <a id="q20"></a> Qn 20

**Question**  
What is the primary use of the Scikit-learn library?

**Options**  

1. Data visualization  
2. Machine learning algorithms  
3. Web scraping  
4. Database management  

**Answer**  
Machine learning algorithms

**Explanation**  
Scikit-learn provides simple and efficient tools for predictive data analysis
  and machine learning.

[↑ Go to TOC](#toc)


### <a id="q21"></a> Qn 21

**Question**  
What is the difference between classification and regression?

**Options**  

1. Classification predicts categories, regression predicts continuous values  
2. Classification uses unsupervised learning, regression uses supervised  
3. Classification is for small datasets, regression for large  
4. There is no difference  

**Answer**  
Classification predicts categories, regression predicts continuous values

**Explanation**  
Classification predicts discrete class labels, while regression predicts
  continuous numerical values.

[↑ Go to TOC](#toc)


### <a id="q22"></a> Qn 22

**Question**  
What is a confusion matrix used for?

**Options**  

1. Visualizing high-dimensional data  
2. Evaluating the performance of a classification model  
3. Storing large datasets efficiently  
4. Performing matrix calculations in linear algebra  

**Answer**  
Evaluating the performance of a classification model

**Explanation**  
A confusion matrix shows true/false positives/negatives, helping evaluate
  classification model performance.

[↑ Go to TOC](#toc)


### <a id="q23"></a> Qn 23

**Question**  
What does ETL stand for in data engineering?

**Options**  

1. Extract, Transform, Load  
2. Evaluate, Test, Learn  
3. Extract, Test, Load  
4. Explore, Transform, Label  

**Answer**  
Extract, Transform, Load

**Explanation**  
ETL refers to the process of extracting data from sources, transforming it, and
  loading it into a destination system.

[↑ Go to TOC](#toc)


### <a id="q24"></a> Qn 24

**Question**  
Which of these is a dimensionality reduction technique?

**Options**  

1. Linear Regression  
2. Decision Trees  
3. Principal Component Analysis (PCA)  
4. K-Nearest Neighbors  

**Answer**  
Principal Component Analysis (PCA)

**Explanation**  
PCA reduces dimensionality by transforming data to a new coordinate system with
  fewer dimensions.

[↑ Go to TOC](#toc)


### <a id="q25"></a> Qn 25

**Question**  
What is the purpose of cross-validation?

**Options**  

1. To increase the size of the training set  
2. To reduce the need for a test set  
3. To get more reliable estimates of model performance  
4. To speed up model training  

**Answer**  
To get more reliable estimates of model performance

**Explanation**  
Cross-validation provides more robust performance estimates by using multiple
  train/test splits of the data.

[↑ Go to TOC](#toc)


### <a id="q26"></a> Qn 26

**Question**  
What is the main advantage of using a Jupyter Notebook?

**Options**  

1. It provides the fastest execution speed  
2. It combines code, visualizations, and narrative text  
3. It automatically documents all code  
4. It doesn't require any programming knowledge  

**Answer**  
It combines code, visualizations, and narrative text

**Explanation**  
Jupyter Notebooks allow interactive development with code, visualizations, and
  explanatory text in a single document.

[↑ Go to TOC](#toc)


### <a id="q27"></a> Qn 27

**Question**  
What is the purpose of one-hot encoding?

**Options**  

1. To compress large datasets  
2. To convert categorical variables to numerical format  
3. To normalize numerical data  
4. To handle missing values  

**Answer**  
To convert categorical variables to numerical format

**Explanation**  
One-hot encoding converts categorical variables to a binary (0/1) numerical
  format that machine learning algorithms can process.

[↑ Go to TOC](#toc)


### <a id="q28"></a> Qn 28

**Question**  
Which metric would you use for an imbalanced classification problem?

**Options**  

1. Accuracy  
2. Precision-Recall curve  
3. Mean Squared Error  
4. R-squared  

**Answer**  
Precision-Recall curve

**Explanation**  
For imbalanced classes, accuracy can be misleading. Precision-Recall curves
  provide better insight into model performance.

[↑ Go to TOC](#toc)


### <a id="q29"></a> Qn 29

**Question**  
What is feature engineering?

**Options**  

1. Creating new features from existing data  
2. Selecting the most important features  
3. Building machine learning models  
4. Visualizing data features  

**Answer**  
Creating new features from existing data

**Explanation**  
Feature engineering involves creating new input features from existing data to
  improve model performance.

[↑ Go to TOC](#toc)


### <a id="q30"></a> Qn 30

**Question**  
What is the purpose of a ROC curve?

**Options**  

1. To visualize the trade-off between true positive and false positive rates  
2. To show the distribution of a single variable  
3. To compare regression models  
4. To perform clustering analysis  

**Answer**  
To visualize the trade-off between true positive and false positive rates

**Explanation**  
ROC curves show the diagnostic ability of a binary classifier by plotting true
  positive rate vs false positive rate.

[↑ Go to TOC](#toc)


### <a id="q31"></a> Qn 31

**Question**  
What is the main advantage of using a random forest over a single decision tree?

**Options**  

1. It's always more accurate  
2. It reduces overfitting by averaging multiple trees  
3. It requires less computational power  
4. It works better with small datasets  

**Answer**  
It reduces overfitting by averaging multiple trees

**Explanation**  
Random forests combine multiple decision trees to reduce variance and
  overfitting compared to a single tree.

[↑ Go to TOC](#toc)


### <a id="q32"></a> Qn 32

**Question**  
What is the purpose of the 'iloc' method in Pandas?

**Options**  

1. To select data by integer position  
2. To select data by label  
3. To perform interpolation  
4. To handle missing values  

**Answer**  
To select data by integer position

**Explanation**  
iloc is primarily integer-location based indexing for selection by position.

[↑ Go to TOC](#toc)


### <a id="q33"></a> Qn 33

**Question**  
What is the difference between deep learning and traditional machine learning?

**Options**  

1. Deep learning always performs better  
2. Deep learning automatically learns feature hierarchies from raw data  
3. Deep learning requires less data  
4. There is no difference  

**Answer**  
Deep learning automatically learns feature hierarchies from raw data

**Explanation**  
Deep learning models can learn hierarchical feature representations directly
  from data, while traditional ML often requires manual feature engineering.

[↑ Go to TOC](#toc)


### <a id="q34"></a> Qn 34

**Question**  
What is the purpose of a learning curve in machine learning?

**Options**  

1. To visualize model performance over time  
2. To show the relationship between training set size and model performance  
3. To track the learning rate during training  
4. To compare different optimization algorithms  

**Answer**  
To show the relationship between training set size and model performance

**Explanation**  
Learning curves plot model performance (e.g., accuracy) against training set
  size or training iterations.

[↑ Go to TOC](#toc)


### <a id="q35"></a> Qn 35

**Question**  
What is the bias-variance tradeoff?

**Options**  

1. The balance between model complexity and generalization  
2. The choice between supervised and unsupervised learning  
3. The decision to use Python or R  
4. The selection of training vs test data size  

**Answer**  
The balance between model complexity and generalization

**Explanation**  
The bias-variance tradeoff refers to balancing a model's simplicity (bias)
  against its sensitivity to training data (variance) to achieve good
  generalization.

[↑ Go to TOC](#toc)


### <a id="q36"></a> Qn 36

**Question**  
What is the purpose of regularization in machine learning?

**Options**  

1. To speed up model training  
2. To reduce overfitting by penalizing complex models  
3. To handle missing data  
4. To normalize input features  

**Answer**  
To reduce overfitting by penalizing complex models

**Explanation**  
Regularization techniques like L1/L2 add penalty terms to prevent overfitting by
  discouraging overly complex models.

[↑ Go to TOC](#toc)


### <a id="q37"></a> Qn 37

**Question**  
What is transfer learning in deep learning?

**Options**  

1. Training multiple models simultaneously  
2. Using a pre-trained model as a starting point for a new task  
3. Transferring data between different formats  
4. Moving models between different hardware  

**Answer**  
Using a pre-trained model as a starting point for a new task

**Explanation**  
Transfer learning leverages knowledge gained from solving one problem and
  applies it to a different but related problem.

[↑ Go to TOC](#toc)


### <a id="q38"></a> Qn 38

**Question**  
What is the purpose of a word embedding in NLP?

**Options**  

1. To count word frequencies  
2. To represent words as dense vectors capturing semantic meaning  
3. To correct spelling errors  
4. To translate between languages  

**Answer**  
To represent words as dense vectors capturing semantic meaning

**Explanation**  
Word embeddings represent words as numerical vectors where similar words have
  similar vector representations.

[↑ Go to TOC](#toc)


### <a id="q39"></a> Qn 39

**Question**  
What is the main advantage of using SQL databases over NoSQL?

**Options**  

1. Better scalability  
2. More flexible schema  
3. Stronger consistency guarantees  
4. Faster write speeds  

**Answer**  
Stronger consistency guarantees

**Explanation**  
SQL databases provide ACID transactions and strong consistency, while NoSQL
  prioritizes scalability and flexibility.

[↑ Go to TOC](#toc)


### <a id="q40"></a> Qn 40

**Question**  
What is the purpose of A/B testing?

**Options**  

1. To compare two machine learning models  
2. To test two different versions of a product feature  
3. To analyze variance in datasets  
4. To balance class distributions  

**Answer**  
To test two different versions of a product feature

**Explanation**  
A/B testing compares two versions (A and B) to determine which performs better
  on a specific metric.

[↑ Go to TOC](#toc)


### <a id="q41"></a> Qn 41

**Question**  
What is the main purpose of the 'apply' function in Pandas?

**Options**  

1. To apply mathematical operations to a DataFrame  
2. To apply a function along an axis of a DataFrame  
3. To apply CSS styles to a DataFrame display  
4. To apply machine learning models to data  

**Answer**  
To apply a function along an axis of a DataFrame

**Explanation**  
The apply() function applies a function along an axis (rows or columns) of a
  DataFrame or Series.

[↑ Go to TOC](#toc)


### <a id="q42"></a> Qn 42

**Question**  
What is the difference between batch gradient descent and stochastic gradient descent?

**Options**  

1. Batch uses all data per update, stochastic uses one sample  
2. Batch is for classification, stochastic for regression  
3. Batch is faster but less accurate  
4. There is no difference  

**Answer**  
Batch uses all data per update, stochastic uses one sample

**Explanation**  
Batch GD computes gradients using the entire dataset, while SGD uses a single
  random sample per iteration.

[↑ Go to TOC](#toc)


### <a id="q43"></a> Qn 43

**Question**  
What is the purpose of the 'dropna' method in Pandas?

**Options**  

1. To drop columns with missing values  
2. To drop rows or columns with missing values  
3. To replace missing values with zeros  
4. To count missing values  

**Answer**  
To drop rows or columns with missing values

**Explanation**  
dropna() removes missing values (NaN) from a DataFrame, either by rows or
  columns.

[↑ Go to TOC](#toc)


### <a id="q44"></a> Qn 44

**Question**  
What is the main advantage of using a pipeline in Scikit-learn?

**Options**  

1. To speed up model training  
2. To chain multiple processing steps into a single object  
3. To visualize model performance  
4. To handle large datasets that don't fit in memory  

**Answer**  
To chain multiple processing steps into a single object

**Explanation**  
Pipelines sequentially apply transforms and a final estimator, ensuring steps
  are executed in the right order.

[↑ Go to TOC](#toc)


### <a id="q45"></a> Qn 45

**Question**  
What is the purpose of the 'value_counts' method in Pandas?

**Options**  

1. To count the number of unique values in a Series  
2. To calculate the mean of numerical columns  
3. To count missing values  
4. To enumerate all values in a DataFrame  

**Answer**  
To count the number of unique values in a Series

**Explanation**  
value_counts() returns a Series containing counts of unique values in descending
  order.

[↑ Go to TOC](#toc)


### <a id="q46"></a> Qn 46

**Question**  
What is the main purpose of feature scaling?

**Options**  

1. To remove unimportant features  
2. To ensure all features contribute equally to distance-based algorithms  
3. To reduce the number of features  
4. To handle categorical variables  

**Answer**  
To ensure all features contribute equally to distance-based algorithms

**Explanation**  
Feature scaling normalizes the range of features so that features with larger
  scales don't dominate algorithms like KNN or SVM.

[↑ Go to TOC](#toc)


### <a id="q47"></a> Qn 47

**Question**  
What is the difference between 'fit' and 'transform' in Scikit-learn?

**Options**  

1. 'fit' learns parameters, 'transform' applies them  
2. 'fit' trains models, 'transform' makes predictions  
3. 'fit' is for classification, 'transform' for regression  
4. There is no difference  

**Answer**  
'fit' learns parameters, 'transform' applies them

**Explanation**  
fit() learns model parameters from training data, while transform() applies the
  learned transformation to data.

[↑ Go to TOC](#toc)


### <a id="q48"></a> Qn 48

**Question**  
What is the purpose of the 'merge' function in Pandas?

**Options**  

1. To combine DataFrames based on common columns  
2. To concatenate DataFrames vertically  
3. To merge cells in a DataFrame  
4. To combine Series objects  

**Answer**  
To combine DataFrames based on common columns

**Explanation**  
merge() combines DataFrames using database-style joins on columns or indices.

[↑ Go to TOC](#toc)


### <a id="q49"></a> Qn 49

**Question**  
What is the main advantage of using a dictionary for vectorization in NLP?

**Options**  

1. It preserves word order  
2. It's more memory efficient than other methods  
3. It creates a fixed-length representation regardless of document length  
4. It automatically handles spelling errors  

**Answer**  
It creates a fixed-length representation regardless of document length

**Explanation**  
Dictionary-based vectorization (like CountVectorizer) creates consistent-length
  vectors from variable-length texts.

[↑ Go to TOC](#toc)


### <a id="q50"></a> Qn 50

**Question**  
What is the purpose of the 'pivot_table' function in Pandas?

**Options**  

1. To rotate a DataFrame 90 degrees  
2. To create a spreadsheet-style pivot table as a DataFrame  
3. To pivot between different data types  
4. To transform wide data to long format  

**Answer**  
To create a spreadsheet-style pivot table as a DataFrame

**Explanation**  
pivot_table() creates a multi-dimensional summary table similar to Excel pivot
  tables, aggregating data.

[↑ Go to TOC](#toc)


---

*Automatically generated from [data_science_questions.json](data_science_questions.json)*
*Updated: 2025-05-18 13:57*
