# Pandas Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: Which method efficiently applies a function along an axis of a DataFrame?](#q01)  
- [Qn 02: What's the correct way to merge two DataFrames on multiple columns?](#q02)  
- [Qn 03: How do you handle missing values in a DataFrame column?](#q03)  
- [Qn 04: What does the method `groupby().agg()` allow you to do?](#q04)  
- [Qn 05: Which of the following transforms a DataFrame to a long format?](#q05)  
- [Qn 06: How can you efficiently select rows where a column value meets a complex condition?](#q06)  
- [Qn 07: What's the most efficient way to calculate a rolling 7-day average of a time series?](#q07)  
- [Qn 08: How do you perform a pivot operation in pandas?](#q08)  
- [Qn 09: Which method can reshape a DataFrame by stacking column labels to rows?](#q09)  
- [Qn 10: How do you efficiently concatenate many DataFrames with identical columns?](#q10)  
- [Qn 11: What's the correct way to create a DatetimeIndex from a column containing date strings?](#q11)  
- [Qn 12: Which method performs a cross-tabulation of two factors?](#q12)  
- [Qn 13: How do you calculate cumulative statistics in pandas?](#q13)  
- [Qn 14: Which approach efficiently calculates the difference between consecutive rows in a DataFrame?](#q14)  
- [Qn 15: How do you create a MultiIndex DataFrame from scratch?](#q15)  
- [Qn 16: Which method is most appropriate for performing complex string operations on DataFrame columns?](#q16)  
- [Qn 17: What's the best way to compute percentiles for grouped data?](#q17)  
- [Qn 18: How do you efficiently implement a custom aggregation function that requires the entire group?](#q18)  
- [Qn 19: What's the most memory-efficient way to read a large CSV file with pandas?](#q19)  
- [Qn 20: Which method is correct for resampling time series data to monthly frequency?](#q20)  
- [Qn 21: How do you efficiently identify and remove duplicate rows in a DataFrame?](#q21)  
- [Qn 22: Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?](#q22)  
- [Qn 23: How do you sample data from a DataFrame with weights?](#q23)  
- [Qn 24: What's the correct way to use the pd.cut() function for binning continuous data?](#q24)  
- [Qn 25: How do you efficiently perform a custom window operation in pandas?](#q25)  
- [Qn 26: Which approach can create a lagged feature in a time series DataFrame?](#q26)  
- [Qn 27: What's the best way to explode a DataFrame column containing lists into multiple rows?](#q27)  
- [Qn 28: How do you efficiently compute a weighted mean in pandas?](#q28)  
- [Qn 29: Which method correctly identifies the top-k values in each group?](#q29)  
- [Qn 30: What's the best way to add a new column based on a categorical mapping of an existing column?](#q30)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
Which method efficiently applies a function along an axis of a DataFrame?

**Options**  

1. df.map(func)  
2. df.apply(func, axis=0)  
3. df.transform(func)  
4. df.aggregate(func)  

**Answer**  
df.apply(func, axis=0)

**Explanation**  
The apply() method allows applying a function along an axis (rows or columns) of
  a DataFrame.

[↑ Go to TOC](#toc)


### <a id="q02"></a> Qn 02

**Question**  
What's the correct way to merge two DataFrames on multiple columns?

**Options**  

1. pd.merge(df1, df2, on=['col1', 'col2'])  
2. pd.join(df1, df2, keys=['col1', 'col2'])  
3. df1.merge(df2, how='inner', left_on=['col1', 'col2'], right_on=['col1', 'col2'])  
4. Both A and C  

**Answer**  
Both A and C

**Explanation**  
Both pd.merge() and DataFrame.merge() methods can merge on multiple columns
  specified as lists.

[↑ Go to TOC](#toc)


### <a id="q03"></a> Qn 03

**Question**  
How do you handle missing values in a DataFrame column?

**Options**  

1. df['column'].fillna(0)  
2. df['column'].dropna()  
3. df['column'].replace(np.nan, 0)  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All listed methods can handle missing values: fillna() replaces NaNs, dropna()
  removes rows with NaNs, and replace() can substitute NaNs with specified
  values.

[↑ Go to TOC](#toc)


### <a id="q04"></a> Qn 04

**Question**  
What does the method `groupby().agg()` allow you to do?

**Options**  

1. Group data and apply a single aggregation function  
2. Group data and apply different aggregation functions to different columns  
3. Group data and apply multiple aggregation functions to the same column  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
The agg() method is versatile and can apply single or multiple functions to
  grouped data, either to all columns or selectively.

[↑ Go to TOC](#toc)


### <a id="q05"></a> Qn 05

**Question**  
Which of the following transforms a DataFrame to a long format?

**Options**  

1. df.stack()  
2. df.melt()  
3. pd.wide_to_long(df)  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
stack(), melt(), and wide_to_long() all convert data from wide format to long
  format, albeit with different approaches and parameters.

[↑ Go to TOC](#toc)


### <a id="q06"></a> Qn 06

**Question**  
How can you efficiently select rows where a column value meets a complex condition?

**Options**  

1. df.loc[df['column'] > 5 & df['column'] < 10]  
2. df.loc[(df['column'] > 5) & (df['column'] < 10)]  
3. df.query('column > 5 and column < 10')  
4. Both B and C  

**Answer**  
Both B and C

**Explanation**  
Both loc with boolean indexing (with proper parentheses) and query() method can
  filter data based on complex conditions.

[↑ Go to TOC](#toc)


### <a id="q07"></a> Qn 07

**Question**  
What's the most efficient way to calculate a rolling 7-day average of a time series?

**Options**  

1. df['rolling_avg'] = df['value'].rolling(window=7).mean()  
2. df['rolling_avg'] = df['value'].resample('7D').mean()  
3. df['rolling_avg'] = df.groupby(pd.Grouper(freq='7D'))['value'].transform('mean')  
4. df['rolling_avg'] = pd.rolling_mean(df['value'], window=7)  

**Answer**  
df['rolling_avg'] = df['value'].rolling(window=7).mean()

**Explanation**  
The rolling() method with a window of 7 followed by mean() calculates a rolling
  average over a 7-period window.

[↑ Go to TOC](#toc)


### <a id="q08"></a> Qn 08

**Question**  
How do you perform a pivot operation in pandas?

**Options**  

1. df.pivot(index='A', columns='B', values='C')  
2. pd.pivot_table(df, index='A', columns='B', values='C')  
3. df.pivot_table(index='A', columns='B', values='C')  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All three methods can perform pivot operations, with pivot_table being more
  flexible as it can aggregate duplicate entries.

[↑ Go to TOC](#toc)


### <a id="q09"></a> Qn 09

**Question**  
Which method can reshape a DataFrame by stacking column labels to rows?

**Options**  

1. df.unstack()  
2. df.pivot()  
3. df.stack()  
4. df.melt()  

**Answer**  
df.stack()

**Explanation**  
stack() method pivots the columns of a DataFrame to become the innermost index
  level, creating a Series with a MultiIndex.

[↑ Go to TOC](#toc)


### <a id="q10"></a> Qn 10

**Question**  
How do you efficiently concatenate many DataFrames with identical columns?

**Options**  

1. pd.join([df1, df2, df3])  
2. pd.merge([df1, df2, df3])  
3. pd.concat([df1, df2, df3])  
4. df1.append([df2, df3])  

**Answer**  
pd.concat([df1, df2, df3])

**Explanation**  
pd.concat() is designed to efficiently concatenate pandas objects along a
  particular axis with optional set logic.

[↑ Go to TOC](#toc)


### <a id="q11"></a> Qn 11

**Question**  
What's the correct way to create a DatetimeIndex from a column containing date strings?

**Options**  

1. pd.to_datetime(df['date_col'])  
2. df.set_index(pd.to_datetime(df['date_col']))  
3. df.set_index('date_col', inplace=True); df.index = pd.to_datetime(df.index)  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All methods will correctly convert date strings to datetime objects, with
  different approaches to setting them as the index.

[↑ Go to TOC](#toc)


### <a id="q12"></a> Qn 12

**Question**  
Which method performs a cross-tabulation of two factors?

**Options**  

1. pd.crosstab(df['A'], df['B'])  
2. df.pivot_table(index='A', columns='B', aggfunc='count')  
3. df.groupby(['A', 'B']).size().unstack()  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All methods can create cross-tabulations, though crosstab() is specifically
  designed for this purpose.

[↑ Go to TOC](#toc)


### <a id="q13"></a> Qn 13

**Question**  
How do you calculate cumulative statistics in pandas?

**Options**  

1. df.cumsum(), df.cumprod(), df.cummax(), df.cummin()  
2. df.rolling().sum(), df.rolling().prod(), df.rolling().max(), df.rolling().min()  
3. df.expanding().sum(), df.expanding().prod(), df.expanding().max(), df.expanding().min()  
4. df.aggregate(['sum', 'prod', 'max', 'min'])  

**Answer**  
df.cumsum(), df.cumprod(), df.cummax(), df.cummin()

**Explanation**  
The cum- methods (cumsum, cumprod, cummax, cummin) calculate cumulative
  statistics along an axis.

[↑ Go to TOC](#toc)


### <a id="q14"></a> Qn 14

**Question**  
Which approach efficiently calculates the difference between consecutive rows in a DataFrame?

**Options**  

1. df - df.shift(1)  
2. df.diff()  
3. df.rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0])  
4. Both A and B  

**Answer**  
Both A and B

**Explanation**  
Both subtracting a shifted DataFrame and using diff() calculate element-wise
  differences between consecutive rows.

[↑ Go to TOC](#toc)


### <a id="q15"></a> Qn 15

**Question**  
How do you create a MultiIndex DataFrame from scratch?

**Options**  

1. pd.DataFrame(data, index=pd.MultiIndex.from_tuples([('A',1), ('A',2), ('B',1), ('B',2)]))  
2. pd.DataFrame(data, index=pd.MultiIndex.from_product([['A', 'B'], [1, 2]]))  
3. pd.DataFrame(data, index=pd.MultiIndex.from_arrays([['A', 'A', 'B', 'B'], [1, 2, 1, 2]]))  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All three methods create equivalent MultiIndex objects using different
  approaches: from_tuples, from_product, and from_arrays.

[↑ Go to TOC](#toc)


### <a id="q16"></a> Qn 16

**Question**  
Which method is most appropriate for performing complex string operations on DataFrame columns?

**Options**  

1. df['col'].apply(lambda x: x.upper())  
2. df['col'].str.upper()  
3. df['col'].map(str.upper)  
4. All of the above work, but B is most efficient  

**Answer**  
All of the above work, but B is most efficient

**Explanation**  
While all methods can transform strings, the .str accessor provides vectorized
  string functions that are generally more efficient than apply() or map().

[↑ Go to TOC](#toc)


### <a id="q17"></a> Qn 17

**Question**  
What's the best way to compute percentiles for grouped data?

**Options**  

1. df.groupby('group').quantile([0.25, 0.5, 0.75])  
2. df.groupby('group').agg(lambda x: np.percentile(x, [25, 50, 75]))  
3. df.groupby('group').describe(percentiles=[0.25, 0.5, 0.75])  
4. Both A and C  

**Answer**  
Both A and C

**Explanation**  
Both quantile() and describe() can compute percentiles for grouped data, with
  describe() providing additional statistics. For option B, While this approach
  uses the right function (numpy's percentile), there's an issue with how it's
  implemented in the context of pandas GroupBy. This would likely raise errors
  because the lambda function returns arrays rather than scalars, which is
  problematic for the standard aggregation pipeline.

[↑ Go to TOC](#toc)


### <a id="q18"></a> Qn 18

**Question**  
How do you efficiently implement a custom aggregation function that requires the entire group?

**Options**  

1. df.groupby('group').agg(custom_func)  
2. df.groupby('group').apply(custom_func)  
3. df.groupby('group').transform(custom_func)  
4. df.groupby('group').aggregate(custom_func)  

**Answer**  
df.groupby('group').apply(custom_func)

**Explanation**  
apply() is designed for operations that need the entire group as a DataFrame,
  whereas agg() is better for operations that can be vectorized.

[↑ Go to TOC](#toc)


### <a id="q19"></a> Qn 19

**Question**  
What's the most memory-efficient way to read a large CSV file with pandas?

**Options**  

1. pd.read_csv('file.csv', nrows=1000)  
2. pd.read_csv('file.csv', chunksize=1000)  
3. pd.read_csv('file.csv', usecols=['needed_col1', 'needed_col2'])  
4. pd.read_csv('file.csv', dtype={'col1': 'category', 'col2': 'int8'})  

**Answer**  
pd.read_csv('file.csv', dtype={'col1': 'category', 'col2': 'int8'})

**Explanation**  
Specifying appropriate dtypes, especially using 'category' for string columns
  with repeated values, significantly reduces memory usage.

[↑ Go to TOC](#toc)


### <a id="q20"></a> Qn 20

**Question**  
Which method is correct for resampling time series data to monthly frequency?

**Options**  

1. df.resample('M').mean()  
2. df.groupby(pd.Grouper(freq='M')).mean()  
3. df.asfreq('M')  
4. Both A and B  

**Answer**  
Both A and B

**Explanation**  
Both resample() and groupby() with Grouper can aggregate time series data to
  monthly frequency, though asfreq() only changes frequency without aggregation.

[↑ Go to TOC](#toc)


### <a id="q21"></a> Qn 21

**Question**  
How do you efficiently identify and remove duplicate rows in a DataFrame?

**Options**  

1. df[~df.duplicated()]  
2. df.drop_duplicates()  
3. df.loc[~df.index.duplicated()]  
4. Both A and B  

**Answer**  
Both A and B

**Explanation**  
Both df[~df.duplicated()] and df.drop_duplicates() remove duplicate rows, with
  the latter being more readable and offering more options.

[↑ Go to TOC](#toc)


### <a id="q22"></a> Qn 22

**Question**  
Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?

**Options**  

1. df.pipe(custom_func)  
2. df.transform(custom_func)  
3. df.apply(custom_func)  
4. custom_func(df)  

**Answer**  
df.pipe(custom_func)

**Explanation**  
pipe() is designed for functions that take and return a DataFrame, creating
  readable method chains when applying multiple functions.

[↑ Go to TOC](#toc)


### <a id="q23"></a> Qn 23

**Question**  
How do you sample data from a DataFrame with weights?

**Options**  

1. df.sample(n=5, weights='probability_column')  
2. df.sample(frac=0.1, weights=df['probability_column'])  
3. df.sample(n=5, weights=df['probability_column']/df['probability_column'].sum())  
4. Both B and C  

**Answer**  
Both B and C

**Explanation**  
Both approaches correctly sample with weights, though weights don't need to be
  normalized as pandas normalizes them internally.

[↑ Go to TOC](#toc)


### <a id="q24"></a> Qn 24

**Question**  
What's the correct way to use the pd.cut() function for binning continuous data?

**Options**  

1. pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Middle', 'Senior'])  
2. pd.cut(df['age'], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  
3. pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  
4. All of the above are valid uses  

**Answer**  
All of the above are valid uses

**Explanation**  
All approaches are valid: using explicit bin edges, equal-width bins (cut), or
  equal-frequency bins (qcut).

[↑ Go to TOC](#toc)


### <a id="q25"></a> Qn 25

**Question**  
How do you efficiently perform a custom window operation in pandas?

**Options**  

1. df.rolling(window=3).apply(custom_func, raw=True)  
2. df.rolling(window=3).apply(custom_func)  
3. df.apply(lambda x: [custom_func(x[i:i+3]) for i in range(len(x)-2)])  
4. Both A and B  

**Answer**  
Both A and B

**Explanation**  
Both approaches work for custom window operations, but using raw=True can be
  more efficient for numerical operations by passing a NumPy array instead of a
  Series.

[↑ Go to TOC](#toc)


### <a id="q26"></a> Qn 26

**Question**  
Which approach can create a lagged feature in a time series DataFrame?

**Options**  

1. df['lagged'] = df['value'].shift(1)  
2. df['lagged'] = df['value'].shift(-1)  
3. df['lagged'] = df['value'].rolling(window=2).apply(lambda x: x.iloc[0])  
4. Both A and B  

**Answer**  
Both A and B

**Explanation**  
shift(1) creates a lag (past values), while shift(-1) creates a lead (future
  values), both useful for time series analysis.

[↑ Go to TOC](#toc)


### <a id="q27"></a> Qn 27

**Question**  
What's the best way to explode a DataFrame column containing lists into multiple rows?

**Options**  

1. pd.DataFrame([[i, x] for i, y in df['list_col'].iteritems() for x in y])  
2. df.explode('list_col')  
3. df.assign(list_col=df['list_col']).explode('list_col')  
4. Both B and C  

**Answer**  
Both B and C

**Explanation**  
explode() transforms each element of a list-like column into a row, with the
  original index duplicated as needed.

[↑ Go to TOC](#toc)


### <a id="q28"></a> Qn 28

**Question**  
How do you efficiently compute a weighted mean in pandas?

**Options**  

1. (df['value'] * df['weight']).sum() / df['weight'].sum()  
2. df['value'].mean(weights=df['weight'])  
3. np.average(df['value'], weights=df['weight'])  
4. Both A and C  

**Answer**  
Both A and C

**Explanation**  
Both manually computing weighted mean and using np.average() work efficiently,
  though pandas Series doesn't have a weights parameter for mean().

[↑ Go to TOC](#toc)


### <a id="q29"></a> Qn 29

**Question**  
Which method correctly identifies the top-k values in each group?

**Options**  

1. df.groupby('group')['value'].nlargest(k)  
2. df.groupby('group').apply(lambda x: x.nlargest(k, 'value'))  
3. df.sort_values('value', ascending=False).groupby('group').head(k)  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All three methods can get the top-k values within each group, with different
  syntax but similar results.

[↑ Go to TOC](#toc)


### <a id="q30"></a> Qn 30

**Question**  
What's the best way to add a new column based on a categorical mapping of an existing column?

**Options**  

1. df['category'] = df['value'].map({'low': 1, 'medium': 2, 'high': 3})  
2. df['category'] = df['value'].replace({'low': 1, 'medium': 2, 'high': 3})  
3. df['category'] = pd.Categorical(df['value']).map({'low': 1, 'medium': 2, 'high': 3})  
4. All of the above  

**Answer**  
All of the above

**Explanation**  
All methods can map values to new ones, though map() is generally preferred for
  dictionary-based mappings.

[↑ Go to TOC](#toc)


---

*Automatically generated from [pandas_questions.json](pandas_questions.json)*
*Updated: 2025-05-18 13:57*
