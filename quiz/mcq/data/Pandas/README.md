# Pandas Quiz

## Table of Contents
- [Qn 01: Which method efficiently applies a function along an axis of a DataFrame?](#1)
- [Qn 02: What's the correct way to merge two DataFrames on multiple columns?](#2)
- [Qn 03: How do you handle missing values in a DataFrame column?](#3)
- [Qn 04: What does the method `groupby().agg()` allow you to do?](#4)
- [Qn 05: Which of the following transforms a DataFrame to a long format?](#5)
- [Qn 06: How can you efficiently select rows where a column value meets a complex condition?](#6)
- [Qn 07: What's the most efficient way to calculate a rolling 7-day average of a time series?](#7)
- [Qn 08: How do you perform a pivot operation in pandas?](#8)
- [Qn 09: Which method can reshape a DataFrame by stacking column labels to rows?](#9)
- [Qn 10: How do you efficiently concatenate many DataFrames with identical columns?](#10)
- [Qn 11: What's the correct way to create a DatetimeIndex from a column containing date strings?](#11)
- [Qn 12: Which method performs a cross-tabulation of two factors?](#12)
- [Qn 13: How do you calculate cumulative statistics in pandas?](#13)
- [Qn 14: Which approach efficiently calculates the difference between consecutive rows in a DataFrame?](#14)
- [Qn 15: How do you create a MultiIndex DataFrame from scratch?](#15)
- [Qn 16: Which method is most appropriate for performing complex string operations on DataFrame columns?](#16)
- [Qn 17: What's the best way to compute percentiles for grouped data?](#17)
- [Qn 18: How do you efficiently implement a custom aggregation function that requires the entire group?](#18)
- [Qn 19: What's the most memory-efficient way to read a large CSV file with pandas?](#19)
- [Qn 20: Which method is correct for resampling time series data to monthly frequency?](#20)
- [Qn 21: How do you efficiently identify and remove duplicate rows in a DataFrame?](#21)
- [Qn 22: Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?](#22)
- [Qn 23: How do you sample data from a DataFrame with weights?](#23)
- [Qn 24: What's the correct way to use the pd.cut() function for binning continuous data?](#24)
- [Qn 25: How do you efficiently perform a custom window operation in pandas?](#25)
- [Qn 26: Which approach can create a lagged feature in a time series DataFrame?](#26)
- [Qn 27: What's the best way to explode a DataFrame column containing lists into multiple rows?](#27)
- [Qn 28: How do you efficiently compute a weighted mean in pandas?](#28)
- [Qn 29: Which method correctly identifies the top-k values in each group?](#29)
- [Qn 30: What's the best way to add a new column based on a categorical mapping of an existing column?](#30)

---

### 1. Qn 01: Which method efficiently applies a function along an axis of a DataFrame?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df.apply(func, axis=0)

**Explanation:** The apply() method allows applying a function along an axis (rows or columns) of a DataFrame.


[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: What's the correct way to merge two DataFrames on multiple columns?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and C

**Explanation:** Both pd.merge() and DataFrame.merge() methods can merge on multiple columns specified as lists.


[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: How do you handle missing values in a DataFrame column?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All listed methods can handle missing values: fillna() replaces NaNs, dropna() removes rows with NaNs, and replace() can substitute NaNs with specified values.


[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: What does the method `groupby().agg()` allow you to do?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** The agg() method is versatile and can apply single or multiple functions to grouped data, either to all columns or selectively.


[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: Which of the following transforms a DataFrame to a long format?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** stack(), melt(), and wide_to_long() all convert data from wide format to long format, albeit with different approaches and parameters.


[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: How can you efficiently select rows where a column value meets a complex condition?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both B and C

**Explanation:** Both loc with boolean indexing (with proper parentheses) and query() method can filter data based on complex conditions.


[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: What's the most efficient way to calculate a rolling 7-day average of a time series?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df['rolling_avg'] = df['value'].rolling(window=7).mean()

**Explanation:** The rolling() method with a window of 7 followed by mean() calculates a rolling average over a 7-period window.


[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: How do you perform a pivot operation in pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All three methods can perform pivot operations, with pivot_table being more flexible as it can aggregate duplicate entries.


[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: Which method can reshape a DataFrame by stacking column labels to rows?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df.stack()

**Explanation:** stack() method pivots the columns of a DataFrame to become the innermost index level, creating a Series with a MultiIndex.


[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: How do you efficiently concatenate many DataFrames with identical columns?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** pd.concat([df1, df2, df3])

**Explanation:** pd.concat() is designed to efficiently concatenate pandas objects along a particular axis with optional set logic.


[Go to TOC](#table-of-contents)

</details>

---
### 11. Qn 11: What's the correct way to create a DatetimeIndex from a column containing date strings?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All methods will correctly convert date strings to datetime objects, with different approaches to setting them as the index.


[Go to TOC](#table-of-contents)

</details>

---
### 12. Qn 12: Which method performs a cross-tabulation of two factors?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All methods can create cross-tabulations, though crosstab() is specifically designed for this purpose.


[Go to TOC](#table-of-contents)

</details>

---
### 13. Qn 13: How do you calculate cumulative statistics in pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df.cumsum(), df.cumprod(), df.cummax(), df.cummin()

**Explanation:** The cum- methods (cumsum, cumprod, cummax, cummin) calculate cumulative statistics along an axis.


[Go to TOC](#table-of-contents)

</details>

---
### 14. Qn 14: Which approach efficiently calculates the difference between consecutive rows in a DataFrame?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and B

**Explanation:** Both subtracting a shifted DataFrame and using diff() calculate element-wise differences between consecutive rows.


[Go to TOC](#table-of-contents)

</details>

---
### 15. Qn 15: How do you create a MultiIndex DataFrame from scratch?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All three methods create equivalent MultiIndex objects using different approaches: from_tuples, from_product, and from_arrays.


[Go to TOC](#table-of-contents)

</details>

---
### 16. Qn 16: Which method is most appropriate for performing complex string operations on DataFrame columns?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above work, but B is most efficient

**Explanation:** While all methods can transform strings, the .str accessor provides vectorized string functions that are generally more efficient than apply() or map().


[Go to TOC](#table-of-contents)

</details>

---
### 17. Qn 17: What's the best way to compute percentiles for grouped data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and C

**Explanation:** Both quantile() and describe() can compute percentiles for grouped data, with describe() providing additional statistics. For option B, While this approach uses the right function (numpy's percentile), there's an issue with how it's implemented in the context of pandas GroupBy. This would likely raise errors because the lambda function returns arrays rather than scalars, which is problematic for the standard aggregation pipeline.


[Go to TOC](#table-of-contents)

</details>

---
### 18. Qn 18: How do you efficiently implement a custom aggregation function that requires the entire group?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df.groupby('group').apply(custom_func)

**Explanation:** apply() is designed for operations that need the entire group as a DataFrame, whereas agg() is better for operations that can be vectorized.


[Go to TOC](#table-of-contents)

</details>

---
### 19. Qn 19: What's the most memory-efficient way to read a large CSV file with pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** pd.read_csv('file.csv', dtype={'col1': 'category', 'col2': 'int8'})

**Explanation:** Specifying appropriate dtypes, especially using 'category' for string columns with repeated values, significantly reduces memory usage.


[Go to TOC](#table-of-contents)

</details>

---
### 20. Qn 20: Which method is correct for resampling time series data to monthly frequency?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and B

**Explanation:** Both resample() and groupby() with Grouper can aggregate time series data to monthly frequency, though asfreq() only changes frequency without aggregation.


[Go to TOC](#table-of-contents)

</details>

---
### 21. Qn 21: How do you efficiently identify and remove duplicate rows in a DataFrame?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and B

**Explanation:** Both df[~df.duplicated()] and df.drop_duplicates() remove duplicate rows, with the latter being more readable and offering more options.


[Go to TOC](#table-of-contents)

</details>

---
### 22. Qn 22: Which method is most efficient for applying a custom function to a DataFrame that returns a scalar?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** df.pipe(custom_func)

**Explanation:** pipe() is designed for functions that take and return a DataFrame, creating readable method chains when applying multiple functions.


[Go to TOC](#table-of-contents)

</details>

---
### 23. Qn 23: How do you sample data from a DataFrame with weights?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both B and C

**Explanation:** Both approaches correctly sample with weights, though weights don't need to be normalized as pandas normalizes them internally.


[Go to TOC](#table-of-contents)

</details>

---
### 24. Qn 24: What's the correct way to use the pd.cut() function for binning continuous data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above are valid uses

**Explanation:** All approaches are valid: using explicit bin edges, equal-width bins (cut), or equal-frequency bins (qcut).


[Go to TOC](#table-of-contents)

</details>

---
### 25. Qn 25: How do you efficiently perform a custom window operation in pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and B

**Explanation:** Both approaches work for custom window operations, but using raw=True can be more efficient for numerical operations by passing a NumPy array instead of a Series.


[Go to TOC](#table-of-contents)

</details>

---
### 26. Qn 26: Which approach can create a lagged feature in a time series DataFrame?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and B

**Explanation:** shift(1) creates a lag (past values), while shift(-1) creates a lead (future values), both useful for time series analysis.


[Go to TOC](#table-of-contents)

</details>

---
### 27. Qn 27: What's the best way to explode a DataFrame column containing lists into multiple rows?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both B and C

**Explanation:** explode() transforms each element of a list-like column into a row, with the original index duplicated as needed.


[Go to TOC](#table-of-contents)

</details>

---
### 28. Qn 28: How do you efficiently compute a weighted mean in pandas?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Both A and C

**Explanation:** Both manually computing weighted mean and using np.average() work efficiently, though pandas Series doesn't have a weights parameter for mean().


[Go to TOC](#table-of-contents)

</details>

---
### 29. Qn 29: Which method correctly identifies the top-k values in each group?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All three methods can get the top-k values within each group, with different syntax but similar results.


[Go to TOC](#table-of-contents)

</details>

---
### 30. Qn 30: What's the best way to add a new column based on a categorical mapping of an existing column?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** All of the above

**Explanation:** All methods can map values to new ones, though map() is generally preferred for dictionary-based mappings.


[Go to TOC](#table-of-contents)

</details>

---
