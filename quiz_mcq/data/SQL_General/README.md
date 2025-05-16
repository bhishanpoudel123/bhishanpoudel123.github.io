# Sql General Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: Which SQL statement is used to extract data from a database?](#q01)  
- [Qn 02: Which SQL clause is used to filter records?](#q02)  
- [Qn 03: What does the COUNT() function do in SQL?](#q03)  
- [Qn 04: Which SQL keyword is used to sort the result-set?](#q04)  
- [Qn 05: Which command is used to remove all records from a table in SQL without deleting the table?](#q05)  
- [Qn 06: Which SQL clause is used with aggregate functions to group result-set by one or more columns?](#q06)  
- [Qn 07: Which SQL keyword is used to retrieve only distinct values?](#q07)  
- [Qn 08: Which of the following is a DDL command?](#q08)  
- [Qn 09: What does the SQL INNER JOIN keyword do?](#q09)  
- [Qn 10: What will the result of the query 'SELECT * FROM employees WHERE department IS NULL;' be?](#q10)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
Which SQL statement is used to extract data from a database?

**Options**  
1. GET  
2. SELECT  
3. OPEN  
4. EXTRACT  

**Answer**  
SELECT

**Explanation**  
The SELECT statement is used to extract data from a database table.

[↑ Go to TOC](#toc)

  

### <a id="q02"></a> Qn 02

**Question**  
Which SQL clause is used to filter records?

**Options**  
1. WHERE  
2. GROUP BY  
3. ORDER BY  
4. HAVING  

**Answer**  
WHERE

**Explanation**  
The WHERE clause is used to filter records based on specific conditions.

[↑ Go to TOC](#toc)

  

### <a id="q03"></a> Qn 03

**Question**  
What does the COUNT() function do in SQL?

**Options**  
1. Counts rows with NULLs  
2. Counts only numeric values  
3. Counts non-NULL rows  
4. Counts unique values  

**Answer**  
Counts non-NULL rows

**Explanation**  
COUNT() returns the number of non-NULL values in a specified column.

[↑ Go to TOC](#toc)

  

### <a id="q04"></a> Qn 04

**Question**  
Which SQL keyword is used to sort the result-set?

**Options**  
1. SORT BY  
2. ORDER BY  
3. GROUP BY  
4. ARRANGE BY  

**Answer**  
ORDER BY

**Explanation**  
ORDER BY is used to sort the results of a SELECT query.

[↑ Go to TOC](#toc)

  

### <a id="q05"></a> Qn 05

**Question**  
Which command is used to remove all records from a table in SQL without deleting the table?

**Options**  
1. DELETE  
2. REMOVE  
3. DROP  
4. TRUNCATE  

**Answer**  
TRUNCATE

**Explanation**  
TRUNCATE removes all records from a table but retains the table structure.

[↑ Go to TOC](#toc)

  

### <a id="q06"></a> Qn 06

**Question**  
Which SQL clause is used with aggregate functions to group result-set by one or more columns?

**Options**  
1. GROUP BY  
2. ORDER BY  
3. WHERE  
4. HAVING  

**Answer**  
GROUP BY

**Explanation**  
GROUP BY groups rows that have the same values into summary rows.

[↑ Go to TOC](#toc)

  

### <a id="q07"></a> Qn 07

**Question**  
Which SQL keyword is used to retrieve only distinct values?

**Options**  
1. UNIQUE  
2. ONLY  
3. DISTINCT  
4. SEPARATE  

**Answer**  
DISTINCT

**Explanation**  
DISTINCT is used to return only different (distinct) values.

[↑ Go to TOC](#toc)

  

### <a id="q08"></a> Qn 08

**Question**  
Which of the following is a DDL command?

**Options**  
1. INSERT  
2. UPDATE  
3. DELETE  
4. CREATE  

**Answer**  
CREATE

**Explanation**  
CREATE is a DDL (Data Definition Language) command used to create a new table or
  database.

[↑ Go to TOC](#toc)

  

### <a id="q09"></a> Qn 09

**Question**  
What does the SQL INNER JOIN keyword do?

**Options**  
1. Returns rows when there is a match in both tables  
2. Returns all rows from the left table  
3. Returns all rows from the right table  
4. Combines all rows from both tables  

**Answer**  
Returns rows when there is a match in both tables

**Explanation**  
INNER JOIN selects records that have matching values in both tables.

[↑ Go to TOC](#toc)

  

### <a id="q10"></a> Qn 10

**Question**  
What will the result of the query 'SELECT * FROM employees WHERE department IS NULL;' be?

**Options**  
1. It will throw an error  
2. It selects employees with empty strings  
3. It selects employees with NULL department  
4. It selects all employees  

**Answer**  
It selects employees with NULL department

**Explanation**  
IS NULL checks for columns that contain NULL values.

[↑ Go to TOC](#toc)



---

*Automatically generated from [sql_general_questions.json](sql_general_questions.json)*  
*Updated: 2025-05-16 15:26*
