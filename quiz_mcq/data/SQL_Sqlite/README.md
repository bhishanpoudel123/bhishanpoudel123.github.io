# Sql Sqlite Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: What is the purpose of the `WITHOUT ROWID` clause in SQLite?](#q01)  
- [Qn 02: Which function would you use in SQLite to get the current timestamp?](#q02)  
- [Qn 03: What is the default data type of a column in SQLite if not specified?](#q03)  
- [Qn 04: How are boolean values stored in SQLite?](#q04)  
- [Qn 05: Which of the following is true about SQLite's `VACUUM` command?](#q05)  
- [Qn 06: Which SQLite command lists all tables in the database?](#q06)  
- [Qn 07: Which SQLite command allows you to see the schema of a table?](#q07)  
- [Qn 08: How does SQLite handle foreign key constraints by default?](#q08)  
- [Qn 09: How does SQLite implement AUTOINCREMENT?](#q09)  
- [Qn 10: What pragma statement turns on write-ahead logging in SQLite?](#q10)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
What is the purpose of the `WITHOUT ROWID` clause in SQLite?

**Options**  
1. To create a view without row numbers  
2. To disable rowid for performance reasons  
3. To create a table without the implicit ROWID column  
4. To avoid storing duplicate rows  

**Answer**  
To create a table without the implicit ROWID column

**Explanation**  
`WITHOUT ROWID` creates a table without the implicit `ROWID`, useful for certain
  optimizations.

[↑ Go to TOC](#toc)

  

### <a id="q02"></a> Qn 02

**Question**  
Which function would you use in SQLite to get the current timestamp?

**Options**  
1. NOW()  
2. CURRENT_TIMESTAMP  
3. GETDATE()  
4. SYSDATE  

**Answer**  
CURRENT_TIMESTAMP

**Explanation**  
`CURRENT_TIMESTAMP` returns the current date and time in SQLite.

[↑ Go to TOC](#toc)

  

### <a id="q03"></a> Qn 03

**Question**  
What is the default data type of a column in SQLite if not specified?

**Options**  
1. TEXT  
2. NUMERIC  
3. ANY  
4. NONE  

**Answer**  
NONE

**Explanation**  
If no type is specified, SQLite assigns it an affinity of NONE.

[↑ Go to TOC](#toc)

  

### <a id="q04"></a> Qn 04

**Question**  
How are boolean values stored in SQLite?

**Options**  
1. As TRUE/FALSE literals  
2. As 1 and 0 integers  
3. As BIT type  
4. As TEXT 'true'/'false'  

**Answer**  
As 1 and 0 integers

**Explanation**  
SQLite does not have a separate BOOLEAN type; it uses integers 1 (true) and 0
  (false).

[↑ Go to TOC](#toc)

  

### <a id="q05"></a> Qn 05

**Question**  
Which of the following is true about SQLite's `VACUUM` command?

**Options**  
1. It deletes rows with NULL values  
2. It compacts the database file  
3. It removes duplicate records  
4. It optimizes table indexes  

**Answer**  
It compacts the database file

**Explanation**  
`VACUUM` rebuilds the database file to defragment it and reduce its size.

[↑ Go to TOC](#toc)

  

### <a id="q06"></a> Qn 06

**Question**  
Which SQLite command lists all tables in the database?

**Options**  
1. SHOW TABLES  
2. SELECT * FROM sqlite_master WHERE type='table'  
3. .list tables  
4. DESCRIBE  

**Answer**  
SELECT * FROM sqlite_master WHERE type='table'

**Explanation**  
SQLite uses the `sqlite_master` table to store metadata about the database,
  including table names.

[↑ Go to TOC](#toc)

  

### <a id="q07"></a> Qn 07

**Question**  
Which SQLite command allows you to see the schema of a table?

**Options**  
1. PRAGMA schema  
2. SHOW TABLE  
3. DESCRIBE  
4. .schema  

**Answer**  
.schema

**Explanation**  
`.schema` is a command in the SQLite shell that shows the schema for tables.

[↑ Go to TOC](#toc)

  

### <a id="q08"></a> Qn 08

**Question**  
How does SQLite handle foreign key constraints by default?

**Options**  
1. They are enforced automatically  
2. They must be manually triggered  
3. They are off by default and must be enabled  
4. They are not supported in SQLite  

**Answer**  
They are off by default and must be enabled

**Explanation**  
SQLite supports foreign keys, but enforcement must be enabled with `PRAGMA
  foreign_keys = ON`.

[↑ Go to TOC](#toc)

  

### <a id="q09"></a> Qn 09

**Question**  
How does SQLite implement AUTOINCREMENT?

**Options**  
1. Using INTEGER PRIMARY KEY  
2. Using IDENTITY  
3. Using SERIAL  
4. Using a sequence object  

**Answer**  
Using INTEGER PRIMARY KEY

**Explanation**  
SQLite uses `INTEGER PRIMARY KEY AUTOINCREMENT` to create an auto-incrementing
  ID.

[↑ Go to TOC](#toc)

  

### <a id="q10"></a> Qn 10

**Question**  
What pragma statement turns on write-ahead logging in SQLite?

**Options**  
1. PRAGMA enable_wal = ON  
2. PRAGMA journal_mode = WAL  
3. PRAGMA log_mode = ON  
4. PRAGMA write_ahead = TRUE  

**Answer**  
PRAGMA journal_mode = WAL

**Explanation**  
`PRAGMA journal_mode = WAL` enables write-ahead logging in SQLite.

[↑ Go to TOC](#toc)



---

*Automatically generated from [sql_sqlite_questions.json](sql_sqlite_questions.json)*  
*Updated: 2025-05-16 15:26*
