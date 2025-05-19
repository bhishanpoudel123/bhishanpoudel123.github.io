# SQL_Sqlite Quiz

## Table of Contents
- [Qn 01: What is the purpose of the `WITHOUT ROWID` clause in SQLite?](#1)
- [Qn 02: Which function would you use in SQLite to get the current timestamp?](#2)
- [Qn 03: What is the default data type of a column in SQLite if not specified?](#3)
- [Qn 04: How are boolean values stored in SQLite?](#4)
- [Qn 05: Which of the following is true about SQLite's `VACUUM` command?](#5)
- [Qn 06: Which SQLite command lists all tables in the database?](#6)
- [Qn 07: Which SQLite command allows you to see the schema of a table?](#7)
- [Qn 08: How does SQLite handle foreign key constraints by default?](#8)
- [Qn 09: How does SQLite implement AUTOINCREMENT?](#9)
- [Qn 10: What pragma statement turns on write-ahead logging in SQLite?](#10)

---

### 1. Qn 01: What is the purpose of the `WITHOUT ROWID` clause in SQLite?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** To create a table without the implicit ROWID column

**Explanation:** `WITHOUT ROWID` creates a table without the implicit `ROWID`, useful for certain optimizations.


[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: Which function would you use in SQLite to get the current timestamp?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** CURRENT_TIMESTAMP

**Explanation:** `CURRENT_TIMESTAMP` returns the current date and time in SQLite.


[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: What is the default data type of a column in SQLite if not specified?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** NONE

**Explanation:** If no type is specified, SQLite assigns it an affinity of NONE.


[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: How are boolean values stored in SQLite?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** As 1 and 0 integers

**Explanation:** SQLite does not have a separate BOOLEAN type; it uses integers 1 (true) and 0 (false).


[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: Which of the following is true about SQLite's `VACUUM` command?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** It compacts the database file

**Explanation:** `VACUUM` rebuilds the database file to defragment it and reduce its size.


[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: Which SQLite command lists all tables in the database?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** SELECT * FROM sqlite_master WHERE type='table'

**Explanation:** SQLite uses the `sqlite_master` table to store metadata about the database, including table names.


[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: Which SQLite command allows you to see the schema of a table?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** .schema

**Explanation:** `.schema` is a command in the SQLite shell that shows the schema for tables.


[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: How does SQLite handle foreign key constraints by default?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** They are off by default and must be enabled

**Explanation:** SQLite supports foreign keys, but enforcement must be enabled with `PRAGMA foreign_keys = ON`.


[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: How does SQLite implement AUTOINCREMENT?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Using INTEGER PRIMARY KEY

**Explanation:** SQLite uses `INTEGER PRIMARY KEY AUTOINCREMENT` to create an auto-incrementing ID.


[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: What pragma statement turns on write-ahead logging in SQLite?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** PRAGMA journal_mode = WAL

**Explanation:** `PRAGMA journal_mode = WAL` enables write-ahead logging in SQLite.


[Go to TOC](#table-of-contents)

</details>

---
