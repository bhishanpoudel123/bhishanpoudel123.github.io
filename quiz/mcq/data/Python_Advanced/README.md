# Python_Advanced Quiz

## Table of Contents
- [Qn 01: What is the time complexity of inserting an element at the beginning of a Python list?](#1)
- [Qn 02: Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?](#2)
- [Qn 03: Which Python library provides decorators and context managers to handle retries with exponential backoff?](#3)
- [Qn 04: What is a key difference between multiprocessing and threading in Python?](#4)
- [Qn 05: What is the purpose of Python's `__slots__` declaration?](#5)
- [Qn 06: What will `functools.lru_cache` do?](#6)
- [Qn 07: What does the `@staticmethod` decorator do in Python?](#7)
- [Qn 08: How can you profile memory usage in a Python function?](#8)
- [Qn 09: Which built-in function returns the identity of an object?](#9)
- [Qn 10: What happens when you use the `is` operator between two equal strings in Python?](#10)

---

### 1. Qn 01: What is the time complexity of inserting an element at the beginning of a Python list?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** O(n)

**Explanation:** Inserting at the beginning of a Python list requires shifting all elements, hence O(n).


[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** NumPy arrays

**Explanation:** NumPy arrays are memory efficient and optimized for numerical operations.


[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: Which Python library provides decorators and context managers to handle retries with exponential backoff?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** tenacity

**Explanation:** Tenacity provides powerful retry strategies including exponential backoff.


[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: What is a key difference between multiprocessing and threading in Python?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Processes can utilize multiple CPUs

**Explanation:** Due to the GIL, threads are limited; multiprocessing uses separate memory space and cores.


[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: What is the purpose of Python's `__slots__` declaration?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Reduce memory usage by preventing dynamic attribute creation

**Explanation:** `__slots__` limits attribute assignment and avoids `__dict__` overhead.


[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: What will `functools.lru_cache` do?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Cache function output to speed up subsequent calls

**Explanation:** `lru_cache` stores results of expensive function calls for reuse.


[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: What does the `@staticmethod` decorator do in Python?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Defines a method that takes no self or cls argument

**Explanation:** `@staticmethod` defines a method that does not receive an implicit first argument.

**Learning Resources:**
- [qn_07_answer_long_01](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io/tree/main/quiz/mcq/data/Python_Advanced/questions/qn_07/markdown/qn_07_answer_01.md) (markdown)

[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: How can you profile memory usage in a Python function?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Using tracemalloc

**Explanation:** `tracemalloc` tracks memory allocations in Python.


[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: Which built-in function returns the identity of an object?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** id()

**Explanation:** `id()` returns the identity (memory address) of an object.


[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: What happens when you use the `is` operator between two equal strings in Python?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** It compares object identity

**Explanation:** `is` checks whether two variables point to the same object, not if their values are equal.


[Go to TOC](#table-of-contents)

</details>

---
