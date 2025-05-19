# Python Advanced Study Guide <a id="toc"></a>

## Table of Contents
- [Qn 01: What is the time complexity of inserting an element at the beginning of a Python list?](#q01)  
- [Qn 02: Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?](#q02)  
- [Qn 03: Which Python library provides decorators and context managers to handle retries with exponential backoff?](#q03)  
- [Qn 04: What is a key difference between multiprocessing and threading in Python?](#q04)  
- [Qn 05: What is the purpose of Python's `__slots__` declaration?](#q05)  
- [Qn 06: What will `functools.lru_cache` do?](#q06)  
- [Qn 07: What does the `@staticmethod` decorator do in Python?](#q07)  
- [Qn 08: How can you profile memory usage in a Python function?](#q08)  
- [Qn 09: Which built-in function returns the identity of an object?](#q09)  
- [Qn 10: What happens when you use the `is` operator between two equal strings in Python?](#q10)

## Questions
### <a id="q01"></a> Qn 01

**Question**  
What is the time complexity of inserting an element at the beginning of a Python list?

**Options**  

1. O(1)  
2. O(n)  
3. O(log n)  
4. O(n log n)  

**Answer**  
O(n)

**Explanation**  
Inserting at the beginning of a Python list requires shifting all elements,
  hence O(n).

[↑ Go to TOC](#toc)


### <a id="q02"></a> Qn 02

**Question**  
Which of the following is the most memory-efficient way to handle large numerical data arrays in Python?

**Options**  

1. List of lists  
2. Pandas DataFrame  
3. NumPy arrays  
4. Tuples  

**Answer**  
NumPy arrays

**Explanation**  
NumPy arrays are memory efficient and optimized for numerical operations.

[↑ Go to TOC](#toc)


### <a id="q03"></a> Qn 03

**Question**  
Which Python library provides decorators and context managers to handle retries with exponential backoff?

**Options**  

1. retrying  
2. backoff  
3. tenacity  
4. retry  

**Answer**  
tenacity

**Explanation**  
Tenacity provides powerful retry strategies including exponential backoff.

[↑ Go to TOC](#toc)


### <a id="q04"></a> Qn 04

**Question**  
What is a key difference between multiprocessing and threading in Python?

**Options**  

1. Threads use more memory  
2. Processes can utilize multiple CPUs  
3. Threads can run in parallel on multiple cores  
4. Multiprocessing is slower than threading  

**Answer**  
Processes can utilize multiple CPUs

**Explanation**  
Due to the GIL, threads are limited; multiprocessing uses separate memory space
  and cores.

[↑ Go to TOC](#toc)


### <a id="q05"></a> Qn 05

**Question**  
What is the purpose of Python's `__slots__` declaration?

**Options**  

1. Reduce memory usage by preventing dynamic attribute creation  
2. Enable dynamic typing  
3. Improve readability  
4. Create private attributes  

**Answer**  
Reduce memory usage by preventing dynamic attribute creation

**Explanation**  
`__slots__` limits attribute assignment and avoids `__dict__` overhead.

[↑ Go to TOC](#toc)


### <a id="q06"></a> Qn 06

**Question**  
What will `functools.lru_cache` do?

**Options**  

1. Cache function output to speed up subsequent calls  
2. Limit recursion depth  
3. Create memory leaks  
4. Parallelize function calls  

**Answer**  
Cache function output to speed up subsequent calls

**Explanation**  
`lru_cache` stores results of expensive function calls for reuse.

[↑ Go to TOC](#toc)


### <a id="q07"></a> Qn 07

**Question**  
What does the `@staticmethod` decorator do in Python?

**Options**  

1. Defines a method that takes no self or cls argument  
2. Makes method private  
3. Allows inheritance  
4. Makes the method static across all classes  

**Answer**  
Defines a method that takes no self or cls argument

**Explanation**  
`@staticmethod` defines a method that does not receive an implicit first
  argument.

**Detailed Explanation**  
See detailed documentation: [qn_07_answer_long_01.md](data/Python_Advanced/qn_07_answer_long_01.md)

[↑ Go to TOC](#toc)


### <a id="q08"></a> Qn 08

**Question**  
How can you profile memory usage in a Python function?

**Options**  

1. Using memoryview  
2. Using tracemalloc  
3. Using psutil only  
4. Using timeit  

**Answer**  
Using tracemalloc

**Explanation**  
`tracemalloc` tracks memory allocations in Python.

[↑ Go to TOC](#toc)


### <a id="q09"></a> Qn 09

**Question**  
Which built-in function returns the identity of an object?

**Options**  

1. type()  
2. id()  
3. hash()  
4. repr()  

**Answer**  
id()

**Explanation**  
`id()` returns the identity (memory address) of an object.

[↑ Go to TOC](#toc)


### <a id="q10"></a> Qn 10

**Question**  
What happens when you use the `is` operator between two equal strings in Python?

**Options**  

1. It checks value equality  
2. It compares object identity  
3. It converts them to integers  
4. It raises an exception  

**Answer**  
It compares object identity

**Explanation**  
`is` checks whether two variables point to the same object, not if their values
  are equal.

[↑ Go to TOC](#toc)


---

*Automatically generated from [python_advanced_questions.json](python_advanced_questions.json)*
*Updated: 2025-05-18 13:57*
