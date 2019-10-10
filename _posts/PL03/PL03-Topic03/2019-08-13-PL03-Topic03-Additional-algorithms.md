---
layout : post
title : PL03-Topic03, Additional algorithms
categories: [PL03-Topic03]
comments : true
tags : [PL03-Topic03]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Algorithm.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/'>post1</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **maths**

### ***basic_maths***

<br><br><br>

---




### ***find***

#### find_max

```python
```

<br><br><br>


#### find_min

```python
```

<br><br><br>

#### find_lcm

<br><br><br>

---

### ***abs***

#### abs

```python
def abs_val(num):
    return -num if num < 0 else num
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
>>> abs_val(-5)
5
>>> abs_val(0)
0
```
<hr class='division3'>
</details>
<br><br><br>

#### abs_max

```python
from typing import List

def abs_max(x: List[int]) -> int:
    j = x[0]
    for i in x:
        if abs(i) > abs(j):
            j = i
    return j
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
>>> abs_max([0,5,1,11])
11
>>> abs_max([3,-10,-2])
-10
```
<hr class='division3'>
</details>
<br>
```python
def abs_max_sort(x):
    return sorted(x, key=abs)[-1]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
>>> abs_max_sort([0,5,1,11])
11
>>> abs_max_sort([3,-10,-2])
-10
```
<hr class='division3'>
</details>
<br><br><br>

#### abs_min

```python
```

<br><br><br>

---

### ***aggregation_function***

#### average_mean

```python
def average(nums):
    avg = sum(nums) / len(nums)
    return avg
```

<br><br><br>

#### average_median

```python
def median(nums):
    sorted_list = sorted(nums)
    med = None
    if len(sorted_list) % 2 == 0:
        mid_index_1 = len(sorted_list) // 2
        mid_index_2 = (len(sorted_list) // 2) - 1
        med = (sorted_list[mid_index_1] + sorted_list[mid_index_2]) / float(2)
    else:
        mid_index = (len(sorted_list) - 1) // 2
        med = sorted_list[mid_index]
    return med
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
>>> median([0])
0
>>> median([4,1,3,2])
2.5
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***factorial_recursive***

```python
def fact(n):
    return 1 if n <= 1 else n*fact(n-1)
```

<br><br><br>

---

### ***fibonacci***

<br><br><br>

---

### ***Gaussian***

```python
from numpy import pi, sqrt, exp

def gaussian(x, mu: float = 0.0, sigma: float = 1.0) -> int:
    return 1 / sqrt(2 * pi * sigma ** 2) * exp(-(x - mu) ** 2 / 2 * sigma ** 2)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
>>> gaussian(1)
0.24197072451914337
  
>>> gaussian(24)
3.342714441794458e-126
Supports NumPy Arrays
Use numpy.meshgrid with this to generate gaussian blur on images.
>>> import numpy as np
>>> x = np.arange(15)
>>> gaussian(x)
array([3.98942280e-01, 2.41970725e-01, 5.39909665e-02, 4.43184841e-03,
       1.33830226e-04, 1.48671951e-06, 6.07588285e-09, 9.13472041e-12,
       5.05227108e-15, 1.02797736e-18, 7.69459863e-23, 2.11881925e-27,
       2.14638374e-32, 7.99882776e-38, 1.09660656e-43])
           
>>> gaussian(15)
5.530709549844416e-50
>>> gaussian([1,2, 'string'])
Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for -: 'list' and 'float'
>>> gaussian('hello world')
Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for -: 'str' and 'float'
>>> gaussian(10**234) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
OverflowError: (34, 'Result too large')
>>> gaussian(10**-326)
0.3989422804014327
>>> gaussian(2523, mu=234234, sigma=3425)
0.0
```
<hr class='division3'>
</details>

<br><br><br>

---

### **derivative**

#### simpson_rule

```python
```

<br><br><br>

#### trapezoidal_rule

```python
```

<br><br><br>

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://github.com/TheAlgorithms/Python' target="_blank">TheAlgorithms</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```

```
<hr class='division3'>
</details>

