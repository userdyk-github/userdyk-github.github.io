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
```python
def abs_max_sort(x):
    return sorted(x, key=abs)[-1]
```

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
```

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
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>

