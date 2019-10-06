---
layout : post
title : MATH02, Permutation and combination
categories: [MATH02]
comments : true
tags : [MATH02]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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

## **Permutation**
`Method 1 : by Built-in function`
```python
def permute(arr):
    result = [arr[:]]
    c = [0] * len(arr)
    i = 0
    while i < len(arr):
        if c[i] < i:
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            result.append(arr[:])
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return result
```
<br><br><br>
`Method 2 : by itertools module`
```python
import itertools

pool = ['A', 'B', 'C']
print(list(map(''.join, itertools.permutations(pool, r=3)))) # progression through 3 elements
print(list(map(''.join, itertools.permutations(pool, r=2)))) # progression through 2 elements
```
```
['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']
['AB', 'AC', 'BA', 'BC', 'CA', 'CB']
```
<br><br><br>
<hr class="division2">

## **Combination**

```python
import itertools

pool = ['A', 'B', 'C']
print(list(map(''.join, itertools.combinations(pool, r=3)))) # progression through 3 elements
print(list(map(''.join, itertools.combinations(pool, r=2)))) # progression through 2 elements
```
```
['ABC']
['AB', 'AC', 'BC']
```

<br><br><br>
<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://programmers.co.kr/learn/courses/4008/lessons/12836'>programmers</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---
