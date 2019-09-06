---
layout : post
title : PL03-Topic02, Pandas
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) <br>
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

## **Series**

### ***One-column***

`CREATING METHOD1 : sperately`
```python
import pandas as pd

s = pd.Series([909976, 8615246, 2872086, 2273305])
s.name = "Population"
s.index = ["Stockholm", "London", "Rome", "Paris"] 

s
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Stockholm     909976
London       8615246
Rome         2872086
Paris        2273305
Name: Population, dtype: int64
```
<hr class='division3'>
</details>
<br>

```python
type(s)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
<class 'pandas.core.series.Series'> 
```
<hr class='division3'>
</details>

<br>
```python
type(s.name)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
<class 'str'> 
```
<hr class='division3'>
</details>

<br>
```python
type(s.index)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
<class 'pandas.core.indexes.base.Index'>
```
<hr class='division3'>
</details>


<br><br><br>
`CREATING METHOD2 : all at once`
```python
import pandas as pd

s = pd.Series([909976, 8615246, 2872086, 2273305], 
              name="Population" ,
              index=["Stockholm", "London", "Rome", "Paris"])
s
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Stockholm     909976
London       8615246
Rome         2872086
Paris        2273305
Name: Population, dtype: int64
```
<hr class='division3'>
</details>

<br><br><br>

---

#### A series] searching index(key) & name of a column
`Searching all index`
```python
import pandas as pd

s = pd.Series([909976, 8615246, 2872086, 2273305], 
              name="Population" ,
              index=["Stockholm", "London", "Rome", "Paris"])
              
s.index
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Index(['Stockholm', 'London', 'Rome', 'Paris'], dtype='object')
```
<hr class='division3'>
</details>
<br>

<br><br><br>

---

#### A series] searching values based on index(key)

```python
import pandas as pd

# method0 searching single value 
s[1]
print('\n method0 searching single value, \n s[1] : \n',
      s[1])

# method1 searching single value 
s["London"]
print('\n method1 searching single value, \n s["London"] : \n',
      s["London"])

# method2 searching single value 
s.London
print('\n method2 searching single value, \n s.London : \n',
      s.London)





# method0 searching multi-values 
s[[1,2]]
print('\n\n method0 searching multi-values, \n s[[1,2]] : \n',
      s[[1,2]])

# method1 searching multi-values 
s[["London","Rome"]]
print('\n method1 searching multi-values, \n s[["London","Rome"]] : \n',
      s[["London","Rome"]])

# method2 searching multi-values 
s[1:3]
print('\n method2 searching multi-values, \n s[1:2] : \n',
      s[1:3])

# method3 searching multi-values 
s["London":"Rome"]
print('\n method3 searching multi-values, \n s["London":"Rome"] : \n',
      s["London":"Rome"])





#  method searching all values
s.values
print('\n\n method searching all values, \n s.values : \n',
      s.values)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
 method0 searching single value, 
 s[1] : 
 8615246

 method1 searching single value, 
 s["London"] : 
 8615246

 method2 searching single value, 
 s.London : 
 8615246


 method0 searching multi-values, 
 s[[1,2]] : 
 London    8615246
Rome      2872086
Name: Population, dtype: int64

 method1 searching multi-values, 
 s[["London","Rome"]] : 
 London    8615246
Rome      2872086
Name: Population, dtype: int64

 method2 searching multi-values, 
 s[1:2] : 
 London    8615246
Rome      2872086
Name: Population, dtype: int64

 method3 searching multi-values, 
 s["London":"Rome"] : 
 London    8615246
Rome      2872086
Name: Population, dtype: int64


 method searching all values, 
 s.values : 
 [ 909976 8615246 2872086 2273305]
```
<hr class='division3'>
</details>
<br><br><br>

---

#### A series] analysis

```python
import pandas as pd

s = pd.Series([1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4], 
              name="Population")
s.head()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
0    1
1    1
2    1
3    2
4    2
Name: Population, dtype: int64
```
<hr class='division3'>
</details>

```python
s.shape
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
(21,)
```
<hr class='division3'>
</details>

```python
s.unique()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([1, 2, 3, 4], dtype=int64)
```
<hr class='division3'>
</details>

```python
s.value_counts()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
3    10
2     5
4     3
1     3
Name: Population, dtype: int64
```
<hr class='division3'>
</details>
<br><br><br>

---

#### A series] statistics

```python
import pandas as pd

s = pd.Series([1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4], 
              name="Population")
              
s.median(), s.mean(), s.std(), s.min(), s.max()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
(3.0, 2.619047619047619, 0.9206622874969125, 1, 4)
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Several columns***

---

<hr class="division2">

## **DataFrame**

### ***One-dataframe***

---

### ***Several dataframes***

---

<hr class="division2">

## **Covert Data-Type**

### ***DataFrame to Series***

---

### ***Series to DataFrame***

---

### ***DataFrame to numpy***

---

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---


