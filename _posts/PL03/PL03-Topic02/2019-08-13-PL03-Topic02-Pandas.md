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

#### Creating and searching

`CREATING METHOD1 : sperately`
```python
import pandas as pd

s = pd.Series([909976, 8615246, 2872086, 2273305])
s.name = "Population"
s.index = ["Stockholm", "London", "Rome", "Paris"] 
```
```python
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
```
```python
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
```python 
s.name
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
'Population'
```
<hr class='division3'>
</details>
<br>

```python 
s.values
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method searching all values 
```
array([ 909976, 8615246, 2872086, 2273305], dtype=int64)
```
<hr class='division3'>
</details>
<br>

```python 
s[1]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method0 : searching single value 
```
8615246
```
<hr class='division3'>
</details>
<br>

```python 
s["London"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 : searching single value 
```
8615246
```
<hr class='division3'>
</details>
<br>

```python 
s.London
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 : searching single value 
```
8615246
```
<hr class='division3'>
</details>
<br>

```python 
s[[1,2]]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method0 : searching multi-values
```
London    8615246
Rome      2872086
Name: Population, dtype: int64
```
<hr class='division3'>
</details>
<br>

```python 
s[["London","Rome"]]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 : searching multi-values 
```
London    8615246
Rome      2872086
Name: Population, dtype: int64
```
<hr class='division3'>
</details>
<br>

```python 
s[1:3]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 : searching multi-values
```
London    8615246
Rome      2872086
Name: Population, dtype: int64
```
<hr class='division3'>
</details>
<br>

```python 
s["London":"Rome"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method3 : searching multi-values 
```
London    8615246
Rome      2872086
Name: Population, dtype: int64
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


