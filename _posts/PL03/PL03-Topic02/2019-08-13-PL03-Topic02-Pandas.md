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





#### Analysis

```python
import pandas as pd

s = pd.Series([1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4], 
              name="Population")
```
```python
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
<br>

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
<br>

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
<br>

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

#### Statistics

```python
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series([1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4], 
              name="Population")
```
```python
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
<br>

```python
s.describe()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
count    21.000000
mean      2.619048
std       0.920662
min       1.000000
25%       2.000000
50%       3.000000
75%       3.000000
max       4.000000
Name: Population, dtype: float64
```
<hr class='division3'>
</details>
<br>

```python
fig, axes = plt.subplots(1,4, figsize=(12, 3))
s.plot(ax=axes[0], kind='line', title='line')
s.plot(ax=axes[1], kind='bar', title='bar')
s.plot(ax=axes[2], kind='box', title='box')
s.plot(ax=axes[3], kind='pie', title='pie')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/64455025-7d710780-d127-11e9-9734-4b88a3bf9f37.png)
<hr class='division3'>
</details>


<br><br><br>

---

### ***Several columns***

#### Creating and concatenate

```python
import pandas as pd

s1 = pd.Series([909976, 8615246, 2872086, 2273305], 
              name="Population1" ,
              index=["Stockholm1", "London1", "Rome1", "Paris1"])
s2 = pd.Series([909976, 8615246, 2872086, 2273305], 
              name="Population2" ,
              index=["Stockholm2", "London2", "Rome2", "Paris2"])
s3 = pd.Series([909976, 8615246, 2872086, 2273305], 
              name="Population3" ,
              index=["Stockholm3", "London3", "Rome3", "Paris3"])
```
```python
s1, s2, s3
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Stockholm1     909976
London1       8615246
Rome1         2872086
Paris1        2273305
Name: Population1, dtype: int64 

 Stockholm2     909976
London2       8615246
Rome2         2872086
Paris2        2273305
Name: Population2, dtype: int64 

 Stockholm3     909976
London3       8615246
Rome3         2872086
Paris3        2273305
Name: Population3, dtype: int64
```
<hr class='division3'>
</details>
<br>

```python
pd.concat([s1, s2, s3], axis=0)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Stockholm1     909976
London1       8615246
Rome1         2872086
Paris1        2273305
Stockholm2     909976
London2       8615246
Rome2         2872086
Paris2        2273305
Stockholm3     909976
London3       8615246
Rome3         2872086
Paris3        2273305
dtype: int64
```
<hr class='division3'>
</details>
<br>

```python
pd.concat([s1, s2, s3], axis=1)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
            Population1  Population2  Population3
London1       8615246.0          NaN          NaN
London2             NaN    8615246.0          NaN
London3             NaN          NaN    8615246.0
Paris1        2273305.0          NaN          NaN
Paris2              NaN    2273305.0          NaN
Paris3              NaN          NaN    2273305.0
Rome1         2872086.0          NaN          NaN
Rome2               NaN    2872086.0          NaN
Rome3               NaN          NaN    2872086.0
Stockholm1     909976.0          NaN          NaN
Stockholm2          NaN     909976.0          NaN
Stockholm3          NaN          NaN     909976.0
```
<hr class='division3'>
</details>
<br>

```python
pd.concat([s1, s2, s3], axis=1, ignore_index=True)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
                    0          1          2
London1     8615246.0        NaN        NaN
London2           NaN  8615246.0        NaN
London3           NaN        NaN  8615246.0
Paris1      2273305.0        NaN        NaN
Paris2            NaN  2273305.0        NaN
Paris3            NaN        NaN  2273305.0
Rome1       2872086.0        NaN        NaN
Rome2             NaN  2872086.0        NaN
Rome3             NaN        NaN  2872086.0
Stockholm1   909976.0        NaN        NaN
Stockholm2        NaN   909976.0        NaN
Stockholm3        NaN        NaN   909976.0
```
<hr class='division3'>
</details>
<br>

```python
pd.concat([s1, s2, s3], axis=1, keys=['C0', 'C1', 'C2'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
                   C0         C1         C2
London1     8615246.0        NaN        NaN
London2           NaN  8615246.0        NaN
London3           NaN        NaN  8615246.0
Paris1      2273305.0        NaN        NaN
Paris2            NaN  2273305.0        NaN
Paris3            NaN        NaN  2273305.0
Rome1       2872086.0        NaN        NaN
Rome2             NaN  2872086.0        NaN
Rome3             NaN        NaN  2872086.0
Stockholm1   909976.0        NaN        NaN
Stockholm2        NaN   909976.0        NaN
Stockholm3        NaN        NaN   909976.0
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **DataFrame**

### ***One-dataframe***

#### Creating and searching

`Creating method1 based on row`
```python
import pandas as pd

df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United Kingdom"],
                   [2872086, "Italy"],
                   [2273305, "France"]])
df.index = ["Stockholm", "London", "Rome", "Paris"]
df.columns = ["Population", "State"] 
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
           Population           State
Stockholm      909976          Sweden
London        8615246  United Kingdom
Rome          2872086           Italy
Paris         2273305          France
```
<hr class='division3'>
</details>
<br><br><br>



`Creating method2 based on row`
```python
import pandas as pd

df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United Kingdom"],
                   [2872086, "Italy"],
                   [2273305, "France"]])
df.index = ["Stockholm", "London", "Rome", "Paris"]
df.rename(columns={0:"Population", 1:"State"}, inplace=True)
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
           Population           State
Stockholm      909976          Sweden
London        8615246  United Kingdom
Rome          2872086           Italy
Paris         2273305          France
```
<hr class='division3'>
</details>
<br><br><br>



`Creating method3 based on row, all at once`
```python
import pandas as pd

df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United Kingdom"],
                   [2872086, "Italy"],
                   [2273305, "France"]],
                 index=["Stockholm", "London", "Rome", "Paris"],
                 columns=["Population", "State"])
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
           Population           State
Stockholm      909976          Sweden
London        8615246  United Kingdom
Rome          2872086           Italy
Paris         2273305          France
```
<hr class='division3'>
</details>
<br><br><br>




`Creating method1 based on columns, all at once`
```python
import pandas as pd

df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
                   "State": ["Sweden", "United Kingdom", "Italy", "France"]},
                  index=["Stockholm", "London", "Rome", "Paris"])
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
           Population           State
Stockholm      909976          Sweden
London        8615246  United Kingdom
Rome          2872086           Italy
Paris         2273305          France
```
<hr class='division3'>
</details>
<br>

```python
df.index
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching all index
```
Index(['Stockholm', 'London', 'Rome', 'Paris'], dtype='object', name='index')
```
<hr class='division3'>
</details>
<br>

```python
df.columns
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching all columns
```
Index(['Population', 'State'], dtype='object')
```
<hr class='division3'>
</details>
<br>

searching row or values of row
```python
df.loc["Stockholm"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching single row
```
Population    909976
State         Sweden
Name: Stockholm, dtype: object
```
<br>
`SUPPLEMENT`
```python
type(df.loc["Stockholm"])
```
OUTPUT : <class 'pandas.core.series.Series'>
<hr class='division3'>
</details>
<br>

```python
df.loc[["Paris","Rome"]]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method1 multi-rows
```
       Population   State                  
Paris     2273305  France
Rome      2872086   Italy
```
<br>
`SUPPLEMENT`
```python
type(df.loc[["Paris","Rome"]])
```
OUTPUT : <class 'pandas.core.frame.DataFrame'>
<hr class='division3'>
</details>
<br>

```python
df[2:4]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method2 multi-rows
```
       Population   State                
Rome      2872086   Italy
Paris     2273305  France
```
<br>
`SUPPLEMENT`
```python
type(df[2:4])
```
OUTPUT : <class 'pandas.core.frame.DataFrame'>
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm","Population"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method1 single value
```
909976
```
<br>
`SUPPLEMENT`
```python
type(df.loc["Stockholm","Population"])
```
OUTPUT : <class 'numpy.int64'>
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"][0]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method2 single value
```
909976
```
<br>
`SUPPLEMENT`
```python
type(df.loc["Stockholm"][0])
```
OUTPUT : <class 'numpy.int64'>
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"]["Population"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method3 single value
```
909976
```
<br>
`SUPPLEMENT`
```python
type(df.loc["Stockholm"]["Population"])
```
OUTPUT : <class 'numpy.int64'>
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"].Population
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching method4 single value
```
909976
```
<br>
`SUPPLEMENT`
```python
type(df.loc["Stockholm"].Population)
```
OUTPUT :  <class 'numpy.int64'>
<hr class='division3'>
</details>
<br>

```python
df.loc[["Paris","Rome"],"Population"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
searching multi-values
```
Paris    2273305
Rome     2872086
Name: Population, dtype: int64
```
<br>
`SUPPLEMENT`
```python
type(df.loc[["Paris","Rome"],"Population"])
```
OUTPUT : <class 'pandas.core.series.Series'>
<hr class='division3'>
</details>







<br><br><br>





`Creating method2 based on columns`
```python
import pandas as pd

df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
                   "State": ["Sweden", "United Kingdom", "Italy", "France"],
                   "index": ["Stockholm", "London", "Rome", "Paris"]})
df = df.set_index("index")
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
           Population           State
index                                
Stockholm      909976          Sweden
London        8615246  United Kingdom
Rome          2872086           Italy
Paris         2273305          France
```
<hr class='division3'>
</details>
<br><br><br>





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


