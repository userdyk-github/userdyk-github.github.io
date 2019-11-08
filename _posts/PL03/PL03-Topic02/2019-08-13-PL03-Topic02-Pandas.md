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

## **Installation**
### ***For linux***
```bash
$ 
```
<br><br><br>

### ***For windows***
```dos

```
<br><br><br>


### ***Version Control***
```python

```
<br><br><br>


<hr class="division2">

## **Series**

### ***One-column***

#### Creating and searching
<span class="frame3">METHOD 1, Creating Series : sperately</span>
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
<span class="frame3">METHOD2, Creating Series : all at once</span>
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

#### Creating

<span class="frame3">Concatenate</span>

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

<span class="frame3">get_dummies</span>
```python
import pandas as pd

s1 = pd.Series(list('abca'))
s2 = pd.get_dummies(s1)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
s1
```
```
0    a
1    b
2    c
3    a
dtype: object
```
<br>
```python
s2
```
```
	a	b	c
0	1	0	0
1	0	1	0
2	0	0	1
3	1	0	0
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **DataFrame**

### ***One-dataframe***

#### Creating and searching
<span class="frame3">METHOD 1, Creating frame based on row</span>
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


<span class="frame3">METHOD 2, Creating frame based on row</span>
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


<span class="frame3">METHOD 3, Creating frame based on row, all at once</span>
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



<span class="frame3">METHOD 1, Creating frame based on columns, all at once</span>
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
df.index.difference(['Stockholm'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Index(['London', 'Paris', 'Rome'], dtype='object')
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

```python
df.columns.difference(['Population'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Index(['State'], dtype='object')
```
<hr class='division3'>
</details>

<br>

**Searching row or values of row**
```python
df.loc["Stockholm"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method searching single row
```
Population    909976
State         Sweden
Name: Stockholm, dtype: object
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc["Stockholm"])
OUTPUT : <class 'pandas.core.series.Series'>
```
<hr class='division3'>
</details>
<br>

```python
df.loc[["Paris","Rome"]]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 searching multi-rows
```
       Population   State                  
Paris     2273305  France
Rome      2872086   Italy
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc[["Paris","Rome"]])
OUTPUT : <class 'pandas.core.frame.DataFrame'>
```
<hr class='division3'>
</details>
<br>

```python
df[2:4]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 searching multi-rows
```
       Population   State                
Rome      2872086   Italy
Paris     2273305  France
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df[2:4])
OUTPUT : <class 'pandas.core.frame.DataFrame'>
```
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm","Population"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc["Stockholm","Population"])
OUTPUT : <class 'numpy.int64'>
```
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"][0]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc["Stockholm"][0])
OUTPUT : <class 'numpy.int64'>
```
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"]["Population"]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method3 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc["Stockholm"]["Population"])
OUTPUT : <class 'numpy.int64'>
```
<hr class='division3'>
</details>
<br>

```python
df.loc["Stockholm"].Population
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method4 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc["Stockholm"].Population)
OUTPUT :  <class 'numpy.int64'>
```
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
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.loc[["Paris","Rome"],"Population"])
OUTPUT : <class 'pandas.core.series.Series'>
```
<hr class='division3'>
</details>











<br>

**Searching columns or values of columns**
```python
df['Population']
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 searching single column
```
Stockholm     909976
London       8615246
Rome         2872086
Paris        2273305
Name: Population, dtype: int64
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df['Population'])
OUTPUT : <class 'pandas.core.series.Series'>
```
<hr class='division3'>
</details>
<br>

```python
df.Population
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 searching single column
```
Stockholm     909976
London       8615246
Rome         2872086
Paris        2273305
Name: Population, dtype: int64
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
**Data-type**
```
 INPUT : type(df.Population)
OUTPUT : <class 'pandas.core.series.Series'>
```
<hr class='division3'>
</details>
<br>

```python
df['Population'][0]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method1 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
<hr class='division3'>
</details>
<br>

```python
df.Population[0]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method2 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
<hr class='division3'>
</details>
<br>

```python
df['Population']['Stockholm']
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method3 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
<hr class='division3'>
</details>
<br>

```python
df['Population'].Stockholm
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
method4 searching single value
```
909976
```
<br>
<br>
**Original dataset**

|             |Population |          State|
|:------------|:----------|:--------------|
|Stockholm    | 909976    |        Sweden |
|London       | 8615246   |United Kingdom |
|Rome         | 2872086   |         Italy |
|Paris        | 2273305   |        France |

<br>
<hr class='division3'>
</details>





<br><br><br>




<span class="frame3">METHOD 2, Creating frame based on columns</span>
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
<br>
```python
df.iloc[0,:]
```
```
Population    909976
State         Sweden
Name: Stockholm, dtype: object
```
<br>
```python
df.iloc[:,0]
```
```
index
Stockholm     909976
London       8615246
Rome         2872086
Paris        2273305
Name: Population, dtype: int64
```
<br><br><br>


---

#### Arrangement
<span class="frame3">STEP1</span>
```python
import pandas as pd

# creating dataset
df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305,123234,123444,23333,343434],
                   "State": ["Sweden", "United Kingdom", "Italy","Seoul","Suwon", "France","Korea", "Japan"],
                   "Alphabet" : ["a","b","x","d","a","a","b","c"],
                   "rank" : [1,2,3,4,6,5,7,8]})
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   Population           State Alphabet  rank
0      909976          Sweden        a     1
1     8615246  United Kingdom        b     2
2     2872086           Italy        x     3
3     2273305           Seoul        d     4
4      123234           Suwon        a     6
5      123444          France        a     5
6       23333           Korea        b     7
7      343434           Japan        c     8
```
<hr class='division3'>
</details>
<br>
<span class="frame3">STEP2</span>
```python
df = df.sort_index(axis=1)
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
  Alphabet  Population           State  rank
0        a      909976          Sweden     1
1        b     8615246  United Kingdom     2
2        x     2872086           Italy     3
3        d     2273305           Seoul     4
4        a      123234           Suwon     6
5        a      123444          France     5
6        b       23333           Korea     7
7        c      343434           Japan     8
```
<hr class='division3'>
</details>
<br>
<span class="frame3">STEP3</span>
```python
df = df.set_index(['Alphabet','rank'])
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
a        1         909976          Sweden
b        2        8615246  United Kingdom
x        3        2872086           Italy
d        4        2273305           Seoul
a        6         123234           Suwon
         5         123444          France
b        7          23333           Korea
c        8         343434           Japan
```
<hr class='division3'>
</details>
<br>
<span class="frame3">STEP4</span>
```python
df = df.sort_index()
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
a        1         909976          Sweden
         5         123444          France
         6         123234           Suwon
b        2        8615246  United Kingdom
         7          23333           Korea
c        8         343434           Japan
d        4        2273305           Seoul
x        3        2872086           Italy
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Based on rank</span>
```python
df.sort_values("rank", ascending=False)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
c        8         343434           Japan
b        7          23333           Korea
a        6         123234           Suwon
         5         123444          France
d        4        2273305           Seoul
x        3        2872086           Italy
b        2        8615246  United Kingdom
a        1         909976          Sweden
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Based on Population</span>
```python
df.sort_values("Population", ascending=False)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
b        2        8615246  United Kingdom
x        3        2872086           Italy
d        4        2273305           Seoul
a        1         909976          Sweden
c        8         343434           Japan
a        5         123444          France
         6         123234           Suwon
b        7          23333           Korea
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Based on State</span>
```python
df.sort_values("State", ascending=False)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
b        2        8615246  United Kingdom
a        1         909976          Sweden
         6         123234           Suwon
d        4        2273305           Seoul
b        7          23333           Korea
c        8         343434           Japan
x        3        2872086           Italy
a        5         123444          France
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Based on Alphabet</span>
```python
df = df.sort_values("Alphabet", ascending=False)
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
x        3        2872086           Italy
d        4        2273305           Seoul
c        8         343434           Japan
b        2        8615246  United Kingdom
         7          23333           Korea
a        1         909976          Sweden
         6         123234           Suwon
         5         123444          France
```
<hr class='division3'>
</details>

<br><br><br>

---


#### Deleting
<span class="frame3">drop</span>
```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
```
<hr class='division3'>
</details>
<br>

```python
# single column drop
df.drop('A', axis=1)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   B   C   D
0  1   2   3
1  5   6   7
2  9  10  11
```
<hr class='division3'>
</details>
<br>

```python
# multi-columns drop
df.drop(['B', 'C'], axis=1)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   A   D
0  0   3
1  4   7
2  8  11
```
<hr class='division3'>
</details>
<br>

```python
# single row drop
df.drop(1, axis=0)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   A  B   C   D
0  0  1   2   3
2  8  9  10  11
```
<hr class='division3'>
</details>
<br>

```python
# multi-row drop
df.drop([1,2], axis=0)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   A  B  C  D
0  0  1  2  3
```
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">drop_duplicates</span>
```python
import pandas as pd

df = pd.DataFrame({"phone": [909976, 8615246, 2872086, 2273305,2273305,2273305,2273305]})
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
     phone
0   909976
1  8615246
2  2872086
3  2273305
4  2273305
5  2273305
6  2273305
```
<hr class='division3'>
</details>
<br>

```python
df.drop_duplicates('phone',keep='first')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
     phone
0   909976
1  8615246
2  2872086
3  2273305
```
<hr class='division3'>
</details>
<br>

```python
df.drop_duplicates('phone',keep='last')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
     phone
0   909976
1  8615246
2  2872086
6  2273305
```
<hr class='division3'>
</details>

<br><br><br>
<span class="frame3">dropna</span>
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'1C':[None, -0.738247,  0.598380,  0.727832],
                   '2C':[None,  0.073079,  1.182290, -0.138224],
                   '3C':[0.554677, -0.530208, -0.397182,  0.990026],
                   '4C':[-0.332384, -1.979684,  0.560655,  0.833487]})
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	1C		2C		3C		4C
0	NaN		NaN		0.554677	-0.332384
1	-0.738247	0.073079	-0.530208	-1.979684
2	0.598380	1.182290	-0.397182	0.560655
3	0.727832	-0.138224	0.990026	0.833487
```
<hr class='division3'>
</details>
<br>

```python
df.dropna()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	1C		2C		3C		4C
1	-0.738247	0.073079	-0.530208	-1.979684
2	0.598380	1.182290	-0.397182	0.560655
3	0.727832	-0.138224	0.990026	0.833487
```
<hr class='division3'>
</details>
<br>
```python
df.dropna(axis=0)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	1C		2C		3C		4C
1	-0.738247	0.073079	-0.530208	-1.979684
2	0.598380	1.182290	-0.397182	0.560655
3	0.727832	-0.138224	0.990026	0.833487
```
<hr class='division3'>
</details>
<br>
```python
df.dropna(axis=1)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	3C		4C
0	0.554677	-0.332384
1	-0.530208	-1.979684
2	-0.397182	0.560655
3	0.990026	0.833487
```
<hr class='division3'>
</details>




<br><br><br>


---



#### Analysis

```python
import pandas as pd
df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305,123234,123444,23333,343434],
                   "State": ["Sweden", "United Kingdom", "Italy","Seoul","Suwon", "France","Korea", "Japan"],
                   "Alphabet" : ["a","b","x","d","a","a","b","c"],
                   "rank" : [1,2,3,4,6,5,7,8]})
```
```python
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   Population           State Alphabet  rank
0      909976          Sweden        a     1
1     8615246  United Kingdom        b     2
2     2872086           Italy        x     3
3     2273305           Seoul        d     4
4      123234           Suwon        a     6
5      123444          France        a     5
6       23333           Korea        b     7
7      343434           Japan        c     8
```
<hr class='division3'>
</details>
<br>

```python
df.shape
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
(8, 4)
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
```
Index(['Population', 'State', 'Alphabet', 'rank'], dtype='object')
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
```
RangeIndex(start=0, stop=8, step=1)
```
<hr class='division3'>
</details>
<br>

```python
df['Alphabet'].unique()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array(['a', 'b', 'x', 'd', 'c'], dtype=object)
```
<hr class='division3'>
</details>
<br>

```python
df['Alphabet'].value_counts()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
a    3
b    2
c    1
x    1
d    1
Name: Alphabet, dtype: int64
```
<hr class='division3'>
</details>
<br><br><br>

---

#### Statistics

```python
import pandas as pd

df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305,123234,123444,23333,343434],
                   "State": ["Sweden", "United Kingdom", "Italy","Seoul","Suwon", "France","Korea", "Japan"],
                   "Alphabet" : ["a","b","x","d","a","a","b","c"],
                   "rank" : [1,2,3,4,6,5,7,8]})
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	Population	State	          Alphabet	rank
0	909976	  	Sweden	          a	        1
1	8615246	        United Kingdom	  b	        2
2	2872086   	Italy	          x       	3
3	2273305   	Seoul	          d       	4
4	123234	    	Suwon	          a       	6
5	123444	  	France	          a       	5
6	23333	    	Korea	          b       	7
7	343434	        Japan	          c       	8
```
<hr class='division3'>
</details>
```python
df.count(axis=0)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Population    8
State         8
Alphabet      8
rank          8
dtype: int64
```
<hr class='division3'>
</details>
```python
df.count(axis=1)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
0    4
1    4
2    4
3    4
4    4
5    4
6    4
7    4
dtype: int64
```
<hr class='division3'>
</details>
```python
df.corr()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
	          Population	rank
Population	 1.000000  	-0.573738
rank	      -0.573738    1.000000
```
<hr class='division3'>
</details>

```python
df = df.set_index(["Alphabet","rank"]).sort_index()
df
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
               Population           State
Alphabet rank                            
a        1         909976          Sweden
         5         123444          France
         6         123234           Suwon
b        2        8615246  United Kingdom
         7          23333           Korea
c        8         343434           Japan
d        4        2273305           Seoul
x        3        2872086           Italy
```
<hr class='division3'>
</details>
<br>
```python
df.loc['a'].std()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[454165.09584217647]
```
<hr class='division3'>
</details>

<br><br><br>






---

### ***Several dataframes***


#### Concatenating

```python
import pandas as pd

df1 = pd.DataFrame([[909976, "Sweden1"],
                    [8615246, "United Kingdom1"],
                    [2872086, "Italy1"],
                    [2273305, "France1"]],
                  index=["Stockholm1", "London1", "Rome1", "Paris1"],
                  columns=["Population1", "State1"])

df2 = pd.DataFrame([[909976, "Sweden2"],
                    [8615246, "United Kingdom2"],
                    [2872086, "Italy2"],
                    [2273305, "France2"]],
                  index=["Stockholm2", "London2", "Rome2", "Paris2"],
                  columns=["Population2", "State2"])
```
```python
df1, df2
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
            Population1           State1
Stockholm1       909976          Sweden1
London1         8615246  United Kingdom1
Rome1           2872086           Italy1
Paris1          2273305          France1 

             Population2           State2
Stockholm2       909976          Sweden2
London2         8615246  United Kingdom2
Rome2           2872086           Italy2
Paris2          2273305          France2
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Concat for left and right</span>
```python
df = pd.concat([df1, df2], axis=1)
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
            Population1           State1  Population2           State2
London1       8615246.0  United Kingdom1          NaN              NaN
London2             NaN              NaN    8615246.0  United Kingdom2
Paris1        2273305.0          France1          NaN              NaN
Paris2              NaN              NaN    2273305.0          France2
Rome1         2872086.0           Italy1          NaN              NaN
Rome2               NaN              NaN    2872086.0           Italy2
Stockholm1     909976.0          Sweden1          NaN              NaN
Stockholm2          NaN              NaN     909976.0          Sweden2
```
<hr class='division3'>
</details>
<br>

```python
df = pd.concat([df1, df2], axis=1, ignore_index=True)
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
                    0                1          2                3
London1     8615246.0  United Kingdom1        NaN              NaN
London2           NaN              NaN  8615246.0  United Kingdom2
Paris1      2273305.0          France1        NaN              NaN
Paris2            NaN              NaN  2273305.0          France2
Rome1       2872086.0           Italy1        NaN              NaN
Rome2             NaN              NaN  2872086.0           Italy2
Stockholm1   909976.0          Sweden1        NaN              NaN
Stockholm2        NaN              NaN   909976.0          Sweden2
```
<hr class='division3'>
</details>
<br>

```python
df = pd.concat([df1, df2], axis=1, keys=['C0', 'C1'])
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
                    C0                           C1                 
           Population1           State1 Population2           State2
London1      8615246.0  United Kingdom1         NaN              NaN
London2            NaN              NaN   8615246.0  United Kingdom2
Paris1       2273305.0          France1         NaN              NaN
Paris2             NaN              NaN   2273305.0          France2
Rome1        2872086.0           Italy1         NaN              NaN
Rome2              NaN              NaN   2872086.0           Italy2
Stockholm1    909976.0          Sweden1         NaN              NaN
Stockholm2         NaN              NaN    909976.0          Sweden2
```
<hr class='division3'>
</details>
<br>
<span class="frame3">Concat for up and down</span>
```python
df = pd.concat([df1, df2], axis=0)
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
            Population1  Population2           State1           State2
Stockholm1     909976.0          NaN          Sweden1              NaN
London1       8615246.0          NaN  United Kingdom1              NaN
Rome1         2872086.0          NaN           Italy1              NaN
Paris1        2273305.0          NaN          France1              NaN
Stockholm2          NaN     909976.0              NaN          Sweden2
London2             NaN    8615246.0              NaN  United Kingdom2
Rome2               NaN    2872086.0              NaN           Italy2
Paris2              NaN    2273305.0              NaN          France2
```
<hr class='division3'>
</details>
<br>

```python
df = pd.concat([df1, df2], axis=0, ignore_index=True)
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
   Population1  Population2           State1           State2
0     909976.0          NaN          Sweden1              NaN
1    8615246.0          NaN  United Kingdom1              NaN
2    2872086.0          NaN           Italy1              NaN
3    2273305.0          NaN          France1              NaN
4          NaN     909976.0              NaN          Sweden2
5          NaN    8615246.0              NaN  United Kingdom2
6          NaN    2872086.0              NaN           Italy2
7          NaN    2273305.0              NaN          France2
```
<hr class='division3'>
</details>
<br>

```python
df = pd.concat([df1, df2], axis=0, keys=['C0', 'C1'])
df
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<span class='jb-small'>United Kingdom = UK</span>
```
               Population1  Population2      State1      State2
C0 Stockholm1     909976.0          NaN     Sweden1         NaN
   London1       8615246.0          NaN         UK1         NaN
   Rome1         2872086.0          NaN      Italy1         NaN
   Paris1        2273305.0          NaN     France1         NaN
C1 Stockholm2          NaN     909976.0         NaN     Sweden2
   London2             NaN    8615246.0         NaN         UK2
   Rome2               NaN    2872086.0         NaN      Italy2
   Paris2              NaN    2273305.0         NaN     France2
```
<hr class='division3'>
</details>
<br><br><br>

---

#### Merging

<br><br><br>

---


#### Analysis

```python
import pandas as pd
idx1 = pd.Index([2, 1, 3, 4])
idx2 = pd.Index([3, 4, 5, 6])
```
```python
idx1.difference(idx2)
```
<span class='jb-medium'>Int64Index([1, 2], dtype='int64')</span>
```python
idx1.difference(idx2, sort=False)
```
<span class='jb-medium'>Int64Index([2, 1], dtype='int64')</span>


<br><br><br>

<hr class="division2">

## **Input/Output**
<span class="frame3">Input dataset</span>
```python
import pandas as pd

df = pd.read_csv(r'C:\Users\userd\Desktop\dataset\iris.csv')
df.head()
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/64994858-bc1b7480-d914-11e9-8c5b-e374bd2d6929.JPG)
<hr class='division3'>
</details>
<br>

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\userd\Desktop\dataset\iris.csv', index_col=0)
df.head()
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/64994804-91c9b700-d914-11e9-82b5-696f4ccb73ef.JPG)
<hr class='division3'>
</details>
<br>

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\userd\Desktop\dataset\iris.csv', index_col='sepal.length')
df.head()
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/64994804-91c9b700-d914-11e9-82b5-696f4ccb73ef.JPG)
<hr class='division3'>
</details>
<br>

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\userd\Desktop\dataset\iris.csv', index_col='variety')
df.head()
```
<details markdown='1'>
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/64994767-7bbbf680-d914-11e9-91b5-9a96982e652e.JPG)
<hr class='division3'>
</details>

<br><br><br>
<span class="frame3">Output dataset</span>
```python
import pandas as pd

df = pd.DataFrame([[4.78232104, 5.82145535],
                   [6.48127781, 6.33186404],
                   [4.63813463, 5.68560883]])

#saving in excel format
filepath = 'pixel_values.xlsx'
df.to_excel(filepath, index=False)  
```

<hr class="division2">

## **Covert Data-Type**

### ***DataFrame to Series***

```
>>> import pandas as pd

# based on column
>>> df = pd.DataFrame({'phone': [1001, 1002, 1003, 1004, 1005, 1006, 1007]})
>>> type(df)
pandas.core.frame.DataFrame

>>> df = df['phone']
>>> type(df)
pandas.core.series.Series


# based on row
>>> df = pd.DataFrame([[909976, 2872086, 8615246, 2872086]])
>>> type(df)
pandas.core.frame.DataFrame

>>> df = df.loc[0]
>>> type(df)
pandas.core.series.Series
```
<br><br><br>

---

### ***Series to DataFrame***

```
>>> import pandas as pd
>>> s = pd.Series([1,2,3,4,5])
>>> type(s)
pandas.core.series.Series

>>> s= pd.DataFrame(s)
>>> type(s)
pandas.core.frame.DataFrame
```
<br><br><br>

---

### ***DataFrame to numpy***

```
>>> import pandas as pd
>>> df = pd.DataFrame({
        'phone': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
        '이름': ['둘리', '도우너', '또치', '길동', '희동', '마이콜', '영희']})
>>> type(df)
pandas.core.frame.DataFrame

>>> df = df.values
>>> type(df)
numpy.ndarray
```
<br><br><br>

---

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/c5ccddd6716042ee8be3e5436081778b/' target="_blank">데이터 사이언스 스쿨</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---


[1]:{{ site.url }}/download/PL03/iris.csv
