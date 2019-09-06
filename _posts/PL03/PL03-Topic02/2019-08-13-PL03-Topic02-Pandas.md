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

#### Arrangement
`STEP1`
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
`STEP2`
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
`STEP3`
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
`STEP4`
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
`Based on rank`
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
`Based on Population`
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
`Based on State`
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
`Based on Alphabet`
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
df = df.set_index(["Alphabet","rank"]).sort_index()
```
```python
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

#### Deleting

`drop`
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
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
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
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
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
1  4  5   6   7
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
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
```
<hr class='division3'>
</details>
<br><br><br>


`drop_duplicates`
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







---

#### Concatenating

```python
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



---

#### Merging

---

#### Drop_duplicates


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


