---
layout : post
title : MATH05, Time series
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a> | <a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/MATH05/2019-08-13-MATH05-Time-series.md" target="_blank">page management</a><br>
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

## **pandas basic about time series**
```python
import pandas as pd
import numpy as np

t_series01 = pd.date_range("2015-1-1", periods=31)
T_series01 = pd.Series(np.arange(31), index=t_series01)
print(t_series01)
print(T_series01)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
DatetimeIndex(['2015-01-01', '2015-01-02', '2015-01-03', '2015-01-04',
               '2015-01-05', '2015-01-06', '2015-01-07', '2015-01-08',
               '2015-01-09', '2015-01-10', '2015-01-11', '2015-01-12',
               '2015-01-13', '2015-01-14', '2015-01-15', '2015-01-16',
               '2015-01-17', '2015-01-18', '2015-01-19', '2015-01-20',
               '2015-01-21', '2015-01-22', '2015-01-23', '2015-01-24',
               '2015-01-25', '2015-01-26', '2015-01-27', '2015-01-28',
               '2015-01-29', '2015-01-30', '2015-01-31'],
              dtype='datetime64[ns]', freq='D')
2015-01-01     0
2015-01-02     1
2015-01-03     2
2015-01-04     3
2015-01-05     4
2015-01-06     5
2015-01-07     6
2015-01-08     7
2015-01-09     8
2015-01-10     9
2015-01-11    10
2015-01-12    11
2015-01-13    12
2015-01-14    13
2015-01-15    14
2015-01-16    15
2015-01-17    16
2015-01-18    17
2015-01-19    18
2015-01-20    19
2015-01-21    20
2015-01-22    21
2015-01-23    22
2015-01-24    23
2015-01-25    24
2015-01-26    25
2015-01-27    26
2015-01-28    27
2015-01-29    28
2015-01-30    29
2015-01-31    30
Freq: D, dtype: int32
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
# timestamp object
print(T_series01.index[2])
print(T_series01.index[2].year,
      T_series01.index[2].month,
      T_series01.index[2].day,
      T_series01.index[2].nanosecond)

# datetime object
print(T_series01.index[2].to_pydatetime())
```
```
# timestamp object
2015-01-03 00:00:00
2015 1 3 0

# datetime object
2015-01-03 00:00:00
```
<br><br><br>
<span class="frame3">datetime object</span>
```python
import pandas as pd
import numpy as np
import datetime

T_series = pd.Series(np.random.rand(2),
                     index=[datetime.datetime(2015, 1, 1), datetime.datetime(2015, 2, 1)])
print(T_series)
```
```
2015-01-01    0.972084
2015-02-01    0.301809
dtype: float64
```
<hr class='division3'>
</details><br>



```python
import pandas as pd
import numpy as np

t_series02 = pd.date_range("2015-1-1 00:00", "2015-1-1 12:00", freq="H")
T_series02 = pd.Series(np.arange(13), index=t_series02)
print(t_series02)
print(T_series02)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
               '2015-01-01 02:00:00', '2015-01-01 03:00:00',
               '2015-01-01 04:00:00', '2015-01-01 05:00:00',
               '2015-01-01 06:00:00', '2015-01-01 07:00:00',
               '2015-01-01 08:00:00', '2015-01-01 09:00:00',
               '2015-01-01 10:00:00', '2015-01-01 11:00:00',
               '2015-01-01 12:00:00'],
              dtype='datetime64[ns]', freq='H')
2015-01-01 00:00:00     0
2015-01-01 01:00:00     1
2015-01-01 02:00:00     2
2015-01-01 03:00:00     3
2015-01-01 04:00:00     4
2015-01-01 05:00:00     5
2015-01-01 06:00:00     6
2015-01-01 07:00:00     7
2015-01-01 08:00:00     8
2015-01-01 09:00:00     9
2015-01-01 10:00:00    10
2015-01-01 11:00:00    11
2015-01-01 12:00:00    12
Freq: H, dtype: int32
```
<hr class='division3'>
</details>
<br>

```python
import pandas as pd
import numpy as np

t_series03 = pd.PeriodIndex([pd.Period('2015-01'), pd.Period('2015-02'), pd.Period('2015-03')])
T_series03 = pd.Series(np.random.rand(3), index=t_series03)
print(T_series03)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
2015-01    0.075913
2015-02    0.550537
2015-03    0.971680
Freq: M, dtype: float64
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
# PeriodIndex object
print(ts2.to_period('M'))
```
```
2015-01 0.683801
2015-02 0.916209
Freq: M, dtype: float64
```
<hr class='division3'>
</details>

<br>

<br><br><br>
<hr class="division2">

## title2

<hr class="division2">

## title3

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/9987e98ec60946c79a8a7f37cb7ae9cc/' target="_blank">Time series</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

