---
layout : post
title : MATH05, Regression analysis
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)  ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a><br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Python Libraries</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Ordinary Least Squares**

```python
import statsmodels.api as sm

X = range(1,8)
X = sm.add_constant(X)
Y = [1,3,4,5,2,3,4]

model = sm.OLS(Y,X)
results = model.fit()
results.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT 1</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/67741410-5e726000-fa5c-11e9-8c65-a95a041fe96c.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT 2</summary>
<hr class='division3'>
<a href="https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
" target="_blank">RegressionResults</a>
<br><br><br>
```python
results.tvalues
```
```
array([ 1.87867287,  0.98019606])
```
<br>
```python
results.t_test([1, 0])
```
```
<class 'statsmodels.stats.contrast.ContrastResults'>
                             Test for Constraints                             
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
c0             2.1429      1.141      1.879      0.119      -0.789       5.075
==============================================================================
```
<br>
```python
import numpy as np

results.f_test(np.identity(2))
```
```
<class 'statsmodels.stats.contrast.ContrastResults'>
<F test: F=array([[19.46078431]]), p=0.004372505910945688, df_denom=5, df_num=2>
```
<hr class='division3'>
</details>



<br><br><br>
<hr class="division2">

## title2

<hr class="division2">

## title3

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/9987e98ec60946c79a8a7f37cb7ae9cc/' target="_blank">Regression analysis</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

