---
layout : post
title : MATH05, Analysis of variance
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a><br>
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

## **One-way analysis of variance**
![ANOVA_oneway](https://user-images.githubusercontent.com/52376448/66703411-507aca80-ed4d-11e9-8ae9-db644dd8b89f.JPG)
```python
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_regression

X0, y, coef = make_regression(n_samples = 100, n_features=1, noise=30, coef=True, random_state=0)
dfX0 = pd.DataFrame(X0, columns=["X"])
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(y, columns=["Y"])
df = pd.concat([dfX, dfy], axis=1)

model = sm.OLS.from_formula("Y ~ X", data=df)
result = model.fit()

print("TSS = ", result.uncentered_tss)
print("ESS = ", result.mse_model)
print("RSS = ", result.ssr)
print("ESS + RSS = ", result.mse_model + result.ssr)
print("R squared = ", result.rsquared)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
TSS =  291345.7578983061
ESS =  188589.61349210917
RSS =  102754.33755137534
ESS + RSS =  291343.9510434845
R squared =  0.6473091780922585
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.distplot(y,
             kde=False, fit=stats.norm, hist_kws={"color": "r", "alpha": 0.2}, fit_kws={"color": "r"},
             label="TSS")
sns.distplot(result.fittedvalues,
             kde=False, hist_kws={"color": "g", "alpha": 0.2}, fit=stats.norm, fit_kws={"color": "g"},
             label="ESS")
sns.distplot(result.resid,
             kde=False, hist_kws={"color": "b", "alpha": 0.2}, fit=stats.norm, fit_kws={"color": "b"},
             label="RSS")
plt.legend()
plt.show()
```
![download](https://user-images.githubusercontent.com/52376448/67594401-37d3d100-f79f-11e9-8052-d02248ab7dcb.png)
<hr class='division3'>
</details>
<br><br><br>

### ***Coefficient of Determination(R2)***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***The hypothesis test***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Regression F-test and ANOVA Relationship***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Coefficient of Determination(R2) and Correlation Coefficient***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Model without constant***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Comparing Model Using F Test***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Comparison of importance of variables using F test***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Adjusted Coefficient of Determination(Adjusted R2)***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

### ***Information Criterion***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<br><br><br>

---

<hr class="division2">










## **Two-way analysis of variance**

### ***Two-way ANOVA with repetition***
![ANOVA_twoway_with_repetition](https://user-images.githubusercontent.com/52376448/66703412-507aca80-ed4d-11e9-9dcb-66728cfe44ae.JPG)

<br><br><br>

---

### ***Two-way ANOVA without repetition***
![ANOVA_twoway_without_repetition](https://user-images.githubusercontent.com/52376448/66703413-507aca80-ed4d-11e9-8801-e7f73c9e9fee.JPG)

<br><br><br>
<hr class="division2">

## **Multivariate analysis of variance (MANOVA)**

<br><br><br>
<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/a60e97ad90164e07ad236095ca74e657/' target="_blank">분산 분석과 모형 성능</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

