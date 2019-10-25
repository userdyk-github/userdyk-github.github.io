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
<summary class='jb-small' style="color:blue">OUTPUT 1</summary>
<hr class='division3'>
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/44a8b57e2a4335f02faa2bd5003d94979af4f408" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:12.7ex; height:6.843ex;" alt="{\bar {y}}={\frac {1}{n}}\sum _{i=1}^{n}y_{i}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/aec2d91094ee54fbf0f7912d329706ff016ec1bd" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.303ex; height:5.509ex;" alt="SS_{\text{tot}}=\sum _{i}(y_{i}-{\bar {y}})^{2},">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/107a9fb71364b9db3cf481e956ad2af11cba10a1" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.398ex; height:5.509ex;" alt="SS_{\text{reg}}=\sum _{i}(f_{i}-{\bar {y}})^{2},">  
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2669c9340581d55b274d3b8ea67a7deb2225510b" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:30.579ex; height:5.509ex;" alt="{\displaystyle SS_{\text{res}}=\sum _{i}(y_{i}-f_{i})^{2}=\sum _{i}e_{i}^{2}\,}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0ab5cc13b206a34cc713e153b192f93b685fa875" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.338ex; width:16.401ex; height:5.843ex;" alt="{\displaystyle R^{2}\equiv 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}\,}">
</div>
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
<summary class='jb-small' style="color:blue">OUTPUT 2 : Regression F-test and ANOVA Relationship</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/67594520-79fd1280-f79f-11e9-8a55-6f8fede7c13a.JPG)

```python
sm.stats.anova_lm(result)
```
```
	        df	  sum_sq	      mean_sq	      F	          PR(>F)
X	        1.0	  188589.613492	188589.613492	179.863766	6.601482e-24
Residual	98.0	102754.337551	1048.513648	  NaN	        NaN
```
<br>
```python
print(result.summary())
```
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.647
Model:                            OLS   Adj. R-squared:                  0.644
Method:                 Least Squares   F-statistic:                     179.9
Date:                Fri, 25 Oct 2019   Prob (F-statistic):           6.60e-24
Time:                        19:14:57   Log-Likelihood:                -488.64
No. Observations:                 100   AIC:                             981.3
Df Residuals:                      98   BIC:                             986.5
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -2.4425      3.244     -0.753      0.453      -8.880       3.995
X             43.0873      3.213     13.411      0.000      36.712      49.463
==============================================================================
Omnibus:                        3.523   Durbin-Watson:                   1.984
Prob(Omnibus):                  0.172   Jarque-Bera (JB):                2.059
Skew:                          -0.073   Prob(JB):                        0.357
Kurtosis:                       2.312   Cond. No.                         1.06
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
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

### ***Coefficient of Determination(R2) and Correlation Coefficient***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/44a8b57e2a4335f02faa2bd5003d94979af4f408" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:12.7ex; height:6.843ex;" alt="{\bar {y}}={\frac {1}{n}}\sum _{i=1}^{n}y_{i}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/aec2d91094ee54fbf0f7912d329706ff016ec1bd" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.303ex; height:5.509ex;" alt="SS_{\text{tot}}=\sum _{i}(y_{i}-{\bar {y}})^{2},">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/107a9fb71364b9db3cf481e956ad2af11cba10a1" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.398ex; height:5.509ex;" alt="SS_{\text{reg}}=\sum _{i}(f_{i}-{\bar {y}})^{2},">  
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2669c9340581d55b274d3b8ea67a7deb2225510b" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:30.579ex; height:5.509ex;" alt="{\displaystyle SS_{\text{res}}=\sum _{i}(y_{i}-f_{i})^{2}=\sum _{i}e_{i}^{2}\,}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0ab5cc13b206a34cc713e153b192f93b685fa875" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.338ex; width:16.401ex; height:5.843ex;" alt="{\displaystyle R^{2}\equiv 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}\,}">  
</div>
<div class="frame1">

</div>
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

