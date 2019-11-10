---
layout : post
title : MATH05, Analysis of variance
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a><br>
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

## **One-way analysis of variance**

![image](https://user-images.githubusercontent.com/52376448/68536347-b0887f00-0394-11ea-8889-7f91b1ed1aeb.png)

![ANOVA_oneway](https://user-images.githubusercontent.com/52376448/66703411-507aca80-ed4d-11e9-8ae9-db644dd8b89f.JPG)


<span class="frame3">Load Dataset</span><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">Dataset description</summary>
<hr class='division3'>
<b>Get and sort sample data</b><br>

Twenty-two patients undergoing cardiac bypass surgery were randomized to one of three ventilation groups:

- Group I: Patients received 50% nitrous oxide and 50% oxygen mixture continuously for 24 h.
- Group II: Patients received a 50% nitrous oxide and 50% oxygen mixture only dirng the operation.
- Group III: Patients received no nitrous oxide but received 35-50% oxygen for 24 h.

The data show red cell folate levels for the three groups after 24h' ventilation.

<hr class='division3'>
</details>

```python
import urllib

# Get the data
inFile = 'altman_910.txt'
url_base = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/'

url = url_base + inFile
data = genfromtxt(urllib.request.urlopen(url), delimiter=',')

# Sort them into groups, according to column 1
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Data</summary>
<hr class='division3'>
```python
data
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
array([[243.,   1.],
       [251.,   1.],
       [275.,   1.],
       [291.,   1.],
       [347.,   1.],
       [354.,   1.],
       [380.,   1.],
       [392.,   1.],
       [206.,   2.],
       [210.,   2.],
       [226.,   2.],
       [249.,   2.],
       [255.,   2.],
       [273.,   2.],
       [285.,   2.],
       [295.,   2.],
       [309.,   2.],
       [241.,   3.],
       [258.,   3.],
       [270.,   3.],
       [293.,   3.],
       [328.,   3.]])
```
<hr class='division3_1'>
</details>

<br>
```python
group1
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
array([243., 251., 275., 291., 347., 354., 380., 392.])
```
<hr class='division3_1'>
</details>

<br>
```python
group2
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
array([206., 210., 226., 249., 255., 273., 285., 295., 309.])
```
<hr class='division3_1'>
</details>

<br>
```python
group3
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
array([241., 258., 270., 293., 328.])
```
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>
<br>

<span class="frame3">Levene test for equal-variance</span>
```python
# check if the variances are equal with the "Levene"-test
from scipy import stats

(W,p) = stats.levene(group1, group2, group3)
if p<0.05:
    
    print('Warning: the p-value of the Levene test is <0.05: p={0}'.format(p))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Warning: the p-value of the Levene test is <0.05: p=0.045846812634186246
```
<hr class='division3'>
</details>
<br>

<span class="frame3">One-way ANOVA with scipy</span>
```python
F_statistic, pVal = stats.f_oneway(group1, group2, group3)

print('The results from the one-way ANOVA, with the data from Altman 910: F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('One of the groups is significantly different.')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
The results from the one-way ANOVA, with the data from Altman 910: F=3.7, p=0.04359
One of the groups is significantly different.
```
<hr class='division3'>
</details>
<br>

<span class="frame3">One-way ANOVA with statsmodel</span>
```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# move dataset to dataframe of pandas
df = pd.DataFrame(data, columns=['value', 'treatment'])    

# the "C" indicates categorical data
model = ols('value ~ C(treatment)', df).fit()

print(anova_lm(model))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
                df        sum_sq      mean_sq         F    PR(>F)
C(treatment)   2.0  15515.766414  7757.883207  3.711336  0.043589
Residual      19.0  39716.097222  2090.320906       NaN       NaN
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Tukey’s multi-comparison method, Holm-Bonferroni Method***
<a href="https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/" target="_blank">URL</a>

<span class="frame3">Setting up the data, and running an ANOVA</span><br>
```python
import numpy as np
import scipy.stats as stats

# Create four random groups of data with a mean difference of 1

mu, sigma = 10, 3 # mean and standard deviation
group1 = np.random.normal(mu, sigma, 50)

mu, sigma = 11, 3 # mean and standard deviation
group2 = np.random.normal(mu, sigma, 50)

mu, sigma = 12, 3 # mean and standard deviation
group3 = np.random.normal(mu, sigma, 50)

mu, sigma = 13, 3 # mean and standard deviation
group4 = np.random.normal(mu, sigma, 50)

# Show the results for Anova

F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4)

print ('P value:')
print (pVal)
```
```
P value:
1.6462001201818463e-08
```
```python
# Put into dataframe

df = pd.DataFrame()
df['treatment1'] = group1
df['treatment2'] = group2
df['treatment3'] = group3
df['treatment4'] = group4

# Stack the data (and rename columns):

stacked_data = df.stack().reset_index()
stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})
# Show the first 8 rows:

print (stacked_data.head(8))
```
```
   id   treatment     result
0   0  treatment1  12.980445
1   0  treatment2   8.444603
2   0  treatment3  10.713692
3   0  treatment4  10.777762
4   1  treatment1  14.350560
5   1  treatment2   9.436072
6   1  treatment3  12.715509
7   1  treatment4  15.016419

```
<span class="frame3">Tukey’s multi-comparison method</span><br>
```python
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# Set up the data for comparison (creates a specialised object)
MultiComp = MultiComparison(stacked_data['result'],
                            stacked_data['treatment'])

# Show all pair-wise comparisons:

# Print the comparisons

print(MultiComp.tukeyhsd().summary())
```
```
 Multiple Comparison of Means - Tukey HSD,FWER=0.05 
====================================================
  group1     group2   meandiff  lower  upper  reject
----------------------------------------------------
treatment1 treatment2  1.5021  -0.0392 3.0435 False 
treatment1 treatment3   1.47   -0.0714 3.0113 False 
treatment1 treatment4  3.8572   2.3159 5.3985  True 
treatment2 treatment3 -0.0322  -1.5735 1.5091 False 
treatment2 treatment4  2.355    0.8137 3.8963  True 
treatment3 treatment4  2.3872   0.8459 3.9285  True 
```
<span class="frame3">Holm-Bonferroni Method</span><br>
```python
comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
print (comp[0])
```
```
Test Multiple Comparison ttest_rel 
FWER=0.05 method=Holm
alphacSidak=0.01, alphacBonf=0.008
=====================================================
  group1     group2     stat   pval  pval_corr reject
-----------------------------------------------------
treatment1 treatment2 -2.1234 0.0388   0.0776  False 
treatment1 treatment3 -2.4304 0.0188   0.0564  False 
treatment1 treatment4 -6.4443  0.0      0.0     True 
treatment2 treatment3  0.0457 0.9637   0.9637  False 
treatment2 treatment4 -3.7878 0.0004   0.0017   True 
treatment3 treatment4 -5.0246  0.0      0.0     True 
```
<br><br><br>




---


### ***One-way ANOVA for regression analysis, model with constant***

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

print("TSS = ", result.uncentered_tss)       # TSS: total sum of square
print("ESS = ", result.mse_model)            # ESS: explained sum of squares
print("RSS = ", result.ssr)                  # RSS: residual sum of squares
print("ESS + RSS = ", result.mse_model + result.ssr)
print("R squared = ", result.rsquared)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT 1 : ANOVA table for fixed model, single factor</summary>
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
<summary class='jb-small' style="color:blue">Visualization for OUTPUT 1</summary>
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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT 2 : Regression F-test and ANOVA Relationship</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/67594520-79fd1280-f79f-11e9-8a55-6f8fede7c13a.JPG)

```python
sm.stats.anova_lm(result)
```
```
	        df	  sum_sq	      mean_sq	      F	                PR(>F)
X	        1.0	  188589.613492	      188589.613492   179.863766	6.601482e-24
Residual	98.0	  102754.337551	      1048.513648     NaN	        NaN
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
<summary class='jb-small' style="color:blue">Visualization : Coefficient of Determination(R2) and Correlation Coefficient</summary>
<hr class='division3'>
<a href="http://mathworld.wolfram.com/CorrelationCoefficient.html" target="_blank">mathworld : correlation coefficient</a><br><br>
<span class="frame3">Coefficient of Determination(R2)</span>
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/44a8b57e2a4335f02faa2bd5003d94979af4f408" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:12.7ex; height:6.843ex;" alt="{\bar {y}}={\frac {1}{n}}\sum _{i=1}^{n}y_{i}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/aec2d91094ee54fbf0f7912d329706ff016ec1bd" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.303ex; height:5.509ex;" alt="SS_{\text{tot}}=\sum _{i}(y_{i}-{\bar {y}})^{2},">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/107a9fb71364b9db3cf481e956ad2af11cba10a1" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:21.398ex; height:5.509ex;" alt="SS_{\text{reg}}=\sum _{i}(f_{i}-{\bar {y}})^{2},">  
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2669c9340581d55b274d3b8ea67a7deb2225510b" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:30.579ex; height:5.509ex;" alt="{\displaystyle SS_{\text{res}}=\sum _{i}(y_{i}-f_{i})^{2}=\sum _{i}e_{i}^{2}\,}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0ab5cc13b206a34cc713e153b192f93b685fa875" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.338ex; width:16.401ex; height:5.843ex;" alt="{\displaystyle R^{2}\equiv 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}\,}">  
</div>
<br>

<span class="frame3">Correlation Coefficient</span>
<div class="frame1">
For a Population,
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/93185aed3047ef42fa0f1b6e389a4e89a5654afa" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.171ex; width:57.998ex; height:6.009ex;" alt="{\displaystyle \rho _{X,Y}=\operatorname {corr} (X,Y)={\operatorname {cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={\operatorname {E} [(X-\mu _{X})(Y-\mu _{Y})] \over \sigma _{X}\sigma _{Y}}}">
For a Sample,
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/332ae9dcde34d03f30ed6e1880af8b43327dd49c" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -7.338ex; width:59.755ex; height:14.343ex;" alt="{\displaystyle r_{xy}\quad {\overset {\underset {\mathrm {def} }{}}{=}}\quad {\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{(n-1)s_{x}s_{y}}}={\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sqrt {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})^{2}\sum \limits _{i=1}^{n}(y_{i}-{\bar {y}})^{2}}}},}">
</div>
<br>

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.jointplot(result.fittedvalues, y)
plt.show()
```
![download (1)](https://user-images.githubusercontent.com/52376448/67597855-1d055a80-f7a7-11e9-9411-c0235633160b.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***One-way ANOVA for regression analysis, model without constant***

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_regression

X0, y, coef = make_regression(
    n_samples=100, n_features=1, noise=30, bias=100, coef=True, random_state=0)
dfX = pd.DataFrame(X0, columns=["X"])
dfy = pd.DataFrame(y, columns=["Y"])
df = pd.concat([dfX, dfy], axis=1)

model2 = sm.OLS.from_formula("Y ~ X + 0", data=df)
result2 = model2.fit()
result2.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/67632432-88a21180-f8e6-11e9-8c56-31ceadba9821.JPG)
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

