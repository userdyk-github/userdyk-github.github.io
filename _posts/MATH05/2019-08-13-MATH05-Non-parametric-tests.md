---
layout : post
title : MATH05, Non parametric tests
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)  ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a><br>
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

## **Non parametric**

```python
from scipy import stats
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

X = stats.chi2(df=5) 
X_samples = X.rvs(100)

kde = stats.kde.gaussian_kde(X_samples)
kde_low_bw = stats.kde.gaussian_kde(X_samples, bw_method=0.25)

x = np.linspace(0, 20, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 3)) 
axes[0].hist(X_samples, normed=True, alpha=0.5, bins=25)   
axes[1].plot(x, kde(x), label="KDE")   
axes[1].plot(x, kde_low_bw(x), label="KDE (low bw)")   
axes[1].plot(x, X.pdf(x), label="True PDF")  
axes[1].legend()   
sns.distplot(X_samples, bins=25, ax=axes[2])
```
![download (23)](https://user-images.githubusercontent.com/52376448/66985781-79f27800-f0f8-11e9-9096-61d254a04c21.png)
<details markdown="1">
<summary class='jb-small' style="color:blue">additional kde</summary>
<hr class='division3'>
```python
kde.resample(10)
```
```
array([[ 2.21027713,  2.86300834,  5.63643055,  9.93925447, 11.0112984 ,
         5.53754038,  4.57539167,  0.18351943,  5.84327588,  5.67924786]])
```
<br>
```python
def _kde_cdf(x): 
    return kde.integrate_box_1d(-np.inf, x)

def _kde_ppf(q):
    return optimize.fsolve(lambda x, q: kde_cdf(x) - q, kde. dataset.mean(), args=(q,))[0] 
    
kde_cdf = np.vectorize(_kde_cdf)
kde_ppf = np.vectorize(_kde_ppf)
kde_ppf([0.05, 0.95])
```
```
array([0.53427617, 8.06347491])
```
<hr class='division3'>
</details>
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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

