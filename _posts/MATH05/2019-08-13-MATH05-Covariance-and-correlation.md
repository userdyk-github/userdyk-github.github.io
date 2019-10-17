---
layout : post
title : MATH05, Covariance and correlation
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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

## **sample covariance**
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4d158b1ec5a3c6d1de84b9d59f604d8170a51407" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:38.104ex; height:7.343ex;" alt=" q_{jk}=\frac{1}{N-1}\sum_{i=1}^{N}\left(  x_{ij}-\bar{x}_j \right)  \left( x_{ik}-\bar{x}_k \right), ">

<br><br><br>
<hr class="division2">

## **sample correlation coefficient**
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/332ae9dcde34d03f30ed6e1880af8b43327dd49c" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -7.338ex; width:59.755ex; height:14.343ex;" alt="{\displaystyle r_{xy}\quad {\overset {\underset {\mathrm {def} }{}}{=}}\quad {\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{(n-1)s_{x}s_{y}}}={\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sqrt {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})^{2}\sum \limits _{i=1}^{n}(y_{i}-{\bar {y}})^{2}}}},}">

<br><br><br>

<hr class="division2">

## **covariance and correlation**
<span class="frame3">covariance</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/435d5740798f0ec6c3fdf5cf70c82fa78c2e0f77" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:33.229ex; height:2.843ex;" alt="{\displaystyle \operatorname {Cov} (X,Y)=\operatorname {E} \left((X-\mu )(Y-\nu )\right)\,}">
```python
from sklearn.datasets import load_iris
from scipy import stats

X = load_iris().data
x1 = X[:, 0]  # 꽃받침의 길이
x2 = X[:, 1]  # 꽃받침의 폭
x3 = X[:, 2]  # 꽃잎의 길이
x4 = X[:, 3]  # 꽃잎의 폭

stats.pearsonr(x1, x3)[0]
```
```
0.8717537758865832
```
<br><br><br>

<span class="frame3">correlation</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/93185aed3047ef42fa0f1b6e389a4e89a5654afa" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.171ex; width:57.998ex; height:6.009ex;" alt="{\displaystyle \rho _{X,Y}=\operatorname {corr} (X,Y)={\operatorname {cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={\operatorname {E} [(X-\mu _{X})(Y-\mu _{Y})] \over \sigma _{X}\sigma _{Y}}}">
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
corrs = [1, 0.7, 0.3, 0, -0.3, -0.7, -1]
plt.figure(figsize=(len(corrs), 2))
for i, r in enumerate(corrs):
    x, y = np.random.multivariate_normal([0, 0], [[1, r], [r, 1]], 1000).T
    plt.subplot(1, len(corrs), i + 1)
    plt.plot(x, y, 'ro', ms=1)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.title(r"$\rho$={}".format(r))

plt.suptitle("scatter plot about correlation", y=1.1)
plt.tight_layout()
plt.show()
```
![download (15)](https://user-images.githubusercontent.com/52376448/66980870-17e04580-f0ed-11e9-87a5-559f785ebccf.png)
<br><br><br>
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
slope = [1, 0.7, 0.3, 0, -0.3, -0.7, -1]
plt.figure(figsize=(len(slope), 2))
for i, s in enumerate(slope):
    plt.subplot(1, len(slope), i + 1)
    x, y = np.random.multivariate_normal([0, 0], [[1, 1], [1, 1]], 100).T
    y2 = s * y
    plt.plot(x, y2, 'ro', ms=1)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    if s > 0:
        plt.title(r"$\rho$=1")
    if s < 0:
        plt.title(r"$\rho$=-1")

plt.suptitle("correlation and slope are independent", y=1.1)
plt.tight_layout()
plt.show()
```
![download (17)](https://user-images.githubusercontent.com/52376448/66980997-76a5bf00-f0ed-11e9-8069-17c62022109c.png)
<br><br><br>
<hr class="division2">

## **non-linear correlation**

<br><br><br>

<hr class="division2">

## **Frank Anscombe data**

<br><br><br>


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

