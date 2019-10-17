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
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7120384a1c843727d9589e2b33dbc33901d14f42" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -1.005ex; width:40.233ex; height:3.176ex;" alt="{\displaystyle \operatorname {cov} (X,Y)=\operatorname {E} {{\big [}(X-\operatorname {E} [X])(Y-\operatorname {E} [Y]){\big ]}},}">
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

