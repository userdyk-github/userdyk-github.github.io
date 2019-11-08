---
layout : post
title : MATH05, Covariance and correlation
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

## **sample covariance**
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4d158b1ec5a3c6d1de84b9d59f604d8170a51407" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:38.104ex; height:7.343ex;" alt=" q_{jk}=\frac{1}{N-1}\sum_{i=1}^{N}\left(  x_{ij}-\bar{x}_j \right)  \left( x_{ik}-\bar{x}_k \right), "><br>
<span class="frame3">One column</span>
```python
import numpy as np

def sample_covariance(rv):
    Cov = 0
    for i in range(rv.shape[0]):
        Cov = Cov + (rv[i]-rv.mean())**2
    Cov = Cov/(rv.shape[0]-1)
    return Cov

np.random.seed(2019)
rv = np.random.RandomState(2019)
rv = rv.normal(10,100,(100,1))

sample_covariance(rv)
```
```
10066.80475325
```
<br>
```python
import numpy as np

def sample_covariance(rv):
    Cov = 0
    for i in range(rv.shape[0]):
        Cov = Cov + (rv[i][0]-rv[:,0].mean())*(rv[i][1]-rv[:,1].mean())
    Cov = Cov/(rv.shape[0]-1)
    return Cov


np.random.seed(2019)
rv = np.random.RandomState(2019)
rv = rv.normal(5,1,size=(5000,2))
sample_covariance(rv)
```
```
0.004201816972783285
```
<details markdown="1">
<summary class='jb-small' style="color:blue">np.cov</summary>
<hr class='division3'>
```python
np.cov(rv[:,0],rv[:,1])
```
```
array([[0.98179804, 0.00420182],
       [0.00420182, 1.03103471]])
```
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">Two columns</span>
```python

```
```
```
<br><br><br>

<span class="frame3">Several columns</span>
```python

```
```
```
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
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 500
np.random.seed(1)

plt.subplot(221)
x1 = np.random.uniform(-1, 1, n)
y1 = 2*x1**2 + np.random.uniform(-0.1, 0.1, n)
plt.scatter(x1, y1)
r1 = stats.pearsonr(x1, y1)[0]
plt.title(r"non-linear correlation 1: r={:3.1f}".format(r1))

plt.subplot(222)
x2 = np.random.uniform(-1, 1, n)
y2 = 4*(x2**2-0.5)**2 + 0.1 * np.random.uniform(-1, 1, n)
plt.scatter(x2, y2)
r2 = stats.pearsonr(x2, y2)[0]
plt.title(r"non-linear correlation 2: r={:3.1f}".format(r2))

plt.subplot(223)
x3 = np.random.uniform(-1, 1, n)
y3 = np.cos(x3 * np.pi) + np.random.uniform(0, 1/8, n)
x3 = np.sin(x3 * np.pi) + np.random.uniform(0, 1/8, n)
plt.scatter(x3, y3)
r3 = stats.pearsonr(x3, y3)[0]
plt.title(r"non-linear correlation 3: r={:3.1f}".format(r3))

plt.subplot(224)
x4 = np.random.uniform(-1, 1, n)
y4 = (x4**2 + np.random.uniform(0, 0.1, n)) * \
    np.array([-1, 1])[np.random.random_integers(0, 1, size=n)]
plt.scatter(x4, y4)
r4 = stats.pearsonr(x4, y4)[0]
plt.title(r"non-linear correlation 4: r={:3.1f}".format(r4))


plt.tight_layout()
plt.show()
```
![download (18)](https://user-images.githubusercontent.com/52376448/66981159-e6b44500-f0ed-11e9-88ef-66bf2b4025e0.png)
<br><br><br>

<hr class="division2">

## **Frank Anscombe data**
```python
import statsmodels.api as sm

data = sm.datasets.get_rdataset("anscombe")
df = data.data
df[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]]
```
```
	x1	y1	x2	y2	x3	y3	x4	y4
0	10	8.04	10	9.14	10	7.46	8	6.58
1	8	6.95	8	8.14	8	6.77	8	5.76
2	13	7.58	13	8.74	13	12.74	8	7.71
3	9	8.81	9	8.77	9	7.11	8	8.84
4	11	8.33	11	9.26	11	7.81	8	8.47
5	14	9.96	14	8.10	14	8.84	8	7.04
6	6	7.24	6	6.13	6	6.08	8	5.25
7	4	4.26	4	3.10	4	5.39	19	12.50
8	12	10.84	12	9.13	12	8.15	8	5.56
9	7	4.82	7	7.26	7	6.42	8	7.91
10	5	5.68	5	4.74	5	5.73	8	6.89
```
<br>
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.subplot(221)
sns.regplot(x="x1", y="y1", data=df)
plt.subplot(222)
sns.regplot(x="x2", y="y2", data=df)
plt.subplot(223)
sns.regplot(x="x3", y="y3", data=df)
plt.subplot(224)
sns.regplot(x="x4", y="y4", data=df)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle("Frank Anscombe data")
plt.show()
```
![download (19)](https://user-images.githubusercontent.com/52376448/66981414-9ab5d000-f0ee-11e9-8677-3ee65a43d69e.png)
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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
