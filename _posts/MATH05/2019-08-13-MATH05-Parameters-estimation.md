---
layout : post
title : MATH05, Parameters estimation
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Python Libraries</a>
- <a href='https://en.wikipedia.org/wiki/List_of_probability_distributions' target="_blank">List of probability distributions</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **method of moment**

### ***estimate for beta distribution***

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/5fc18388353b219c482e8e35ca4aae808ab1be81" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -14.049ex; margin-bottom: -0.289ex; width:38.853ex; height:29.843ex;" alt="{\displaystyle {\begin{aligned}f(x;\alpha ,\beta )&amp;=\mathrm {constant} \cdot x^{\alpha -1}(1-x)^{\beta -1}\\[3pt]&amp;={\frac {x^{\alpha -1}(1-x)^{\beta -1}}{\displaystyle \int _{0}^{1}u^{\alpha -1}(1-u)^{\beta -1}\,du}}\\[6pt]&amp;={\frac {\Gamma (\alpha +\beta )}{\Gamma (\alpha )\Gamma (\beta )}}\,x^{\alpha -1}(1-x)^{\beta -1}\\[6pt]&amp;={\frac {1}{\mathrm {B} (\alpha ,\beta )}}x^{\alpha -1}(1-x)^{\beta -1}\end{aligned}}}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e03c03f31b903a1bc73ea8b637e3134b110a85a2" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:36.574ex; height:7.343ex;" alt="\operatorname{E}[X^k]= \frac{\alpha^{(k)}}{(\alpha + \beta)^{(k)}} = \prod_{r=0}^{k-1} \frac{\alpha+r}{\alpha+\beta+r}">

```python
from scipy import stats
import numpy as np

def estimate_beta(x):
    x_bar = x.mean()
    s2 = x.var()
    a = x_bar * (x_bar * (1 - x_bar) / s2 - 1)
    b = (1 - x_bar) * (x_bar * (1 - x_bar) / s2 - 1)
    return a, b
    
np.random.seed(0)
x = stats.beta(15, 12).rvs(10000)
estimate_beta(x)
```
<span class="jb-medium">(15.346682046700685, 12.2121537049535)</span>
<hr class="division2">

## **likelihood function**

<hr class="division2">

## **Bayesian estimation**

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
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>

