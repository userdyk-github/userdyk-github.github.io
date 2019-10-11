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

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/6f2ccc34699ad28c71419340168b2b51c683a93d" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -14.924ex; margin-bottom: -0.247ex; width:43.592ex; height:31.509ex;" alt="{\displaystyle {\begin{aligned}M_{X}(\alpha ;\beta ;t)&amp;=\operatorname {E} \left[e^{tX}\right]\\[4pt]&amp;=\int _{0}^{1}e^{tx}f(x;\alpha ,\beta )\,dx\\[4pt]&amp;={}_{1}F_{1}(\alpha ;\alpha +\beta ;t)\\[4pt]&amp;=\sum _{n=0}^{\infty }{\frac {\alpha ^{(n)}}{(\alpha +\beta )^{(n)}}}{\frac {t^{n}}{n!}}\\[4pt]&amp;=1+\sum _{k=1}^{\infty }\left(\prod _{r=0}^{k-1}{\frac {\alpha +r}{\alpha +\beta +r}}\right){\frac {t^{k}}{k!}}\end{aligned}}}">

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

