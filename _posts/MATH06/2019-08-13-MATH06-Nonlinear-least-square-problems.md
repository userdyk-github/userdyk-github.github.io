---
layout : post
title : MATH06, Nonlinear least square problems
categories: [MATH06]
comments : true
tags : [MATH06]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[MATH06](https://userdyk-github.github.io/math06/MATH06-Contents.html)<br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/math05/MATH05-Curve-fitting.html' target="_blank">Curve fitting</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## the Levenberg-Marquardt method for nonlinear least square problems
In general, a least square problem can be viewed as an optimization problem with the objective function. Nonlinear least square optimization problem has a specific structure, and several methods that are tailored to solve this particular optimization problem have
been developed. One example, **the Levenberg-Marquardt method is based on the idea of successive linearizations of the problem in each iteration.**

```python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

beta = (0.25, 0.75, 0.5)

# true model : f(xdata, *beta)
def f(x, b0, b1, b2):
    return b0 + b1 * np.exp(-b2 * x**2)

xdata = np.linspace(0, 5, 50)
y = f(xdata, *beta)

# input data : ydata
ydata = y + 0.05 * np.random.randn(len(xdata))

# residual(deviation) : g(beta) = ydata - f(xdata, *beta)
def g(beta):
    return ydata - f(xdata, *beta)

# optimization for beta : beta_opt
beta_start = (1, 1, 1)
beta_opt, beta_cov = optimize.leastsq(g, beta_start)

# visualization
fig, ax = plt.subplots()
ax.scatter(xdata, ydata, label='samples')
ax.plot(xdata, y, 'r', lw=2, label='true model')
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2, label='fitted model')
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
ax.legend()
plt.show()
```
![다운로드 (13)](https://user-images.githubusercontent.com/52376448/65291248-66053600-db8d-11e9-85b5-36d1b8d32770.png)

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
beta_opt
```
`OUTPUT` : <span style="font-size: 70%;">$$[0.24852741, 0.77109938, 0.49358439]$$</span>

```python
beta_cov
```
`OUTPUT` : <span style="font-size: 70%;">$$1$$</span>

<hr class='division3'>
</details>

<br><br><br>
<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- [post1](https://userdyk-github.github.io/)
- <a href='https://darkpgmr.tistory.com/142' target="_blank">the Levenberg-Marquardt method</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---
