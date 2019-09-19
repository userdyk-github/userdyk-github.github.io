---
layout : post
title : MATH06, Univariate optimization
categories: [MATH06]
comments : true
tags : [MATH06]
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

## **Method for root finding**

|bisection method||
|Newton’s method||
|hybrid method||

<br><br><br>
<hr class="division2">

## **The classic optimization problem: Minimize the area of a cylinder with unit volume**

<div style="font-size: 70%; text-align: center;">
  $$the objective function : f ([r,h]) = 2πr^{2} + 2πrh$$
  $$the equality constraint : g([r,h]) = πr^{2}h − 1 = 0$$
</div>

```python
from scipy import optimize
import numpy as np                     # after executing numpy, and then execute cvxopt!
import cvxopt                          # Just add path 'C:\Python36\Library\bin' to PATH environment variable
import matplotlib.pyplot as plt 
import sympy 
sympy.init_printing()

r, h = sympy.symbols("r, h") 
Area = 2 * sympy.pi * r**2 + 2 * sympy.pi * r * h 
Volume = sympy.pi * r**2 * h 
h_r = sympy.solve(Volume - 1)[0]
Area_r = Area.subs(h_r)
rsol = sympy.solve(Area_r.diff(r))[0] 
rsol
```
`OUTPUT` : <span style="font-size: 70%;">$$\frac{2^{\frac{2}{3}}}{2 \sqrt[3]{\pi}}$$</span>
```python
_.evalf()
```
`OUTPUT` : <span style="font-size: 70%;">0.541926070139289<span>
  
<br><br><br>
<hr class="division2">

## title2

<hr class="division2">

## title3

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
