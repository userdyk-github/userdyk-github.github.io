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

## **Implementation with sympy for simple case**
The classic optimization problem: Minimize the area of a cylinder with unit volume
<div style="font-size: 70%; text-align: center;">
  $$the\ objective\ function\ :\ f ([r,h]) = 2πr^{2} + 2πrh$$
  $$the\ equality\ constraint\ :\ g([r,h]) = πr^{2}h − 1 = 0$$
</div>

```python
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

<details markdown="1">
<summary class='jb-small' style="color:blue">Numerical value</summary>
<hr class='division3'>
```python
_.evalf()
```
`OUTPUT` : <span style="font-size: 70%;">0.541926070139289<span>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Verification</summary>
<hr class='division3'>
```python
Area_r.diff(r, 2).subs(r, rsol)
```
`OUTPUT` : <span style="font-size:70%;">12π</span>

```python
Area_r.subs(r, rsol)  
```
`OUTPUT` : <span style="font-size:70%;">$$3 \sqrt[3]{2} \sqrt[3]{\pi}$$</span>

```python
_.evalf()
```
`OUTPUT` : <span style="font-size:70%;">$$5.53581044593209$$</span>
<hr class='division3'>
</details>
  
  
<br><br><br>
<hr class="division2">

## **Implementation with scipy for more realistic problems**

<div style="font-size: 70%; text-align: center;">
  $$the\ objective\ function\ :\ f (r) = 2πr^{2} + \frac{2}{r}$$
</div>
```python
from scipy import optimize
import numpy as np                     # after executing numpy, and then execute cvxopt!
import cvxopt                          # Just add path 'C:\Python36\Library\bin' to PATH environment variable
import matplotlib.pyplot as plt 

# object function
def f(r):   
    return 2 * np.pi * r**2 + 2 / r

# optimization
r_min = optimize.brent(f, brack=(0.1, 4)) 
r_min, f(r_min)
```
`OUTPUT` : <span style="font-size: 70%;">$$\left ( 0.5419260772557135, \quad 5.535810445932086\right )$$</span>
<br><br><br>
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
