---
layout : post
title : MATH06, Univariate optimization
categories: [MATH06]
comments : true
tags : [MATH06]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-Matplotlib.html' target="_blank">Matplotlib</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Method for root finding**

### ***Bracketing methods***

#### Bisection method

<br><br><br>

---

#### False position (regula falsi)

<br><br><br>

---


### ***Interpolation***

<br><br><br>

---


### ***Iterative methods***

#### Newton's method (and similar derivative-based methods)

<br><br><br>

---

#### Secant method

<br><br><br>

---

#### Steffensen's method

<br><br><br>

---

#### Inverse interpolation

<br><br><br>

---

### ***Combinations of methods***

#### Brent's method
<br><br><br>

---



### ***Roots of polynomials***

#### Finding one root

<br><br><br>

---

#### Finding roots in pairs

<br><br><br>

---

#### Finding all roots at once

<br><br><br>

---

#### Exclusion and enclosure methods

<br><br><br>

---

#### Real-root isolation

<br><br><br>

---

#### Finding multiple roots of polynomials

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
<span class="frame2">Method1</span>
```python
from scipy import optimize
import numpy as np

# object function
def f(r):   
    return 2 * np.pi * r**2 + 2 / r

# optimization
r_min = optimize.brent(f, brack=(0.1, 4)) 
r_min, f(r_min)
```
`OUTPUT` : <span style="font-size: 70%;">$$\left ( 0.5419260772557135, \quad 5.535810445932086\right )$$</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
**brack keyword argument** specify a starting interval for the algorithm
<hr class='division3'>
</details>
<br><br><br>



<span class="frame2">Method2</span>
```python
from scipy import optimize
import numpy as np

# object function
def f(r):   
    return 2 * np.pi * r**2 + 2 / r

# optimization
optimize.minimize_scalar(f, bracket=(0.1, 4))
```
`OUTPUT` :
```
     fun: 5.535810445932086
    nfev: 19
     nit: 15
 success: True
       x: 0.5419260772557135
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt
import numpy as np

# main graph
r = np.linspace(0.1,2,100)
y = 2*np.pi*r**2 + 2/r
plt.plot(r,y)
plt.ylim([0,30])

# optimization point
plt.plot(0.5419260772557135, 5.535810445932086, marker='*', ms=15, mec='r')
plt.annotate("Optimization point", fontsize=14, family="serif", xy=(0.5419260772557135, 5.535810445932086), xycoords="data", xytext=(+20, +50), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))

plt.show()
```
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/65272947-08f08c80-db5b-11e9-8260-a2e75271c595.png)
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
- <a href='https://en.wikipedia.org/wiki/List_of_algorithms' target="_blank">List of algorithms</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---



