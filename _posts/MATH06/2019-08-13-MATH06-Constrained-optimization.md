---
layout : post
title : MATH06, Constrained optimization
categories: [MATH06]
comments : true
tags : [MATH06]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[Optimization](https://userdyk-github.github.io/math06/MATH06-Contents.html) <br>
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

## **Bounded optimization problem with the L-BFGS-B algorithm**
<div style="font-size: 70%; text-align: center;">
    $$the\ objective\ function\ :\ f(x) = (x_{1}-1)^{2}-(x_{2}-1)^{2}$$
    $$s.t. \qquad 2<x_{1}<3,\ 0 \le x_{2} \le 2$$
</div>
```python
from scipy import optimize

# objective function
def f(X):   
    x, y = X   
    return (x - 1)**2 + (y - 1)**2 

# constraints
bnd_x1, bnd_x2 = (2, 3), (0, 2) 

# optimization of obejective function considering constraints
optimize.minimize(f, [1, 1], method='L-BFGS-B', 
                  bounds=[bnd_x1, bnd_x2]).x 
```
`OUTPUT` : <span style="font-size: 70%;">$$optimal\ point\ with\ constraints\ :\ (2., 1.)$$</span>

<details markdown="1">
<summary class='jb-small' style="color:blue">VISUALIZATION</summary>
<hr class='division3'>
```python
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt 

# objective function
def f(X):   
    x, y = X   
    return (x - 1)**2 + (y - 1)**2 

# optimization of objective function
x_opt = optimize.minimize(f, [1, 1], method='BFGS').x 

# constraints
bnd_x1, bnd_x2 = (2, 3), (0, 2) 

# optimization of obejective function considering constraints
x_cons_opt = optimize.minimize(f, [1, 1], method='L-BFGS-B',   
                               bounds=[bnd_x1, bnd_x2]).x 


def func_X_Y_to_XY(f, X, Y):   
    """   
    Wrapper for f(X, Y) -> f([X, Y])   
    """  
    s = np.shape(X)  
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s) 


x_ = y_ = np.linspace(-1, 3, 100)   
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots(figsize=(6, 4))   
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)   
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)   
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)  
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]),    
                           bnd_x1[1] - bnd_x1[0], bnd_x2[1] -  bnd_x2[0], facecolor="grey")   
ax.add_patch(bound_rect)    
ax.set_xlabel(r"$x_1$", fontsize=18)    
ax.set_ylabel(r"$x_2$", fontsize=18) 
plt.colorbar(c, ax=ax)
```
![다운로드](https://user-images.githubusercontent.com/52376448/65370629-0f375380-dc96-11e9-9e79-aba55cae09ee.png)
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Optimization problem using Lagrange multipliers**
Using the Lagrange multipliers, it is possible to convert a constrained optimization problem to an unconstrained problem by introducing additional variables. 
<div style="font-size: 70%; text-align: center;">
    $$the\ objective\ function\ :\ f(x) = x_{1}x_{2}x_{3}$$
    $$s.t. \qquad g(x) = 2x_{1}x_{2} +2x_{0}x_{2}+2x_{1}x_{0}-1=0$$
</div>
```python
import sympy 
sympy.init_printing()

x = x0, x1, x2, l = sympy.symbols("x_0, x_1, x_2, lambda") 
f = x0 * x1 * x2 
g = 2 * (x0 * x1 + x1 * x2 + x2 * x0) - 1
L = f + l * g 

grad_L = [sympy.diff(L, x_) for x_ in x]
sympy.solve(grad_L) 
```
`OUTPUT` : 
<div style="font-size: 70%;">
    $$optimal\ point\ with\ constraints\ using\ Lagrange\ multipliers$$
    $$\left [ \left \{ \lambda : - \frac{\sqrt{6}}{24}, \quad x_{0} : \frac{\sqrt{6}}{6}, \quad x_{1} : \frac{\sqrt{6}}{6}, \quad x_{2} : \frac{\sqrt{6}}{6}\right \}, \quad \left \{ \lambda : \frac{\sqrt{6}}{24}, \quad x_{0} : - \frac{\sqrt{6}}{6}, \quad x_{1} : - \frac{\sqrt{6}}{6}, \quad x_{2} : - \frac{\sqrt{6}}{6}\right \}\right ]$$
</div>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
g.subs(sols[0])
```
`OUTPUT` : <span style="font-size: 70%;">$$0$$</span>
```python
f.subs(sols[0])
```
`OUTPUT` : <span style="font-size: 70%;">$$\frac{\sqrt{6}}{36}$$</span>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Sequential least square programming(SLSQP) to handle inequality constraints**  
`Equality constraints`
<div style="font-size: 70%; text-align: center;">
    $$the\ objective\ function\ :\ f(x) = -x_{0}x_{1}x_{2}$$
    $$s.t. \qquad g(x) = 2(x_{0}x_{1}+x{1}x{2}+x{2}x{0})-1 = 0$$
</div>
```python
from scipy import optimize

# objective function
def f(X):   
    return -X[0] * X[1] * X[2] 

# constraints
def g(X):   
    return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1

# optimization
constraint = dict(type='eq', fun=g) 
optimize.minimize(f, [0.5, 1, 1.5], method='SLSQP', constraints=[constraint]).x
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
`OUTPUT` : <span style="font-size: 70%;">$$[0.40824188, 0.40825127, 0.40825165]$$</span>
<hr class='division3'>
</details>

<br>

`Inequality constraints`
<div style="font-size: 70%; text-align: center;">
    $$the\ objective\ function\ :\ f(x) = (x_{0}-1)**2 + (x_{1}-1)**2$$
    $$s.t. \qquad g(x) = x_{1}-1.75-(x_{0}-0.75)**4 \ge 0$$
</div>
```python
from scipy import optimize

# objective function
def f(X):  
    return (X[0] - 1)**2 + (X[1] - 1)**2
    
# constraints
def g(X):  
    return X[1] - 1.75 - (X[0] - 0.75)**4 
    
# optimization
constraints = [dict(type='ineq', fun=g)]
optimize.minimize(f, (0, 0), method='SLSQP', constraints=constraints).x
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
`OUTPUT` : <span style="font-size: 70%;">$$[0.96857656, 1.75228252]$$</span>
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">VISULALIZATION</summary>
<hr class='division3'>
```python
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# objective function
def f(X):  
    return (X[0] - 1)**2 + (X[1] - 1)**2
    
# constraints    
def g(X):  
    return X[1] - 1.75 - (X[0] - 0.75)**4 
    
# optimization  
constraints = [dict(type='ineq', fun=g)]
x_opt = optimize.minimize(f, (0, 0), method='BFGS').x 
x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP', constraints=constraints).x

# visualization
x_ = y_ = np.linspace(-1, 3, 100)   
X, Y = np.meshgrid(x_, y_)  

fig, ax = plt.subplots(figsize=(6, 4)) 
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)   
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)  
ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)  
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color='grey')   
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15) 

ax.set_ylim(-1, 3) 
ax.set_xlabel(r"$x_0$", fontsize=18)   
ax.set_ylabel(r"$x_1$", fontsize=18)   
plt.colorbar(c, ax=ax)
```
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65371853-ebc6d580-dca2-11e9-8dd2-0512e6374b04.png)
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
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---


