---
layout : post
title : MATH06, Unconstrained multivariate optimization
categories: [MATH06]
comments : true
tags : [MATH06]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-Matplotlib.html' target="_blank">Matplotlib</a>
- <a href='https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-NumPy.html' target="_blank">NumPy</a>
- <a href='https://userdyk-github.github.io/math06/MATH06-Univariate-optimization.html'>Univariate optimization</a>


---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Gradient and Hessian**

<div style="font-size: 70%; text-align:center;"> $$the\ objective\ function\ :\ f(x) = (x_{1} + 10)^{2} + 5(x_{2} - 9)^{2} - 2x_{1}x_{2}$$</div>

```python
import sympy
sympy.init_printing()

x1, x2 = sympy.symbols("x_1, x_2") 
f_sym = (x1+10)**2 + 5 * (x2-9)**2 - 2*x1*x2 

# Gradient
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]
Gradient = sympy.Matrix(fprime_sym)

# Hessian  
fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)] 
Hessian = sympy.Matrix(fhess_sym)

print(Gradient, '\n', Hessian)
```
`OUTPUT` :
<span style="font-size: 70%;"> $$\left[\begin{matrix}2 x_{1} - 2 x_{2} + 20\\- 2 x_{1} + 10 x_{2} - 90\end{matrix}\right], \left[\begin{matrix}2 & -2\\-2 & 10\end{matrix}\right]$$</span>

<br><br><br>
<hr class="division2">


## **Optimization process using the Newton-CG method with Gradient&Hessian**

<div class='jb-medium'>If both the gradient and the Hessian are known, then Newton’s method is the method with <strong>the fastest convergence in general.</strong></div> 

<div style="font-size: 70%; text-align:center;"> $$the\ objective\ function\ :\ f(x) = (x_{1} - 1)^{4} + 5(x_{2} - 1)^{2} - 2x_{1}x_{2}$$</div>

```python
from scipy import optimize
import sympy
sympy.init_printing()

x1, x2 = sympy.symbols("x_1, x_2") 

# Object function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2 
# Gradient
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]
# Hessian  
fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)] 

# Convert sympy function to numpy function
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy') 
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
fhess_lmbda = sympy.lambdify((x1, x2), fhess_sym, 'numpy')

# Unpacking for multivariate function
def func_XY_to_X_Y(f):    
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) 
f = func_XY_to_X_Y(f_lmbda) 
fprime = func_XY_to_X_Y(fprime_lmbda) 
fhess = func_XY_to_X_Y(fhess_lmbda)

# Optimization
optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Optimization terminated successfully.
         Current function value: -3.867223
         Iterations: 8
         Function evaluations: 10
         Gradient evaluations: 17
         Hessian evaluations: 8
array([1.88292613, 1.37658523])
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
Gradient
```
sympy.Matrix(fprime_sym)
```
`OUTPUT` :
<span style="font-size: 70%;"> $$\left[\begin{matrix}- 2 x_{2} + 4 \left(x_{1} - 1\right)^{3}\\- 2 x_{1} + 10 x_{2} - 10\end{matrix}\right]$$</span><br>
Hessian
```
sympy.Matrix(fhess_sym)
```
`OUTPUT` :
<span style="font-size: 70%;"> $$\left[\begin{matrix}12 \left(x_{1} - 1\right)^{2} & -2\\-2 & 10\end{matrix}\right]$$</span>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">VISUALLIZATION</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

x_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)  
x_ = y_ = np.linspace(-1, 4, 100)  
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots(figsize=(6, 4)) 
c = ax.contour(X, Y, f_lmbda(X, Y), 100)   
plt.colorbar(c, ax=ax)

ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)   
ax.set_xlabel(r"$x_1$", fontsize=18)   
ax.set_ylabel(r"$x_2$", fontsize=18)    
plt.show()
```
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/65281242-1747a400-db6d-11e9-9be4-635a88f95011.png)
<hr class='division3'>
</details>
<br><br><br>
<hr class="division2">

<br><br><br>

## **Optimization process using the BFGS algorithm with Gradient**

<div class="medium">Although the BFGS and the conjugate gradient methods theoretically have slower convergence than Newton’s method, they can sometimes offer <strong>improved stability</strong> and can therefore be preferable. </div>

```python
from scipy import optimize
import sympy
sympy.init_printing()
import numpy as np

x1, x2 = sympy.symbols("x_1, x_2") 

# Object function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2 
# Gradient
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]

# Convert sympy function to numpy function
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy') 
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')

# Unpacking for multivariate function
def func_XY_to_X_Y(f):    
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) 
f = func_XY_to_X_Y(f_lmbda) 
fprime = func_XY_to_X_Y(fprime_lmbda) 

# Optimization
optimize.fmin_bfgs(f, (0, 0), fprime=fprime)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Optimization terminated successfully.
         Current function value: -3.867223
         Iterations: 9
         Function evaluations: 13
         Gradient evaluations: 13
array([1.88292645, 1.37658596])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
Gradient
```
sympy.Matrix(fprime_sym)
```
`OUTPUT` :
<span style="font-size: 70%;">$$\left[\begin{matrix}- 2 x_{2} + 4 \left(x_{1} - 1\right)^{3}\\- 2 x_{1} + 10 x_{2} - 10\end{matrix}\right]$$</span><br>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">VISUALLIZATION</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

x_opt = optimize.fmin_bfgs(f, (0, 0), fprime=fprime)
x_ = y_ = np.linspace(-1, 4, 100)  
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots(figsize=(6, 4)) 
c = ax.contour(X, Y, f_lmbda(X, Y), 100)   
plt.colorbar(c, ax=ax)

ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)   
ax.set_xlabel(r"$x_1$", fontsize=18)   
ax.set_ylabel(r"$x_2$", fontsize=18)    
plt.show()
```
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/65285866-b3c37380-db78-11e9-84b2-f00ef7001f7c.png)
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Optimization process using a nonlinear conjugate gradient algorithm with Gradient**

<div class="medium">Although the BFGS and the conjugate gradient methods theoretically have slower convergence than Newton’s method, they can sometimes offer <strong>improved stability</strong> and can therefore be preferable. </div>

```python
from scipy import optimize
import sympy
sympy.init_printing()
import numpy as np

x1, x2 = sympy.symbols("x_1, x_2") 

# Object function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2 
# Gradient
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]

# Convert sympy function to numpy function
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy') 
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')

# Unpacking for multivariate function
def func_XY_to_X_Y(f):    
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) 
f = func_XY_to_X_Y(f_lmbda) 
fprime = func_XY_to_X_Y(fprime_lmbda) 

# Optimization
optimize.fmin_cg(f, (0, 0), fprime=fprime)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Optimization terminated successfully.
         Current function value: -3.867223
         Iterations: 8
         Function evaluations: 18
         Gradient evaluations: 18
array([1.88292612, 1.37658523])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
Gradient
```
sympy.Matrix(fprime_sym)
```
`OUTPUT` :
<span style="font-size: 70%;">$$\left[\begin{matrix}- 2 x_{2} + 4 \left(x_{1} - 1\right)^{3}\\- 2 x_{1} + 10 x_{2} - 10\end{matrix}\right]$$</span>
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">VISUALLIZATION</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

x_opt = optimize.fmin_cg(f, (0, 0), fprime=fprime)
x_ = y_ = np.linspace(-1, 4, 100)  
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots(figsize=(6, 4)) 
c = ax.contour(X, Y, f_lmbda(X, Y), 100)   
plt.colorbar(c, ax=ax)

ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)   
ax.set_xlabel(r"$x_1$", fontsize=18)   
ax.set_ylabel(r"$x_2$", fontsize=18)    
plt.show()
```
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/65286094-54199800-db79-11e9-9877-1e639502119b.png)
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Optimization process using the BFGS algorithm without Gradient&Hessian**

<div class='jb-medium'>In general, the BFGS method is often a good first approach to try, in particular if neither the gradient nor the Hessian is known. If only the gradient is known, then the BFGS method is still <strong>the generally recommended method</strong> to use, although the conjugate gradient method is often a competitive alternative to the BFGS method.</div>

```python
from scipy import optimize
import sympy
sympy.init_printing()
import numpy as np

x1, x2 = sympy.symbols("x_1, x_2") 

# Object function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2 

# Convert sympy function to numpy function
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy') 

# Unpacking for multivariate function
def func_XY_to_X_Y(f):    
    '''Wrapper for f(X) -> f(X[0], X[1])'''
    return lambda X: np.array(f(X[0], X[1])) 
f = func_XY_to_X_Y(f_lmbda) 

# Optimization
optimize.fmin_bfgs(f, (0, 0))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Optimization terminated successfully.
         Current function value: -3.867223
         Iterations: 9
         Function evaluations: 52
         Gradient evaluations: 13
array([1.88292645, 1.37658596])
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">VISUALLIZATION</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

x_opt = optimize.fmin_bfgs(f, (0, 0))
x_ = y_ = np.linspace(-1, 4, 100)  
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots(figsize=(6, 4)) 
c = ax.contour(X, Y, f_lmbda(X, Y), 100)   
plt.colorbar(c, ax=ax)

ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)   
ax.set_xlabel(r"$x_1$", fontsize=18)   
ax.set_ylabel(r"$x_2$", fontsize=18)    
plt.show()
```
![다운로드 (11)](https://user-images.githubusercontent.com/52376448/65286198-a5c22280-db79-11e9-8dbf-bd764dd63a48.png)
<hr class='division3'>
</details>

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

