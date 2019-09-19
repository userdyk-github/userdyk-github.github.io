---
layout : post
title : MATH01, Linear equation systems
categories: [MATH01]
comments : true
tags : [MATH01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/math01/MATH01-Basic-concepts.html'>Basic concepts</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Equation**

```python
import sympy
#sympy.init_printing()

x = sympy.Symbol("x")
sympy.solve(x**2 + 2*x - 3)
```
`OUTPUT` : <span style="font-size: 70%;">$$[-3, 1]$$</span>
<br><br><br>

```python
import sympy
#sympy.init_printing()

x = sympy.Symbol("x")
a, b, c = sympy.symbols("a, b, c")
sympy.solve(a * x**2 + b * x + c, x)
```
`OUTPUT` : <span style="font-size: 70%;">$$\left [ \frac{1}{2 a} \left(- b + \sqrt{- 4 a c + b^{2}}\right), \quad - \frac{1}{2 a} \left(b + \sqrt{- 4 a c + b^{2}}\right)\right ]$$</span>

<br><br><br>

```python
import sympy
#sympy.init_printing()

x = sympy.Symbol("x")
sympy.solve(sympy.sin(x) - sympy.cos(x), x)
```
`OUTPUT` : <span style="font-size: 70%;">$$\left [ - \frac{3 \pi}{4}, \quad \frac{\pi}{4}\right ]$$</span>

<br><br><br>

```python
import sympy
#sympy.init_printing()

x = sympy.Symbol("x")
sympy.solve(sympy.exp(x) + 2 * x, x)
```
`OUTPUT` : <span style="font-size: 70%;">$$\left [ - \operatorname{LambertW}{\left (\frac{1}{2} \right )}\right ]$$</span>

<br><br><br>

```python
import sympy
#sympy.init_printing()

x = sympy.Symbol("x")
sympy.solve(x**5 - x**2 + 1, x) 
```
`OUTPUT` :

<div style="color:black; font-size: 70%; text-align: center;">
  $$[ \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 0\right)}, \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 1\right)}, $$ 
  $$\operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 2\right)}, \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 3\right)}, $$
  $$\operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 4\right)}]$$
</div>

<br><br><br>


```python
import sympy
sympy.init_printing()

x = sympy.Symbol("x")
y = sympy.Symbol("y")

eq1 = x + 2 * y - 1
eq2 = x - y + 1

sympy.solve([eq1, eq2], [x, y], dict=True) 
```

`OUTPUT` : <span style="font-size: 70%;">$$\left [ \left \{ x : - \frac{1}{3}, \quad y : \frac{2}{3}\right \}\right ]$$</span>
<br><br><br>

```python
import sympy
sympy.init_printing()

x = sympy.Symbol("x")
y = sympy.Symbol("y")

eq1 = x**2 - y    
eq2 = y**2 - x 

sympy.solve([eq1, eq2], [x, y], dict=True) 
```

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
import sympy
sympy.init_printing()

x = sympy.Symbol("x")
y = sympy.Symbol("y")

eq1 = x**2 - y    
eq2 = y**2 - x 
sols = sympy.solve([eq1, eq2], [x, y], dict=True) 
[eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 for sol in sols]
```
`OUTPUT` : [True, True, True, True]
<hr class='division3'>
</details>

`OUTPUT` : 
<div style="font-size: 70%; text-align: center;">
  $$[ \left \{ x : 0, \quad y : 0\right \}, \quad \left \{ x : 1, \quad y : 1\right \},$$
  $$\quad \left \{ x : \left(- \frac{1}{2} - \frac{\sqrt{3} i}{2}\right)^{2}, \quad y : - \frac{1}{2} - \frac{\sqrt{3} i}{2}\right \}, \quad \left \{ x : \left(- \frac{1}{2} + \frac{\sqrt{3} i}{2}\right)^{2}, \quad y : - \frac{1}{2} + \frac{\sqrt{3} i}{2}\right \} ]$$
</div>


<br><br><br>
<hr class="division2">


## **Square Systems**
<div style="color:black; font-size: 80%; text-align: center;">
  $$Ax=b$$ 
  $$\begin{pmatrix}2 & 3 \\5 & 4\end{pmatrix}x=\begin{pmatrix}4 \\ 3 \end{pmatrix}$$
</div>

<span class='frame2'>Main code : method1</span>
```python
from sympy import Matrix
#sympy.init_printing()

A = Matrix([[2,3],[5,4]])
b = Matrix([4,3])
x = A.solve(b)

print(x)
```
`OUTPUT` : Matrix([[-1], [2]])

<span class='frame2'>Main code : method1</span>
```python
import numpy as np
from scipy import linalg as la

A = np.array([[2,3],[5,4]])
b = np.array([4,3])
x = la.solve(A,b)

print(x)
```
`OUTPUT` : [-1.  2.]

<br><br><br>


<hr class="division2">

## **Rectangular Systems**
<div style="color:black; font-size: 80%; text-align: center;">
  $$Ax=b$$ 
  $$\begin{pmatrix}1 & 2 & 3 \\4 & 5 &6\end{pmatrix}\begin{pmatrix}x_{1} \\ x_{2} \\ x_{3} \end{pmatrix}=\begin{pmatrix}7 \\ 8 \end{pmatrix}$$
</div>

<span class='frame2'>Main code</span>

```python
from sympy import symbols, Matrix, solve
#sympy.init_printing()

x_vars = symbols("x_1, x_2, x_3")
A = Matrix([[1, 2, 3], [4, 5, 6]])
x = Matrix(x_vars)
b = Matrix([7, 8])
solution = solve(A*x - b, x_vars)

print(solution)
```
<br><br>

### ***Data fitting***

<div style="color:black; font-size: 80%; text-align: center;">
  $$\begin{pmatrix}1 & x_{1} & x_{1}^{2} \\\vdots & \vdots& \vdots \\1 & x_{m} & x_{m}^{2}\end{pmatrix}\begin{pmatrix}A \\B \\C \end{pmatrix}=\begin{pmatrix}y_{1} \\\vdots \\y_{m} \end{pmatrix} $$
</div>

<span class='frame2'>Main code</span>
```python
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

# define true model parameters
x = np.linspace(-1, 1, 100)
a, b, c = 1, 2, 3
y_exact = a + b * x + c * x**2

# simulate noisy data
m = 100
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + np.random.randn(m)

# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2])  # see np.vander for alternative
sol, r, rank, sv = la.lstsq(A.T, Y)

y_fit = sol[0] + sol[1] * x + sol[2] * x**2   
fig, ax = plt.subplots(figsize=(12, 4))  

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')   
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')  
ax.plot(x, y_fit, 'b', lw=2, label='Leat square fit')   
ax.set_xlabel(r"$x$", fontsize=18)    
ax.set_ylabel(r"$y$", fontsize=18)   
ax.legend(loc=2)
plt.show()
```
![Figure_1](https://user-images.githubusercontent.com/52376448/63553433-53d7ae00-c575-11e9-9b90-c2057b8860fc.png)
<br><br><br>

<span class='frame2'>Main code</span>
```python
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

# define true model parameters
x = np.linspace(-1, 1, 100)
a, b, c = 1, 2, 3
y_exact = a + b * x + c * x**2

# simulate noisy data
m = 100
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + np.random.randn(m)

# fit the data to the model using linear least square:    
# 1st order polynomial   
A = np.vstack([X**n for n in range(2)])   
sol, r, rank, sv = la.lstsq(A.T, Y)  
y_fit1 = sum([s * x**n for n, s in enumerate(sol)])   

# 15th order polynomial    
A = np.vstack([X**n for n in range(16)])    
sol, r, rank, sv = la.lstsq(A.T, Y)   
y_fit15 = sum([s * x**n for n, s in enumerate(sol)])  

fig, ax = plt.subplots(figsize=(12, 4))   
ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')   
ax.plot(x, y_fit1, 'b', lw=2, label='Least square fit [1st order]')    
ax.plot(x, y_fit15, 'm', lw=2, label='Least square fit [15th order]')   
ax.set_xlabel(r"$x$", fontsize=18)  
ax.set_ylabel(r"$y$", fontsize=18)   
ax.legend(loc=2)
plt.show()
```
![Figure_1](https://user-images.githubusercontent.com/52376448/63554103-7f5b9800-c577-11e9-8dd6-55d34a2eab6b.png)

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
