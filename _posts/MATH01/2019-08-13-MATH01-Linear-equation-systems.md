---
layout : post
title : MATH01, Linear equation systems
categories: [MATH01]
comments : true
tags : [MATH01]
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

---

$$Ax=b$$

## Square Systems

$$\begin{pmatrix}2 & 3 \\5 & 4\end{pmatrix}x=\begin{pmatrix}4 \\ 3 \end{pmatrix}$$

<div class='frame1'>Main code : symbolic method</div>
```python
from sympy import Matrix

A = Matrix([[2,3],[5,4]])
b = Matrix([4,3])
x = A.solve(b)

print(x)
```


<div class='frame1'>Main code : numerical method</div>
```python
import numpy as np
from scipy import linalg as la

A = np.array([[2,3],[5,4]])
b = np.array([4,3])
x = la.solve(A,b)

print(x)
```



---

## Rectangular Systems

$$\begin{pmatrix}1 & 2 & 3 \\4 & 5 &6\end{pmatrix}
\begin{pmatrix}x_{1} \\ x_{2} \\ x_{3} \end{pmatrix}
=\begin{pmatrix}7 \\ 8 \end{pmatrix}$$

<div class='frame1'>Main code</div>

```python
from sympy import symbols, Matrix, solve

x_vars = symbols("x_1, x_2, x_3")
A = Matrix([[1, 2, 3], [4, 5, 6]])
x = Matrix(x_vars)
b = Matrix([7, 8])
solution = solve(A*x - b, x_vars)

print(solution)
```

### data fitting

$$\begin{pmatrix}1 & x_{1} & x_{1}^{2} \\\vdots & \vdots& \vdots \\1 & x_{m} & x_{m}^{2}\end{pmatrix}\begin{pmatrix}A \\B \\C \end{pmatrix}=\begin{pmatrix}y_{1} \\\vdots \\y_{m} \end{pmatrix} $$

<div class='frame1'>Main code : symbolic method</div>
```python
import numpy as np
from scipy import linalg as la

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
```


---

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
