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

---

## title3

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
