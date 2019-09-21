---
layout : post
title : MATH06, Linear programming
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

## Linear programming
The solution to a linear optimization problem must necessarily lie on a constraint boundary, so it is sufficient to search the vertices of the intersections of the linear constraint functions.
- Simplex algorithm
<div style="font-size: 70%; text-align: center;">standard form for linear programming : $$min_{x}c^{T}x$$</div>
```python
import numpy as np
import cvxopt

c = np.array([-1.0, 2.0, -3.0])
A = np.array([[ 1.0, 1.0, 0.0],
              [-1.0, 3.0, 0.0],
              [ 0.0, -1.0, 1.0]])
b = np.array([1.0, 2.0, 3.0])
A_ = cvxopt.matrix(A)  
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)

cvxopt.solvers.lp(c_, A_, b_)
```
`OUTPUT`
```
Optimal solution found.
{'dual infeasibility': 1.4835979218054372e-16,
 'dual objective': -10.0,
 'dual slack': 0.0,
 'gap': 0.0,
 'iterations': 0,
 'primal infeasibility': 0.0,
 'primal objective': -10.0,
 'primal slack': -0.0,
 'relative gap': 0.0,
 'residual as dual infeasibility certificate': None,
 'residual as primal infeasibility certificate': None,
 's': <3x1 matrix, tc='d'>,
 'status': 'optimal',
 'x': <3x1 matrix, tc='d'>,
 'y': <0x1 matrix, tc='d'>,
 'z': <3x1 matrix, tc='d'>}
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`INPUT`
```python
x = np.array(sol['x'])
x
```
`OUTPUT`
```
array([[ 0.25],
       [ 0.75],
       [ 3.75]])
```
<br>
`INPUT`
```python
sol['primal objective']
```
`OUTPUT`
```
-10.0
```

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



