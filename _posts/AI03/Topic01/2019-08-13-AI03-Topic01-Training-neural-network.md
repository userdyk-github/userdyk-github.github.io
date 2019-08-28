---
layout : post
title : AI03-Topic01, Training neural network
categories: [AI03-Topic01]
comments : true
tags : [AI03-Topic01]
---
[Back to the previous page](https://userdyk-github.github.io/ai03/AI03-Fundamental-of-deep-learning.html) <br>
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

## **Loss function**

### ***Mean squared error, MSE***
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:24.729ex; height:6.843ex;" alt="{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}">
```python
import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)
    
y = np.array([1,2,3])
t = np.array([3,4,7])

mean_squared_error(y,t)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
12.0
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Cross entropy error, CEE***
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c6b895514e10a3ce88773852cba1cb1e248ed763" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:28.839ex; height:5.676ex;" alt="{\displaystyle H(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)}">
```python
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
    
y = np.array([0,2,3,8,9])
t = np.array([3,4,7,8,9])

cross_entropy_error(y,t)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
1.4808580471604245
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Mini batch training***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement cross entropy error***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">




## **Neumerical derivative**

### ***Derivative***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Example for numerical derivative***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Partial derivative***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">





## **Gradient**

### ***Gradient descent method***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Gradient at neural network***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">




## **Implement Learning Algorithms**

### ***Implement two layer neural network class***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement mini batch training***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Evaluation on test dataset***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
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
