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
import numpy as np

# definition of derivative
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# test
def f(x):
    return 0.01*x**2 + 0.1*x
x = np.linspace(-1,1,100)

numerical_diff(f, x)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([0.08      , 0.08040404, 0.08080808, 0.08121212, 0.08161616,
       0.0820202 , 0.08242424, 0.08282828, 0.08323232, 0.08363636,
       0.0840404 , 0.08444444, 0.08484848, 0.08525253, 0.08565657,
       0.08606061, 0.08646465, 0.08686869, 0.08727273, 0.08767677,
       0.08808081, 0.08848485, 0.08888889, 0.08929293, 0.08969697,
       0.09010101, 0.09050505, 0.09090909, 0.09131313, 0.09171717,
       0.09212121, 0.09252525, 0.09292929, 0.09333333, 0.09373737,
       0.09414141, 0.09454545, 0.09494949, 0.09535354, 0.09575758,
       0.09616162, 0.09656566, 0.0969697 , 0.09737374, 0.09777778,
       0.09818182, 0.09858586, 0.0989899 , 0.09939394, 0.09979798,
       0.10020202, 0.10060606, 0.1010101 , 0.10141414, 0.10181818,
       0.10222222, 0.10262626, 0.1030303 , 0.10343434, 0.10383838,
       0.10424242, 0.10464646, 0.10505051, 0.10545455, 0.10585859,
       0.10626263, 0.10666667, 0.10707071, 0.10747475, 0.10787879,
       0.10828283, 0.10868687, 0.10909091, 0.10949495, 0.10989899,
       0.11030303, 0.11070707, 0.11111111, 0.11151515, 0.11191919,
       0.11232323, 0.11272727, 0.11313131, 0.11353535, 0.11393939,
       0.11434343, 0.11474747, 0.11515152, 0.11555556, 0.1159596 ,
       0.11636364, 0.11676768, 0.11717172, 0.11757576, 0.1179798 ,
       0.11838384, 0.11878788, 0.11919192, 0.11959596, 0.12      ])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">CAUTION</summary>
<hr class='division3'>
```
>>> import numpy as np
>>> np.float32(1e-50)
0.0
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
