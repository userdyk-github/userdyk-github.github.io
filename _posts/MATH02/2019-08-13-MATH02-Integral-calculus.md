---
layout : post
title : MATH02, Integral calculus
categories: [MATH02]
comments : true
tags : [MATH02]
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

## Indefinite integral

### Single-variable function

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, integrate

x = symbols("x")
f = Function("f")(x)

integrate(f)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int f{\left(x \right)}\, dx$$</span>
<br><br><br>

---

### Explicit function

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, integrate

x = symbols("x")

integrate(sin(x))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle - \cos{\left(x \right)}$$</span>
<br><br><br>
 


<hr class="division2">

## Definite integral

### Single-variable function

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, integrate

a, b, x, y = symbols("a, b, x, y")
f = Function("f")(x)

integrate(f, (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int\limits_{a}^{b} f{\left(x \right)}\, dx$$</span>
<br><br><br>

---

### Explicit function

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, integrate

a, b, x, y = symbols("a, b, x, y")

integrate(sin(x), (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \cos{\left(a \right)} - \cos{\left(b \right)}$$</span>
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
