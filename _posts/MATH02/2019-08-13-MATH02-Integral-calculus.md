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

## integrate

### Indefinite integral

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, integrate

a, b, x, y = sympy.symbols("a, b, x, y")
f = sympy.Function("f")(x)

integrate(f)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int f{\left(x \right)}\, dx$$</span>
<br><br><br>

---

### Definite integral

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, integrate

a, b, x, y = sympy.symbols("a, b, x, y")
f = sympy.Function("f")(x)

integrate(f, (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int\limits_{a}^{b} f{\left(x \right)}\, dx$$</span>
<br><br><br>

<hr class="division2">

## title2

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
