---
layout : post
title : MATH02, Derivative calculus
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

## Derivative of an abstract function

### First-order derivative

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, diff

x = symbols('x')
f = Function('f')(x)

diff(f, x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{d}{d x} f{\left(x \right)}$$</span>
<br><br><br>

---

### Higher-order derivative

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, diff

x = symbols('x')
f = Function('f')(x)

diff(f, x, 3)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{d^{3}}{d x^{3}} f{\left(x \right)}$$</span>
<br><br><br>

<hr class="division2">

## Derivative of an specific function

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
