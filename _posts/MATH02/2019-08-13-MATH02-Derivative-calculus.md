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

## diff : Derivative of an abstract function

### First-order derivative

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, diff

x = symbols('x')
f = Function('f')(x)

f.diff(x)              # equivalent to diff(f, x)
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

f.diff(x, 3)           # equivalent to f.diff(x, x, x)
                       # equivalent to diff(f, x, 3)
                       # equivalent to diff(f, x, x, x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{d^{3}}{d x^{3}} f{\left(x \right)}$$</span>
<br><br><br>

---

### Derivative of multivariate functions

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, diff

x, y = symbols('x, y')
g = sympy.Function('g')(x, y)

g2 = g.diff(x, y)           # equivalent to diff(g, x, y)
g5 = g.diff(x, 3, y, 2)     # equivalent to g.diff(x, x, x, y, y)
                            # equivalent to diff(g, x, 3, y, 2)
                            # equivalent to diff(g, x, x, x, y, y)

g2, g5
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \left( \frac{\partial^{2}}{\partial y\partial x} g{\left(x,y \right)}, \  \frac{\partial^{5}}{\partial y^{2}\partial x^{3}} g{\left(x,y \right)}\right)$$</span>
<br><br><br>

<hr class="division2">

## diff : Derivative of an specific function

### Polynomials

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, diff

x = symbols('x')
expr = x**4 + x**3 + x**2 + x + 1
expr.diff(x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle 4 x^{3} + 3 x^{2} + 2 x + 1$$</span>
<br><br><br>

---

### Trigonometric

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, diff, cos, sin

x, y = symbols('x, y')
expr = sin(x * y) * cos(x / 2)
expr.diff(x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle y \cos{\left(\frac{x}{2} \right)} \cos{\left(x y \right)} - \frac{\sin{\left(\frac{x}{2} \right)} \sin{\left(x y \right)}}{2}$$</span>
<br><br><br>

---

### Special function

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, diff

x = symbols('x')
expr = sympy.special.polynomials.hermite(x, 0)
expr.diff(x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{2^{x} \sqrt{\pi} \operatorname{polygamma}{\left(0,\frac{1}{2} - \frac{x}{2} \right)}}{2 \Gamma\left(\frac{1}{2} - \frac{x}{2}\right)} + \frac{2^{x} \sqrt{\pi} \log{\left(2 \right)}}{\Gamma\left(\frac{1}{2} - \frac{x}{2}\right)}$$</span>
<br><br><br>

<hr class="division2">

## Derivative

### Symbolically represent a derivative

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Derivative, exp, cos

x = symbols('x')
Derivative(exp(cos(x)), x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{d}{d x} e^{\cos{\left(x \right)}}$$</span>
<br><br><br>

---

### Evalutation for a derivative

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Derivative, exp, cos

x = symbols('x')
d = Derivative(exp(cos(x)), x)
d.doit()
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle - e^{\cos{\left(x \right)}} \sin{\left(x \right)}$$</span>
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
