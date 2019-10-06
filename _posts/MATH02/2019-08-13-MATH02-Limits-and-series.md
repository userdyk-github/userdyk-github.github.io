---
layout : post
title : MATH02, Limits and series
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

## Limits

### Limit of a function

```python
import sympy
#symypy.init_printing()
from sympy import symbols, limit, sin

x = symbols('x')
limit(sin(x) / x, x, 0)
```

`OUTPUT` : <span class='jb-small'>$$1$$</span>
<br><br><br>

---

### Derivatives using limit

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, limit, cos

f = Function('f')
x, h = symbols("x, h")

diff_limit = (f(x + h) - f(x))/h
limit(diff_limit.subs(f, cos), h, 0)
```

`OUTPUT` : <span class='jb-small'>$$- \sin{\left (x \right )}$$</span>
<br><br><br>

---

### Asymptotic behavior

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, limit, oo

f = Function('f')
x = symbols("x")

expr = (x**2 - 3*x) / (2*x - 2)
p = limit(expr/x, x, oo)
q = limit(expr - p*x, x, oo)

p, q
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left( \frac{1}{2}, \  -1\right)$$</span>
<br><br><br>

## Series for an unspecified function

### Maclaurin series

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, series

x = symbols("x")
f = Function("f")(x)

series(f, x, n=3)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle f{\left(0 \right)} + x \left. \frac{d}{d x} f{\left(x \right)} \right|_{\substack{ x=0 }} + \frac{x^{2} \left. \frac{d^{2}}{d x^{2}} f{\left(x \right)} \right|_{\substack{ x=0 }}}{2} + O\left(x^{3}\right)$$</span>
<br><br><br>

---

### Taylor series

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, series

x, x0 = symbols("x, {x_0}")
f = Function("f")(x)

f.series(x, x0, n=2)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle f{\left({x_0} \right)} + \left(x - {x_0}\right) \left. \frac{d}{d \xi_{1}} f{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={x_0} }} + O\left(\left(x - {x_0}\right)^{2}; x\rightarrow {x_0}\right)$$</span>
<br><br><br>

---

### Approximation on taylor series

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, series

x, x0 = symbols("x, {x_0}")
f = Function("f")(x)

f.series(x, x0, n=2).removeO()
```
`OUTPUT` : <span class='jb-small'>$$ f{\left({x_0} \right)} + \displaystyle \left(x - {x_0}\right) \left. \frac{d}{d \xi_{1}} f{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={x_0} }}$$</span>
<br><br><br>

<hr class="division2">

## Series for an specified function

### Univariate

```python
import sympy
#symypy.init_printing()
from sympy import symbols, cos, series

x = symbols("x")

cos(x).series(n=10)
```
`OUTPUT` : <span class='jb-small'>$$ \displaystyle 1 - \frac{x^{2}}{2} + \frac{x^{4}}{24} - \frac{x^{6}}{720} + \frac{x^{8}}{40320} + O\left(x^{10}\right)$$</span>
<br><br><br>

---

### Multivariate

```python
import sympy
#symypy.init_printing()
from sympy import symbols, cos, sin, series

x, y = symbols("x, y")

expr = cos(x) / (1 + sin(x * y))
X = expr.series(x, n=2)
Y = expr.series(y, n=2)

X, Y
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \left( 1 - x y + O\left(x^{2}\right), \  \cos{\left(x \right)} - x y \cos{\left(x \right)} + O\left(y^{2}\right)\right)$$</span>
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
