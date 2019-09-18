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

```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, integrate

x = symbols("x")

integrate(sin(x))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle - \cos{\left(x \right)}$$</span>
<br><br><br>
 
---

### Fail to evaluate an integral, Representing the formal integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, ,cos, integrate

x = symbols("x")

integrate(sin(x * cos(x)))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int \sin{\left(x \cos{\left(x \right)} \right)}\, dx$$</span>
<br><br><br>

---

### Symbolically represent an integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Integral

x = symbols("x")

Integral(x)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int x\, dx$$</span>
<br><br><br>

---

### Evalutation for an integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Integral

x = symbols("x")

i = Integral(x)
i.doit()
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{x^{2}}{2}$$</span>
<br><br><br>

---


### Integral of a multivariable expression

```python
import sympy
#symypy.init_printing()
from sympy import symbols, integrate

x, y = symbols("x, y")
expr = (x + y)**2

integrate(expr, x, y)
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{x^{3} y}{3} + \frac{x^{2} y^{2}}{2} + \frac{x y^{3}}{3}$$</span>
<br><br><br>


<hr class="division2">

## Definite integral

### Single-variable function

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Function, integrate

a, b, x = symbols("a, b, x")
f = Function("f")(x)

integrate(f, (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int\limits_{a}^{b} f{\left(x \right)}\, dx$$</span>
<br><br><br>

---

### Explicit function

```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, integrate

a, b, x = symbols("a, b, x")

integrate(sin(x), (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \cos{\left(a \right)} - \cos{\left(b \right)}$$</span>
<br><br><br>

---

### Infinity

```python
import sympy
#symypy.init_printing()
from sympy import symbols, exp, integrate, oo

x = symbols("x")

integrate(exp(-x**2), (x, 0, oo))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{\sqrt{\pi}}{2}$$</span>
<br><br><br>

---

### Fail to evaluate an integral, Representing the formal integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, sin, cos, integrate, oo

a, b, x = symbols("a, b, x")

integrate(sin(x * cos(x)),(x,a,b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int\limits_{a}^{b} \sin{\left(x \cos{\left(x \right)} \right)}\, dx$$</span>
<br><br><br>

---

### Symbolically represent an integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Integral

a, b, x = symbols("a, b, x")

Integral(x, (x, a, b))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \int\limits_{a}^{b} x\, dx$$</span>
<br><br><br>

---

### Evalutation for an integral

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Integral

a, b, x = symbols("a, b, x")

i = Integral(x, (x, a, b))
i.doit()
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle - \frac{a^{2}}{2} + \frac{b^{2}}{2}$$</span>
<br><br><br>

---

### Integral of a multivariable expression

```python
import sympy
#symypy.init_printing()
from sympy import symbols, integrate

x, y = symbols("x, y")
expr = (x + y)**2

integrate(expr, (x, 0, 1), (y, 0, 1))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{7}{6}$$</span>
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
