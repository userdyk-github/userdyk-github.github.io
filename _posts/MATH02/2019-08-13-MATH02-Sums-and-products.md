---
layout : post
title : MATH02, Sums and products
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

## Sum

### Sum

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Sum, oo

n = symbols("n", integer=True)
Sum(1/(n**2), (n, 1, oo))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \sum_{n=1}^{\infty} \frac{1}{n^{2}}$$</span>
<br><br><br>

---

### Evaluation for sum

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Sum, oo

n = symbols("n", integer=True)
x = Sum(1/(n**2), (n, 1, oo))
x.doit()
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{\pi^{2}}{6}$$</span>
<br><br><br>

<hr class="division2">

## Product

### Product

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Product

n = symbols("n", integer=True)
Product(n, (n, 1, 7))
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \prod_{n=1}^{7} n$$</span>
<br><br><br>

---

### Evaluation for product

```python
import sympy
#symypy.init_printing()
from sympy import symbols, Product

n = symbols("n", integer=True)
x = Product(n, (n, 1, 7))
x.doit()
```
`OUTPUT` : <span class='jb-small'>$$5040$$</span>
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
