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

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Sum, oo

n = symbols("n", integer=True)
x = Sum(1/(n**2), (n, 1, oo))
x
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \sum_{n=1}^{\infty} \frac{1}{n^{2}}$$</span>

---

### Evaluation for sum

`INPUT`
```python
import sympy
#symypy.init_printing()
from sympy import symbols, Sum, oo

n = symbols("n", integer=True)
x = Sum(1/(n**2), (n, 1, oo))
x.doit()
```
`OUTPUT` : <span class='jb-small'>$$\displaystyle \frac{\pi^{2}}{6}$$</span>

<hr class="division2">

## Product


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
