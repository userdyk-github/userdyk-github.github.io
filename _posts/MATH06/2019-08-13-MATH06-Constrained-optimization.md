---
layout : post
title : MATH06, Constrained optimization
categories: [MATH06]
comments : true
tags : [MATH06]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[Optimization](https://userdyk-github.github.io/math06/MATH06-Contents.html) <br>
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

## **Bounded optimization problem with the L-BFGS-B algorithm**
<div style="font-size: 70%; text-align: center;">
    $$the\ objective\ function\ :\ f(x) = (x_{1}-1)^{2}-(x_{2}-1)^{2}$$
    $$s.t. \qquad 2<x_{1}<3,\ 0 \le x_{2} \le 2$$
</div>
```python
from scipy import optimize

# objective function
def f(X):   
    x, y = X   
    return (x - 1)**2 + (y - 1)**2 

# constraints
bnd_x1, bnd_x2 = (2, 3), (0, 2) 

# optimization of obejective function considering constraints
optimize.minimize(f, [1, 1], method='L-BFGS-B', 
                  bounds=[bnd_x1, bnd_x2]).x 
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

![다운로드](https://user-images.githubusercontent.com/52376448/65370629-0f375380-dc96-11e9-9e79-aba55cae09ee.png)
<hr class='division3'>
</details>

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


