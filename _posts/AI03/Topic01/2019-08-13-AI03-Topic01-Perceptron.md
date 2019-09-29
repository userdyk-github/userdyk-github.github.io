---
layout : post
title : AI03-Topic01, Perceptron
categories: [AI03-Topic01]
comments : true
tags : [AI03-Topic01]
---
[Back to the previous page](https://userdyk-github.github.io/ai03/AI03-Fundamental-of-deep-learning.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Python Libraries</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **What is Perceptron?**

<hr class="division2">

## **Simple logic circuit**

### ***AND Gate***

```python
import pandas as pd

AND = pd.DataFrame({'$x_{1}$':[0,1,0,1], '$x_{2}$':[0,0,1,1], 'AND':[0,0,0,1]})
AND.set_index(['$x_{1}$','$x_{2}$'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***NAND Gate***

```python
import pandas as pd

NAND = pd.DataFrame({'$x_{1}$':[0,1,0,1], '$x_{2}$':[0,0,1,1], 'NAND':[1,1,1,0]})
NAND.set_index(['$x_{1}$','$x_{2}$'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***OR Gate***

```python
import pandas as pd

OR = pd.DataFrame({'$x_{1}$':[0,1,0,1], '$x_{2}$':[0,0,1,1], 'OR':[0,0,0,1]})
OR.set_index(['$x_{1}$','$x_{2}$'])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Implementing Perceptron**

### ***From a simple implementation***

```python
def AND(x1,x2):
    w1,w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print('AND(0,0) : ',AND(0,0))
print('AND(1,0) : ',AND(1,0))
print('AND(0,1) : ',AND(0,1))
print('AND(1,1) : ',AND(1,1))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Institute weights and bias***

```python
import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])

b = -0.7
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
w * x
```
```
array([0. , 0.5])
```
<br>
```python
np.sum(w*x)
```
```
0.5
```
<br>
```python
np.sum(w*x) + b
```
```
-0.19999999999999996
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement weights and bias***

```python
import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tem <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tem = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = - 0.2
    tem = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

<hr class="division2">

## **Limit of Percentron**

<hr class="division2">

## **What if the multilayer perceptron crashes?**

<hr class="division2">

## **From NAND to Computer**

<hr class="division2">

## **Reference Codes**

### ******
```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
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


<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


