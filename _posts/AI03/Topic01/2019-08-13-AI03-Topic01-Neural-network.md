---
layout : post
title : AI03-Topic01, Neural network
categories: [AI03-Topic01]
comments : true
tags : [AI03-Topic01]
---
[Back to the previous page](https://userdyk-github.github.io/ai03/AI03-Fundamental-of-deep-learning.html) <br>
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

## **Activation function**

### ***Implement the step function***

```python
import numpy as np


def step_function1(x):
  if x > 0:
    return 1
  else:
    return 0
x1 = 10.0
print(step_function1(x1))


def step_function2(x):
  y = x > 0
  return y.astype(np.int)
x2 = np.array([-10.0, 5.0, 10.0])
print(step_function2(x2))
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  1<br>
  [0 1 1]
</p>
<hr class='division3'>
</details>
<br>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
>>> import numpy as np
>>> np.array([-10.0, 10.0, 10.0])
array([-10.0, 10.0, 10.0])

>>> np.array([-10.0, 10.0, 10.0])>0
array([False,  True])

>>> np.array([False, True]).astype(np.int)
array([0, 1])
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***Graph of the step function***

```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/63807679-26b24380-c959-11e9-8acd-acacbefec72c.png)
<hr class='division3'>
</details>
<br>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
>>> import numpy as np
>>> def step_function(x):
...   return np.array(x > 0, dtype=np.int)

>>> x = np.array([-10, 10, 20])
>>> step_function(x)
array([0, 1, 1])

>>> np.arange(-5.0, 5.0, 1)
array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***Implement the sigmoid function***

```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
  return 1/(1 + np.exp(-x))

a = np.array([-1.0, 1.0, 2.0])
print(sigmoid(a))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[0.26894142 0.73105858 0.88079708]
```
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/63809030-44cd7300-c95c-11e9-9ef3-58ecac07ea93.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***ReLU function***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Calculation of multidimensional array**

### ***Multidimensional array***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Matrix multiplication***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Matrix multiplication in Neural Networks***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Implement a three-layer neural network**

### ***Implement a neuronal signal transduction at each layer***

```python
X = np.array([1.0, 0.5])   #
W1 = np.array()            #
B1 = np.array()       

A1 = np.dot(X, W1) + B1
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Summary for Implement***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Design the output layer**

### ***Implement identity and softmax function***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Cautions for implementing softmax function***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Characteristics of Softmax function***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Handwriting number recognition**

### ***MNIST dataset***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Inference processing of neural network***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Batch processing***

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
