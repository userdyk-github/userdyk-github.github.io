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
<div><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a32e22e6cb7bd6418442d4ab3af89ee1341aa102" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.505ex; width:23.637ex; height:6.176ex;" alt="{\displaystyle \chi _{A}(x)={\begin{cases}1&amp;{\text{if }}x\in A,\\0&amp;{\text{if }}x\notin A.\\\end{cases}}}"></div>
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
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9537e778e229470d85a68ee0b099c08298a1a3f6" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.171ex; width:26.95ex; height:5.509ex;" alt="{\displaystyle S(x)={\frac {1}{1+e^{-x}}}={\frac {e^{x}}{e^{x}+1}}.}">
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
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
  return 1/(1 + np.exp(-x))
def identity_function(x):
    return x
  
X = np.array([1.0, 0.5])

W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3)

Y = identity_function(A3)
Y
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([0.21682708, 0.49627909])
```
<hr class='division3'>
</details>
<br>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/63815273-f5dd0900-c96e-11e9-849a-7f59ff1aa6cb.JPG)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Summary for Implement***

```python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.2,0.3],[0.2,0.3,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.2],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[0.31067024 0.68352896]
```
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Design the output layer**

### ***Implement identity and softmax function***
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/46d00b0bc5d63bf06b74d6d34234e063e03a1d26" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.505ex; width:58.066ex; height:7.343ex;" alt="{\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{\beta z_{i}}}{\sum _{j=1}^{K}e^{\beta z_{j}}}}{\text{ or }}\sigma (\mathbf {z} )_{i}={\frac {e^{-\beta z_{i}}}{\sum _{j=1}^{K}e^{-\beta z_{j}}}}{\text{ for }}i=1,\dotsc ,K}">
```python
def softmax1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
a1 = np.array([1010,1000,990])
y1 = softmax1(a1)
print(y1)


def softmax2(a):
    c =np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
a2 = np.array([1010,1000,990])
y2 = softmax2(a2)
print(y2)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[nan nan nan]
[9.99954600e-01 4.53978686e-05 2.06106005e-09]
```
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
