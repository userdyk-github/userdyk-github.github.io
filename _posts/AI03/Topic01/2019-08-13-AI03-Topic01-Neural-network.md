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
A3 = np.dot(Z2, W3) + B3

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
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    
a1 = np.array([0.3,2.9,4.0])
a2 = np.array([1010,1000,990])
y1 = softmax(a1)
y2 = softmax(a2)
print(y1, y2)

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[0.01821127 0.24519181 0.73659691] [nan nan nan]
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
>>> import numpy as np
>>> a = np.array([1010,1000,990])
>>> np.exp(a)/np.sum(np.exp(a))
array([nan, nan, nan])
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Cautions for implementing softmax function***

```python
def softmax(a):
    c =np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    
a1 = np.array([0.3,2.9,4.0])
a2 = np.array([1010,1000,990])
y1 = softmax(a1)
y2 = softmax(a2)
print(y1, y2)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[0.01821127 0.24519181 0.73659691] [9.99954600e-01 4.53978686e-05 2.06106005e-09]
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
>>> import numpy as np
>>> a = np.array([1010,1000,990])
>>> c = np.max(a)
>>> a - c
array([  0, -10, -20])

>>> np.exp(a-c)/np.sum(np.exp(a-c))
array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Characteristics of Softmax function***

```
>>> import numpy as np
>>> def softmax(a):
...   c =np.max(a)
...   exp_a = np.exp(a-c)
...   sum_exp_a = np.sum(exp_a)
...   y = exp_a / sum_exp_a
...   return y

>>> a = np.array([0.3,2.9,4.0])
>>> y = softmax(a)
>>> print(y)
[0.01821127 0.24519181 0.73659691]

>>> np.sum(y)
1.0
```
<br><br><br>


<hr class="division2">

## **Handwriting number recognition**

### ***MNIST dataset***

<details markdown="1">
<summary class='jb-small' style="color:blue">ADVANCDED PREPERATION</summary>
<hr class='division3'>
[dataset.zip][1]

[1]:{{ site.url }}/download/AI03/AI03-Topic01/dataset.zip
<hr class='division3'>
</details>


```python
import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.unit8(img))
    pil_img.show()


# load dataset
(x_train, t_train),(x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 0th sample test
img, label = x_train[0], t_train[0]
img = img.reshape(28, 28)
img_show(img)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/63860073-71789d80-c9e3-11e9-9b49-44daf7e1ad4a.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```
>>> from dataset.mnist import load_mnist
>>> (x_train, t_train),(x_test, t_test) = load_mnist(flatten=True, normalize=False)

>>> type(x_train)
<class 'numpy.ndarray'>
>>> type(t_train)
<class 'numpy.ndarray'>
>>> type(x_test)
<class 'numpy.ndarray'>
>>> type(t_test)
<class 'numpy.ndarray'>

>>> x_train.shape
(60000, 784)
>>> t_train.shape
(60000,)
>>> x_test.shape
(10000, 784)
>>> t_test.shape
(10000,)

>>> x_train.dtype
dtype('uint8')
>>> t_train.dtype
dtype('uint8')
>>> x_test.dtype
dtype('uint8')
>>> t_test.dtype
dtype('uint8')
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Inference processing of neural network***
<details markdown="1">
<summary class='jb-small' style="color:blue">ADVANCDED PREPERATION</summary>
<hr class='division3'>
[dataset.zip][1] <br>
[common.zip][2] <br>
[sample_weight.pkl][3]


[1]:{{ site.url }}/download/AI03/AI03-Topic01/dataset.zip
[2]:{{ site.url }}/download/AI03/AI03-Topic01/common.zip
[3]:{{ site.url }}/download/AI03/AI03-Topic01/sample_weight.pkl
<hr class='division3'>
</details>

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y
    
 
accuracy_cnt = 0
x, t = get_data()
network = init_network()

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # getting a index with the higher probability
    if p == t[i] :
        accuracy_cnt += 1
        
print('Accuracy :' + str(float(accuracy_cnt)/len(x)))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Accuracy :0.9352
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```
>>> from dataset.mnist import load_mnist
>>> def get_data():
...     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
...     return x_test, t_test
>>> x, t = get_data()

>>> type(x)
<class 'numpy.ndarray'>
>>> type(x[0])
<class 'numpy.ndarray'>
>>> type(t)
<class 'numpy.ndarray'>
>>> type(t[0])
<class 'numpy.uint8'>

>>> x.shape
(10000, 784)
>>> len(x)
10000
>>> x[0].shape
(784,)
>>> t.shape
(10000,)
>>> t[0].shape
()
>>> t[0]
7

>>> import pickle
>>> def init_network():
...     with open('sample_weight.pkl', 'rb') as f:
...             network = pickle.load(f)
...     return network

>>> network = init_network()
>>> type(network)
<class 'dict'>
>>> network.keys()
dict_keys(['b2', 'W1', 'b1', 'W2', 'W3', 'b3'])

>>> network.get('b1')
array([-0.06750315,  0.0695926 , -0.02730473,  0.02256093, -0.22001474,
       -0.22038847,  0.04862635,  0.13499236,  0.23342554, -0.0487357 ,
        0.10170191, -0.03076038,  0.15482435,  0.05212503,  0.06017235,
       -0.03364862, -0.11218343, -0.26460695, -0.03323386,  0.13610415,
        0.06354368,  0.04679805, -0.01621654, -0.05775835, -0.03108677,
        0.10366164, -0.0845938 ,  0.11665157,  0.21852103,  0.04437255,
        0.03378392, -0.01720384, -0.07383765,  0.16152057, -0.10621249,
       -0.01646949,  0.00913961,  0.10238428,  0.00916639, -0.0564299 ,
       -0.10607515,  0.09892716, -0.07136887, -0.06349134,  0.12461706,
        0.02242282, -0.00047972,  0.04527043, -0.15179175,  0.10716812],
      dtype=float32)
      
>>> network.get('W1')
array([[-0.00741249, -0.00790439, -0.01307499, ...,  0.01978721,
        -0.04331266, -0.01350104],
       [-0.01029745, -0.01616653, -0.01228376, ...,  0.01920228,
         0.02809811,  0.01450908],
       [-0.01309184, -0.00244747, -0.0177224 , ...,  0.00944778,
         0.01387301,  0.03393568],
       ...,
       [ 0.02242565, -0.0296145 , -0.06326169, ..., -0.01012643,
         0.01120969,  0.01027199],
       [-0.00761533,  0.02028973, -0.01498873, ...,  0.02735376,
        -0.01229855,  0.02407041],
       [ 0.00027915, -0.06848375,  0.00911191, ..., -0.03183098,
         0.00743086, -0.04021148]], dtype=float32)    
         
```
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
