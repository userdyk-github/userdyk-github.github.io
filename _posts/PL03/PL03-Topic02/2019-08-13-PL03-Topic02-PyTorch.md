---
layout : post
title : PL03-Topic02, PyTorch
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/2019-08-13-PL03-Topic02-PyTorch.md" target="_blank">page management</a> <br>
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

## **Installation**
<a href="https://pytorch.org/get-started/locally/" target="_blank">URL</a>
<a href="https://anaconda.org/pytorch/pytorch">anaconda torch url</a>
<a href="https://anaconda.org/pytorch/torchvision">anaconda torchvision url</a>

### ***For linux***
```bash
$ 
```
<br><br><br>

### ***For windows***
```dos

```
<br><br><br>

### ***Version Control***
```python

```
<br><br><br>


<hr class="division2">

## **Tensor & Tensor operations**

### ***Using Tensors***
<a href="https://pytorch.org/docs/stable/tensors.html" target="_blank">torch.tensor api</a><br>
```python
import torch

x = [12,23,34,45,56,67,78]
print(torch.is_tensor(x))
print(torch.is_storage(x))
```
```
False
False
```
<br><br><br>
```python
import torch

x = torch.tensor([1])
print(torch.is_tensor(x))
print(torch.is_storage(x))
```
```
True
False
```
<details markdown="1">
<summary class='jb-small' style="color:blue">requires_grad</summary>
<hr class='division3'>
```python
import torch

x = torch.tensor([1])
print(x.requires_grad)
```
```
False
```
<br><br><br>
```python
import torch

x = torch.tensor([1], dtype=torch.float, requires_grad=True)
print(x.requires_grad)
print(x.detach().requires_grad)
```
```
True
False
```
<hr class='division3'>
</details>

<br><br><br>
```python
import torch

x = torch.randn(1,2,3,4,5)
print(torch.is_tensor(x))
print(torch.is_storage(x))
print(torch.numel(x))         #number of elements
```
```
True
False
120
```
<br><br><br>
```python
import torch

x = torch.eye(3,3)
print(x)
```
```
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
```
<br><br><br>
```python
import numpy as np
import torch

x = [12,23,34,45,56,67]
y = np.array(x)
z = torch.from_numpy(y)

print(x)
print(y)
print(z)
```
```
[12, 23, 34, 45, 56, 67]
[12 23 34 45 56 67]
tensor([12, 23, 34, 45, 56, 67], dtype=torch.int32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
import torch

x = torch.tensor([1])
print(x.numpy())
```
```
[1]
```
<hr class='division3'>
</details>

<br><br><br>
```python
import torch

x = torch.linspace(2,10,25)
y = torch.logspace(2,10,25)

print(x)
print(y)
```
```
tensor([ 2.0000,  2.3333,  2.6667,  3.0000,  3.3333,  3.6667,  4.0000,  4.3333,
         4.6667,  5.0000,  5.3333,  5.6667,  6.0000,  6.3333,  6.6667,  7.0000,
         7.3333,  7.6667,  8.0000,  8.3333,  8.6667,  9.0000,  9.3333,  9.6667,
        10.0000])
tensor([1.0000e+02, 2.1544e+02, 4.6416e+02, 1.0000e+03, 2.1544e+03, 4.6416e+03,
        1.0000e+04, 2.1544e+04, 4.6416e+04, 1.0000e+05, 2.1544e+05, 4.6416e+05,
        1.0000e+06, 2.1544e+06, 4.6416e+06, 1.0000e+07, 2.1544e+07, 4.6416e+07,
        1.0000e+08, 2.1544e+08, 4.6416e+08, 1.0000e+09, 2.1544e+09, 4.6416e+09,
        1.0000e+10])
```
<br><br><br>
```python
import torch

x = torch.rand(10)        # random numbers 10 from a uniform distribution between 0 and 1
y = torch.rand(4,5)       # random numbers 20 = 4*5 from a uniform distribution between 0 and 1
z = torch.randn(10)       # random numbers 10 from a normal distribution (0,1)
```
```
tensor([0.0329, 0.8617, 0.1021, 0.3931, 0.8998, 0.8649, 0.1870, 0.9334, 0.5804,
        0.9534])
tensor([[0.1078, 0.4410, 0.2292, 0.3280, 0.2127],
        [0.0472, 0.0099, 0.0181, 0.4200, 0.0257],
        [0.6366, 0.9422, 0.1212, 0.1833, 0.1107],
        [0.3173, 0.8371, 0.5419, 0.5221, 0.0068]])
tensor([ 0.2746, -0.8012,  0.7291, -1.0866,  1.3591,  0.3519,  1.3433,  0.1243,
         0.0065,  0.1567])
```
<br><br><br>

```python
import torch

x = torch.randperm(10)      # random permutation
y = torch.arange(10,40,2)   # step size = 2
z = torch.arange(10,40)     # step size = 1
```
```
tensor([8, 6, 0, 4, 9, 7, 5, 3, 1, 2])
tensor([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
```
<br><br><br>
```python
import torch

x = torch.randn(4,5)


print(x)
print(torch.argmin(x))
print(torch.argmin(x, dim=1))

print(torch.argmax(x))
print(torch.argmax(x, dim=1))
```
```
tensor([[-0.6006,  0.5420, -0.7122,  0.8044,  0.5344],
        [ 0.1702, -0.2696, -0.3626,  0.5435,  0.9020],
        [ 0.5961, -0.7445, -0.3796, -0.6009,  1.2564],
        [ 0.7729, -1.9188, -0.3456,  0.3841, -0.0653]])
tensor(16)
tensor([2, 2, 1, 1])
tensor(14)
tensor([3, 4, 4, 0])
```
<br><br><br>
```python
import torch

x = torch.zeros(4,5)
y = torch.zeros(10)

print(x)
print(y)
```
```
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```
<br><br><br>
```python
import torch

x = torch.randn(1,4)
p = torch.cat((x,x))
q = torch.cat((x,x),0)
r = torch.cat((x,x,x), 1)


print(x)
print(p)
print(q)
print(r)
```
```
tensor([[ 0.2394, -2.9119,  0.1089,  0.6426]])
tensor([[ 0.2394, -2.9119,  0.1089,  0.6426],
        [ 0.2394, -2.9119,  0.1089,  0.6426]])
tensor([[ 0.2394, -2.9119,  0.1089,  0.6426],
        [ 0.2394, -2.9119,  0.1089,  0.6426]])
tensor([[ 0.2394, -2.9119,  0.1089,  0.6426,  0.2394, -2.9119,  0.1089,  0.6426,
          0.2394, -2.9119,  0.1089,  0.6426]])
```
<br><br><br>
```python
import torch

x = torch.randn(4,4)
p = torch.chunk(x, 2)
q = torch.chunk(x,2,0)
r = torch.chunk(x,2,1)

print(x)
print(p)
print(q)
print(r)
```
```
tensor([[-0.7438, -0.2451,  0.2383,  0.0779],
        [-1.3219, -0.2667,  0.1635,  1.2190],
        [ 1.0349,  0.6819,  0.9239,  0.8569],
        [-2.8974, -0.5763, -0.2475, -0.8700]])
(tensor([[-0.7438, -0.2451,  0.2383,  0.0779],
        [-1.3219, -0.2667,  0.1635,  1.2190]]), tensor([[ 1.0349,  0.6819,  0.9239,  0.8569],
        [-2.8974, -0.5763, -0.2475, -0.8700]]))
(tensor([[-0.7438, -0.2451,  0.2383,  0.0779],
        [-1.3219, -0.2667,  0.1635,  1.2190]]), tensor([[ 1.0349,  0.6819,  0.9239,  0.8569],
        [-2.8974, -0.5763, -0.2475, -0.8700]]))
(tensor([[-0.7438, -0.2451],
        [-1.3219, -0.2667],
        [ 1.0349,  0.6819],
        [-2.8974, -0.5763]]), tensor([[ 0.2383,  0.0779],
        [ 0.1635,  1.2190],
        [ 0.9239,  0.8569],
        [-0.2475, -0.8700]]))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division2">

## **Probability Distributions**

### ***Sampling Tensors***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Variable Tensors***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Basic Statistics***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Gradient Computation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Tensor Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Tensor Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Distributions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


<hr class="division2">

## **CNN and RNN**

### ***Setting Up a Loss Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Estimating the Derivative of the Loss Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Fine-Tuning a Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Selecting an Optimization Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Further Optimizing the Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Implementing a Convolutional Neural Network (CNN)***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Reloading a Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Implementing a Recurrent Neural Network (RNN)***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Implementing a RNN for Regression Problems***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Using PyTorch Built-in Functions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Working with Autoencoders***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Fine-Tuning Results Using Autoencoder***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Visualizing the Encoded Data in a 3D Plot***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Restricting Model Overfitting***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Visualizing the Model Overfit***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Initializing Weights in the Dropout Rate***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Adding Math Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Embedding Layers in RNN***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division2">

## **Neural Networks**

### ***Working with Activation Functions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Visualizing the Shape of Activation Functions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Basic Neural Network Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Tensor Differentiation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division2">

## **Supervised Learning**

### ***Data Preparation for the Supervised Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Forward and Backward Propagation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Optimization and Gradient Computation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Viewing Predictions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Supervised Model Logistic Regression***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division2">

## **Fine-Tuning Deep Learning Models**

### ***Building Sequential Neural Networks***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Deciding the Batch Size***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Deciding the Learning Rate***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***Performing Parallel Training***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division2">

## **Natural Language Processing**

### ***Word Embedding***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***CBOW Model in PyTorch***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


### ***LSTM Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

---


<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Pradeepta Mishra, PyTorch Recipes, 2019
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---




