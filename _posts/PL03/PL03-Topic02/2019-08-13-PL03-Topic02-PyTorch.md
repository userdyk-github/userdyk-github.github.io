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

print(x.detach().requires_grad)  # not in-place
print(x.requires_grad)

print(x.detach_().requires_grad) # in-place
print(x.requires_grad)
```
```
True

False
True

False
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
<summary class='jb-small' style="color:blue">to numpy</summary>
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
<br><br><br>

<a href="http://www.programmersought.com/article/81801261179/;jsessionid=848520548A8855A35D2F4B97F617EE2B" target="_blank">URL</a>
```python
import torch

b = torch.Tensor([[1,2,3],[4,5,6]])
print(b)

index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print(torch.gather(b, dim=1, index=index_1))   # 'dim = 1' means axis-column
print(torch.gather(b, dim=0, index=index_2))   # 'dim = 0' means axis-row
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[1., 2.],
        [6., 4.]])
tensor([[1., 5., 6.],
        [1., 2., 3.]])
```
<hr class='division3'>
</details>
<br>
```python
import torch

a = torch.randn(4,4)
indices = torch.LongTensor([0,2])

result1 = torch.index_select(a, 0, indices)
result2 = torch.index_select(a, 1, indices)
print("a",a)
print("dim=0(row[0:2]) \n", result1)
print("dim=1(column[0:2]) \n", result2)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
a tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
        [-0.7123,  0.1369, -0.3352,  1.5771],
        [ 1.2470,  0.5784, -0.1455,  1.5894],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
dim=0(row[0:2])
 tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
        [ 1.2470,  0.5784, -0.1455,  1.5894]])
dim=1(column[0:2])
 tensor([[-0.9946, -0.9979],
        [-0.7123, -0.3352],
        [ 1.2470, -0.1455],
        [ 0.4785,  0.2051]])
```
<hr class='division3'>
</details>
<br>
```python
import torch

a = torch.tensor([10, 0, 2, 0, 0])
non_zero = torch.nonzero(a)
print(non_zero)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
tensor([[0],
        [2]])
```
<hr class='division3'>
</details>
<br>
```python
import torch

a = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
split_2 = torch.split(a,2)
split_3 = torch.split(a,3)
print(split_2)
print(split_3)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
(tensor([11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18]), tensor([19, 20]))
(tensor([11, 12, 13]), tensor([14, 15, 16]), tensor([17, 18, 19]), tensor([20]))
```
<hr class='division3'>
</details>
<br><br><br>


```python
import torch

a = torch.tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
                  [-0.7123,  0.1369, -0.3352,  1.5771],
                  [ 1.2470,  0.5784, -0.1455,  1.5894],
                  [ 0.4785, -0.3342,  0.2051, -0.5731]])

print(a)
print(a.t())
print(a.transpose(1,0))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
        [-0.7123,  0.1369, -0.3352,  1.5771],
        [ 1.2470,  0.5784, -0.1455,  1.5894],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
tensor([[-0.9946, -0.7123,  1.2470,  0.4785],
        [ 0.9729,  0.1369,  0.5784, -0.3342],
        [-0.9979, -0.3352, -0.1455,  0.2051],
        [-1.1015,  1.5771,  1.5894, -0.5731]])
tensor([[-0.9946, -0.7123,  1.2470,  0.4785],
        [ 0.9729,  0.1369,  0.5784, -0.3342],
        [-0.9979, -0.3352, -0.1455,  0.2051],
        [-1.1015,  1.5771,  1.5894, -0.5731]])
```
<hr class='division3'>
</details>
<br><br><br>
```python
import torch

a = torch.tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
                  [-0.7123,  0.1369, -0.3352,  1.5771],
                  [ 1.2470,  0.5784, -0.1455,  1.5894],
                  [ 0.4785, -0.3342,  0.2051, -0.5731]])

print(a)
print(torch.unbind(a,1))    # dim = 1 removing a column
print(torch.unbind(a))      # dim = 0 removing a row
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
        [-0.7123,  0.1369, -0.3352,  1.5771],
        [ 1.2470,  0.5784, -0.1455,  1.5894],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
(tensor([-0.9946, -0.7123,  1.2470,  0.4785]), tensor([ 0.9729,  0.1369,  0.5784, -0.3342]), tensor([-0.9979, -0.3352, -0.1455,  0.2051]), tensor([-1.1015,  1.5771,  1.5894, -0.5731]))
(tensor([-0.9946,  0.9729, -0.9979, -1.1015]), tensor([-0.7123,  0.1369, -0.3352,  1.5771]), tensor([ 1.2470,  0.5784, -0.1455,  1.5894]), tensor([ 0.4785, -0.3342,  0.2051, -0.5731]))
```
<hr class='division3'>
</details>

<br><br><br>

#### mathematical functions
```python
import torch

a = torch.tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
                  [-0.7123,  0.1369, -0.3352,  1.5771],
                  [ 1.2470,  0.5784, -0.1455,  1.5894],
                  [ 0.4785, -0.3342,  0.2051, -0.5731]])

print("a\n", a)
print("add\n", torch.add(a,100))
print("mul\n", torch.mul(a,100))
print("ceil\n", torch.ceil(a))
print("floor\n", torch.floor(a))
print("clamp\n", torch.clamp(a, min=-0.8, max=0.8))
print("exp\n", torch.exp(a))
print("frac\n", torch.frac(a))
print("log\n", torch.log(a))
print("pow\n", torch.pow(a,2))
print("sigmoid\n", torch.sigmoid(a))
print("sqrt\n", torch.sqrt(a))
```
```
a
 tensor([[-0.9946,  0.9729, -0.9979, -1.1015],
        [-0.7123,  0.1369, -0.3352,  1.5771],
        [ 1.2470,  0.5784, -0.1455,  1.5894],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
add
 tensor([[ 99.0054, 100.9729,  99.0021,  98.8985],
        [ 99.2877, 100.1369,  99.6648, 101.5771],
        [101.2470, 100.5784,  99.8545, 101.5894],
        [100.4785,  99.6658, 100.2051,  99.4269]])
mul
 tensor([[ -99.4600,   97.2900,  -99.7900, -110.1500],
        [ -71.2300,   13.6900,  -33.5200,  157.7100],
        [ 124.7000,   57.8400,  -14.5500,  158.9400],
        [  47.8500,  -33.4200,   20.5100,  -57.3100]])
ceil
 tensor([[-0.,  1., -0., -1.],
        [-0.,  1., -0.,  2.],
        [ 2.,  1., -0.,  2.],
        [ 1., -0.,  1., -0.]])
floor
 tensor([[-1.,  0., -1., -2.],
        [-1.,  0., -1.,  1.],
        [ 1.,  0., -1.,  1.],
        [ 0., -1.,  0., -1.]])
clamp
 tensor([[-0.8000,  0.8000, -0.8000, -0.8000],
        [-0.7123,  0.1369, -0.3352,  0.8000],
        [ 0.8000,  0.5784, -0.1455,  0.8000],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
exp
 tensor([[0.3699, 2.6456, 0.3687, 0.3324],
        [0.4905, 1.1467, 0.7152, 4.8409],
        [3.4799, 1.7832, 0.8646, 4.9008],
        [1.6137, 0.7159, 1.2276, 0.5638]])
frac
 tensor([[-0.9946,  0.9729, -0.9979, -0.1015],
        [-0.7123,  0.1369, -0.3352,  0.5771],
        [ 0.2470,  0.5784, -0.1455,  0.5894],
        [ 0.4785, -0.3342,  0.2051, -0.5731]])
log
 tensor([[    nan, -0.0275,     nan,     nan],
        [    nan, -1.9885,     nan,  0.4556],
        [ 0.2207, -0.5475,     nan,  0.4634],
        [-0.7371,     nan, -1.5843,     nan]])
pow
 tensor([[0.9892, 0.9465, 0.9958, 1.2133],
        [0.5074, 0.0187, 0.1124, 2.4872],
        [1.5550, 0.3345, 0.0212, 2.5262],
        [0.2290, 0.1117, 0.0421, 0.3284]])
sigmoid
 tensor([[0.2700, 0.7257, 0.2694, 0.2495],
        [0.3291, 0.5342, 0.4170, 0.8288],
        [0.7768, 0.6407, 0.4637, 0.8305],
        [0.6174, 0.4172, 0.5511, 0.3605]])
sqrt
 tensor([[   nan, 0.9864,    nan,    nan],
        [   nan, 0.3700,    nan, 1.2558],
        [1.1167, 0.7605,    nan, 1.2607],
        [0.6917,    nan, 0.4529,    nan]])
```
<br><br><br>

---


### ***GPU control***
```python
import torch
 
#  Returns a bool indicating if CUDA is currently available.
torch.cuda.is_available()
#  True
 
#  Returns the index of a currently selected device.
torch.cuda.current_device()
#  0
 
#  Returns the number of GPUs available.
torch.cuda.device_count()
#  1
 
#  Gets the name of a device.
torch.cuda.get_device_name(0)
#  'GeForce GTX 1060'
 
#  Context-manager that changes the selected device.
#  device (torch.device or int) – device index to select. 
torch.cuda.device(0)
```
```python
import torch
 
# Default CUDA device
cuda = torch.device('cuda')
 
# allocates a tensor on default GPU
a = torch.tensor([1., 2.], device=cuda)
 
# transfers a tensor from 'C'PU to 'G'PU
b = torch.tensor([1., 2.]).cuda()
 
# Same with .cuda()
b2 = torch.tensor([1., 2.]).to(device=cuda)
```
<br><br><br>
<hr class="division2">

## **Probability Distributions**

### ***Sampling Tensors***

```python
import torch

torch.manual_seed(1234)
a = torch.randn(4,4)
print(a)
```
```
tensor([[-0.1117, -0.4966,  0.1631, -0.8817],
        [ 0.0539,  0.6684, -0.0597, -0.4675],
        [-0.2153,  0.8840, -0.7584, -0.3689],
        [-0.3424, -1.4020,  0.3206, -1.0219]])
```
<br><br><br>

```python
import torch

torch.manual_seed(1234)
a = torch.Tensor(4,4).uniform_(0,1)
print(a)
```
```
tensor([[0.0290, 0.4019, 0.2598, 0.3666],
        [0.0583, 0.7006, 0.0518, 0.4681],
        [0.6738, 0.3315, 0.7837, 0.5631],
        [0.7749, 0.8208, 0.2793, 0.6817]])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Variable Tensors***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Basic Statistics***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Gradient Computation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Tensor Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Tensor Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Distributions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>


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
<br><br><br>

---


### ***Estimating the Derivative of the Loss Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Fine-Tuning a Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Selecting an Optimization Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Further Optimizing the Function***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Implementing a Convolutional Neural Network (CNN)***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Reloading a Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Implementing a Recurrent Neural Network (RNN)***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Implementing a RNN for Regression Problems***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Using PyTorch Built-in Functions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Working with Autoencoders***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Fine-Tuning Results Using Autoencoder***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Visualizing the Encoded Data in a 3D Plot***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Restricting Model Overfitting***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Visualizing the Model Overfit***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Initializing Weights in the Dropout Rate***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Adding Math Operations***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Embedding Layers in RNN***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

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
<br><br><br>

---


### ***Visualizing the Shape of Activation Functions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Basic Neural Network Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Tensor Differentiation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

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
<br><br><br>

---


### ***Forward and Backward Propagation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Optimization and Gradient Computation***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Viewing Predictions***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Supervised Model Logistic Regression***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

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
<br><br><br>

---


### ***Deciding the Batch Size***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Deciding the Learning Rate***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Performing Parallel Training***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

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
<br><br><br>

---


### ***CBOW Model in PyTorch***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***LSTM Model***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Pradeepta Mishra, PyTorch Recipes, 2019
- <a href='https://wikidocs.net/book/2788' target="_blank">wikidocs, pytorch</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---




