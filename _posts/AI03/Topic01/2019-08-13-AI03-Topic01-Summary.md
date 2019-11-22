---
layout : post
title : AI03-Topic01, Summary
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


## **Regression**
### ***Linear regression***
![image](https://user-images.githubusercontent.com/52376448/69369878-291a0480-0ce0-11ea-8615-28ce7d19a464.png)
![image](https://user-images.githubusercontent.com/52376448/69403808-e0456880-0d3e-11ea-9764-7a88e0a4f342.png)

```python
class LinearNeuron:
    def __init__(self):
        self.w = None
        self.b = None
    
    def forpass(self, x):
        y_hat = np.sum(x*self.w) + self.b
        return y_hat
    
    def backprop(self, x, err_p):
        w_grad = x*err_p
        b_grad = 1*err_p
        return w_grad, b_grad
    
    def fit(self,x,y,epochs=100, learning_rate=0.00001):
        self.w = np.ones(x.shape[1])            # x.shape[1] : dimension of dataset
        self.b = 1
        for _ in range(epochs):
            for x_i, y_i in zip(x,y):
                y_hat = self.forpass(x_i)
                err_p = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= learning_rate*w_grad
                self.b -= learning_rate*b_grad
                #print(self.w)
```
```python
import numpy as np

x = np.linspace(0,100,10000).reshape(10000,1)
y = lambda x : 3*x + 5

a = LinearNeuron()
a.fit(x,y(x))
```
<br><br><br>

### ***Logistic regression***
![image](https://user-images.githubusercontent.com/52376448/69402086-28ae5780-0d3a-11ea-9524-632ce29de793.png)
![image](https://user-images.githubusercontent.com/52376448/69403870-0c60e980-0d3f-11ea-95a9-96b0ce2b4cf5.png)

```python
class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x, err_p):
        w_grad = x*err_p
        b_grad = 1*err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1/(1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, learning_rate=1):
        self.w = np.ones(x.shape[1])            # x.shape[1] : dimension of dataset
        self.b = 1
        for i in range(epochs):    
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= learning_rate*w_grad
                self.b -= learning_rate*b_grad
                #print(self.w)
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
```python
import numpy as np

x = np.linspace(0,100,10000).reshape(10000,1)
y = lambda x : 1/(1+np.exp(-3*x))

a = LogisticNeuron()
a.fit(x,y(x))
```

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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>


