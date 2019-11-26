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
        y_hat = x*self.w + self.b
        return y_hat
    
    def backprop(self, x, err_p):
        w_grad = x*err_p
        b_grad = 1*err_p
        return w_grad, b_grad
    
    def fit(self,x,y,epochs=100, rate_w=0.0001, rate_b=0.001):
        self.w = 1.0
        self.b = 1.0
        for _ in range(epochs):
            for x_i, y_i in zip(x,y):
                y_hat = self.forpass(x_i)
                err_p = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= rate_w*w_grad
                self.b -= rate_b*b_grad
                print(self.w, self.b)
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
    
    def fit(self, x, y, epochs=100, rate_w=1, rate_b=1):
        self.w = np.ones(x.shape[1])            # x.shape[1] : dimension of dataset
        self.b = 1
        for i in range(epochs):    
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= rate_w*w_grad
                self.b -= rate_b*b_grad
                print(self.w, self.b)
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
```python
import numpy as np

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

a = LogisticNeuron()
a.fit(x,y(x1,x2))
```

<br><br><br>

### ***SingleLayer***
```python
class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []
    
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def add_biad(self, x):
        return np.c_p[np.ones((x.shape[0],1)),x]
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= w_grad
                self.b -= b_grad
                a = np.clip(a, 1e-10, 1 - 1e-10)
                print(np.round(self.w, 2))
                
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array() > 0
    
    def score(self, x, y):
        return np.mean(self,predict(x) == y)
```
```python
import numpy as np

rv = np.random.RandomState(120)
x = rv.normal(0, 1, (569,30))
w = list(range(1,31))
l=[]
for i in range(len(x)):
    l.append(np.dot(x[i],w))
l = np.array(l).reshape(569,1)
y = lambda x : 0.5 > 1/(1+np.exp(-x))


layer = SingleLayer()
layer.fit(x, y(l))
layer.score
```
<details markdown="1">
<summary class='jb-small' style="color:blue">by scikit-learn</summary>
<hr class='division3'>
```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss="log", max_iter=100, tol=1e-3, random_state=42)
sgd.fit(x,y(l))
sgd.score(x,y(l))
```
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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


