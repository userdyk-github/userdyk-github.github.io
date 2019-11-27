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
    def __init__(self, learning_rate=0.001):
        self.w = None
        self.b = None
        self.lr = learning_rate
    
    def forpass(self, x):
        y_hat = np.sum(x*self.w) + self.b
        return y_hat
    
    def backprop(self, x, err_p):
        w_grad = x*err_p
        b_grad = 1*err_p
        return w_grad, b_grad
    
    def fit(self,x,y,epochs=100, rate_b=0.001):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for _ in range(epochs):
            for x_i, y_i in zip(x,y):
                y_hat = self.forpass(x_i)
                err_p = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                print(self.w, self.b)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np

x = np.linspace(0,100,10000).reshape(10000,1)
y = lambda x : 3*x + 5

layer = LinearNeuron()
layer.fit(x,y(x))
```
```python
import numpy as np

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 3*x1 + 5*x2 + 10

layer = LinearNeuron()
layer.fit(x,y(x1,x2))
```
<br><br><br>

### ***Logistic regression***
![image](https://user-images.githubusercontent.com/52376448/69402086-28ae5780-0d3a-11ea-9524-632ce29de793.png)
![image](https://user-images.githubusercontent.com/52376448/69403870-0c60e980-0d3f-11ea-95a9-96b0ce2b4cf5.png)

```python
class LogisticNeuron:
    def __init__(self, learning_rate=0.001):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
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
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Add : Function 1, traing loss history</summary>
<hr class='division3'>
```python
class metric():
    def __init__(self):
        """<<<F1[1]>>>"""
        self.losses = []
        """<<<F1[1]>>>"""
        
    """<<<F1[4]>>>"""    
    def loss(self):
        plt.clf()
        plt.grid(True)
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        display.display(plt.gcf())
        display.clear_output(wait=True)

    def loss_save(self):
        np.savetxt('loss.txt', self.losses)
        plt.clf()
        plt.grid(True)
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss.jpg')
    """<<<F1[4]>>>"""    
        
class LogisticNeuron(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate
        
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
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            """<<<F1[2]>>>"""
            loss = 0
            """<<<F1[2]>>>"""
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<F1[3]"""        
                a = np.clip(a, 1e-10, 1 - 1e-10)
                loss += -(y_i*np.log(a)+(1-y_i)*np.log(1-a))
            self.losses.append(loss/len(y))
            self.loss()
        self.loss_save()
        """F1[3]>>>"""
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
```python
import numpy as np

rv = np.random.RandomState(19)
x = rv.normal(0,1,(1000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

layer = LogisticNeuron()
layer.fit(x,y(x1,x2))
```
```python
plt.plot(layer.losses)
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Add : Function 2, Weight</summary>
<hr class='division3'>
```python
class metric():
    def __init__(self):
        """<<<F2[1]>>>"""
        self.w_histories = []
        """<<<F2[1]>>>"""
        
    """<<<F2[3]>>>"""    
    def w_history(self):
        print(*self.w, self.b)
        display.clear_output(wait=True)

    def w_history_save(self):
        np.savetxt('weight.txt', self.w_histories)
    """<<<F2[3]>>>"""


class LogisticNeuron(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate
        
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
    
    def fit(self, x, y, epochs=300, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
            """<<<F2[2]"""
            self.w_histories.append([*self.w, self.b])
            self.w_history()
        self.w_history_save()
        """F2[2]>>>"""
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Add : Function 3, Bias</summary>
<hr class='division3'>
```python
class LogisticNeuron:
    def __init__(self, learning_rate=0.001):
        self.w = None
        self.b = None
        self.lr = learning_rate        
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x, err_p):
        w_grad = x*err_p
        b_grad = 1*err_p
        return w_grad, b_grad
    
    """<<<F3[1]>>>"""
    def add_bias(self, x):
        return np.c_p[np.ones((x.shape[0],1)),x]
    """<<<F3[1]>>>"""
    
    def activation(self, z):
        a = 1/(1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):    
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err_p = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Add : Function 4, Shuffle</summary>
<hr class='division3'>
```python
class LogisticNeuron:
    def __init__(self, learning_rate=0.001):
        self.w = None
        self.b = None
        self.lr = learning_rate        
        
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
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            """<<<F4[1]>>>"""
            indexes = np.random.permutation(np.arange(len(x))) 
            for i in indexes:                                  
                z = self.forpass(x[i])                         
                a = self.activation(z)                         
                err_p = -(y[i] - a)                            
                w_grad, b_grad = self.backprop(x[i], err_p)    
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
            """<<<F4[1]>>>"""
            
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
<hr class='division3'>
</details>
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

layer = LogisticNeuron()
layer.fit(x,y(x1,x2))
```
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

layer=LogisticNeuron()
layer.fit(x_train,y_train)
```
<br><br><br>

### ***Latest Version SingleLayer***
```python
class metric():
    def __init__(self):
        self.losses = []
        self.w_histories = []
        
    def loss(self):
        plt.clf()
        plt.grid(True)
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        display.display(plt.gcf())
        #display.clear_output(wait=True)

    def loss_save(self):
        np.savetxt('loss.txt', self.losses)
        plt.clf()
        plt.grid(True)
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss.jpg')
        
    def w_history(self):
        print(*self.w, self.b)
        display.clear_output(wait=True)

    def w_history_save(self):
        np.savetxt('weight.txt', self.w_histories)


class SingleLayer(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate                
    
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def add_bias(self, x):
        return np.c_p[np.ones((x.shape[0],1)),x]
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            loss = 1.0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                a = np.clip(a, 1e-10, 1 - 1e-10)                
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            self.loss()
            self.w_histories.append([*self.w, self.b])
            self.w_history()
        self.loss_save()
        self.w_history_save()
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

layer = SingleLayer()
layer.fit(x,y(x1,x2))
```
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">by scikit-learn</summary>
<hr class='division3'>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
sgd.fit(x_train, y_train)
sgd.score(x_test,y_test)
```
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **Training skills on sigle layer**

### ***Version 0 : Basic model of single layer***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---


### ***Version 1 : Update rate according to weight history for train-dataset***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        """<<<V1>>>"""
        self.w_history = []
        """<<<V1>>>"""
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        """<<<V1>>>"""
        self.w_history.append(self.w.copy())
        """<<<V1>>>"""
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<V1>>>"""
                self.w_history.append(self.w.copy())
                """<<<V1>>>"""
                print(self.w, self.b)
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 2 : Standardization***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 3 : Update rate according to weight history for validation-dataset***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 4 : Early stopping***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 5 : Regularization(L1)***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 6 : Regularization(L2)***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 7 : Vectorization***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err_p):
        w_grad = x * err_p
        b_grad = 1 * err_p
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=1, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err_p = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err_p)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```
<span class="frame3">Artificial Dataset</span><br>
```python
import numpy as np
from sklearn.model_selection import train_test_split

rv = np.random.RandomState(19)
x = rv.normal(0,1,(10000,2)); x1 = x[:,0]; x2 = x[:,1]
y = lambda x1, x2 : 1/(1+np.exp(-3*x1 -5*x2 - 10))

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y(x1,x2), test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<span class="frame3">Real Dataset</span><br>
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

loaded_dataset = load_breast_cancer()
x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

layer=SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **Multi Layer**
### ***Dual Layer***
<br><br><br>

<hr class="division2">

## **Classification**
<br><br><br>

<hr class="division2">

## **Convolutional neural network**
<br><br><br>

<hr class="division2">

## **Recurrent neural network**
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


