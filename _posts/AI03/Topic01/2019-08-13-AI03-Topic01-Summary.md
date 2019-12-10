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

## **GPU Resource**
```bash
$ nvidia-smi   
$ nvidia-smi -l 1
$ watch -n 1 -d nvidia-smi
```
```bash
$ python train.py | tee log.txt | less
```

<br><br><br>
<hr class="division2">

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
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad
    
    def fit(self,x,y,epochs=100, rate_b=0.001):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for _ in range(epochs):
            for x_i, y_i in zip(x,y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i,err)
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
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
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
                err = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Add : Function 1, training loss history</summary>
<hr class='division3'>
```python
class metric():
    def __init__(self):
        """<<<+++F1[1]+++>>>"""
        self.losses = []
        """<<<+++F1[1]+++>>>"""
        
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
    """<<<+++F1[4]+++>>>"""    
        
class LogisticNeuron(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1/(1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            """<<<+++F1[2]+++>>>"""
            loss = 0
            """<<<+++F1[2]+++>>>"""
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<+++F1[3]"""        
                a = np.clip(a, 1e-10, 1 - 1e-10)
                loss += -(y_i*np.log(a)+(1-y_i)*np.log(1-a))
            self.losses.append(loss/len(y))
            self.loss()
        self.loss_save()
        """F1[3]+++>>>"""
        
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
        """<<<+++F2[1]+++>>>"""
        self.weights = []
        """<<<+++F2[1]+++>>>"""
        
    """<<<+++F2[3]+++>>>"""    
    def w_history(self):
        print(*self.w, self.b)
        display.clear_output(wait=True)

    def w_history_save(self):
        np.savetxt('weight.txt', self.weights)
    """<<<+++F2[3]+++>>>"""


class LogisticNeuron(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
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
                err = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
            """<<<+++F2[2]"""
            self.weights.append([*self.w, self.b])
            self.w_history()
        self.w_history_save()
        """F2[2]+++>>>"""
        
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
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad
    
    """<<<+++F3[1]+++>>>"""
    def add_bias(self, x):
        return np.c_p[np.ones((x.shape[0],1)),x]
    """<<<+++F3[1]+++>>>"""
    
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
                err = -(y_i - a)
                w_grad, b_grad = self.backprop(x_i,err)
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
    
    def backprop(self, x, err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1/(1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 1.0
        for i in range(epochs):
            """<<<+++F4[1]+++>>>"""
            indexes = np.random.permutation(np.arange(len(x))) 
            for i in indexes:                                  
                z = self.forpass(x[i])                         
                a = self.activation(z)                         
                err = -(y[i] - a)                            
                w_grad, b_grad = self.backprop(x[i], err)    
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
            """<<<+++F4[1]+++>>>"""
            
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

<hr class="division2">

## **Sigle layer : binaray classification**

### ***Basic model of single layer***
Bias(F3) + Shuffle(F4)

```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
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
<br>
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


### ***Version 1 : Update loss according to weight history about train-dataset***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        """<<<+++V1+++>>>"""
        self.losses = []
        self.weights = []
        """<<<+++V1+++>>>"""
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        """<<<+++V1+++>>>"""
        self.weights.append(self.w.copy())
        """<<<+++V1+++>>>"""
        for i in range(epochs):
            """<<<+++V1+++>>>"""
            loss = 0
            """<<<+++V1+++>>>"""
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<+++V1"""
                self.weights.append(self.w.copy())
                a = np.clip(a, 1e-10, 1 - 1e-10)                
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            """V1+++>>>"""
        
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
<br>
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
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
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

"""<<<+++V2+++>>>"""
x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
"""<<<+++V2+++>>>"""

layer = SingleLayer()
layer.fit(x_train,y_train)
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>
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

"""<<<+++V2+++>>>"""
x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
"""<<<+++V2+++>>>"""

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

### ***Version 3 : Update loss according to weight history about validation-dataset***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        """<<<+++V3+++>>>"""
        self.val_losses = []
        self.weights = []
        """<<<+++V3+++>>>"""
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    """<<<+++V3"""
    def fit(self, x, y, epochs=100, rate_b=1, x_val=None, y_val=None):
        """V3+++>>>"""
        self.w = np.ones(x.shape[1])
        self.b = 0
        """<<<+++V3+++>>>"""
        self.weights.append(self.w.copy())
        """<<<+++V3+++>>>"""
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<+++V3"""
                self.weights.append(self.w.copy())
            self.update_val_loss(x_val, y_val)
            """V3+++>>>"""
            
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
        """<<<+++V3+++>>>"""
    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a) + (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val))
    """<<<+++V3+++>>>"""
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
"""<<<+++V3+++>>>"""
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
"""<<<+++V3+++>>>"""
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>
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
"""<<<+++V3+++>>>"""
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
"""<<<+++V3+++>>>"""
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
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
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
"""<<<+++V4+++>>>"""
layer.fit(x_train,y_train, epochs=20)
"""<<<+++V4+++>>>"""
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>
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
"""<<<+++V4+++>>>"""
layer.fit(x_train,y_train, epochs=20)
"""<<<+++V4+++>>>"""
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 5 : Regularization(L1, L2)***
```python
class SingleLayer:
    """<<<+++V5"""
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        """V5+++>>>"""
        self.w = None
        self.b = None
        self.lr = learning_rate
        """<<<+++V5+++>>>"""
        self.losses = []
        self.val_losses = []
        self.l1 = l1
        self.l2 = l2
        """<<<+++V5+++>>>"""
            
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1, x_val=None, y_val=None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            """<<<+++V5+++>>>"""
            loss = 0
            """<<<+++V5+++>>>"""
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                """<<<+++V5+++>>>"""
                w_grad += self.l1*np.sign(self.w) + self.l2*self.w
                """<<<+++V5+++>>>"""
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                """<<<+++V5"""
                a = np.clip(a, 1e-10, 1 - 1e-10)                
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y) + self.reg_loss())
            self.update_val_loss(x_val, y_val)
            """V5+++>>>"""
        
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
        """<<<+++V5+++>>>"""    
    def reg_loss(self):
        return self.l1*np.sum(np.abs(self.w)) + self.l2/2*np.sum(self.w**2)
        
    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a) + (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val) + self.reg_loss())
    """<<<+++V5+++>>>"""
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

"""<<<+++V5+++>>>"""
layer = SingleLayer(l1=0.01,l2=0)
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
"""<<<+++V5+++>>>"""
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>
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

"""<<<+++V5+++>>>"""
layer=SingleLayer(l1=0.01,l2=0)
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
"""<<<+++V5+++>>>"""
layer.score(x_test,y_test)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Version 6 : k-fold validation***
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
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
<br>
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
<details markdown="1">
<summary class='jb-small' style="color:blue">Vectorization method</summary>
<hr class='division3'>
<span class="frame3">Dot product</span><br>
```python
# Dot product 
import time 
import numpy 
import array 

# 8 bytes size int 
a = array.array('q') 
for i in range(100000): 
    a.append(i); 

b = array.array('q') 
for i in range(100000, 200000): 
    b.append(i) 

# classic dot product of vectors implementation 
tic = time.process_time() 
dot = 0.0; 

for i in range(len(a)): 
    dot += a[i] * b[i] 

toc = time.process_time() 

print("dot_product = "+ str(dot)); 
print("Computation time = " + str(1000*(toc - tic )) + "ms", '\n\n') 



n_tic = time.process_time() 
n_dot_product = numpy.dot(a, b) 
n_toc = time.process_time() 

print("n_dot_product = "+str(n_dot_product)) 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 
```
<p>
dot_product = 833323333350000.0<br>
Computation time = 46.50096700000006ms<br><br> 

n_dot_product = 833323333350000<br>
Computation time = 1.0852390000000156ms
</p>
<br><br><br>
<span class="frame3">Outer product</span><br>
```python
# Outer product 
import time 
import numpy 
import array 

a = array.array('i') 
for i in range(200): 
    a.append(i); 

b = array.array('i') 
for i in range(200, 400): 
    b.append(i) 

# classic outer product of vectors implementation 
tic = time.process_time() 
outer_product = numpy.zeros((200, 200)) 

for i in range(len(a)): 
    for j in range(len(b)): 
        outer_product[i][j]= a[i]*b[j] 

toc = time.process_time() 

print("outer_product = "+ str(outer_product)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms", '\n\n') 



n_tic = time.process_time() 
outer_product = numpy.outer(a, b) 
n_toc = time.process_time() 

print("outer_product = "+str(outer_product)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 
```
<p>
outer_product = [[    0.     0.     0. ...     0.     0.     0.]<br>
 [  200.   201.   202. ...   397.   398.   399.]<br>
 [  400.   402.   404. ...   794.   796.   798.]<br>
 ...<br>
 [39400. 39597. 39794. ... 78209. 78406. 78603.]<br>
 [39600. 39798. 39996. ... 78606. 78804. 79002.]<br>
 [39800. 39999. 40198. ... 79003. 79202. 79401.]]<br>
Computation time = 32.9991500000002ms <br><br>


outer_product = [[    0     0     0 ...     0     0     0]<br>
 [  200   201   202 ...   397   398   399]<br>
 [  400   402   404 ...   794   796   798]<br>
 ...<br>
 [39400 39597 39794 ... 78209 78406 78603]<br>
 [39600 39798 39996 ... 78606 78804 79002]<br>
 [39800 39999 40198 ... 79003 79202 79401]]<br>
Computation time = 0.24932600000049376ms
</p>
<br><br><br>
<span class="frame3">Element-wise multiplication</span><br>
```python
# Element-wise multiplication 
import time 
import numpy 
import array 

a = array.array('i') 
for i in range(50000): 
    a.append(i); 

b = array.array('i') 
for i in range(50000, 100000): 
    b.append(i) 

# classic element wise product of vectors implementation 
vector = numpy.zeros((50000)) 

tic = time.process_time() 

for i in range(len(a)): 
    vector[i]= a[i]*b[i] 

toc = time.process_time() 

print("Element wise Product = "+ str(vector)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms", '\n\n') 



n_tic = time.process_time() 
vector = numpy.multiply(a, b) 
n_toc = time.process_time() 

print("Element wise Product = "+str(vector)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 
```
<p>
Element wise Product = [0.00000000e+00 5.00010000e+04 1.00004000e+05 ... 4.99955001e+09<br>
 4.99970000e+09 4.99985000e+09]<br>
Computation time = 24.53195000000008ms<br><br>


Element wise Product = [        0     50001    100004 ... 704582713 704732708 704882705]<br>
Computation time = 0.7946960000007053ms
</p>
<hr class='division3'>
</details>

```python
class SingleLayer:
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate
        
    def forpass(self, x):
        """<<<---V7--->>>
        z = np.sum(x*self.w) + self.b
        <<<---V7--->>>"""
        """<<<+++V7+++>>>"""
        z = np.dot(x, self.w) + self.b
        """<<<+++V7+++>>>"""
        return z
    
    def backprop(self, x ,err):
        """<<<+++V7+++>>>"""
        m = len(x)
        """<<<+++V7+++>>>"""
        """<<<---V7--->>>
        w_grad = x * err
        b_grad = 1 * err
        <<<---V7--->>>"""
        """<<<+++V7+++>>>"""
        w_grad = np.dot(x.T, err) / m
        b_grad = np.sum(err) / m
        """<<<+++V7+++>>>"""
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1):
        """<<<+++V7+++>>>"""
        y = y.reshape(-1, 1)
        m = len(x)
        """<<<+++V7+++>>>"""
        """<<<---V7--->>>
        self.w = np.ones(x.shape[1])
        <<<---V7--->>>"""
        """<<<+++V7+++>>>"""
        self.w = np.ones((x.shape[1],1))
        """<<<+++V7+++>>>"""
        self.b = 0
        for i in range(epochs):
            """<<<---V7--->>>
            indexes = np.random.permutation(np.arange(len(x)))            
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
            <<<---V7--->>>"""
            """<<<+++V7+++>>>"""
            z = self.forpass(x)
            a = self.activation(z)
            err = -(y - a)
            w_grad, b_grad = self.backprop(x, err)
            self.w -= self.lr*w_grad
            self.b -= rate_b*b_grad
            """<<<+++V7+++>>>"""
            
    def predict(self, x):
        """<<<---V7--->>>
        z = [self.forpass(x_i) for x_i in x]
        <<<---V7--->>>"""
        """<<<+++V7+++>>>"""
        z = self.forpass(x)
        """<<<+++V7+++>>>"""
        return z > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1,1))
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
<br>
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

### ***Latest model of singleLayer***
#### Stochastic
```python
class SingleLayer:
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.weights = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
        
    def forpass(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        m = len(x)
        w_grad = np.dot(x.T, err) / m
        b_grad = np.sum(err) / m
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def fit(self, x, y, epochs=100, rate_b=1, x_val=None, y_val=None):
        y = y.reshape(-1, 1)
        y_val = y_val.reshape(-1,1)
        m = len(x)
        self.w = np.ones((x.shape[1],1))
        self.b = 0
        self.weights.append(self.w.copy())
        for _ in range(epochs):
            z = self.forpass(x)
            a = self.activation(z)
            err = -(y - a)
            w_grad, b_grad = self.backprop(x, err)
            w_grad += (self.l1*np.sign(self.w) + self.l2*self.w)/m
            self.w -= self.lr*w_grad
            self.b -= rate_b*b_grad
            self.weights.append(self.w.copy())
            a = np.clip(a, 1e-10, 1 - 1e-10)                
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss())/m)
            self.update_val_loss(x_val, y_val)
            
    def predict(self, x):
        z = self.forpass(x)
        return z > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1,1))
    
    def reg_loss(self):
        return self.l1*np.sum(np.abs(self.w)) + self.l2/2*np.sum(self.w**2)
        
    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.activation(z)
        a = np.clip(a, 1e-10, 1-1e-10)
        val_loss = np.sum(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
        self.val_losses.append((val_loss + self.reg_loss())/len(y_val))
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

layer = SingleLayer(l1=0.01,l2=0)
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
layer.score(x_test,y_test)
```
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

layer=SingleLayer(l1=0.01,l2=0)
layer.fit(x_train,y_train,x_val=x_val,y_val=y_val)
layer.score(x_test,y_test)
```

<br><br><br>
#### Mini-batch

<br><br><br>


#### Batch

<br><br><br>


---

### ***Custumized model of singleLayer***
```python
class metric():
    def __init__(self):
        self.losses = []
        self.weights = []
        
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
        np.savetxt('weight.txt', self.weights)


class SingleLayer(metric):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.w = None
        self.b = None
        self.lr = learning_rate                
    
    def forpass(self, x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x ,err):
        w_grad = x * err
        b_grad = 1 * err
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
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                self.w -= self.lr*w_grad
                self.b -= rate_b*b_grad
                a = np.clip(a, 1e-10, 1 - 1e-10)                
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            self.loss()
            self.weights.append([*self.w, self.b])
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

---


### ***Example***

<br><br><br>
<hr class="division2">

## **Dual Layer**
### ***Basic model of dual layer***
<br><br><br>

---

### ***Version 1 : Inheritance from singlelayer***
<br><br><br>

---

### ***Version 2 : mini-batch***
<br><br><br>

---

### ***Latest model of dual layer(multiclass classification)***
```python
class DualLayer:
    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
        self.units = units
        self.batch_size = batch_size
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.a1 = None
        self.losses = []
        self.val_losses = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
        
    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        return z2
    
    def backprop(self, x, err):
        m = len(x)
        w2_grad = np.dot(self.a1.T, err)
        b2_grad = np.sum(err) / m
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        w1_grad = np.dot(x.T, err_to_hidden) / m
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad
    
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1).reshape(-1,1)
    
    def init_weights(self, n_features, n_classes):
        self.w1 = np.random.normal(0, 1, (n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.normal(0, 1, (self.units, n_classes))
        self.b2 = np.zeros(n_classes)
        
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        self.init_weights(x.shape[1], y.shape[1])
        for _ in range(epochs):
            loss = 0
            print('.', end='')
            for x_batch, y_batch in self.gen_batch(x,y):
                a = self.training(x_batch, y_batch)
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += np.sum(-y_batch*np.log(a))
            self.losses.append((loss + self.reg_loss()) / len(x))
            self.update_val_loss(x_val, y_val)
    
    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size
        if length % self.batch_size:
            bins += 1
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]
    
    def training(self, x, y):
        m = len(x)
        z = self.forpass(x)
        a = self.softmax(z)
        err = -(y - a)
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a
    
    def predict(self, x):
        z = self.forpass(x)
        return np.argmax(z, axis=1)
    
    def score(self, x, y):
        return np.mean(self.predict(x) == np.argmax(y, axis=1))
    
    def reg_loss(self):
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
                self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
    
    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.softmax(z)
        a = np.clip(a, 1e-10 , 1-1e-10)
        val_loss = np.sum(-y_val*np.log(a))
        self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))
```
<br><br><br>

---

### ***Custumized model of dual layer***
<br><br><br>

---

### ***Example : minist***
<br><br><br>

<hr class="division2">

## **Multi Layer**
### ***Basic model of multi layer***
<br><br><br>

---

### ***Latest model of multi layer***
<br><br><br>

---

### ***Custumized model of multi layer***

<br><br><br>

---

### ***Example***

<br><br><br>
<hr class="division2">


## **Convolutional neural network : computer vision**

### ***Basic model of CNN***
#### conv1d
```python
def conv1d(x,w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0 :
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad,x_padded,zero_pad])
    res = []
    for i in range(0, int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
        
    return np.array(res)
```

<br><br><br>

#### conv2d
```python
def conv2d(X,W, p=(0,0), s=(1,1)):
    W_rot = np.array(W)[::-1,::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],
             p[1]:p[1]+X_orig.shape[1]] = X_orig
    
    res = []
    for i in range(0, int((X_padded.shape[0]-W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1]-W_rot.shape[1])/s[1])+1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                             j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub*W_rot))
    return np.array(res)
```



<br><br><br>

---

### ***Latest model of CNN***
<br><br><br>

---

### ***Custumized model of CNN***
<br><br><br>

---

### ***Example***

<br><br><br>

<hr class="division2">

## **Recurrent neural network : natural language processing**

### ***Basic model of RNN***
<br><br><br>

---

### ***Latest model of RNN***
```python
class RecurrentNetwork:
    def __init__(self, n_cells=10, batch_size=32, learning_rate=0.1):
        self.n_cells = n_cells
        self.batch_size = batch_size
        self.w1h = None
        self.w1x = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.h = None
        self.losses = []
        self.val_losses = []
        self.lr = learning_rate

    def forpass(self, x):
        self.h = [np.zeros((x.shape[0], self.n_cells))]
        seq = np.swapaxes(x,0,1)
        for x in seq:
            z1 = np.dot(x, self.w1x) + np.dot(self.h[-1], self.w1h) + self.b1
            h = np.tanh(z1)
            self.h.append(h)
            z2 = np.dot(h,self.w2) + self.b2
        return z2

    def backprop(self, x, err):
        m = len(x)
        w2_grad = np.dot(self.h[-1].T, err) / m
        b2_grad = np.sum(err) / m
        seq = np.swapaxes(x, 0, 1)

        w1h_grad = w1x_grad = b1_grad = 0
        err_to_cell = np.dot(err, self.w2.T)*(1 - self.h[-1]**2)
        for x,h in zip(seq[::1][:10], self.h[:-1][::-1][:10]):
            w1h_grad += np.dot(h.T, err_to_cell)
            w1x_grad += np.dot(x.T, err_to_cell)
            b1_grad += np.sum(err_to_cell, axis=0)
            err_to_cell = np.dot(err_to_cell, self.w1h)*(1-h**2)

        w1h_grad /= m
        w1x_grad /= m
        b1_grad /= m

        return w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad

    def sigmoid(self, z):
        a = 1/(1+np.exp(-z))
        return a
    def init_weights(self, n_features, n_classes):
        orth_init = tf.initializers.Orthogonal()
        glorot_init = tf.initializers.GlorotUniform()

        self.w1h = orth_init((self.n_cells, self.n_cells)).numpy()
        self.w1x = glorot_init((n_features, self.n_cells)).numpy()
        self.b1 = np.zeros(self.n_cells)
        self.w2 = glorot_init((self.n_cells, n_classes)).numpy()
        self.b2 = np.zeros(n_classes)

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y = y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        self.init_weights(x.shape[2], y.shape[1])
        for i in range(epochs):
            print('epochs', i, end='')
            batch_losses = []
            for x_batch, y_batch in self.gen_batch(x,y):
                print('.', end='')
                a = self.training(x_batch,  y_batch)
                a = np.clip(a, 1e-10, 1-1e-10)
                loss = np.mean(-(y_batch*np.log(a) + (1-y_batch)*np.log(1-a)))
                batch_losses.append(loss)
            print()
            self.losses.append(np.mean(batch_losses))
            self.update_val_loss(x_val, y_val)

    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size
        if length % self.batch_size:
            bins += 1
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def training(self, x, y):
        m = len(x)
        z = self.forpass(x)
        a = self.sigmoid(z)
        err = -(y - a)
        w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad = self.backprop(x,err)
        self.w1h -= self.lr * w1h_grad
        self.w1x -= self.lr * w1x_grad
        self.b1 -= self.lr * b1_grad
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a

    def predict(self, x):
        z = self.forpass(x)
        return z > 0

    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1,1))

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.sigmoid(z)
        a = np.clip(a, 1e-10, 1-1e-10)
        val_loss = np.mean(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
        self.val_losses.append(val_loss)
```
<br><br><br>

---

### ***Custumized model of RNN***
<br><br><br>

---

### ***Example***

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


