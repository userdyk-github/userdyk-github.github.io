---
layout : post
title : AI03-Topic01, Training neural network
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

## **Loss function**

### ***Mean squared error, MSE***
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:24.729ex; height:6.843ex;" alt="{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}">
```python
import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)
    
y = np.array([1,2,3])
t = np.array([3,4,7])

mean_squared_error(y,t)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
12.0
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Cross entropy error, CEE***
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c6b895514e10a3ce88773852cba1cb1e248ed763" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:28.839ex; height:5.676ex;" alt="{\displaystyle H(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)}">
```python
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
    
y = np.array([0,2,3,8,9])
t = np.array([3,4,7,8,9])

cross_entropy_error(y,t)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
1.4808580471604245
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Mini batch training***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement cross entropy error***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">




## **Neumerical derivative**

### ***Derivative***

```python
import numpy as np

# 1st-definition of derivative
def numerical_diff1(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x)) / (h)
    
# 2rd-definition of derivative
def numerical_diff2(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">CAUTION</summary>
<hr class='division3'>
```
>>> import numpy as np
>>> np.float32(1e-50)
0.0
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Example for numerical derivative***

```python
import numpy as np

# definition of derivative
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    diff = (f(x+h) - f(x-h)) / (2*h)
    
    # Put the results during calculation
    diff_result = {}  
    for i in range(len(diff)):
         diff_result['x=%f'%x[i]]= diff[i]
        
    return diff_result
    
# test
def f(x):
    return 0.01*x**2 + 0.1*x
x = np.linspace(-1,1,100)

numerical_diff(f, x)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
{'x=-1.000000': 0.07999999999994123,
 'x=-0.979798': 0.08040404040399185,
 'x=-0.959596': 0.08080808080818125,
 'x=-0.939394': 0.08121212121209309,
 'x=-0.919192': 0.08161616161614371,
 'x=-0.898990': 0.08202020202012494,
 'x=-0.878788': 0.08242424242424495,
 'x=-0.858586': 0.08282828282829557,
 'x=-0.838384': 0.0832323232322768,
 'x=-0.818182': 0.08363636363632743,
 'x=-0.797980': 0.08404040404044744,
 'x=-0.777778': 0.08444444444449806,
 'x=-0.757576': 0.08484848484847929,
 'x=-0.737374': 0.08525252525252991,
 'x=-0.717172': 0.08565656565658053,
 'x=-0.696970': 0.08606060606056176,
 'x=-0.676768': 0.08646464646461238,
 'x=-0.656566': 0.08686868686862831,
 'x=-0.636364': 0.08727272727274832,
 'x=-0.616162': 0.08767676767676424,
 'x=-0.595960': 0.08808080808081487,
 'x=-0.575758': 0.0884848484847961,
 'x=-0.555556': 0.08888888888888141,
 'x=-0.535354': 0.08929292929289734,
 'x=-0.515152': 0.08969696969698265,
 'x=-0.494949': 0.09010101010096389,
 'x=-0.474747': 0.0905050505050492,
 'x=-0.454545': 0.09090909090909982,
 'x=-0.434343': 0.09131313131315044,
 'x=-0.414141': 0.09171717171713167,
 'x=-0.393939': 0.09212121212121699,
 'x=-0.373737': 0.09252525252523291,
 'x=-0.353535': 0.09292929292928354,
 'x=-0.333333': 0.09333333333333416,
 'x=-0.313131': 0.09373737373736743,
 'x=-0.292929': 0.0941414141414007,
 'x=-0.272727': 0.09454545454545132,
 'x=-0.252525': 0.0949494949494846,
 'x=-0.232323': 0.09535353535351787,
 'x=-0.212121': 0.09575757575758584,
 'x=-0.191919': 0.09616161616161911,
 'x=-0.171717': 0.09656565656565239,
 'x=-0.151515': 0.09696969696968566,
 'x=-0.131313': 0.0973737373737276,
 'x=-0.111111': 0.09777777777777823,
 'x=-0.090909': 0.09818181818182017,
 'x=-0.070707': 0.09858585858585778,
 'x=-0.050505': 0.09898989898990407,
 'x=-0.030303': 0.0993939393939395,
 'x=-0.010101': 0.09979797979798037,
 'x=0.010101': 0.10020202020201906,
 'x=0.030303': 0.10060606060605667,
 'x=0.050505': 0.10101010101010513,
 'x=0.070707': 0.10141414141414708,
 'x=0.090909': 0.10181818181818902,
 'x=0.111111': 0.1022222222222223,
 'x=0.131313': 0.10262626262625557,
 'x=0.151515': 0.10303030303028884,
 'x=0.171717': 0.10343434343433947,
 'x=0.191919': 0.10383838383835539,
 'x=0.212121': 0.10424242424242336,
 'x=0.232323': 0.10464646464645663,
 'x=0.252525': 0.10505050505047256,
 'x=0.272727': 0.10545454545452318,
 'x=0.292929': 0.1058585858585738,
 'x=0.313131': 0.10626262626260707,
 'x=0.333333': 0.1066666666666577,
 'x=0.353535': 0.10707070707070832,
 'x=0.373737': 0.10747474747472424,
 'x=0.393939': 0.10787878787877486,
 'x=0.414141': 0.10828282828282548,
 'x=0.434343': 0.10868686868684141,
 'x=0.454545': 0.10909090909089203,
 'x=0.474747': 0.10949494949494265,
 'x=0.494949': 0.10989898989899327,
 'x=0.515152': 0.1103030303030092,
 'x=0.535354': 0.11070707070705982,
 'x=0.555556': 0.11111111111111044,
 'x=0.575758': 0.11151515151512637,
 'x=0.595960': 0.11191919191917699,
 'x=0.616162': 0.11232323232315822,
 'x=0.636364': 0.11272727272727823,
 'x=0.656566': 0.11313131313132885,
 'x=0.676768': 0.11353535353531008,
 'x=0.696970': 0.1139393939393607,
 'x=0.717172': 0.11434343434341132,
 'x=0.737374': 0.11474747474746194,
 'x=0.757576': 0.11515151515151256,
 'x=0.777778': 0.11555555555556318,
 'x=0.797980': 0.11595959595954441,
 'x=0.818182': 0.11636363636366442,
 'x=0.838384': 0.11676767676764566,
 'x=0.858586': 0.11717171717169628,
 'x=0.878788': 0.11757575757581629,
 'x=0.898990': 0.11797979797979752,
 'x=0.919192': 0.11838383838377875,
 'x=0.939394': 0.11878787878782937,
 'x=0.959596': 0.11919191919187999,
 'x=0.979798': 0.119595959596,
 'x=1.000000': 0.11999999999998123}
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Partial derivative***

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">





## **Gradient**

### ***Gradient descent method***

```python
import numpy as np


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # calculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # calculate f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def function(x):
    return x[0] ** 2 + x[1] ** 2


x = np.linspace(-5, 5, 100)
diff = numerical_gradient(function, np.array([3,4]))
print(diff)
```


<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Gradient at neural network***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">




## **Implement Learning Algorithms**

### ***Implement two layer neural network class***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement mini batch training***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Evaluation on test dataset***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
```
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Reference Codes**

### ***gradient_1d.py***
```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

### ***gradient_2d.py***
```python
# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

### ***gradient_method.py***
```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

### ***gradient_simplenet.py***
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

### ***train_neuralnet.py***
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

### ***two_layer_net.py***
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
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
- <a href='http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html' target="_blank">Beomsu Kim</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---
