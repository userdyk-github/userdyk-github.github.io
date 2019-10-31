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

<details markdown="1">
<summary class='jb-small' style="color:blue">ADVANCDED PREPERATION</summary>
<hr class='division3'>
[dataset.zip][1]
<hr class='division3'>
</details>

```python
import numpy as np
from mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_szie = 10

batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
print(x_batch)
print(t_batch)
```
```
(10, 784)
(10, 10)
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
print(x_train.shape,
      t_train.shape,
      x_test.shape,
      t_test.shape)
```
```
(60000, 784) (60000, 10) (10000, 784) (10000, 10)
```
<br>
```python
for i in range(x_train.shape[0]):
    print(x_train[i].shape)
```
```
(784,)
(784,)
(784,)
...
...
(784,)
(784,)
(784,)
```
<br>
```python
for i in range(t_train.shape[0]):
    print(t_train[i].shape)
```
```
(10,)
(10,)
(10,)
...
...
(10,)
(10,)
(10,)
```
<br>
```python
for i in range(x_test.shape[0]):
    print(x_test[i].shape)
```
```
(784,)
(784,)
(784,)
...
...
(784,)
(784,)
(784,)
```
<br>
```python
for i in range(t_test.shape[0]):
    print(t_test[i].shape)
```
```
(10,)
(10,)
(10,)
...
...
(10,)
(10,)
(10,)
```
<br><br><br>
```python
np.random.choice(4,10)
```
```
array([2, 3, 0, 1, 3, 2, 2, 1, 2, 0])
```
<br>


<hr class='division3'>
</details>
<br><br><br>

---

### ***Implement cross entropy error for batch***

<span class="frame3">when labels type are one-hot(binary) format</span>
```python
def cross_entropy_error(y,t):
    
    # 1d array > 2d array
    # ex) [.1,.05,.6,.0,.05,.1,.0,.1,.0,.0] > [[.1,.05,.6,.0,.05,.1,.0,.1,.0,.0]]
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7))/batch_size
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
y = np.array([.1,.05,.6,.0,.05,.1,.0,.1,.0,.0])
t = np.array([0,0,1,0,0,0,0,0,0,0])

cross_entropy_error(y,t)
```
```
0.510825457099338
```
<hr class='division3'>
</details>
<br>
<span class="frame3">when labels type are not one-hot format</span>
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python

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




## **Implement Learning Algorithms with numpy**

```python
# import libraries
import time
import numpy as np


# set hyperparmeter
epsilon = 0.0001


# utlility function
def _t(x):
    return np.transpose(x)
def _m(A, B):
    return np.matmul(A, B)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def mean_squared_error(h, y):
    return 1 / 2 * np.mean(np.square(h - y))


# define neuron
class Neuron:
    def __init__(self, W, b, a):
        # Model Parameter
        self.W = W
        self.b = b
        self.a = a
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.a(_m(_t(self.W), x) + self.b) # activation((W^T)x + b)


# define deep neural network
class DNN:
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))
        
        # Hidden layers
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, x, y, loss_func):
        def get_new_sequence(layer_index, new_neuron):
            new_sequence = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_index:
                    new_sequence.append(new_neuron)
                else:
                    new_sequence.append(layer)
            return new_sequence
        
        def eval_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x
        
        loss = loss_func(self(x), y)
        
        for layer_id, layer in enumerate(self.sequence): # iterate layer
            for w_i, w in enumerate(layer.W): # iterate W (row)
                for w_j, ww in enumerate(w): # iterate W (col)
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon
                    
                    new_neuron = Neuron(W, layer.b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon  # (f(x+eps) - f(x)) / epsilon
                    layer.dW[w_i][w_j] = num_grad
            
                for b_i, bb in enumerate(layer.b): # iterate b
                    b = np.copy(layer.b)
                    b[b_i] = bb + epsilon
                    
                    new_neuron = Neuron(layer.W, b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon  # (f(x+eps) - f(x)) / epsilon
                    layer.db[b_i] = num_grad
        return loss
        

# define gradient descent
def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


# operation
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10, num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))    
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Epoch 0: Test loss 0.5788407666374237
Epoch 1: Test loss 0.5755734143209582
Epoch 2: Test loss 0.5723271009377402
Epoch 3: Test loss 0.5691025121018389
Epoch 4: Test loss 0.565900303856321
Epoch 5: Test loss 0.5627211022628145
Epoch 6: Test loss 0.5595655030828444
Epoch 7: Test loss 0.5564340715483467
Epoch 8: Test loss 0.5533273422196228
Epoch 9: Test loss 0.5502458189274984
Epoch 10: Test loss 0.5471899747975388
...
...
...
Epoch 91: Test loss 0.3874385796049036
Epoch 92: Test loss 0.38632830347838176
Epoch 93: Test loss 0.3852320899123199
Epoch 94: Test loss 0.38414972218996524
Epoch 95: Test loss 0.38308098674951835
Epoch 96: Test loss 0.3820256731742948
Epoch 97: Test loss 0.3809835741802311
Epoch 98: Test loss 0.3799544856008311
Epoch 99: Test loss 0.37893820636984105
47.71986770629883 seconds elapsed.
```
<hr class='division3'>
</details>


<br><br><br>

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

## **Implement Learning Algorithms with tensorflow**

```python
# [0] : import libraries
import tensorflow as tf
import numpy as np

# [1] : Define network architecture
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
            
    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)
        
# [2] : Implement training loop        
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables) # df(x)/dx
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)
    
# [3] : Import and organize dataset    
np.random.seed(0)

pts = list()
labels = list()
center_pts = np.random.uniform(-8.0, 8.0, (10, 2))
for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)


# [4] : create model
model = MyModel()

# [5] : define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# [6] : Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# [7] : do training loop and test
EPOCHS = 1000
for epoch in range(EPOCHS):
    for x, label in train_ds:
        train_step(model, x, label, loss_object, optimizer, train_loss, train_accuracy)
        
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Epoch 1, Loss: 2.2050864696502686, Accuracy: 23.299999237060547
Epoch 2, Loss: 1.8474318981170654, Accuracy: 51.79999923706055
Epoch 3, Loss: 1.6150718927383423, Accuracy: 55.29999542236328
Epoch 4, Loss: 1.446712613105774, Accuracy: 63.0
Epoch 5, Loss: 1.3001761436462402, Accuracy: 71.69999694824219
Epoch 6, Loss: 1.2056496143341064, Accuracy: 72.89999389648438
Epoch 7, Loss: 1.1280295848846436, Accuracy: 75.80000305175781
Epoch 8, Loss: 1.0488293170928955, Accuracy: 79.69999694824219
Epoch 9, Loss: 0.9811602830886841, Accuracy: 78.89999389648438
Epoch 10, Loss: 0.9269685745239258, Accuracy: 85.0999984741211
...
...
...
Epoch 991, Loss: 0.24664713442325592, Accuracy: 89.80000305175781
Epoch 992, Loss: 0.24172385036945343, Accuracy: 90.20000457763672
Epoch 993, Loss: 0.24257495999336243, Accuracy: 89.80000305175781
Epoch 994, Loss: 0.242522731423378, Accuracy: 90.0
Epoch 995, Loss: 0.24358901381492615, Accuracy: 89.70000457763672
Epoch 996, Loss: 0.24744541943073273, Accuracy: 90.5999984741211
Epoch 997, Loss: 0.2455950230360031, Accuracy: 90.0
Epoch 998, Loss: 0.255244642496109, Accuracy: 90.30000305175781
Epoch 999, Loss: 0.26216936111450195, Accuracy: 90.0
Epoch 1000, Loss: 0.24371500313282013, Accuracy: 90.20000457763672
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT : [3]</summary>
<hr class='division3'>
```python
np.random.seed(0)

pts = list()
labels = list()
center_pts = np.random.uniform(-8.0, 8.0, (10, 2))
for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)
```
<details markdown="1">
<summary class='jb-small' style="color:red">SUPPLEMENT</summary>
<hr class='division3_1'>
randn : Gaussian(standard normal distribution)
```
>>> import numpy as np
>>> np.random.randn(10)      # 10 random numbers in range : (-oo,oo)
array([ 0.58711644,  0.04700508, -1.10859032, -0.78977472,  2.64137167,
       -0.01833935,  0.03531587, -1.72592648,  0.66461845, -0.36460468])

>>> np.random.randn(3, 5)    # 15=3*5 random numbers in range : (-oo,oo)
array([[-0.99387193,  0.71975003, -0.719061  , -0.51130777, -0.18149095],
       [-0.95578814,  0.23776812, -1.80650151,  0.86778844, -1.12507707],
       [-0.88193264,  2.44759966, -0.27246929,  1.8909227 , -1.21857409]])
```
<hr class='division3_1'>
</details>
<br>
```python
pts = np.stack(pts, axis=0).astype(np.float32)
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)
```
<details markdown="1">
<summary class='jb-small' style="color:red">SUPPLEMENT</summary>
<hr class='division3_1'>
```
>>> import numpy as np
>>> a = np.array([1,2])
>>> b = np.array([3,4])
>>> list = [a,b]
>>> list
[array([1, 2]), array([3, 4])]

>>> np.stack(list)
array([[1, 2],
       [3, 4]])
```
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Save Parameters</summary>
<hr class='division3'>
```python
np.savez_compressed('ch2_dataset.npz', inputs=pts, labels=labels)

W_h, b_h = model.d1.get_weights()
W_o, b_o = model.d2.get_weights()
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)
np.savez_compressed('ch2_parameters.npz',
                    W_h=W_h,
                    b_h=b_h,
                    W_o=W_o,
                    b_o=b_o)
```
<hr class='division3'>
</details>
<br><br><br>
<hr class="division2">

## **Reference Codes**


<details markdown="1">
<summary class='jb-small' style="color:blue">Pre-define</summary>
<hr class='division3'>
<details markdown="1">
<summary class='jb-small' style="color:red">functions.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">gradient.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">layers.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">multi_layer_net.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """全結合による多層ニューラルネットワーク

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        損失関数の値
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">multi_layer_net_extend.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """拡張版の全結合による多層ニューラルネットワーク
    
    Weiht Decay、Dropout、Batch Normalizationの機能を持つ

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """勾配を求める（数値微分）

        Parameters
        ----------
        X : 入力データ
        T : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">optimizer.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import numpy as np

class SGD:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">trainer.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
```
<hr class='division3_1'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:red">util.py</summary>
<hr class='division3_1'>
```python
# coding: utf-8
import numpy as np


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>
<br><br><br>



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

---

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

---

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

---

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

---

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

---

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

[1]:{{ site.url }}/download/AI03/AI03-Topic01/dataset.zip
