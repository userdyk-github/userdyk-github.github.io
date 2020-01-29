---
layout : post
title : AI03, Artificial neural networks
categories: [AI03]
comments : true
tags : [AI03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜ <a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI03/2019-08-13-AI03-Artificial-neural-networks.md" target="_blank">page management</a><br>
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

## **Resource**
### CPU
<span class="frmae3">CPU Resource info</span>
```bash
# cat /proc/cpuinfo
```
<span class="frmae3">Total number of CPU cores</span>
```bash
$ grep -c processor /proc/cpuinfo
```
<span class="frmae3">Number of CPUs</span>
```bash
$ grep "physical id" /proc/cpuinfo | sort -u | wc -l
```
<span class="frmae3">Number of cores per one CPU</span>
```bash
$ grep "cpu cores" /proc/cpuinfo | tail -1
```
<br><br><br>

---

### GPU
#### GPU Resource info
```bash
$ nvidia-smi
$ watch -n 1 -d nvidia-smi
$ fuser -v /dev/nvidia*
```
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 7812072362293866351
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 12834618334973673973
physical_device_desc: "device: XLA_CPU device"
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 10813738189
locality {
  bus_id: 1
  links {
  }
}
incarnation: 2176570505504160042
physical_device_desc: "device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:3b:00.0, compute capability: 7.5"
, name: "/device:GPU:1"
device_type: "GPU"
memory_limit: 10813738189
locality {
  bus_id: 1
  links {
  }
}
incarnation: 16344150243988831062
physical_device_desc: "device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:5e:00.0, compute capability: 7.5"
, name: "/device:GPU:2"
device_type: "GPU"
memory_limit: 10813738189
locality {
  bus_id: 2
  numa_node: 1
  links {
  }
}
incarnation: 15503034830640890796
physical_device_desc: "device: 2, name: GeForce RTX 2080 Ti, pci bus id: 0000:86:00.0, compute capability: 7.5"
, name: "/device:GPU:3"
device_type: "GPU"
memory_limit: 10812430746
locality {
  bus_id: 2
  numa_node: 1
  links {
  }
}
incarnation: 17206545542125030428
physical_device_desc: "device: 3, name: GeForce RTX 2080 Ti, pci bus id: 0000:af:00.0, compute capability: 7.5"
, name: "/device:XLA_GPU:0"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 3251941024359796176
physical_device_desc: "device: XLA_GPU device"
, name: "/device:XLA_GPU:1"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 14468545947390282029
physical_device_desc: "device: XLA_GPU device"
, name: "/device:XLA_GPU:2"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 759770992281457065
physical_device_desc: "device: XLA_GPU device"
, name: "/device:XLA_GPU:3"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 15023472020250575167
physical_device_desc: "device: XLA_GPU device"
]
```
<hr class='division3'>
</details>
<br><br><br>
#### Deallocate memory on GPU
```bash
$ nvidia-smi --gpu-reset -i 0
```
```
# forcely
$ kill -9 [PID_num]
```

#### Allocate memory on GPU
<span class="frmae3">tensorflow : One GPU(default)</span>
```python
import tensorflow as tf

[Code : data preprocessing]
[Code : data neural net model]
```
<span class="frmae3">tensorflow : One GPU with CPU</span>
```python
import tensroflow as tf

tf.debugging.set_log_device_placement(True)

try:
    with tf.device('/device:CPU:0'):
        [Code : data preprocessing]
    with tf.device('/device:GPU:2'):
        [Code : deep neural net model]
        
except RuntimeError as e:
    print(e)
```
<span class="frmae3">tensorflow : Multi-GPU with CPU</span>
```python
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_logical_devices('GPU')
if gpus:
    with tf.device('/CPU:0'):
        [Code : data preprocessing]

    for gpu in gpus:
        with tf.device(gpu.name):
            [Code : deep neural net model]
```

<span class="frmae3">pytorch</span>

<br><br><br>

---

### ***tensorboard***
```python
import tensorflow as tf
from datetime import datetime
import os

%load_ext tensorboard
%matplotlib inline

[Code : data preprocessing]
[Code : data neural net model]

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('logs',  datetime.now().strftime("%Y%m%d-%H%M%S")), 
    write_graph=True, 
    write_images=True,
    histogram_freq=1
)

%tensorboard --logdir logs --port [port_num]
```

<br><br><br>
<hr class="division2">
## **Numpy**

### ***Regression***
#### Simple Linear regression
```python
import numpy as np
import matplotlib.pyplot as plt

def cost():
    c = 0
    for i in range(len(X)) :
        c += (W * X[i] - Y[i]) ** 2
    return c / len(X)

def gradient_W():
    return np.sum(np.multiply(np.multiply(W, X) + b - Y, X))

def gradient_b():
    return np.sum(np.multiply(np.multiply(W, X) + b - Y, 1))

# data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 2, 3, 4, 5])

# parameters
W = 2.5; b = 1;
alpha = 0.01; beta = 0.1;
fig, axes = plt.subplots(1,2,figsize=(15,5))

# gradient descent
epochs = 5;
curr_cost = []; step = [];
for i in range(epochs):
    # update
    W = W - np.multiply(alpha, gradient_W()); print('W = ', W)
    b = b - np.multiply(beta, gradient_b()); print('b = ', b)
    
    # visualize results
    curr_cost.append(cost())
    step.append(i+1)
    axes[1].plot(X, W*X + b)
axes[0].plot(step, curr_cost)    
axes[1].plot(X,Y, 'o')
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73160858-7924a580-412d-11ea-8c35-722f781481d4.png)


<br><br><br>
#### Multi-variable regression
```python

```

<br><br><br>
#### Logistic regression
```python

```

<br><br><br>
#### Soft-max regression
```python

```

<br><br><br>

---

### ***FCN***
#### FCN through numerical method
```python
import time
import numpy as np

epsilon = 0.0001

def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(h, y):
    return 1 / 2 * np.mean(np.square(h - y))


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

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10, num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))
```
<br><br><br>

#### FCN through backpropagation
```python
import time
import numpy as np

def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)

class Sigmoid:
    def __init__(self):
        self.last_o = 1

    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    def grad(self): # sigmoid(x)(1-sigmoid(x))
        return self.last_o * (1 - self.last_o)

class MeanSquaredError:
    def __init__(self):
        # gradient
        self.dh = 1
        self.last_diff = 1

    def __call__(self, h, y): # 1/2 * mean ((h - y)^2)
        self.last_diff = h - y
        return 1 / 2 * np.mean(np.square(h - y))

    def grad(self): # h - y
        return self.last_diff

class Neuron:
    def __init__(self, W, b, a_obj):
        # Model parameters
        self.W = W
        self.b = b
        self.a = a_obj()

        # gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))

        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)

    def grad(self): # dy/dh = W
        return self.W * self.a.grad()

    def grad_W(self, dh):
        grad = np.ones_like(self.W)
        grad_a = self.a.grad()
        for j in range(grad.shape[1]): # dy/dw = x
            grad[:, j] = dh[j] * grad_a[j] * self.last_x
        return grad

    def grad_b(self, dh): # dy/dh = 1
        return dh * self.a.grad()

class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden Layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)

        # back-prop loop
        for i in range(len(self.sequence) - 1, 0, -1):
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]

            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)

        self.sequence.remove(loss_obj)

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))
```


---

### ***CNN***

<br><br><br>

---

### ***RNN***


<br><br><br>

<hr class="division2">

## **Tensorflow**

### ***Regression***
#### Simple Linear regression
```python
import tensorflow as tf
import matplotlib.pyplot as plt

def cost():
    return tf.reduce_mean(tf.square(W * X + b - Y))

def W_grad():
    return tf.reduce_mean(tf.multiply(tf.multiply(W, X) + b - Y, X))

def b_grad():
    return tf.reduce_mean(tf.multiply(tf.multiply(W, X) + b - Y, X))

# data
X = [1., 2., 3., 4., 5.]
Y = [1., 3., 5., 7., 9.]

# parameters
W = tf.Variable([5.0]); b = tf.Variable([1.0]);
alpha = 0.05; beta = 0.05;
fig, axes = plt.subplots(1,2, figsize=(10,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    curr_grad = W - tf.multiply(alpha, W_grad()); W.assign(curr_grad); print('W = ', W.numpy())
    curr_grad = b - tf.multiply(beta, b_grad()); b.assign(curr_grad); print('b = ', b.numpy())
    
    # visualize results
    step.append(i+1)
    curr_cost.append(cost())
    axes[1].plot(X, W*X + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X, Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73218855-22a37f80-419e-11ea-8949-2c8a6eb94ff8.png)

<span class="frame3">with GradientTape</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt
#tf.enable_eager_execution()

# data
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

# parameters
W = tf.Variable(2.9); b = tf.Variable(0.5);
alpha = 0.03; beta = 0.03;
fig, axes = plt.subplots(1,2, figsize=(10,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = W * X + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(alpha * W_grad); print('W = ', W.numpy())
    b.assign_sub(beta * b_grad); print('b = ', b.numpy())

    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X, W*X + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X, Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73225270-710d4a00-41af-11ea-8e62-6682bcca5f1c.png)

<br><br><br>

#### Multi-variable regression
<span class="frame3">with GradientTape</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
X1 = [1, 0, 3, 0, 5]; X2 = [0, 2, 0, 4, 0]
Y  = [1, 2, 3, 4, 5]

# parameters
W1 = tf.Variable([1.0]); W2 = tf.Variable([1.0]); b = tf.Variable([1.0]);
alpha1 = tf.Variable(0.03); alpha2 = tf.Variable(0.03); beta = tf.Variable(0.03);
fig, axes = plt.subplots(1,3,figsize=(15,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = W1*X1 + W2*X2 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    W1_grad, W2_grad, b_grad = tape.gradient(cost, [W1, W2, b])
    W1.assign_sub(alpha1 * W1_grad); print('W1 = ', W1.numpy())
    W2.assign_sub(alpha2 * W2_grad); print('W2 = ', W2.numpy())
    b.assign_sub(beta * b_grad); print('b = ', b.numpy())
    
    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X1, W1*X1 + W2*X2 + b)
    axes[2].plot(X2, W1*X1 + W2*X2 + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X1, Y, 'x')
axes[2].plot(X2, Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()    
```
![image](https://user-images.githubusercontent.com/52376448/73228320-98691480-41b9-11ea-8db7-770453b0782c.png)
<span class="frame3">with GradientTape, vectorization(matrix)</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
X = [[1., 0., 3., 0., 5.],
     [0., 2., 0., 4., 0.]]
Y  = [1, 2, 3, 4, 5]

# parameters
W = tf.Variable([[1.0, 1.0]]); b = tf.Variable([1.0]);
learning_rate = tf.Variable(0.05)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X) + b # (1, 2) * (2, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad); print(W.numpy())
    b.assign_sub(learning_rate * b_grad); print(b.numpy())
        
    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X[0], W[0][0]*X[0] + W[0][1]*X[1] + b)
    axes[2].plot(X[1], W[0][0]*X[0] + W[0][1]*X[1] + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X[0], Y, 'x')
axes[2].plot(X[1], Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73231221-285f8c00-41c3-11ea-9373-45eda45ded77.png)

<span class="frame3">with GradientTape, vectorization(matrix), optimizer(update)</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
X = [[1., 0., 3., 0., 5.],
     [0., 2., 0., 4., 0.]]
Y  = [1, 2, 3, 4, 5]

# parameters
W = tf.Variable([[1.0, 1.0]]); b = tf.Variable([1.0]);
learning_rate = tf.Variable(0.05)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
optimizer = tf.keras.optimizers.SGD(learning_rate)
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X) + b # (1, 2) * (2, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    optimizer.apply_gradients(grads_and_vars=zip([W_grad, b_grad],[W, b])); print('W = ', W.numpy(),'b = ',b.numpy())
        
    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X[0], W[0][0]*X[0] + W[0][1]*X[1] + b)
    axes[2].plot(X[1], W[0][0]*X[0] + W[0][1]*X[1] + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X[0], Y, 'x')
axes[2].plot(X[1], Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73234429-68c40780-41cd-11ea-862e-27697e67da64.png)

<span class="frame3">with GradientTape, vectorization(matrix), implicit bias(b)</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
X = [[1., 1., 1., 1., 1.],   # bias(b)
     [1., 0., 3., 0., 5.],   # feature 1
     [0., 2., 0., 4., 0.]]   # feature 2
Y  = [1, 2, 3, 4, 5]

# parameters
W = tf.Variable([[1.0, 1.0, 1.0]])
learning_rate = tf.Variable(0.05)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X)   # (1, 3) * (3, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad = tape.gradient(cost, [W])
    W.assign_sub(learning_rate * W_grad[0]);print(W.numpy())
        
    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X[1], W[0][1]*X[1] + W[0][2]*X[2] + W[0][0]*X[0])
    axes[2].plot(X[2], W[0][1]*X[1] + W[0][2]*X[2] + W[0][0]*X[0])
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X[1], Y, 'x')
axes[2].plot(X[2], Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73232988-6f9c4b80-41c8-11ea-806f-478e15f37f55.png)

<span class="frame3">with GradientTape, vectorization(matrix), implicit bias(b), optimizer(update)</span><br>
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
X = [[1., 1., 1., 1., 1.],   # bias(b)
     [1., 0., 3., 0., 5.],   # feature 1
     [0., 2., 0., 4., 0.]]   # feature 2
Y  = [1, 2, 3, 4, 5]

# parameters
W = tf.Variable([[1.0, 1.0, 1.0]])
learning_rate = tf.Variable(0.05)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
optimizer = tf.keras.optimizers.SGD(learning_rate)
for i in range(epochs):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X)   # (1, 3) * (3, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad = tape.gradient(cost, [W])
    optimizer.apply_gradients(grads_and_vars=zip(W_grad,[W])); print(W.numpy())
        
        
    # visualize results
    curr_cost.append(cost)
    step.append(i+1)
    axes[1].plot(X[1], W[0][1]*X[1] + W[0][2]*X[2] + W[0][0]*X[0])
    axes[2].plot(X[2], W[0][1]*X[1] + W[0][2]*X[2] + W[0][0]*X[0])
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X[1], Y, 'x')
axes[2].plot(X[2], Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73234120-2d750900-41cc-11ea-8503-d029baaa9dc6.png)

<br><br><br>
#### Logistic regression
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# data
x_train = tf.constant([[1., 2.],
                       [2., 3.],
                       [3., 1.],
                       [4., 3.],
                       [5., 3.],
                       [6., 2.]])
y_train = tf.constant([[0.],
                       [0.],
                       [0.],
                       [1.],
                       [1.],
                       [1.]])
x_test = tf.constant([[5.,2.]])
y_test = tf.constant([[1.]])
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))#.repeat()

# parameters
W = tf.Variable(tf.zeros([2,1])); b = tf.Variable(tf.zeros([1]));
learning_rate = 0.001
fig, axes = plt.subplots(1,3,figsize=(15,5))

# gradient descent
epochs = 1000
curr_cost = []; step = [];
optimizer = tf.keras.optimizers.SGD(learning_rate)
for i in range(epochs):
    for features, labels  in iter(dataset):
        with tf.GradientTape() as tape:
            hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
            cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
        grads = tape.gradient(cost, [W,b])
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b])); print(W.numpy(), b.numpy())

        # visualize results
        curr_cost.append(cost)
        step.append(i+1)
        axes[1].plot(features[:,0], tf.divide(1., 1. + tf.exp(tf.matmul(features,W) + b)))
        axes[2].plot(features[:,1], tf.divide(1., 1. + tf.exp(tf.matmul(features,W) + b)))
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(x_train[:,0], y_train, 'x')
axes[2].plot(x_train[:,1], y_train, 'x')
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()

hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(x_test, W) + b))
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_test), dtype=tf.int32))
print(predicted)
```
![image](https://user-images.githubusercontent.com/52376448/73317806-b8acd800-427a-11ea-8356-322562b04944.png)

<br><br><br>
#### Soft-max regression
```python

```

<br><br><br>

---

### ***Perceptron***
#### OR
```python

```
<br><br><br>

#### XOR
```python

```

---

### ***FCN***
#### Beginner mode
```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(2000, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1000, activation='relu'),
                                    tf.keras.layers.Dense(500, activation='relu'),
                                    tf.keras.layers.Dense(200, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
<br><br><br>

#### Expert mode
```python
```

---

### ***CNN***
#### Beginner mode
```python
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential

EPOCHS = 10

def MyModel():
    return Sequential([Conv2D(32, (3, 3), padding='same', activation='relu'), # 28x28x32
                       MaxPool2D(), # 14x14x32
                       Conv2D(64, (3, 3), padding='same', activation='relu'), # 14x14x64
                       MaxPool2D(), # 7x7x64
                       Conv2D(128, (3, 3), padding='same', activation='relu'), # 7x7x128
                       Flatten(), # 6272
                       Dense(128, activation='relu'),
                       Dense(10, activation='softmax')]) # 128


fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# NHWC
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32).prefetch(2048)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(2048)


model = MyModel()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)            
```

<br><br><br>
#### Expert mode
```python
import tensorflow as tf
import numpy as np

EPOCHS = 10

class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.MaxPool2D
        self.sequence = list()
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16
        self.sequence.append(maxpool((2,2))) # 14x14x16
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(maxpool((2,2))) # 7x7x32
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        self.sequence.append(tf.keras.layers.Flatten()) # 1568
        self.sequence.append(tf.keras.layers.Dense(128, activation='relu'))
        self.sequence.append(tf.keras.layers.Dense(10, activation='softmax'))

    def call(self, x, training=False, mask=None):
        for layer in self.sequence:
            x = layer(x)
        return x

# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train : (NUM_SAMPLE, 28, 28) -> (NUM_SAMPLE, 28, 28, 1)
x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Create model
model = ConvNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```


<br><br><br>


---

### ***CNN(DNN)***
#### Beginner mode
```python

```

<br><br><br>
#### Expert mode
```python
import tensorflow as tf
import numpy as np

EPOCHS = 10

class DenseUnit(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(DenseUnit, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False, mask=None): # x: (Batch, H, W, Ch_in)
        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h) # h: (Batch, H, W, filter_output)
        return self.concat([x, h]) # (Batch, H, W, (Ch_in + filter_output))

class DenseLayer(tf.keras.Model):
    def __init__(self, num_unit, growth_rate, kernel_size):
        super(DenseLayer, self).__init__()
        self.sequence = list()
        for idx in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

class TransitionLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(TransitionLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, x, training=False, mask=None):
        x = self.conv(x)
        return self.pool(x)

class DenseNet(tf.keras.Model):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu') # 28x28x8

        self.dl1 = DenseLayer(2, 4, (3, 3)) # 28x28x16
        self.tr1 = TransitionLayer(16, (3, 3)) # 14x14x16

        self.dl2 = DenseLayer(2, 8, (3, 3)) # 14x14x32
        self.tr2 = TransitionLayer(32, (3, 3)) # 7x7x32

        self.dl3 = DenseLayer(2, 16, (3, 3)) # 7x7x64

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)

        x = self.dl1(x, training=training)
        x = self.tr1(x)

        x = self.dl2(x, training=training)
        x = self.tr2(x)

        x = self.dl3(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Create model
model = DenseNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```


<br><br><br>

---


### ***RNN(LSTM)***
#### Beginner mode
```python
import tensorflow as tf

EPOCHS = 10
NUM_WORDS = 10000

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        self.rnn = tf.keras.layers.SimpleRNN(32)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)


imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=0,
                                                        padding='pre',
                                                        maxlen=32)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


model = MyModel()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
```

<br><br><br>
#### Expert mode
```python
import tensorflow as tf

EPOCHS = 10
NUM_WORDS = 10000

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        self.rnn = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)

# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=32)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                      value=0,
                                                      padding='pre',
                                                      maxlen=32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Create model
model = MyModel()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


for epoch in range(EPOCHS):
    for seqs, labels in train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_seqs, test_labels in test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```

<br><br><br>

---

### ***GAN***
#### Beginner mode
```python

```

<br><br><br>
#### Expert mode
```python

```


<br><br><br>

---


### ***ResNET***
#### Beginner mode
```python

```

<br><br><br>
#### Expert mode
```python
import tensorflow as tf
import numpy as np

EPOCHS = 10

class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding='same')

    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()
        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu') # 28x28x8

        self.res1 = ResnetLayer(8, (16, 16), (3, 3)) # 28x28x16
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2)) # 14x14x16

        self.res2 = ResnetLayer(16, (32, 32), (3, 3)) # 14x14x32
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2)) # 7x7x32

        self.res3 = ResnetLayer(32, (64, 64), (3, 3)) # 7x7x64

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)

        x = self.res1(x, training=training)
        x = self.pool1(x)
        x = self.res2(x, training=training)
        x = self.pool2(x)
        x = self.res3(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Create model
model = ResNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```

<br><br><br>

---

### ***Attention Net***
#### Beginner mode
```python

```

<br><br><br>
#### Expert mode
[chatbot_data.zip][1]
```python
import random
import tensorflow as tf
from konlpy.tag import Okt

EPOCHS = 200
NUM_WORDS = 2000

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

    def call(self, x, training=False, mask=None):
        x = self.emb(x)
        H, h, c = self.lstm(x)
        return H, h, c

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.att = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, s0, c0, H = inputs
        x = self.emb(x)
        S, h, c = self.lstm(x, initial_state=[s0, c0])

        S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
        A = self.att([S_, H])
        y = tf.concat([S, A], axis=-1)

        return self.dense(y), h, c

class Seq2seq(tf.keras.Model):
    def __init__(self, sos, eos):
        super(Seq2seq, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = sos
        self.eos = eos

    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            H, h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c, H))
            return y
        else:
            x = inputs
            H, h, c = self.enc(x)

            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1))

            seq = tf.TensorArray(tf.int32, 64)

            for idx in tf.range(64):
                y, h, c = self.dec([y, h, c, H])
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                seq = seq.write(idx, y)

                if y == self.eos:
                    break

            return tf.reshape(seq.stack(), (1, 64))

# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    output_labels = labels[:, 1:]
    shifted_labels = labels[:, :-1]
    with tf.GradientTape() as tape:
        predictions = model([inputs, shifted_labels], training=True)
        loss = loss_object(output_labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(output_labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, inputs):
    return model(inputs, training=False)



dataset_file = 'chatbot_data.csv' # acquired from 'http://www.aihub.or.kr' and modified
okt = Okt()

with open(dataset_file, 'r') as file:
    lines = file.readlines()
    seq = [' '.join(okt.morphs(line)) for line in lines]

questions = seq[::2]
answers = ['\t ' + lines for lines in seq[1::2]]

num_sample = len(questions)

perm = list(range(num_sample))
random.seed(0)
random.shuffle(perm)

train_q = list()
train_a = list()
test_q = list()
test_a = list()

for idx, qna in enumerate(zip(questions, answers)):
    q, a = qna
    if perm[idx] > num_sample//5:
        train_q.append(q)
        train_a.append(a)
    else:
        test_q.append(q)
        test_a.append(a)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')

tokenizer.fit_on_texts(train_q + train_a)

train_q_seq = tokenizer.texts_to_sequences(train_q)
train_a_seq = tokenizer.texts_to_sequences(train_a)

test_q_seq = tokenizer.texts_to_sequences(test_q)
test_a_seq = tokenizer.texts_to_sequences(test_a)

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_q_seq,
                                                        value=0,
                                                        padding='pre',
                                                        maxlen=64)
y_train = tf.keras.preprocessing.sequence.pad_sequences(train_a_seq,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=65)

x_test = tf.keras.preprocessing.sequence.pad_sequences(test_q_seq,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=64)
y_test = tf.keras.preprocessing.sequence.pad_sequences(test_a_seq,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=65)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32).prefetch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1).prefetch(1024)


# Create model
model = Seq2seq(sos=tokenizer.word_index['\t'],
                eos=tokenizer.word_index['\n'])

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


for epoch in range(EPOCHS):
    for seqs, labels in train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))

    train_loss.reset_states()
    train_accuracy.reset_states()

for test_seq, test_labels in test_ds:
    prediction = test_step(model, test_seq)
    test_text = tokenizer.sequences_to_texts(test_seq.numpy())
    gt_text = tokenizer.sequences_to_texts(test_labels.numpy())
    texts = tokenizer.sequences_to_texts(prediction.numpy())
    print('_')
    print('q: ', test_text)
    print('a: ', gt_text)
    print('p: ', texts)
   
```


<br><br><br>

---

### ***Transfer learning***
#### Beginner mode
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

EPOCHS = 100

def MyModel():
    feat = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                             include_top=False)
    feat.trainable = False
    
    seq = tf.keras.models.Sequential()
    seq.add(feat) # h x w x c 
    seq.add(tf.keras.layers.GlobalAveragePooling2D()) # c
    seq.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return seq

split = tfds.Split.TRAIN.subsplit(weighted=(8, 2))
dataset, meta = tfds.load('cats_vs_dogs',
                          split=list(split),
                          with_info=True,
                          as_supervised=True)

train_ds, test_ds = dataset

l2s = meta.features['label'].int2str
for img, label in test_ds.take(2):
    plt.figure()
    plt.imshow(img)
    plt.title(l2s(label))

def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, (224, 224))
    return img, label

train_ds = train_ds.map(preprocess).batch(32).prefetch(1024)
test_ds = test_ds.map(preprocess).batch(32).prefetch(1024)


model = MyModel()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
```

<br><br><br>
#### Expert mode
```python

```


<br><br><br>
<hr class="division2">


## **Pytorch**

### ***Regression***
#### Simple Linear regression
```python
import torch
import matplotlib.pyplot as plt

def cost():
    return torch.mean((W*X + b - Y) ** 2)

def W_grad():
    return torch.sum((W*X + b - Y) * X)

def b_grad():
    return torch.sum((W*X + b - Y) * 1)

# data
X = torch.FloatTensor([[1], [2], [3]])
Y = torch.FloatTensor([[1], [2], [3]])

# parameters
W = torch.zeros(1); b = torch.zeros(1);
alpha = 0.1; beta = 0.1;
fig, axes = plt.subplots(1,2, figsize=(10,5))

# gradient descent
epochs = 5
curr_cost = []; step = [];
for i in range(epochs):
    W -= alpha * W_grad(); print('W =', W.item())
    b -= beta * b_grad(); print('b =', b.item())
    
    # visualize results
    step.append(i+1)
    curr_cost.append(cost().item())
    axes[1].plot(X, W*X + b)
axes[0].plot(step, curr_cost, marker='o', ls='-')
axes[1].plot(X, Y, 'x')
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/73321998-6625e880-4287-11ea-92fb-64415ece50b3.png)
<span class="frame3">with optimizer</span><br>
```python

```

<br><br><br>
#### Multi-variable regression
```python

```

<br><br><br>
#### Logistic regression
```python

```

<br><br><br>
#### Soft-max regression
```python

```

<br><br><br>

---

### ***FCN***
```python
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms


seed = 1

lr = 0.001
momentum = 0.5

batch_size = 64
test_batch_size = 64

epochs = 5

no_cuda = False
log_interval = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
torch.manual_seed(seed)

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


for epoch in range(1, epochs + 1):
    # Train Mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # backpropagation 계산하기 전에 0으로 기울기 계산
        output = model(data)
        loss = F.nll_loss(output, target)  # https://pytorch.org/docs/stable/nn.html#nll-loss
        loss.backward()  # 계산한 기울기를 
        optimizer.step()  

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Test mode
    model.eval()  # batch norm이나 dropout 등을 train mode 변환
    test_loss = 0
    correct = 0
    with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # pred와 target과 같은지 확인

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
<br><br><br>

---

### ***CNN***

<br><br><br>

---

### ***RNN***

<br><br><br>

---

### ***GAN***

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


[1]:{{ site.url }}/download/AI03/chatbot_data.zip

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


