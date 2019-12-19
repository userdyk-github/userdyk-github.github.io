---
layout : post
title : AI03, Artificial neural networks
categories: [AI03]
comments : true
tags : [AI03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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
<hr class="division2">
## **Numpy**
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

```

<br><br><br>

---

### ***Attention Net***
#### Beginner mode
```python

```

<br><br><br>
#### Expert mode
```python

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
### ***FCN***
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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


