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
<span class="frmae3">GPU Resource info</span>
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

<span class="frmae3">Deallocate memory on GPU</span>
```bash
$ nvidia-smi --gpu-reset -i 0
```
```
# forcely
$ kill -9 [PID_num]
```




<br><br><br>
<hr class="division2">
## **Numpy**
### ***FCN***

<br><br><br>

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
#### One GPU(default)
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

#### One GPU with CPU
```python
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

try:
    with tf.device('/device:CPU:0'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    with tf.device('/device:GPU:2'):
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

except RuntimeError as e:
    print(e)
```
<br><br><br>

#### Multi-GPU
```python
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

try:
    with tf.device('/device:CPU:0'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    with tf.device('/device:GPU:2'):
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

except RuntimeError as e:
    print(e)
```

<br><br><br>
#### Multi-GPU with CPU
```python
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_logical_devices('GPU')
if gpus:
    with tf.device('/CPU:0'):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    for gpu in gpus:
        with tf.device(gpu.name):
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

---

### ***CNN***
#### One GPU default
```python

```

<br><br><br>
#### One GPU with CPU
```python

```

<br><br><br>

#### Multi-GPU
```python

```

<br><br><br>
#### Multi-GPU with CPU
```python

```

<br><br><br>


---

### ***RNN***
#### One GPU default
```python

```

<br><br><br>
#### One GPU with CPU
```python

```

<br><br><br>

#### Multi-GPU
```python

```

<br><br><br>
#### Multi-GPU with CPU
```python

```

<br><br><br>

---

### ***GAN***
#### One GPU default
```python

```

<br><br><br>
#### One GPU with CPU
```python

```

<br><br><br>

#### Multi-GPU
```python

```

<br><br><br>
#### Multi-GPU with CPU
```python

```

<br><br><br>

<hr class="division2">


## **Pytorch**

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


