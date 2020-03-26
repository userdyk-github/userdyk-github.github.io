---
layout : post
title : PL03-Topic02, TensorFlow_version2
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/2019-08-13-PL03-Topic02-TensorFlow_version2.md" target="_blank">page management</a><br>
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
import tensorflow as tf

print(tf.__version__)
```
<br><br><br>


<hr class="division2">

## **Tutorials**
### ***Dataset : loader***

<br><br><br>

---

### ***Neural net : Custom layers***
#### Sequential 
<a href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential" target="_blank">URL</a>
```python

```
<br><br><br>

#### Module 
<a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model" target="_blank">URL</a>
```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
<br><br><br>

---

### ***Optimization : Training***

<br><br><br>

---

### ***Evaluation : Predicting***

<br><br><br>
<hr class="division2">

## **Tensor operation**

### ***Create Tensor***
<a href="https://www.tensorflow.org/api_docs/python/tf/dtypes/DType" target="_blank">tf.dtypes</a><br>
```
>>> import numpy as np
>>> import tensorflow as tf

# list -> tensor
>>> tf.constant([1, 2, 3])
<tf.Tensor: id=0, shape=(3,), dtype=int32, numpy=array([1, 2, 3])>

# tuple -> tensor
>>> tf.constant(((1,2,3), (1,2,3)))
<tf.Tensor: id=1, shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [1, 2, 3]])>
       
# array -> tensor
>>> tf.constant(np.array([1,2,3]))
<tf.Tensor: id=2, shape=(3,), dtype=int32, numpy=array([1, 2, 3])>
```
<br><br><br>

---

### ***Confirm the information about tensor***

```
>>> import numpy as np
>>> import tensorflow as tf

# check shape
>>> tensor = tf.constant(np.array([1,2,3]))
>>> tensor.shape
TensorShape([3])

# check data type
>>> tensor.dtype
tf.int32

# define data type
>>> tf.constant([1,2,3], dtype=tf.float32)
<tf.Tensor: id=4, shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>

# convert data type
>>> tensor = tf.constant([1,2,3], dtype=tf.float32)
>>> tf.cast(tensor, dtype=tf.unit8)
<tf.Tensor: id=6, shape=(3,), dtype=uint8, numpy=array([1, 2, 3], dtype=uint8)>

# (1) : tensor -> numpy
>>> tensor.numpy()
array([1., 2., 3.], dtype=float32)

# (2) : tensor -> numpy
>>> np.array(tensor)
array([1., 2., 3.], dtype=float32)
```
<br><br><br>

---

### ***Generate random numbers***

```
>>> import tensorflow as tf

# normal distribution
>>> tf.random.normal([3,3])
<tf.Tensor: id=12, shape=(3, 3), dtype=float32, numpy=
array([[ 0.09256658, -0.8121212 , -0.7272139 ],
       [ 0.98095334, -0.5709948 , -1.6302806 ],
       [-1.2910917 , -0.72114223,  0.0984603 ]], dtype=float32)>
       
# uniform distribution
>>> tf.random.uniform([4,4])
<tf.Tensor: id=19, shape=(4, 4), dtype=float32, numpy=
array([[0.86863124, 0.38861847, 0.7144052 , 0.07352793],
       [0.9975059 , 0.08511567, 0.8157798 , 0.39816856],
       [0.7468585 , 0.01785278, 0.00612283, 0.17590272],
       [0.40437186, 0.32082295, 0.03417969, 0.3017025 ]], dtype=float32)>
```

<br><br><br>

<hr class="division2">

## **Load Dataset**

### ***Load data***

```
>>> from tensorflow.keras import datasets
>>> mnist = datasets.mnist
>>> (train_x, train_y), (test_x, test_y) = mnist.load_data()
>>> train_x.shape
(60000, 28, 28)
```
<br><br><br>

---

### ***Image dataset***

```
>>> from tensorflow.keras import datasets
>>> mnist = datasets.mnist
>>> (train_x, train_y), (test_x, test_y) = mnist.load_data()

# extract and check one data
>>> image = train_x[0]
>>> image.shape
(28, 28)

>>> plt.imshow(image, 'gray')
>>> plt.show()
```
![다운로드](https://user-images.githubusercontent.com/52376448/65838278-7ca14f00-e33c-11e9-9b03-9b6225d30600.png)

<br><br><br>

---

### ***Channel***

```
>>> import numpy as np
>>> import tensorflow as tf
>>> from tensorflow.keras import datasets
>>> import matplotlib.pyplot as plt
>>> mnist = datasets.mnist
>>> (train_x, train_y), (test_x, test_y) = mnist.load_data()


# check dataset shape
>>> train_x.shape

# (1) : expand dimension by numpy
>>> dataset_with_channel = np.expand_dims(train_x, 0)
>>> dataset_with_channel.shape
(1, 60000, 28, 28)
>>> dataset_with_channel = np.expand_dims(train_x, 1)
>>> dataset_with_channel.shape
(60000, 1, 28, 28)
>>> dataset_with_channel = np.expand_dims(train_x, 2)
>>> dataset_with_channel.shape
(60000, 28, 1, 28)
>>> dataset_with_channel = np.expand_dims(train_x, -1)
>>> dataset_with_channel.shape
(60000, 28, 28, 1)

# (2) : expand dimension by tensorflow
>>> dataset_with_channel = tf.expand_dims(train_x, -1)
>>> dataset_with_channel.shape
TensorShape([60000, 28, 28, 1])

# (3) : expand dimension by tensorflow
>>> dataset_with_channel = train_x[..., tf.newaxis]
>>> dataset_with_channel.shape
(60000, 28, 28, 1)
>>> dataset_with_channel = train_x[tf.newaxis, ...]
>>> dataset_with_channel.shape
(1, 60000, 28, 28)
>>> dataset_with_channel = train_x[tf.newaxis, tf.newaxis, ...]
>>> dataset_with_channel.shape
(1, 1, 60000, 28, 28)
>>> dataset_with_channel = train_x[..., tf.newaxis, tf.newaxis]
>>> dataset_with_channel.shape
(60000, 28, 28, 1, 1)

# (4) : expand dimension by tensorflow
>>> dataset_with_channel = train_x.reshape([60000, 28, 28, 1])
>>> dataset_with_channel.shape
(60000, 28, 28, 1)

# shrink dimension(1)
>>> dataset_with_channel = train_x.reshape([60000, 28, 28, 1])
>>> dataset_without_channel = np.squeeze(dataset_with_channel)
>>> dataset_without_channel.shape
(60000, 28, 28)

# shrink dimension(2)
>>> dataset_with_channel = train_x.reshape([60000, 28, 28, 1])
>>> dataset_without_channel = dataset_with_channel[:, :, :, 0]
>>> dataset_without_channel.shape
(60000, 28, 28)

# visualization
>>> dataset_with_channel = train_x.reshape([60000, 28, 28, 1])
>>> disp = dataset_with_channel[0, :, :, 0]
>>> plt.imshow(disp, 'gray')
>>> plt.show()
```
![다운로드](https://user-images.githubusercontent.com/52376448/65838278-7ca14f00-e33c-11e9-9b03-9b6225d30600.png)


<br><br><br>

---

### ***OneHot Encoding***

```
>>> from tensorflow.keras.utils import to_categorical
>>> to_categorical(1,8)
array([0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)

>>> to_categorical(1,9)
array([0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

>>> to_categorical(1,10)
array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

>>> to_categorical(2,10)
array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

>>> to_categorical(3,10)
array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)
```

<br><br><br>
<hr class="division2">

## **Layer and Parameters**

```
```

<br><br><br>
<hr class="division2">

## **Optimizaer and Training**

<hr class="division2">

## **Evaluating and Predicting**

<hr class="division2">

## **Framework**

### ***Import and organize dataset***

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

<br><br><br>

---

### ***Define network architecture***

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28,28))
        self.d1 = tf.keras.layers.Dense(128, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
```

<br><br><br>

---

### ***Implement training loop***

```python
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
```

<br><br><br>

---

### ***Implement algorithm test***

```python
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

<br><br><br>

---

### ***Deep learning***

<details markdown="1">
<summary class='jb-small' style="color:blue">Pre-define</summary>
<hr class='division3'>
```python
import tensorflow as tf


# Define network architecture
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28,28))
        self.d1 = tf.keras.layers.Dense(128, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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
    
    
# Import and organize dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```
<hr class='division3'>
</details>

```python
# create model
model = MyModel()

# define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

# do training loop and test
EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
    
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
    
    template = 'Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
```

<br><br><br>

---

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Santanu Pattanayak, Pro Deep Learning with TensorFlow, 2017
- <a href='https://www.tensorflow.org/api_docs/python/tf' target="_blank">tensorflow API</a>
- <a href='https://wikidocs.net/book/2324' target="_blank">wikidocs, tensorflow</a>
- <a href='https://tensorflowkorea.gitbooks.io/tensorflow-kr/' target="_blank">tensorflow-kr</a>

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
