---
layout : post
title : AI01, Scaling image pixel data with keras
categories: [AI01]
comments : true
tags : [AI01]
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

## **MNIST Handwritten Image Classiﬁcation Dataset**

```python
# example of loading the MNIST dataset
from keras.datasets import mnist

# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT 1</summary>
<hr class='division3'>
```python
# summarize dataset type and shape
print(type(train_images), train_images.dtype, train_images.shape)
print(type(train_labels), train_labels.dtype, train_labels.shape)
print(type(test_images), test_images.dtype, test_images.shape)
print(type(test_labels), test_labels.dtype, test_labels.shape)

# summarize pixel values
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
<p>
  <class 'numpy.ndarray'> uint8 (60000, 28, 28)<br>
  <class 'numpy.ndarray'> uint8 (60000,)<br>
  <class 'numpy.ndarray'> uint8 (10000, 28, 28)<br>
  <class 'numpy.ndarray'> uint8 (10000,)<br>
  Train 0 255 33.318421449829934 78.56748998339798<br>
  Test 0 255 33.791224489795916 79.17246322228644<br>
</p>
<hr class='division3'>
</details>

<br>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT 2</summary>
<hr class='division3'>
```python
print(train_images[0].shape)
io.imshow(train_images[0])
```

<p>
  (28, 28)<br>
  <matplotlib.image.AxesImage at 0x23244de4fd0>
</p>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/63792062-bba44500-c937-11e9-9747-e048df95e1a6.png)
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


## **ImageDataGenerator Class for Pixel Scaling**

```python
# create data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split 


"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator()
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)


"""model design"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.fit_generator(train_iterator, validation_data=val_iterator, epochs=10, steps_per_epoch=10, validation_steps=10)


"""evaluation"""
# evaluate model loss on test dataset
result = model.evaluate_generator(test_iterator, steps=10)
for i in range(len(model.metrics_names)):  
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))
    
model.predict_generator(test_iterator)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">(train_images, train_labels), (test_images, test_labels) = mnist.load_data()</summary>
<hr class='division3'>
```python
# summarize dataset shape, pixel values for train
print('Train', train_images.shape, train_labels.shape)
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())

# summarize dataset shape, pixel values for test
print('Test', (test_images.shape, test_labels.shape))
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
```
Train (60000, 28, 28) (60000,)
Train 0 255 33.318421449829934 78.56748998339798
Test ((10000, 28, 28), (10000,))
Test 0 255 33.791224489795916 79.17246322228644
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)</summary>
<hr class='division3'>
```python
# summarize dataset shape, pixel values for train
print('Train', train_images.shape, train_labels.shape)
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())

# summarize dataset shape, pixel values for val
print('Val', valX.shape, valy.shape)
print('Val', valX.min(), valX.max(), valX.mean(), valX.std())

# summarize dataset shape, pixel values for test
print('Test', (test_images.shape, test_labels.shape))
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
```
Train (48000, 28, 28) (48000,)
Train 0 255 33.29773514562075 78.54482970203107
Val (12000, 28, 28) (12000,)
Val 0 255 33.40116666666667 78.65801142483167
Test ((10000, 28, 28), (10000,))
Test 0 255 33.791224489795916 79.17246322228644
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">train_iterator, val_iterator, test_iterator</summary>
<hr class='division3'>
```python
train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=30.790, std=75.816
val batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=34.835, std=80.186
test batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=36.032, std=81.371
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">


## **How to Normalize Images With ImageDataGenerator**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(rescale=1.0/255.0)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(rescale=1.0/255.0)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.130, std=0.307
val batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.126, std=0.302
test batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.123, std=0.302
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=0.000, max=1.000, mean=0.131, std=0.308
val batch(all) shape=(12000, 28, 28, 1), min=0.000, max=1.000, mean=0.131, std=0.308
test batch(all) shape=(10000, 28, 28, 1), min=0.000, max=1.000, mean=0.133, std=0.310
```
<hr class='division3'>
</details><br>


<br><br><br>

---

<hr class="division2">


## **How to Center Images With ImageDataGenerator**
### ***feature-wise centering***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
[[[33.29781]]] None
[[[33.40119]]] None
[[[33.79124]]] None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-1.533, std=77.921
val batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-2.625, std=76.458
test batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-3.413, std=75.195
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.494, std=78.545
val batch(all) shape=(12000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.390, std=78.658
test batch(all) shape=(10000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.000, std=79.172
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***sample-wise centering***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-54.675, max=236.723, mean=0.000, std=79.433
val batch(32) shape=(32, 28, 28, 1), min=-60.829, max=238.736, mean=0.000, std=79.335
test batch(32) shape=(32, 28, 28, 1), min=-62.398, max=242.120, mean=-0.000, std=77.976
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-90.477, max=248.513, mean=-0.000, std=77.764
val batch(all) shape=(12000, 28, 28, 1), min=-101.381, max=247.806, mean=-0.000, std=77.882
test batch(all) shape=(10000, 28, 28, 1), min=-83.435, max=247.832, mean=-0.000, std=78.383
```
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


## **How to Standardize Images With ImageDataGenerator**
### ***feature-wise Standardization***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
[[[33.29781]]] [[[78.54484]]]
[[[33.40119]]] [[[78.65801]]]
[[[33.79124]]] [[[79.172455]]]
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=0.004, std=1.006
val batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=0.029, std=1.034
test batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=-0.026, std=0.968
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.006, std=0.992
val batch(all) shape=(12000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.005, std=0.994
test batch(all) shape=(10000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.000, std=1.000
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***sample-wise Standardization***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-0.600, max=4.298, mean=0.000, std=1.000
val batch(32) shape=(32, 28, 28, 1), min=-0.554, max=4.273, mean=-0.000, std=1.000
test batch(32) shape=(32, 28, 28, 1), min=-0.585, max=4.394, mean=0.000, std=1.000
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-0.777, max=7.770, mean=-0.000, std=1.000
val batch(all) shape=(12000, 28, 28, 1), min=-0.851, max=7.249, mean=0.000, std=1.000
test batch(all) shape=(10000, 28, 28, 1), min=-0.732, max=7.578, mean=0.000, std=1.000
```
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
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---



<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
