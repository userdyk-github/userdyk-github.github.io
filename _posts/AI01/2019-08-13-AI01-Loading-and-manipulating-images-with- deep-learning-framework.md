---
layout : post
title : AI01, Loading and manipulating images with deep learning framework
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Loading-and-manipulating-images-with-%20deep-learning-framework.md" target="_blank">page management</a><br>
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

## **keras**
### ***How to Load an Image with Keras***

```python
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img

# load the image
img = load_img('beach.jpg')

# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)

# show the image
img.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'PIL.JpegImagePlugin.JpegImageFile'><br>
  JPEG<br>
  RGB<br>
  (640, 427)
</p>
![beach](https://user-images.githubusercontent.com/52376448/63721646-9e666c80-c88c-11e9-97ee-096cc2a4f9d1.jpg)
<hr class='division3'>
</details>

<br><br><br>

#### Argument1 : grayscale
```python
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img

# load the image
img = load_img('beach.jpg', grayscale=True)
print(img.mode)
img
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  L
</p>
![다운로드](https://user-images.githubusercontent.com/52376448/63721802-f8ffc880-c88c-11e9-9c03-5999a37ca43b.png)
<hr class='division3'>
</details>

<br><br><br>

---

#### Argument2 : color_mode

```python
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img

# load the image
img = load_img('beach.jpg', color_mode='rgba')
print(img.mode)
img
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>RGBA</p>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63722390-0e292700-c88e-11e9-800d-dcdef10fbacc.png)
<hr class='division3'>
</details>

<br><br><br>

---

#### Argument3 : target_size

```python
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img

# load the image
img = load_img('beach.jpg', target_size=(100,100))
print(img.size)
img
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>(100, 100)</p>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/63722417-1b461600-c88e-11e9-957d-5ba8aae1318b.png)
<hr class='division3'>
</details>

<br><br><br>


<hr class="division2">


### ***How to Convert an Image With Keras***

```python
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
from PIL import Image

# load the image
img = load_img('beach.jpg')
print(type(img))

# convert to numpy array
img_ndarray = np.asarray(img).astype('float32')
img_array = img_to_array(img)
print(img_ndarray.dtype)
print(img_ndarray.shape)
print(img_array.dtype)
print(img_array.shape)

# convert back to image
img_pil1 = array_to_img(img_array)
img_pil2 = img_array.astype(np.uint8)
img_pil2 = Image.fromarray(img_pil2)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'PIL.JpegImagePlugin.JpegImageFile'><br>
  float32<br>
  (427, 640, 3)<br>
  float32<br>
  (427, 640, 3)
</p>
<hr class='division3'>
</details>

<br>

```python
print(type(img_ndarray))
print(img_ndarray.shape)
img_ndarray
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'numpy.ndarray'><br>
  (427, 640, 3)
</p>
```
array([[[ 47., 107., 195.],
        [ 47., 107., 195.],
        [ 46., 106., 194.],
        ...,
        [ 31.,  97., 191.],
        [ 30.,  96., 190.],
        [ 29.,  95., 189.]],

       [[ 46., 106., 194.],
        [ 47., 107., 195.],
        [ 47., 107., 195.],
        ...,
        [ 31.,  97., 191.],
        [ 31.,  97., 191.],
        [ 30.,  96., 190.]],

       [[ 46., 106., 194.],
        [ 48., 108., 196.],
        [ 51., 108., 197.],
        ...,
        [ 30.,  96., 190.],
        [ 31.,  97., 191.],
        [ 30.,  96., 190.]],

       ...,

       [[  1.,   1.,   3.],
        [  1.,   1.,   3.],
        [  3.,   3.,   1.],
        ...,
        [130., 149., 155.],
        [136., 155., 161.],
        [135., 152., 160.]],

       [[  0.,   1.,   0.],
        [  1.,   2.,   0.],
        [  1.,   2.,   0.],
        ...,
        [123., 143., 144.],
        [129., 148., 152.],
        [131., 148., 155.]],

       [[  1.,   0.,   5.],
        [  0.,   0.,   4.],
        [  0.,   1.,   0.],
        ...,
        [122., 142., 141.],
        [126., 146., 145.],
        [129., 147., 149.]]], dtype=float32)
```
<hr class='division3'>
</details>

<br>

```python
print(type(img_array))
print(img_array.shape)
img_array
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'numpy.ndarray'><br>
  (427, 640, 3)
</p>
```
array([[[ 47., 107., 195.],
        [ 47., 107., 195.],
        [ 46., 106., 194.],
        ...,
        [ 31.,  97., 191.],
        [ 30.,  96., 190.],
        [ 29.,  95., 189.]],

       [[ 46., 106., 194.],
        [ 47., 107., 195.],
        [ 47., 107., 195.],
        ...,
        [ 31.,  97., 191.],
        [ 31.,  97., 191.],
        [ 30.,  96., 190.]],

       [[ 46., 106., 194.],
        [ 48., 108., 196.],
        [ 51., 108., 197.],
        ...,
        [ 30.,  96., 190.],
        [ 31.,  97., 191.],
        [ 30.,  96., 190.]],

       ...,

       [[  1.,   1.,   3.],
        [  1.,   1.,   3.],
        [  3.,   3.,   1.],
        ...,
        [130., 149., 155.],
        [136., 155., 161.],
        [135., 152., 160.]],

       [[  0.,   1.,   0.],
        [  1.,   2.,   0.],
        [  1.,   2.,   0.],
        ...,
        [123., 143., 144.],
        [129., 148., 152.],
        [131., 148., 155.]],

       [[  1.,   0.,   5.],
        [  0.,   0.,   4.],
        [  0.,   1.,   0.],
        ...,
        [122., 142., 141.],
        [126., 146., 145.],
        [129., 147., 149.]]], dtype=float32)
```
<hr class='division3'>
</details>

<br>

```python
print(type(img_pil1))
print(img_pil1.format) 
print(img_pil1.mode)
print(img_pil1.size)
img_pil1
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'PIL.Image.Image'><br>
  None<br>
  RGB<br>
  (640, 427)
</p>  
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/63722987-585ed800-c88f-11e9-8edf-586712ad87d1.png)
<hr class='division3'>
</details>

<br>

```python
print(type(img_pil2))
print(img_pil2.format) 
print(img_pil2.mode)
print(img_pil2.size)
img_pil2
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'PIL.Image.Image'><br>
  None<br>
  RGB<br>
  (640, 427)
</p>  
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/63723086-9825bf80-c88f-11e9-8a45-e1e28158a1db.png)
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


### ***How to Save an Image With Keras***

```python
# example of saving an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

# load image as as grayscale
img = load_img('beach.jpg', color_mode='grayscale')

# convert image to a numpy array
img_array = img_to_array(img)

# save the image with a new filename
save_img('bondi_beach_grayscale.jpg', img_array)

# load the image to confirm it was saved correctly
img = load_img('bondi_beach_grayscale.jpg')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
img.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  <class 'PIL.Image.Image'><br>
  None<br>
  RGB<br>
  (640, 427)
</p>
![bondi_beach_grayscale](https://user-images.githubusercontent.com/52376448/63722526-50526880-c88e-11e9-98b3-d8bc432be018.jpg)
<hr class='division3'>
</details>

<br><br><br>


### ***How to Progressively Load Images***
#### flow_from_directory
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
train_iterator = datagen.flow_from_directory('data/train/', class_mode='binary', batch_size=64)
val_iterator = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
test_iterator = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=64)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
Example of proposed directory structure for the image dataset.
<div style="font-size: 70%;">
- data/ <br>
- data/train/<br>
- data/train/red/<br>
- data/train/blue/ <br><br>


- data/test/ <br>
- data/test/red/ <br>
- data/test/blue/ <br><br>


- data/validation/ <br>
- data/validation/red/<br>
- data/validation/blue/<br>
</div>

```python
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model 
from keras.models import Model 
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 


"""data preprocessing"""
# create a data generator 
datagen = ImageDataGenerator()

# Example of creating dataset iterators from an image data generator
# load and iterate training dataset 
train_iterator = datagen.flow_from_directory('data/train/', class_mode='binary', batch_size=64)
val_iterator = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
test_iterator = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=64)


"""model design"""
visible = Input(shape=(256,256,3)) 
conv1 = Conv2D(32, (4,4), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1) 
conv2 = Conv2D(16, (4,4), activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2) 
flat1 = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat1) 
output = Dense(1, activation='sigmoid')(hidden1)

model = Model(inputs=visible, outputs=output) 
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit_generator(train_iterator, epochs=10,steps_per_epoch=16, validation_data=val_it, validation_steps=8)


"""evaluation"""
loss = model.evaluate_generator(test_iterator, steps=24)
yhat = model.predict_generator(test_iterator, steps=24)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
#### flow_from_dataframe
```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

Example of proposed directory structure for the image dataset.
<div style="font-size: 70%;">
- data/ <br>
- data/train/<br>
- data/test/ <br>
- data/validation/ <br>
</div>

```python
from glob import glob
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_class_name(path):
    image_name = os.path.basename(path)
    image_name = image_name.split('_')[-1]
    image_name = image_name.replace('.png','')
    return image_name


'''data preprocessing'''
# path
train_paths = glob('cifar/train/*.png')
test_paths = glob('cifar/test/*.png')

# class name
train_classes_name = [get_class_name(path) for path in train_paths]
test_classes_name = [get_class_name(path) for path in test_paths]

# dataframe : 'path' + 'class name'
train_df = pd.DataFrame({'path':train_paths, 'class':train_classes_name})
test_df = pd.DataFrame({'path':test_paths, 'class':test_classes_name})

# save .csv(format)
train_df.iloc[:40000,:].to_csv('train_dataset.csv', index=False)
train_df.iloc[40000:50000,:].to_csv('val_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# load .csv(format)
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# generator
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.3, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='path', y_col='class', target_size=(32,32,3)[:2], batch_size=32)
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='path', y_col='class', target_size=(32,32,3)[:2], batch_size=32)
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='path', y_col='class', target_size=(32,32,3)[:2], batch_size=32)



'''model design'''
inputs = layers.Input((32,32,3))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.7)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.7)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.7)(net)
net = layers.Dense(10)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimization
              loss='categorical_crossentropy',  # Loss Function 
              metrics=['accuracy'])  # Metrics / Accuracy
model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator))


"""evaluation"""
loss = model.evaluate_generator(test_generator, steps=24)
yhat = model.predict_generator(test_generator, steps=24)
```
<br><br><br>
<hr class="division2">

## **pytorch**

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
