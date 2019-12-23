---
layout : post
title : AI01, Loading large datasets from directories with keras
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

## **How to Progressively Load Images**

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
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
