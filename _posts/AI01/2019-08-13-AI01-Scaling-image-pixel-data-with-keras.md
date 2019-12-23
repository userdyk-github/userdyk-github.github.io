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

# summarize dataset type and shape
print(type(train_images), train_images.dtype, train_images.shape)
print(type(train_labels), train_labels.dtype, train_labels.shape)
print(type(test_images), test_images.dtype, test_images.shape)
print(type(test_labels), test_labels.dtype, test_labels.shape)

# summarize pixel values
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
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

```python
print(train_images[0].shape)
io.imshow(train_images[0])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">


## **How to Normalize Images With ImageDataGenerator**

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


## **How to Center Images With ImageDataGenerator**

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


## **How to Standardize Images With ImageDataGenerator**

```python

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

Text can be **bold**, _italic_, ~~strikethrough~~ or `keyword`.

[Link to another page](another-page).

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

* * *

*   Item foo
*   Item bar
*   Item baz
*   Item zip


1.  Item one
1.  Item two
1.  Item three
1.  Item four

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>


![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)
![](https://guides.github.com/activities/hello-world/branching.png)

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
