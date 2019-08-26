---
layout : post
title : AI01, Image data preparation
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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

## How to Load and Manipulate Images With PIL,Pillow

### How to Load and Display Images

```python
# load and show an image with Pillow 
from PIL import Image 

# load the image 
image = Image.open('opera_house.jpg') 

# summarize some details about the image 
print(image.format)
print(image.mode)
print(image.size)

# show the image 
image.show()
```
`OUTPUT`
<br><br><br>

---

### How to Convert Images to NumPy Arrays and Back

`Matplotlib`
```python
# load and display an image with Matplotlib 
from matplotlib import image
from matplotlib import pyplot

# load image as pixel array 
data = image.imread('opera_house.jpg') 
print(type(data))

# summarize shape of the pixel array 
print(data.dtype) 
print(data.shape) 

# display the array of pixels as an image
pyplot.imshow(data) 
pyplot.show()
```
`OUTPUT`
<br><br><br>

`PIL`
```python
# load image and convert to and from NumPy array 
from PIL import Image
from numpy import asarray

# load the image 
image = Image.open('opera_house.jpg') 
print(type(image))

# convert image to numpy array 
data = asarray(image) 

# summarize shape 
print(type(data))
print(data.shape)

# create Pillow image
image2 = Image.fromarray(data) 
print(type(image2))

# summarize image details 
print(image2.format) 
print(image2.mode)
print(image2.size)

image2
```
`OUTPUT`
<br><br><br>

`Loading all images in a directory`
```python
# load all images in a directory 
from os import listdir
from matplotlib import image

# load all images in a directory 
loaded_images = list() 

for filename in listdir('images'):
    # load image
    img_data = image.imread('images/' + filename) 
    
    # store loaded image 
    loaded_images.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))
```
`OUTPUT`
<br><br><br>

---

### How to Save Images to File

```python

```
`OUTPUT`
<br><br><br>

---

### How to Resize Images

```python

```
`OUTPUT`
<br><br><br>

---

### How to Flip, Rotate, and Crop Images

```python

```
`OUTPUT`
<br><br><br>

---

<hr class="division2">

## How to Manually Scale Image Pixel Data

<hr class="division2">

## How to Load and Manipulate Images with Keras

<hr class="division2">

## How to Scale Image Pixel Data with Keras

<hr class="division2">

## How to Load Large Datasets From Directories with Keras

<hr class="division2">

## How to Use Image Data Augmentation in Keras

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a>Jason Brownlee, Deep Learning for Computer Vision</a>
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post3</a>

---
