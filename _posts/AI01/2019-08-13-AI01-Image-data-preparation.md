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
<p>
    JPEG<br>
    RGB<br>
    (640, 360)<br>
    ![opera_house](https://user-images.githubusercontent.com/52376448/63676217-6a0b9580-c825-11e9-96fd-0d2b96f653c1.jpg)
</p>
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
<p>
    <class 'numpy.ndarray'><br>
    uint8<br>
    (360, 640, 3)<br>
    ![다운로드](https://user-images.githubusercontent.com/52376448/63675998-03867780-c825-11e9-8986-97f3e72827cb.png)
</p>
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
<p>
    <class 'PIL.JpegImagePlugin.JpegImageFile'><br>
    <class 'numpy.ndarray'><br>
    (360, 640, 3)<br>
    <class 'PIL.Image.Image'><br>
    None<br>
    RGB<br>
    (640, 360)<br>
</p>
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
# example of saving an image in another format 
from PIL import Image

# load the image 
image = Image.open('opera_house.jpg')

# save as PNG format 
image.save('opera_house.png', format='PNG') 

# load the image again and inspect the format 
image2 = Image.open('opera_house.png')
print(image2.format)
```
`OUTPUT`
<br><br><br>

---

### How to Resize Images

`Resize images, preserving the aspect ratio`
```python
# create a thumbnail of an image 
from PIL import Image

# load the image 
image = Image.open('opera_house.jpg') 

# report the size of the image
print(image.size)

# create a thumbnail and preserve aspect ratio 
image.thumbnail((100,100)) 

# report the size of the modified image 
print(image.size) 

# show the image 
image.show()
```
`OUTPUT`
<br><br><br>

`Resize images, force the pixels into a new shape`
```python
# resize image and force a new shape 
from PIL import Image

# load the image
image = Image.open('opera_house.jpg') 

# report the size of the image 
print(image.size) 

# resize image and ignore original aspect ratio
img_resized = image.resize((200,200))

# report the size of the thumbnail 
print(img_resized.size)

# show the image 
img_resized.show()
```
`OUTPUT`
<br><br><br>

---

### How to Flip, Rotate, and Crop Images

`Flip images`
```python
# create flipped versions of an image 
from PIL import Image
from matplotlib import pyplot 

# load image 
image = Image.open('opera_house.jpg')

# horizontal flip 
hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT) 

# vertical flip 
ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM) 

# plot all three images using matplotlib 
pyplot.subplot(311)
pyplot.imshow(image)
pyplot.subplot(312)
pyplot.imshow(hoz_flip) 
pyplot.subplot(313) 
pyplot.imshow(ver_flip)
pyplot.show()
```
`OUTPUT`
<br><br><br>

`Rotate images`
```python
# create rotated versions of an image 
from PIL import Image 
from matplotlib import pyplot

# load image 
image = Image.open('opera_house.jpg')

# plot original image 
pyplot.subplot(311) 
pyplot.imshow(image) 

# rotate 45 degrees 
pyplot.subplot(312) 
pyplot.imshow(image.rotate(45))

# rotate 90 degrees
pyplot.subplot(313) 
pyplot.imshow(image.rotate(90))
pyplot.show()
```
`OUTPUT`
<br><br><br>

`Crop images`
```python
# example of cropping an image 
from PIL import Image

# load image
image = Image.open('opera_house.jpg') 

# create a cropped image 
cropped = image.crop((100, 100, 200, 200))

# show cropped image 
cropped.show()
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
