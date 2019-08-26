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

## **How to Load and Manipulate Images With PIL,Pillow**

### ***How to Load and Display Images***

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

<details markdown="1">
<summary class='jb-small'>OUTPUT</summary>
<hr class='division3'>
<p>
    JPEG<br>
    RGB<br>
    (640, 360)
</p>
![opera_house](https://user-images.githubusercontent.com/52376448/63676217-6a0b9580-c825-11e9-96fd-0d2b96f653c1.jpg)
<hr class='division3'>
</details>


<br><br><br>

---

### ***How to Convert Images to NumPy Arrays and Back***

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
    (360, 640, 3)
</p>
![다운로드](https://user-images.githubusercontent.com/52376448/63675998-03867780-c825-11e9-8986-97f3e72827cb.png)
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
    (640, 360)
</p>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63676659-64627f80-c826-11e9-8d08-691a225d1ea0.png)
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
<p>
    > loaded beauty.jpg (150, 120, 3)<br>
    > loaded opera_house.jpg (360, 640, 3)
</p>
<br><br><br>

---

### ***How to Save Images to File***

`Saving images`
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
<p>
    PNG
</p>
![캡처](https://user-images.githubusercontent.com/52376448/63678983-4f3c1f80-c82b-11e9-9553-e2f82d73f67a.JPG)
<br><br><br>

`Showing again saved images after saving images`
```python
# example of saving a grayscale version of a loaded image
from PIL import Image

# load the image 
image = Image.open('opera_house.jpg')

# convert the image to grayscale 
gs_image = image.convert(mode='L')

# save in jpeg format
gs_image.save('opera_house_grayscale.jpg')

# load the image again and show it 
image2 = Image.open('opera_house_grayscale.jpg')

# show the image 
image2.show()
```
`OUTPUT`
![opera_house_grayscale](https://user-images.githubusercontent.com/52376448/63677448-f3bc6280-c827-11e9-997c-4c3d09c99692.jpg)
<br><br><br>

---

### ***How to Resize Images***

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
<p>
    (640, 360)<br>
    (100, 56)
</p>
![resizing_opera_house](https://user-images.githubusercontent.com/52376448/63678027-43e7f480-c829-11e9-9ea4-0dffd961c6a1.png)
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
<p>
    (640, 360)<br>
    (200, 200)
</p>
![resizing_opera_house2](https://user-images.githubusercontent.com/52376448/63678380-0fc10380-c82a-11e9-9f82-a3744ef5237f.png)
<br><br><br>

---

### ***How to Flip, Rotate, and Crop Images***

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
![다운로드](https://user-images.githubusercontent.com/52376448/63678432-32ebb300-c82a-11e9-8c39-2ce84487cac1.png)
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
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63678480-4860dd00-c82a-11e9-98c4-71261d682c57.png)
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
![cropped](https://user-images.githubusercontent.com/52376448/63678860-0ab08400-c82b-11e9-9329-b77b8496da5c.png)
<br><br><br>

<hr class="division2">

## **How to Manually Scale Image Pixel Data**

### ***Normalize Pixel Values***

`STEP 1`
```python
# example of pixel normalization
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)

# confirm pixel range is 0-255
print(pixels.shape)
print(pixels.dtype)
print(pixels.min(), pixels.max())
pixels
```
`OUTPUT1`
<p>
    (856, 1280, 4)<br>
    uint8<br>
    0 255
</p>
```
array([[[221, 223, 226, 255],
        [210, 212, 215, 255],
        [191, 192, 195, 255],
        ...,
        [191, 192, 195, 255],
        [210, 212, 215, 255],
        [221, 223, 226, 255]],

       [[213, 215, 217, 255],
        [190, 192, 194, 255],
        [206, 207, 208, 255],
        ...,
        [206, 207, 208, 255],
        [190, 192, 194, 255],
        [213, 215, 217, 255]],

       [[199, 201, 204, 255],
        [196, 198, 199, 255],
        [236, 234, 236, 255],
        ...,
        [236, 234, 236, 255],
        [196, 198, 199, 255],
        [199, 201, 204, 255]],

       ...,

       [[193, 193, 193, 255],
        [180, 180, 180, 255],
        [151, 152, 152, 255],
        ...,
        [154, 154, 155, 255],
        [180, 180, 180, 255],
        [193, 193, 193, 255]],

       [[197, 197, 197, 255],
        [192, 192, 192, 255],
        [179, 179, 179, 255],
        ...,
        [179, 179, 179, 255],
        [192, 192, 192, 255],
        [197, 197, 197, 255]],

       [[198, 198, 198, 255],
        [196, 196, 196, 255],
        [192, 192, 192, 255],
        ...,
        [192, 192, 192, 255],
        [196, 196, 196, 255],
        [198, 198, 198, 255]]], dtype=uint8)
```
`STEP 2`
```python
# convert from integers to floats
pixels = pixels.astype('float32')

# normalize to the range 0-1
pixels /= 255.0

# confirm the normalization
print(pixels.min(), pixels.max())

pixels
```
`OUTPUT2`
<p>
    0.0 0.003921569
</p>
```
array([[[0.00339869, 0.00342945, 0.00347559, 0.00392157],
        [0.00322953, 0.00326028, 0.00330642, 0.00392157],
        [0.00293733, 0.00295271, 0.00299885, 0.00392157],
        ...,
        [0.00293733, 0.00295271, 0.00299885, 0.00392157],
        [0.00322953, 0.00326028, 0.00330642, 0.00392157],
        [0.00339869, 0.00342945, 0.00347559, 0.00392157]],

       [[0.00327566, 0.00330642, 0.00333718, 0.00392157],
        [0.00292195, 0.00295271, 0.00298347, 0.00392157],
        [0.00316801, 0.00318339, 0.00319877, 0.00392157],
        ...,
        [0.00316801, 0.00318339, 0.00319877, 0.00392157],
        [0.00292195, 0.00295271, 0.00298347, 0.00392157],
        [0.00327566, 0.00330642, 0.00333718, 0.00392157]],

       [[0.00306036, 0.00309112, 0.00313725, 0.00392157],
        [0.00301423, 0.00304498, 0.00306036, 0.00392157],
        [0.00362937, 0.00359862, 0.00362937, 0.00392157],
        ...,
        [0.00362937, 0.00359862, 0.00362937, 0.00392157],
        [0.00301423, 0.00304498, 0.00306036, 0.00392157],
        [0.00306036, 0.00309112, 0.00313725, 0.00392157]],

       ...,

       [[0.00296809, 0.00296809, 0.00296809, 0.00392157],
        [0.00276817, 0.00276817, 0.00276817, 0.00392157],
        [0.00232218, 0.00233756, 0.00233756, 0.00392157],
        ...,
        [0.00236832, 0.00236832, 0.0023837 , 0.00392157],
        [0.00276817, 0.00276817, 0.00276817, 0.00392157],
        [0.00296809, 0.00296809, 0.00296809, 0.00392157]],

       [[0.0030296 , 0.0030296 , 0.0030296 , 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.00275279, 0.00275279, 0.00275279, 0.00392157],
        ...,
        [0.00275279, 0.00275279, 0.00275279, 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.0030296 , 0.0030296 , 0.0030296 , 0.00392157]],

       [[0.00304498, 0.00304498, 0.00304498, 0.00392157],
        [0.00301423, 0.00301423, 0.00301423, 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        ...,
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.00301423, 0.00301423, 0.00301423, 0.00392157],
        [0.00304498, 0.00304498, 0.00304498, 0.00392157]]], dtype=float32)
```
<br><br><br>

---

### ***Center Pixel Values***

```python

```
`OUTPUT`
<br><br><br>


---

### ***Standardize Pixel Values***

```python

```
`OUTPUT`
<br><br><br>

<hr class="division2">

## **How to Load and Manipulate Images with Keras**

### ***Test Image***

```python

```
`OUTPUT`
<br><br><br>

---

### ***How to Load an Image with Keras***

```python

```
`OUTPUT`
<br><br><br>

---

### ***How to Convert an Image With Keras***

```python

```
`OUTPUT`
<br><br><br>

---

### ***How to Save an Image With Keras***

```python

```
`OUTPUT`
<br><br><br>


<hr class="division2">

## **How to Scale Image Pixel Data with Keras**

### ***MNIST Handwritten Image Classiﬁcation Dataset***

```python

```
`OUTPUT`
<br><br><br>

---


### ***ImageDataGenerator Class for Pixel Scaling***

```python

```
`OUTPUT`
<br><br><br>

---


### ***How to Normalize Images With ImageDataGenerator***

```python

```
`OUTPUT`
<br><br><br>

---


### ***How to Center Images With ImageDataGenerator***

```python

```
`OUTPUT`
<br><br><br>

---


### ***How to Standardize Images With ImageDataGenerator***

```python

```
`OUTPUT`
<br><br><br>



<hr class="division2">

## **How to Load Large Datasets From Directories with Keras**

### ***How to Progressively Load Images***

```python

```
`OUTPUT`
<br><br><br>



<hr class="division2">

## **How to Use Image Data Augmentation in Keras**

### ***Sample Image***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Image Augmentation With ImageDataGenerator***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Horizontal and Vertical Shift Augmentation***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Horizontal and Vertical Flip Augmentation***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Random Rotation Augmentation***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Random Brightness Augmentation***

```python

```
`OUTPUT`
<br><br><br>

---

### ***Random Zoom Augmentation***

```python

```
`OUTPUT`
<br><br><br>



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
