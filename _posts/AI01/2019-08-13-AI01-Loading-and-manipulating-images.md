---
layout : post
title : AI01, Loading and manipulating images
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Loading-and-manipulating-images.md" target="_blank">page management</a> <br>
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

## **How to Load and Display Images**

<br>

|Lib for loading an image|Loaded image type|
|:--|:--|
|keras|<class 'PIL.JpegImagePlugin.JpegImageFile'>|
|PIL|<class 'PIL.JpegImagePlugin.JpegImageFile'>|
|skimage|<class 'imageio.core.util.Array'>|
|cv2|<class 'numpy.ndarray'>|
|matplotlib|<class 'numpy.ndarray'>|

<br>

|Loaded image type|Lib for showing an image|
|:--|:--|
|<class 'PIL.JpegImagePlugin.JpegImageFile'>|keras, PIL, matplotlib|
|<class 'imageio.core.util.Array'><br><class 'numpy.ndarray'>|skimage, matplotlib|


<br><br><br>

### ***Loading and showing an image with keras***

```python
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img

# load the image
image = load_img('beach.jpg')

# report details about the image
print(type(image))
print(image.format)
print(image.mode)
print(image.size)

# show the image
image.show()
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

---

### ***Loading and showing an image with Pillow***

```python
# load and show an image with Pillow 
from PIL import Image 

# load the image 
image = Image.open('opera_house.jpg') 

# summarize some details about the image 
print(image.format)
print(image.mode)
print(image.size)
print(type(image))

# show the image 
image.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    JPEG<br>
    RGB<br>
    (640, 360)<br>
    <class 'PIL.JpegImagePlugin.JpegImageFile'>
</p>
![opera_house](https://user-images.githubusercontent.com/52376448/63676217-6a0b9580-c825-11e9-96fd-0d2b96f653c1.jpg)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Loading and showing an image with skimage***
```python
%matplotlib inline
from skimage import io                  # pip install scikit-image

image = io.imread('puppy.jpg')
print(type(image))
print(image.shape)

io.imshow(image)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <class 'imageio.core.util.Array'><br>
    (366, 487, 3)
</p>
<matplotlib.image.AxesImage at 0x22978a710f0>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/63694276-0ac37a80-c851-11e9-89df-295e2b74bb49.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Loading and showing an image with matplotlib***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <class 'numpy.ndarray'><br>
    uint8<br>
    (360, 640, 3)
</p>
![63675998-03867780-c825-11e9-8986-97f3e72827cb](https://user-images.githubusercontent.com/52376448/63707134-28eaa400-c86c-11e9-9ecc-1cfc1e1aa073.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Loading an image with cv2 and showing an image with skimage***

**import required packages**
- pip install scikit-image
- pip install opencv-contrib-python

```python
%matplotlib inline
from skimage import io
import cv2

image = cv2.imread('cat.jpg')
print(type(image))
print(image.shape)

io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <class 'numpy.ndarray'><br>
    (225, 225, 3)<br>
    <matplotlib.image.AxesImage at 0x1f1b00a26a0><br>
</p>
![다운로드](https://user-images.githubusercontent.com/52376448/63705829-f7240e00-c868-11e9-8b18-05a3409b92aa.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Loading all images in a directory***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    > loaded beauty.jpg (150, 120, 3)<br>
    > loaded opera_house.jpg (360, 640, 3)
</p>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **How to Save Images to File**
### ***Saving image with matplotlib***
```python
import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('input_image.jpg')
plt.imshow(img)
plt.figsave('output_image.jpg')
```

<br><br><br>

---

### ***Saving image with keras***

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
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
[bondi_beach_grayscale.jpg][1]

[1]:{{ site.url }}/download/AI01/bondi_beach_grayscale.jpg
<hr class='division3'>
</details>


<br><br><br>

---

### ***Saving image with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    PNG
</p>
[opera_house.png][1]

[1]:{{ site.url }}/download/AI01/opera_house.png
<hr class='division3'>
</details>


<br><br><br>

---

### ***Saving image with skimage***

```python
#Import libraries 
from skimage import io 
from skimage import color 

#Read image 
img = io.imread('puppy.jpg')

#Convert to YPbPr 
img_ypbpr= color.rgb2ypbpr(img)

#Convert back to RGB 
img_rgb= color.ypbpr2rgb(img_ypbpr)

#Save image
io.imsave("puppy_ypbpr.jpg", img_ypbpr)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    PNG
</p>
[puppy_ypbpr.jpg][1]

[1]:{{ site.url }}/download/AI01/puppy_ypbpr.jpg
<hr class='division3'>
</details>


<br><br><br>

---

### ***Saving image with Pandas(format : xlsx)***

```python
from skimage import io                  
import pandas as pd

image = io.imread('puppy.jpg')
df = pd.DataFrame(image.flatten())        # flatten() : convert the three dimensions of an RGB image to a single dimension

# saving in excel format
filepath = 'pixel_values1.xlsx'
df.to_excel(filepath, index=False)      # pip install OpenPyXL
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
[pixel_values1.xlsx][1]

[1]:{{ site.url }}/download/AI01/pixel_values1.xlsx
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **How to Convert Color Space**
### ***RGB to grayscale and Vice Versa***
```python
from PIL import Image

img = Image.open('input_image.jpg').convert('LA')
img.show()
```
<br><br><br>

---

### ***RGB to HSV and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io
from skimage import color
from skimage import data
from pylab import *

img = io.imread('puppy.jpg')         #Read image 
img_hsv = color.rgb2hsv(img)         #Convert to HSV
img_rgb = color.hsv2rgb(img_hsv)     #Convert back to RGB 

#Show both figures 
figure(0)                            
io.imshow(img_hsv)
figure(1)
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x1b99ef0da90>
![다운로드](https://user-images.githubusercontent.com/52376448/63702102-4108f600-c861-11e9-8bda-c3326a0bec65.png)

![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63702107-449c7d00-c861-11e9-98c1-f16f156b00df.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***RGB to XYZ and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io 
from skimage import color
from skimage import data 

#Read image 
img = io.imread('puppy.jpg')

#Convert to XYZ 
img_xyz = color.rgb2xyz(img)

#Convert back to RGB 
img_rgb = color.xyz2rgb(img_xyz)

#Show both figures 
figure(0) 
io.imshow(img_xyz) 
figure(1) 
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x194140a0cf8>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/63702164-639b0f00-c861-11e9-8737-08537e2bc073.png)

![다운로드 (3)](https://user-images.githubusercontent.com/52376448/63702173-6564d280-c861-11e9-857b-bbc28e15932a.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***RGB to LAB and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io
from skimage import color 

#Read image 
img = io.imread('puppy.jpg')

#Convert to LAB 
img_lab = color.rgb2lab(img)

#Convert back to RGB 
img_rgb = color.lab2rgb(img_lab)

#Show both figures 
figure(0) 
io.imshow(img_lab)
figure(1)
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x194141acf98>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/63702214-7d3c5680-c861-11e9-839c-80bc4f8bd825.png)

![다운로드 (5)](https://user-images.githubusercontent.com/52376448/63702221-7f061a00-c861-11e9-93e2-435a552b46a6.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***RGB to YUV and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io 
from skimage import color 

#Read image
img = io.imread('puppy.jpg')

#Convert to YUV 
img_yuv = color.rgb2yuv(img)

#Convert back to RGB
img_rgb = color.yuv2rgb(img_yuv)

#Show both figures 
figure(0) 
io.imshow(img_yuv) 
figure(1) 
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x194142c0940>
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/63702241-8af1dc00-c861-11e9-9fac-480a3545b3bb.png)

![다운로드 (7)](https://user-images.githubusercontent.com/52376448/63702246-8d543600-c861-11e9-8fdc-49164ccfef90.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***RGB to YIQ and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io 
from skimage import color 

#Read image 
img = io.imread('puppy.jpg')

#Convert to YIQ 
img_yiq = color.rgb2yiq(img)

#Convert back to RGB 
img_rgb = color.yiq2rgb(img_yiq)

#Show both figures 
figure(0)
io.imshow(img_yiq) 
figure(1)
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x19415d2a6d8>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/63702272-98a76180-c861-11e9-8d68-82ce82c99353.png)

![다운로드 (9)](https://user-images.githubusercontent.com/52376448/63702277-9b09bb80-c861-11e9-9d00-c27df3b67290.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***RGB to YPbPr and Vice Versa***

```python
%matplotlib inline

#Import libraries 
from skimage import io 
from skimage import color 

#Read image
img = io.imread('puppy.jpg')

#Convert to YPbPr 
img_ypbpr= color.rgb2ypbpr(img)

#Convert back to RGB 
img_rgb= color.ypbpr2rgb(img_ypbpr)

#Show both figures
figure(0)
io.imshow(img_ypbpr) 
figure(1) 
io.imshow(img_rgb)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x19415e8e470>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/63702289-a2c96000-c861-11e9-9ec1-ad7f7bcccd78.png)

![다운로드 (11)](https://user-images.githubusercontent.com/52376448/63702292-a4932380-c861-11e9-9806-88c18c2a9b0b.png)
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **How to Convert Images to NumPy Arrays and Back**

### ***Convert an loaded image with keras to type of ‘numpy.ndarray’***

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

# convert to numpy array(img_ndarray is equal way to img_array)
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
<br><br><br>

---

### ***Convert an loaded image with PIL to type of 'numpy.ndarray'***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
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
<hr class='division3'>
</details>

<br><br><br>

---

### ***Convert an loaded image with skimage to type of 'numpy.ndarray'***

```python
%matplotlib inline
from skimage import io                  # pip install scikit-image
import numpy as np

image = io.imread('puppy.jpg')
image = np.asarray(image)
type(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <class 'numpy.ndarray'>
</p>
<hr class='division3'>
</details>
<br><br><br>



<hr class="division2">


## **How to Resize Images**

### ***Resizing images, preserving the aspect ratio with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (640, 360)<br>
    (100, 56)
</p>
![resizing_opera_house](https://user-images.githubusercontent.com/52376448/63678027-43e7f480-c829-11e9-9ea4-0dffd961c6a1.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Resizing images, forcing the pixels into a new shape with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (640, 360)<br>
    (200, 200)
</p>
![resizing_opera_house2](https://user-images.githubusercontent.com/52376448/63678380-0fc10380-c82a-11e9-9f82-a3744ef5237f.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Resizing images, forcing the pixels into a new shape with skimage***

```python
from skimage import io
from skimage.transform import resize

img = io.imread('puppy.jpg')
img_res = resize(img, (100,100)) 
io.imshow(img_res)
io.imsave("ss.jpg", img_res)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63703411-e7ee9180-c863-11e9-998c-b774033d0b3c.png)
<hr class='division3'>
</details>

<br><br><br>


<hr class="division2">

## **How to Flip, Rotate, and Crop Images**

### ***Flipping images with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/63678432-32ebb300-c82a-11e9-8c39-2ce84487cac1.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Rotating images with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63678480-4860dd00-c82a-11e9-98c4-71261d682c57.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Rotating images with skimage***

```python
%matplotlib inline
from skimage import io 
from skimage.transform import rotate 

img = io.imread('puppy.jpg')
img_rot = rotate(img, 20) 
io.imshow(img_rot)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<matplotlib.image.AxesImage at 0x2708d61b390>
![다운로드](https://user-images.githubusercontent.com/52376448/63703165-6860c280-c863-11e9-9ec1-ee74cd52eee8.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Cropping images with PIL***

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![cropped](https://user-images.githubusercontent.com/52376448/63678860-0ab08400-c82b-11e9-9329-b77b8496da5c.png)
<hr class='division3'>
</details>

<br><br><br>

<hr class='division2'>

## **How to Create Basic Drawings**

### ***Lines***

```python
%matplotlib inline
from skimage import io 
from skimage import draw

img = io.imread('puppy.JPG')
print(img.shape)
x,y = draw.line(0,0,100,100)
img[x, y] = 0
io.imshow(img)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (366, 487, 3)<br>
    <matplotlib.image.AxesImage at 0x25100100908>
</p>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63703992-0acd7580-c865-11e9-8a99-a8fc26c087a3.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Rectangles***

```python
from skimage import io 
from skimage import draw
img = io.imread('puppy.jpg') 

def rectangle(x, y, w, h): 
    rr, cc = [x, x + w, x + w, x], [y, y, y + h, y + h]
    return (draw.polygon(rr, cc))

rr, cc = rectangle(10, 10, 355,355)
img[rr, cc] = 1 
io.imshow(img)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <matplotlib.image.AxesImage at 0x25100046320>
</p>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/63704000-0d2fcf80-c865-11e9-8ff2-5a51305b35ba.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Circles***

```python
#Import libraries 
from skimage import io 
from skimage import draw

#Load image
img = io.imread('puppy.jpg')

#Define circle coordinates and radius 
x, y = draw.circle(10,10, 100)

#Draw circle 
img[x, y] = 1

#Show image
io.imshow(img)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <matplotlib.image.AxesImage at 0x251001516d8>
</p>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/63704008-10c35680-c865-11e9-84d4-c173217b84bd.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Bezier Curve***

```python
#Import libraries
from skimage import io 
from skimage import draw
#Load image 
img = io.imread('puppy.jpg')

#Define Bezier curve coordinates 
x, y = draw.bezier_curve(0,0, 100, 100, 200,300,100)

#Draw Bezier curve 
img[x, y] = 1

#Show image 
io.imshow(img)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <matplotlib.image.AxesImage at 0x251011930b8>
</p>
![다운로드](https://user-images.githubusercontent.com/52376448/63704011-128d1a00-c865-11e9-9cf0-ac6aaa32cde7.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Text***

```python
%matplotlib inline
#import required packages 
import cv2
import numpy as np
from skimage import io

#Read image
image = cv2.imread("cat_1.jpg")

#Define font 
font  = cv2.FONT_HERSHEY_SIMPLEX

#Write on the image 
cv2.putText(image, "I am a Cat", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <matplotlib.image.AxesImage at 0x2386ff17ac8>
</p>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63707520-0d33cd80-c86d-11e9-8571-9c5b570469bb.png)
<hr class='division3'>
</details>
<br><br><br>



<hr class='division2'>

## **How to Gamma Correction(Gamma Encoding)**

```python
%matplotlib inline

from skimage import exposure 
from skimage import io 
from pylab import * 

img = io.imread('puppy.jpg') 
gamma_corrected1 = exposure.adjust_gamma(img, 0.5) 
gamma_corrected2 = exposure.adjust_gamma(img, 5) 

figure(0) 
io.imshow(gamma_corrected1) 
figure(1)
io.imshow(gamma_corrected2)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    <matplotlib.image.AxesImage at 0x1dd37ad2cc0>
</p>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/63704331-cf7f7680-c865-11e9-81f9-479c8b7a7804.png)

![다운로드 (5)](https://user-images.githubusercontent.com/52376448/63704334-d1e1d080-c865-11e9-8f8e-f6b4357cd3a4.png)

<hr class='division3'>
</details>
<br><br><br>

<hr class='division2'>

## **How to Determine Structural Similarity**

***Structural Similarity*** is used to find the index that indicate how much two images are similar.Here, SSIM takes three arguments. **The first** refers to the image; **the second** indicates the range of the pixels (the highest pixel color value less the lowest pixel color value). **The third** argument is multichannel.
```python
from skimage import io 
from skimage.measure import compare_ssim as ssim

img_original = io.imread('puppy.jpg')
img_modified = io.imread('puppy_ypbpr.jpg')

ssim_original = ssim(img_original, img_original, data_range=img_original.max() - img_original.min(), multichannel=True) 
ssim_different = ssim(img_original, img_modified, data_range=img_modified.max() - img_modified.min(), multichannel=True) 

print(ssim_original,ssim_different)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>1.0 0.41821662536853843</p>
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
- <a>Jason Brownlee, Deep Learning for Computer Vision</a>
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post3</a>

---
