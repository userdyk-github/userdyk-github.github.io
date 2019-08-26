---
layout : post
title : AI01, Loading and manipulating images with keras
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

## **How to Load an Image with Keras**

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

### ***Argument1 : grayscale***
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

### ***Argument2 : color_mode***

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

### ***Argument3 : target_size***

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


## **How to Convert an Image With Keras**

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


## **How to Save an Image With Keras**

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
