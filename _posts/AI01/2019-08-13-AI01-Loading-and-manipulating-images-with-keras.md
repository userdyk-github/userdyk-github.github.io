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

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **How to Convert an Image With Keras to Gray Scale**

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
