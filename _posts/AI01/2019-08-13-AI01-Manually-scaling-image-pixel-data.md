---
layout : post
title : AI01, Manually scaling image pixel data
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


## **Normalize Pixel Values**

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT1</summary>
<hr class='division3'>
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
<hr class='division3'>
</details>

<br>

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT2</summary>
<hr class='division3'>
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
<hr class='division3'>
</details>


<br><br><br>

<hr class="division2">


## **Center Pixel Values**

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>


<hr class="division2">

## **Standardize Pixel Values**

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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
