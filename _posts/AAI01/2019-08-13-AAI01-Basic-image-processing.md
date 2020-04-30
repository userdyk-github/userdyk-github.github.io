---
layout : post
title : AAI01, Basic image processing
categories: [AAI01]
comments : true
tags : [AAI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AAI01/2019-08-13-AAI01-Basic-image-processing.md" target="_blank">page management</a><br>
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

## **Building 2d-image with numpy**

### ***dtype = np.uint8***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.uint8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint8<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1d7d630>
</p>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/63788477-86e0bf80-c930-11e9-80ff-3959fe9caf3d.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***dtype = np.int8***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.int8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  int8<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1ddf208>
</p>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/63788479-86e0bf80-c930-11e9-905e-423fad817438.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***dtype = np.uint16***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.uint16)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint16<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1e96208>
</p>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/63788476-86482900-c930-11e9-8fe8-5c9c48218b9d.png)
<hr class='division3'>
</details>
<br><br><br>



<hr class="division2">

## **Building 3d-image with numpy**

### ***dtype = np.uint8***

```python
import numpy as np
from skimage import io

image = np.array([[[255,0,0],[0,255,0],[0,0,255]],
                  [[0,0,0],[128,128,128],[255,255,255]]], dtype=np.uint8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint8<br>
  shape :  (2, 3, 3)<br>
  <matplotlib.image.AxesImage at 0x260c4e91208>
</p>
![다운로드 (12)](https://user-images.githubusercontent.com/52376448/63789907-40409480-c933-11e9-918c-2d215de9965f.png)
<hr class='division3'>
</details>
<br>

```python
import numpy as np
from skimage import io

image = np.array([[[-1,256,256],[256,-1,256],[256,256,-1]],
                  [[0,0,0],[-128,-128,-128],[-1,-1,-1]]], dtype=np.uint8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint8<br>
  shape :  (2, 3, 3)<br>
  <matplotlib.image.AxesImage at 0x260c516d588>
</p>
![다운로드](https://user-images.githubusercontent.com/52376448/63790211-cd83e900-c933-11e9-9451-cdcffef179e7.png)
<hr class='division3'>
</details>


<br><br><br>

---

### ***dtype = np.int8***

```python
import numpy as np
from skimage import io

image = np.array([[[255,0,0],[0,255,0],[0,0,255]],
                  [[0,0,0],[128,128,128],[255,255,255]]], dtype=np.int8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  int8<br>
  shape :  (2, 3, 3)<br>
  <matplotlib.image.AxesImage at 0x260c51ca898>
</p>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/63790589-89ddaf00-c934-11e9-8570-b7487526b92f.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***dtype = np.uint16***

```python
import numpy as np
from skimage import io

image = np.array([[[255,0,0],[0,255,0],[0,0,255]],
                  [[0,0,0],[128,128,128],[255,255,255]]], dtype=np.uint16)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint16<br>
  shape :  (2, 3, 3)<br>
  <matplotlib.image.AxesImage at 0x260c5282358>
</p>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/63790648-aed22200-c934-11e9-84f1-bc4bfdd7c418.png)
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
