---
layout : post
title : AI01, Practical data handling
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Practical-data-handling.md" target="_blank">page management</a><br>
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
## **File I/O**
### ***Image***
```python
import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('input_image.jpg')   # load image
plt.imshow(img)
plt.figsave('output_image.jpg')         # save image
```
<br><br><br>

---

### ***Table***
```python
import pandas as pd

df = pd.read_csv('input_table.csv')    # load table
df.to_excel('output_table.xlsx')       # save table
```
<br><br><br>

---

### ***Text***
```python
with open('input_text.txt','r') as f:  # load text
    text = f.read()
with open('output.txt','w') as f:      # save text
    f.write(text)
```
<br><br><br>
<hr class="division2">

## **From WEB**
<ins>Developer tools</ins><br>
<p>F12 : Elements(Inspector, Ctrl + Shift + c), Networks</p>

<br><br><br>


<hr class="division2">

## **From DB**
<br><br><br>
<hr class="division2">

## **h5**
```python
import h5py
import numpy as np

f = h5py.File('input_big_data.h5','r')    # load big_data
for i in f.keys():                        
    info = f.get(i)                       # show information about big_data
    print(info)                           
    
    data = np.array(info)                 # show big_data
    print(data)
```
<br><br><br>

<hr class="division2">


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
