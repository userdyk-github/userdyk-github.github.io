---
layout : post
title : PL03-Topic02, Matplotlib, Annotation
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-Matplotlib.html) <br>
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


## **Basic annotation**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,10, 1000)
y = np.sin(x-np.pi/2)

fig, ax = plt.subplots()
ax.plot(x,y)

ax.annotate('local max',
            xy=(3, 1), xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right',
            verticalalignment='top')
```
![download](https://user-images.githubusercontent.com/52376448/66709324-754a5e80-ed9c-11e9-8c1b-e271e0c5dfda.png)

<br><br><br>

<hr class="division2">

## **Advanced Annotation**

<br><br><br>

<hr class="division2">

### ***Annotating with Text with Box***

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,10, 1000)
y = np.sin(x-np.pi/2)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.text(0, 0, "Direction",
        ha="center",
        va="center",
        rotation=45,
        size=15,
        bbox={'boxstyle':"rarrow,pad=.1", 
              'fc':"cyan", 
              'ec':"b", 
              'lw':2})
```
![download](https://user-images.githubusercontent.com/52376448/66709336-a6c32a00-ed9c-11e9-8459-23c897402abc.png)
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT : boxstyle</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/66709352-09b4c100-ed9d-11e9-89b6-4f9137c10bf2.JPG)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Annotating with Arrow***
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.annotate("",
            xy=(0.2, 0.2), xycoords='data',
            xytext=(0.8, 0.8), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
```
![download (1)](https://user-images.githubusercontent.com/52376448/66709378-e2aabf00-ed9d-11e9-8eb1-5fca9341c4ad.png)
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT : connectionstyle</summary>
<hr class='division3'>
![s](https://user-images.githubusercontent.com/52376448/66709402-767c8b00-ed9e-11e9-9571-8941b00a2e1b.JPG)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT : arrowstyle</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/66709410-8d22e200-ed9e-11e9-9b9a-52039699c165.JPG)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Placing Artist at the anchored location of the Axes***

```python
import matplotlib.pyplot as plt
from matplotlib import offsetbox

fig, ax = plt.subplots(1,1)
textstr = "hello\nhello"
textbox = offsetbox.AnchoredText(textstr, loc=1)
ax.add_artist(textbox)
```
![download (8)](https://user-images.githubusercontent.com/52376448/66713790-23c8c080-edea-11e9-98db-d9e9390e2998.png)
<br><br><br>

---

### ***Using Complex Coordinates with Annotations***

<br><br><br>

---

### ***Using ConnectionPatch***

<br><br><br>


#### Advanced Topics

<br><br><br>


---

### ***Zoom effect between Axes***

<br><br><br>

---

### ***Define Custom BoxStyle***

<br><br><br>



<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://matplotlib.org/3.1.1/tutorials/text/annotations.html' target="_blank">annotations</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>



