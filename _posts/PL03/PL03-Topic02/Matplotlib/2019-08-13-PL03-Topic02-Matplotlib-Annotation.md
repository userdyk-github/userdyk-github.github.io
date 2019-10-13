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
            xy=(3, 1),
            xycoords='data',
            xytext=(0.8, 0.95),
            textcoords='axes fraction',
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


<br><br><br>

---

### ***Annotating with Arrow***

<br><br><br>

---

### ***Placing Artist at the anchored location of the Axes***

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
- <a href='https://www.tutorialspoint.com/pyqt/index.htm' target="_blank">PyQt Tutorial(official)</a>
- <a href="http://codetorial.net/" target="_blank">PyQt5 Tutorial</a>
- <a href='https://www.youtube.com/playlist?list=PLuTktZ8WcEGTdId-Kjbj6gsZTk65yudJh' target="_blank">Youtube Lecture about Qt designer</a>
- <a href='https://www.youtube.com/watch?v=qiPS70TSvBk' target="_blank">Building .exe file</a>

---




