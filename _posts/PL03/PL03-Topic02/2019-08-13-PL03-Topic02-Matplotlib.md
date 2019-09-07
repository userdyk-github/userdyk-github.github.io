---
layout : post
title : PL03-Topic02, Matplotlib
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) <br>
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

## **Getting Started**

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1,10,10)
y = np.linspace(2,20,10)

plt.plot(x,y, 'rs--')
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/64470527-68c05e00-d17f-11e9-85ee-bb3f4d89bea9.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1,10,100)
y = x**4 + x

plt.plot(x,y)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/64470533-92798500-d17f-11e9-9f63-dd9e76e991e0.png)
<hr class='division3'>
</details>




---

### ***First Steps***

---

### ***Customizing the Color and Styles***

---

### ***Working with Annotations***



<hr class="division2">

## **Figure**

---

### ***Figure object***

**Graphs plot in general**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

plt.title("Plot")
plt.plot(np.random.randn(100))
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/64470563-264b5100-d180-11e9-9360-63438cc1bdb1.png)
<hr class='division3'>
</details>
<br><br><br>

**Graphs plot in principle**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
f1 = plt.figure(figsize=(10, 2))    # At the same time, Resize graph while defining objects

plt.title("Plot")
plt.plot(np.random.randn(100))
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/64470630-88588600-d181-11e9-8fb1-a244466c4da3.png)
<hr class='division3'>
</details>
<br><br><br>

**Identification for the currently allocated figure object**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

f1 = plt.figure(1)

plt.title("Plot")
plt.plot([1, 2, 3, 4], 'ro:')

f2 = plt.gcf()
print(f1, id(f1))             # identification1 for object directly using id
print(f2, id(f2))             # identification2 for object using gcf and id(in principle)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Figure(432x288) 2045494563280
Figure(432x288) 2045494563280
```
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/64470689-89d67e00-d182-11e9-9a82-6c0a0aa44163.png)
<hr class='division3'>
</details>




---

### ***Working with Figures***

---

### ***Working with a File Output***


<hr class="division2">

## **Axes**

---

### ***Subplot commands***

---

### ***Line properties***

---

### ***Text formatting and annotations***

---

### ***Axis propertires***

#### Axis labels and titles

---

#### Axis range

---

#### Axis ticks, tick labels, and grids

---

#### Log plots

---

#### Twin axes

---

#### Spines


<hr class="division2">

## **Advanced Axes Layouts**

### ***Insets***

---

### ***Subplots***

---

### ***Subplot2grid***

---

### ***GridSpec***

<hr class="division2">

## **Colormap Plots**

---

### ***Working with Maps***

<hr class="division2">

## **3D Plots**

---

### ***Working with 3D Figures***

<hr class="division2">

## **User Interface**

<hr class="division2">

## **Korean font**

<hr class="division2">

## **A variety of plot**

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



