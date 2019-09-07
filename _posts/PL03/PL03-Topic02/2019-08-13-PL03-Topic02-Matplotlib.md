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

### ***Figure object and plot commands***

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
f1 = plt.figure(figsize=(10, 2))    # Simultaneously resize graph while defining objects

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
<br><br><br>



---

### ***Working with Figures***

---

### ***Working with a File Output***


<hr class="division2">

## **Axes**

---

### ***Axes object and subplot commands***

**Axes object**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 2, 100)                     # 3th factor mean smoothness 
y =  x**3 + 5*x**2 + 10

fig, ax = plt.subplots()                        # show a picture on screen
ax.plot(x, y, color="blue", label="y(x)")       # here, you can change type of plot,
                                                # if you want, use ax.step, ax.bar, ax.hist, ax.errorbar, ax.scatter, ax.fill_between, ax.quiver instead of ax.plot
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/64471172-c9a06400-d188-11e9-8bcf-e45452da1841.png)
<hr class='division3'>
</details>


<br><br><br>


<details markdown="1">
<summary class='jb-small' style="color:blue">EXPLAINATION</summary>
<hr class='division3'>
<p>
  In some cases, it may be necessary to display multiple plots in an array within a single window, as follows. And each plot in Figure belongs to an object called Axes.
</p>

<p>
  To create Axes within the Figure, you must explicitly acquire Axes objects using the original subplot command(~like plt.subplot). However, using the plot command(~like plt.plot) automatically generates Axes.
</p>

<p>
  The subplot command creates grid-type Axes objects, and you can think of Figure as a matrix and Axes as an element of the matrix. For example, if you have two plots up and down, the row is 2 and the column is 1 is 2x1. The subplot command has three arguments, two numbers for the first two elements to indicate the shape of the entire grid matrix and the third argument to indicate where the graph is. Therefore, to draw two plots up and down in one Figure, you must execute the command as follows. Note that the number pointing to the first plot is not zero but one, since numeric indexing follows Matlab practices rather than Python.
</p>
<hr class='division3'>
</details>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

ax1 = plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
print(ax1)          # Identification for the allocated sub-object

ax2 = plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')
print(ax2)          # Identification for the allocated sub-object

plt.tight_layout()  # The command automatically adjusts the spacing between plots
plt.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
AxesSubplot(0.125,0.536818;0.775x0.343182)
AxesSubplot(0.125,0.125;0.775x0.343182)
```
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/64470780-e4bca500-d183-11e9-8587-666a282f6e87.png)
<hr class='division3'>
</details>
<br><br><br>


If there are four plots in 2x2, draw as follows. The argument (2,2,1) for subplot can be abbreviated a single number of 221. Axes' position counts from top to bottom, from left to right.
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

plt.subplot(221)
plt.plot(np.random.rand(5))
plt.title("axes 1")

plt.subplot(222)
plt.plot(np.random.rand(5))
plt.title("axes 2")

plt.subplot(223)
plt.plot(np.random.rand(5))
plt.title("axes 3")

plt.subplot(224)
plt.plot(np.random.rand(5))
plt.title("axes 4")

plt.tight_layout()
plt.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/64471075-7843a500-d187-11e9-8d7c-79e32119f33c.png)
<hr class='division3'>
</details>
<br><br><br>

You can also create multiple Axes objects simultaneously with the subplots command. Axes objects are returned in two-dimensional ndarray form.
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2)

np.random.seed(0)
axes[0, 0].plot(np.random.rand(5))
axes[0, 0].set_title("axes 1")
axes[0, 1].plot(np.random.rand(5))
axes[0, 1].set_title("axes 2")
axes[1, 0].plot(np.random.rand(5))
axes[1, 0].set_title("axes 3")
axes[1, 1].plot(np.random.rand(5))
axes[1, 1].set_title("axes 4")

plt.tight_layout()
plt.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/64471127-fe5feb80-d187-11e9-93dc-3186c17ecb2f.png)
<hr class='division3'>
</details>
<br><br><br>

The twinx command creates a new Axes object that shares the x-axis.
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

fig, ax0 = plt.subplots()
ax0.set_title("Plot")
ax0.plot([10, 5, 2, 9, 7], 'r-', label="y0")
ax0.set_xlabel("sharing x-axis")
ax0.set_ylabel("y0")
ax0.grid(False)

ax1 = ax0.twinx()
ax1.plot([100, 200, 220, 180, 120], 'g:', label="y1")
ax1.set_ylabel("y1")
ax1.grid(False)

plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (11)](https://user-images.githubusercontent.com/52376448/64471189-374c9000-d189-11e9-97d9-d7c37cfa8a26.png)
<hr class='division3'>
</details>
<br><br><br>

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



