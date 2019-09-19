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
<details open markdown="1">
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
<details open markdown="1">
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
<details open markdown="1">
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
<details open markdown="1">
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
<details open markdown="1">
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

### ***A variety of plot***

#### Line plot

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot : [0,1,2,3] -> [1,4,9,16]
plt.title('Plot')
plt.plot([1,4,9,16])

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (12)](https://user-images.githubusercontent.com/52376448/64471321-14bb7680-d18b-11e9-8b1d-22a71a80138e.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot : [10,20,30,40] -> [1,4,9,16]
plt.title('Plot')
plt.plot([10,20,30,40],[1,4,9,16])

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (13)](https://user-images.githubusercontent.com/52376448/64471326-374d8f80-d18b-11e9-8005-4e6867795c22.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot : [10,20,30,40] -> [1,4,9,16]
# style(simple decoration) : color/marker/line
plt.title("Plot")
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], 'rs--')

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (14)](https://user-images.githubusercontent.com/52376448/64471332-68c65b00-d18b-11e9-895f-67163165eef0.png)
<hr class='division3'>
</details>
<br><br><br>


Style strings are specified in the order of color, marker, and line style. If some of these are omitted, the default value is applied.
- [color ref](https://matplotlib.org/examples/color/named_colors.html)
- [marker ref](https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html)
- [line style ref](https://matplotlib.org/examples/lines_bars_and_markers/line_styles_reference.html)

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot : [10,20,30,40] -> [1,4,9,16]
# details decoration
plt.title("Plot")
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], c="b",
         lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")

plt.show()

# color           : c        : 선 색깔
# linesidth       : lw       : 선 굵기
# linestyle       : ls       : 선 스타일
# marker          : marker   : 마커 종류
# markersize      : ms       : 마커 크기
# markeredgecolor : mec      : 마커 선 색깔
# markeredgewidth : mew      : 마커 선 굵기
# markerfacecolor : mfc      : 마커 내부 색깔
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (15)](https://user-images.githubusercontent.com/52376448/64471377-0752bc00-d18c-11e9-939e-a18d66a0254f.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot : [10,20,30,40] -> [1,4,9,16]
# 그래프 척도(scale, size) 설정 : lim
plt.title("Plot")
plt.plot([10, 20, 30, 40], [1, 4, 9, 16],
         c="b", lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
plt.xlim(0, 50)
plt.ylim(-10, 30)

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (16)](https://user-images.githubusercontent.com/52376448/64471381-218c9a00-d18c-11e9-9feb-b0fc459a908e.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.title("Plot")
plt.plot(x, y)

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (17)](https://user-images.githubusercontent.com/52376448/64471392-3e28d200-d18c-11e9-8e12-274db830c086.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# simple decoration
# style : color/marker/line
x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.title("Plot")
plt.plot(x, y, 'rs--')

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (18)](https://user-images.githubusercontent.com/52376448/64471398-5698ec80-d18c-11e9-813c-bf64699f5d93.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# axes tick set(1)
x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.title("Plot")
plt.plot(x, y, 'rs--')
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
plt.yticks([-1, 0, +1])

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (19)](https://user-images.githubusercontent.com/52376448/64471421-8647f480-d18c-11e9-980b-d240678848a2.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# axes tick set(2) : Latex
x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.title("Plot")
plt.plot(x, y, 'rs--')
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (20)](https://user-images.githubusercontent.com/52376448/64471430-a1b2ff80-d18c-11e9-82bd-a20fae4a9dd0.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# grid 설정 : grid
x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.title("Plot")
plt.plot(x, y, 'rs--')
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pni$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.grid(True)

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (21)](https://user-images.githubusercontent.com/52376448/64471442-b68f9300-d18c-11e9-918c-941319187623.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# multi-plot(1) : can be expressed with 1 plot
t = np.arange(0., 5., 0.2)

plt.title("Plot")
plt.plot(t, t, 'r--', t, 0.5 * t**2, 'bs:', t, 0.2 * t**3, 'g^-')

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (22)](https://user-images.githubusercontent.com/52376448/64471458-e3dc4100-d18c-11e9-8ee7-ae1969b8eaf7.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# list plot
# multi-plot(2) : using several plots
plt.title("Plot")
plt.plot([1, 4, 9, 16],
         c="b", lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
plt.plot([9, 16, 4, 1],
         c="k", lw=3, ls=":", marker="s", ms=10, mec="m", mew=5, mfc="c")

plt.show()

# plt.hold(True)   # <- This code is required in version 1, 5
# plt.hold(False)  # <- This code is required in version 1, 5

# color           : c        : 선 색깔
# linesidth       : lw       : 선 굵기
# linestyle       : ls       : 선 스타일
# marker          : marker   : 마커 종류
# markersize      : ms       : 마커 크기
# markeredgecolor : mec      : 마커 선 색깔
# markeredgewidth : mew      : 마커 선 굵기
# markerfacecolor : mfc      : 마커 내부 색깔
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (23)](https://user-images.githubusercontent.com/52376448/64471485-30278100-d18d-11e9-89fe-3854563aeab7.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# setting legend 
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)

plt.title("Plot")
plt.plot(X, C, ls="--", label="cosine")    # setting legend using label
plt.plot(X, S, ls=":", label="sine")       # setting legend using label
plt.legend(loc=2)                          # lov value means a position of legend in figure

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (24)](https://user-images.githubusercontent.com/52376448/64471500-7381ef80-d18d-11e9-82ef-cb3c32753779.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# numpy plot
# naming axis
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)

plt.title("Cosine Plot")
plt.plot(X, C, label="cosine")
plt.xlabel("time")                # naming x-axis
plt.ylabel("amplitude")           # naming x-axis

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (25)](https://user-images.githubusercontent.com/52376448/64471505-8268a200-d18d-11e9-841f-c5317ecd1864.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Scatter plot

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.title("Scatter Plot")
plt.scatter(X, Y)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/64471579-78936e80-d18e-11e9-9887-08883fe4f740.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

N = 30
np.random.seed(0)
x = np.random.rand(N)
y1 = np.random.rand(N)
y2 = np.random.rand(N)
y3 = np.pi * (15 * np.random.rand(N))**2
plt.title("Bubble Chart")
plt.scatter(x, y1, c=y2, s=y3)   # s : size, c : color
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/64471578-77fad800-d18e-11e9-8dc2-aa0658dd64b8.png)
<hr class='division3'>
</details>
<br><br><br>


---

#### Stem plot

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1, 2 * np.pi, 10)
plt.title("Stem Plot")
plt.stem(x, np.cos(x), '-.')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/64471596-9234b600-d18e-11e9-9b28-ffa9a689f9a3.png)
<hr class='division3'>
</details>
<br><br><br>


---

#### Contour plot

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
XX, YY = np.meshgrid(x, y)
ZZ = f(XX, YY)

plt.title("Contour plots")
plt.contourf(XX, YY, ZZ, alpha=.75, cmap='jet')
plt.contour(XX, YY, ZZ, colors='black')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/64471595-9234b600-d18e-11e9-87c8-5343d794e103.png)
<hr class='division3'>
</details>
<br><br><br>


---

#### Surface plot

```python
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
XX, YY = np.meshgrid(X, Y)
RR = np.sqrt(XX**2 + YY**2)
ZZ = np.sin(RR)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("3D Surface Plot")
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/64471613-b0021b00-d18e-11e9-99b3-e97fbb2dccf3.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Histogram

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = np.random.randn(1000)
plt.title("Histogram")
arrays, bins, patches = plt.hist(x, bins=10)   # bins : Interval to aggregate data
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/64471612-b0021b00-d18e-11e9-918e-97aa64133e62.png)
<hr class='division3'>
</details>

<br>

```python
arrays
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([  9.,  20.,  70., 146., 217., 239., 160.,  86.,  38.,  15.])
```
<hr class='division3'>
</details>

<br>

```python
bins
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([-3.04614305, -2.46559324, -1.88504342, -1.3044936 , -0.72394379,
       -0.14339397,  0.43715585,  1.01770566,  1.59825548,  2.1788053 ,
        2.75935511])
```
<hr class='division3'>
</details>
<br><br><br>

---

#### Bar chart

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

y = [2, 3, 1]
x = np.arange(len(y))
xlabel = ['a', 'b', 'c']
plt.title("Bar Chart")
plt.bar(x, y)
plt.xticks(x, xlabel)
plt.yticks(sorted(y))
plt.xlabel("abc")
plt.ylabel("frequency")
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/64471623-c6a87200-d18e-11e9-8932-e4a3ee3dd8cc.png)
<hr class='division3'>
</details>
<br><br><br>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

people = ['a', 'b', 'c', 'd']
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.title("Barh Chart")
plt.barh(y_pos, performance, xerr=error, alpha=0.4)   # alpha : transparency [0,1]
plt.yticks(y_pos, people)
plt.xlabel('x label')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/64471622-c6a87200-d18e-11e9-90b0-bafe3d85d707.png)
<hr class='division3'>
</details>
<br><br><br>


---

#### Pie chart

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.axis('equal')  # retaining the shape of a circle

labels = ['frog', 'pig', 'dog', 'log']
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.title("Pie Chart")
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/64471621-c6a87200-d18e-11e9-93a0-a1a16c66c922.png)
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

x = np.linspace(-5, 2, 100)                # 3th factor mean smoothness 
y =  x**3 + 5*x**2 + 10

fig, ax = plt.subplots()                   # show a picture on screen
ax.plot(x, y, color="blue", label="y(x)")  # here, you can change type of plot,
                                           # if you want, use ax.step, ax.bar, ax.hist, ax.errorbar, ax.scatter, ax.fill_between, ax.quiver instead of ax.plot
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/64471172-c9a06400-d188-11e9-8bcf-e45452da1841.png)
<hr class='division3'>
</details>


<br><br><br>


<details open markdown="1">
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

<details open markdown="1">
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

<details open markdown="1">
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

<details open markdown="1">
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
<details open markdown="1">
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

```python
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import sympy

fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': '3d'})

def title_and_labels(ax, title):
    ax.set_title(title)  
    ax.set_xlabel("$x$", fontsize=16)   
    ax.set_ylabel("$y$", fontsize=16) 
    ax.set_zlabel("$z$", fontsize=16)  
    
x = y = np.linspace(-3, 3, 74)   
X, Y = np.meshgrid(x, y)   
    
R = np.sqrt(X**2 + Y**2)  
Z = np.sin(4 * R) / R   

norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())  

p =  axes[0].plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, norm=norm, cmap=mpl.cm.Blues)   

cb = fig.colorbar(p, ax=axes[0], shrink=0.6)  
title_and_labels(axes[0], "plot_surface")  

p = axes[1].plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="darkgrey")   
title_and_labels(axes[1], "plot_wireframe")   

cset = axes[2].contour(X, Y, Z, zdir='z', offset=0, norm=norm, cmap=mpl.cm.Blues)   
cset = axes[2].contour(X, Y, Z, zdir='y', offset=3, norm=norm, cmap=mpl.cm.Blues) 
title_and_labels(axes[2], "contour")
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/65240837-24d53d80-db1d-11e9-8841-c5677044b26a.png)
<hr class='division3'>
</details>
<br><br><br>


---

### ***Working with 3D Figures***

<hr class="division2">

## **User Interface**

<hr class="division2">

## **Korean font**


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



