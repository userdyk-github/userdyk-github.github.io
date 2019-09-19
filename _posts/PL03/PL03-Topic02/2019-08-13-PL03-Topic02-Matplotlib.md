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
<br><br><br>



---

### ***First Steps***

---

### ***Working with Annotations***

#### Adding a title

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.title('A polynomial')
plt.plot(X, Y, c = 'k')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/65243246-f6f2f780-db22-11e9-913d-a3b06d49784c.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Using LaTeX-style notations

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.title('$f(x)=\\frac{1}{4}(x+4)(x+1)(x-2)$')
plt.plot(X, Y, c = 'k')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65243291-0a05c780-db23-11e9-87b0-93686192b74f.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding a label to each axis

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.title('Power curve for airfoil KV873')
plt.xlabel('Air speed')
plt.ylabel('Total drag')
plt.plot(X, Y, c = 'k')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65243394-3b7e9300-db23-11e9-8824-809e1553cb84.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding text

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.plot(X, Y, c = 'k')
plt.text(-0.5, -0.25, 'Brackmard minimum')
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Annotate</summary>
<hr class='division3'>
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.plot(X, Y, c = 'k')
plt.text(-0.5, -0.25, 'Brackmard minimum')
plt.annotate("Annotation", fontsize=14, family="serif", xy=(1, 1), xycoords="data", xytext=(+20, +50), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))

plt.show()
```


<hr class='division3'>
</details>
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65243396-3b7e9300-db23-11e9-9409-4f3eec095428.png)
<hr class='division3'>
</details>
<br><br><br>

```python
# Bounding box control
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

box = {
 'facecolor' : '.75',
 'edgecolor' : 'k',
 'boxstyle' : 'round'
}
plt.text(-0.5, -0.20, 'Brackmard minimum', bbox = box)
plt.plot(X, Y, c='k')
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Box options</summary>
<hr class='division3'>
<div class="jb-medium">
- <strong>'facecolor'</strong>: This is the color used for the box. It will be used to set the background and the edge color <br>
- <strong>'edgecolor'</strong>: This is the color used for the edges of the box's shape <br>
- <strong>'alpha'</strong>: This is used to set the transparency level so that the box blends with the background <br>
- <strong>'boxstyle'</strong>: This sets the style of the box, which can either be 'round' or 'square' <br>
- <strong>'pad'</strong>: If 'boxstyle' is set to 'square', it defines the amount of padding between the text and the box's sides <br>
</div>
<hr class='division3'>
</details>
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65243397-3b7e9300-db23-11e9-8e20-ca041b05ebf6.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding arrows

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.annotate('Brackmard minimum',
             ha = 'center', va = 'bottom',
             xytext = (-1.5, 3.),
             xy = (0.75, -2.7),
             arrowprops = { 'facecolor' : 'black', 'shrink' : 0.05 })
plt.plot(X, Y)
plt.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">Arrow options</summary>
<hr class='division3'>
<div class="jb-medium">
- <strong>'arrowstyle'</strong>: The parameters ''<-'', ''<'', ''-'', ''wedge'', ''simple'', and "fancy" control the style of the arrow <br>
- <strong>'facecolor'</strong>: This is the color used for the arrow. It will be used to set the background and the edge color <br>
- <strong>'edgecolor'</strong>: This is the color used for the edges of the arrow's shape <br>
- <strong>'alpha'</strong>: This is used to set the transparency level so that the arrow blends with the background <br>
</div>
<hr class='division3'>
</details>

<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/65243398-3c172980-db23-11e9-8a1a-bef087ffee25.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding a legend

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 6, 1024)
Y1 = np.sin(X)
Y2 = np.cos(X)

plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, Y1, c = 'k', lw = 3., label = 'sin(X)')
plt.plot(X, Y2, c = '.5', lw = 3., ls = '--', label = 'cos(X)')
plt.legend()
plt.show()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">Legend options</summary>
<hr class='division3'>
<div class="jb-medium">
- 'loc': This is the location of the legend. The default value is 'best', which will place it automatically. Other valid values are 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', and 'center'.<br>
- 'shadow': This can be either True or False, and it renders the legend with a shadow effect.<br>
- 'fancybox': This can be either True or False and renders the legend with a rounded box.<br>
- 'title': This renders the legend with the title passed as a parameter.<br>
- 'ncol': This forces the passed value to be the number of columns for the legend<br>
</div>
<hr class='division3'>
</details>

<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/65243393-3ae5fc80-db23-11e9-8bdf-4c6cdae87e0c.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding a grid

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.plot(X, Y, c = 'k')
plt.grid(True, lw = 2, ls = '--', c = '.75')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/65243487-65d05080-db23-11e9-84c8-ac2c4d9f2844.png)
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

---


#### Adding lines

```python
import matplotlib.pyplot as plt

N = 16
for i in range(N):
     plt.gca().add_line(plt.Line2D((0, i), (N - i, 0), color = '.75'))
     
plt.grid(True)
plt.axis('scaled')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/65243488-65d05080-db23-11e9-9435-91613caecef4.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Adding shapes

```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Circle
shape = patches.Circle((0, 0), radius = 1., color = '.75')
plt.gca().add_patch(shape)

# Rectangle
shape = patches.Rectangle((2.5, -.5), 2., 1., color = '.75')
plt.gca().add_patch(shape)

# Ellipse
shape = patches.Ellipse((0, -2.), 2., 1., angle = 45., color =
 '.75')
plt.gca().add_patch(shape)

# Fancy box
shape = patches.FancyBboxPatch((2.5, -2.5), 2., 1., boxstyle =
 'sawtooth', color = '.75')
plt.gca().add_patch(shape)

# Display all
plt.grid(True)
plt.axis('scaled')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">Shape options</summary>
<hr class='division3'>
<div class='jb-small'>
- Circle: This takes the coordinates of its center and the radius as the parameters<br>
- Rectangle: This takes the coordinates of its lower-left corner and its size as the parameters<br>
- Ellipse: This takes the coordinates of its center and the half-length of its two axes as the parameters<br>
- FancyBox: This is like a rectangle but takes an additional boxstyle parameter (either 'larrow', 'rarrow', 'round', 'round4', 'roundtooth', 'sawtooth', or 'square')<br>
</div>
<hr class='division3'>
</details>

<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/65243489-6668e700-db23-11e9-8927-e11104bd8333.png)
<hr class='division3'>
</details>
<br><br><br>

```python
# Working with polygons
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 8)
points = np.vstack((np.cos(theta), np.sin(theta))).transpose()

plt.gca().add_patch(patches.Polygon(points, color = '.75'))
plt.grid(True)
plt.axis('scaled')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/65243490-6668e700-db23-11e9-882c-34f229ba0e51.png)
<hr class='division3'>
</details>
<br><br><br>

```python
# Working with path attributes
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 6)
points = np.vstack((np.cos(theta), np.sin(theta))).transpose()

plt.gca().add_patch(plt.Circle((0, 0), radius = 1., color =
 '.75'))
plt.gca().add_patch(plt.Polygon(points, closed=None, fill=None,
 lw = 3., ls = 'dashed', edgecolor = 'k'))
plt.grid(True)
plt.axis('scaled')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (11)](https://user-images.githubusercontent.com/52376448/65243492-67017d80-db23-11e9-974c-43340698804e.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Controlling tick spacing

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

X = np.linspace(-15, 15, 1024)
Y = np.sinc(X)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.plot(X, Y, c = 'k')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (12)](https://user-images.githubusercontent.com/52376448/65243494-67017d80-db23-11e9-840b-1012419b80f6.png)
<hr class='division3'>
</details>
<br><br><br>

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

X = np.linspace(-15, 15, 1024)
Y = np.sinc(X)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which='both')
plt.plot(X, Y)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (13)](https://user-images.githubusercontent.com/52376448/65243495-67017d80-db23-11e9-85f5-c465da6da3d2.png)
<hr class='division3'>
</details>
<br><br><br>

---


#### Controlling tick labeling

```python
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

name_list = ('Omar', 'Serguey', 'Max', 'Zhou', 'Abidin')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))

plt.bar(pos_list, value_list, color = '.75', align = 'center')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (14)](https://user-images.githubusercontent.com/52376448/65243496-67017d80-db23-11e9-8b44-ad5dff89dc19.png)
<hr class='division3'>
</details>
<br><br><br>

```python
# A simpler way to create bar charts with fixed labels
import numpy as np
import matplotlib.pyplot as plt

name_list = ('Omar', 'Serguey', 'Max', 'Zhou', 'Abidin')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))

plt.bar(pos_list, value_list, color = '.75', align = 'center')
plt.xticks(pos_list, name_list, rotation=30)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (15)](https://user-images.githubusercontent.com/52376448/65243498-679a1400-db23-11e9-8ddb-b26927d67c6a.png)
<hr class='division3'>
</details>
<br><br><br>

```python
# Advanced label generation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

X = np.linspace(0, 1, 256)

def make_label(value, pos):
     return '%0.1f%%' % (100. * value)
     
ax = plt.axes()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_label))
plt.plot(X, np.exp(-10 * X), c ='k')
plt.plot(X, np.exp(-5 * X), c= 'k', ls = '--')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (16)](https://user-images.githubusercontent.com/52376448/65243499-679a1400-db23-11e9-940f-78c299e4302f.png)
<hr class='division3'>
</details>
<br><br><br>

```python
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

X = np.linspace(0, 1, 256)
start_date = datetime.datetime(1998, 1, 1)

def make_label(value, pos):
     time = start_date + datetime.timedelta(days = 365 * value)
     return time.strftime('%b %y')
     
ax = plt.axes()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_label))
plt.plot(X, np.exp(-10 * X), c = 'k')
plt.plot(X, np.exp(-5 * X), c = 'k', ls = '--')
labels = ax.get_xticklabels()
plt.setp(labels, rotation = 30.)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (17)](https://user-images.githubusercontent.com/52376448/65243501-679a1400-db23-11e9-94f5-c1168b57281c.png)
<hr class='division3'>
</details>
<br><br><br>

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
plt.figure(figsize=(10, 2))    # Simultaneously resize graph while defining objects

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

#### Point plot

**Single point plot**
```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(1,1,marker="o")
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/65250077-edbc5780-db2f-11e9-81d1-38e55e3579dd.png)
<hr class='division3'>
</details>
<br><br><br>

**Multiple point plot**

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([1,2,3],[10,5,8],marker="o", lw=0)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/65250284-4c81d100-db30-11e9-993f-b55485e6dabf.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Line plot

**list plot : [0,1,2,3] -> [1,4,9,16]**
```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([1,4,9,16])
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (12)](https://user-images.githubusercontent.com/52376448/65261714-cec7c080-db43-11e9-86a3-4d0c68207254.png)
<hr class='division3'>
</details>
<br><br><br>

**list plot : [10,20,30,40] -> [1,4,9,16]**
```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([10,20,30,40],[1,4,9,16])
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (13)](https://user-images.githubusercontent.com/52376448/65261757-e8690800-db43-11e9-93a6-e811b0ed3580.png)
<hr class='division3'>
</details>
<br><br><br>

**numpy array plot : <span style="font-size: 70%;">$$[-\pi + \pi] \to cos(x)$$ </span>**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.plot(x, y)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (16)](https://user-images.githubusercontent.com/52376448/65262213-cae86e00-db44-11e9-99a8-d772d598bd2b.png)
<hr class='division3'>
</details>
<br><br><br>


**style(simple decoration) : color/marker/line**
```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([10, 20, 30, 40], [1, 4, 9, 16], 'rs--')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (14)](https://user-images.githubusercontent.com/52376448/65261796-fcad0500-db43-11e9-9c55-d146b22da65a.png)
<hr class='division3'>
</details>
<br><br><br>

**details decoration**<br>
<span class="medium">Style strings are specified in the order of color, marker, and line style. If some of these are omitted, the default value is applied.</span> <br>

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([10, 20, 30, 40], [1, 4, 9, 16],
         c="b",
         lw=5, 
         ls="--",
         marker="o", 
         ms=15, 
         mec="g",
         mew=5,
         mfc="r")
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
- <a href="https://matplotlib.org/examples/color/named_colors.html" target="blank">color ref</a><br>
- <a href="https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html" target="blank">marker ref</a><br>
- <a href="https://matplotlib.org/examples/lines_bars_and_markers/line_styles_reference.html" target="blank">line style ref</a><br>

|color           | c|
|linesidth       | lw |
|linestyle       | ls|
|marker          | marker|
|markersize      | ms|
|markeredgecolor | mec|
|markeredgewidth | mew|
|markerfacecolor | mfc|

<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">EXAMPLES</summary>
<hr class='division3'>
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 5)
y = np.ones_like(x)

def axes_settings(fig, ax, title, ymax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, ymax + 1)
    ax.set_title(title)

fig, axes = plt.subplots(1, 4, figsize=(16, 3))

# Line width
linewidths = [0.5, 1.0, 2.0, 4.0]
for n, linewidth in enumerate(linewidths):
    axes[0].plot(x, y + n, color="blue", linewidth=linewidth)
axes_settings(fig, axes[0], "linewidth", len(linewidths))

# Line style
linestyles = ['-', '-.', ':']
for n, linestyle in enumerate(linestyles):
    axes[1].plot(x, y + n, color="blue", lw=2, linestyle=linestyle)
# custom dash style
line, = axes[1].plot(x, y + 3, color="blue", lw=2)
length1, gap1, length2, gap2 = 10, 7, 20, 7
line.set_dashes([length1, gap1, length2, gap2])
axes_settings(fig, axes[1], "linetypes", len(linestyles) + 1)

# marker types
markers = ['+', 'o', '*', 's', '.', '1', '2', '3', '4']
for n, marker in enumerate(markers):
    # lw = shorthand for linewidth, ls = shorthand for linestyle
    axes[2].plot(x, y + n, color="blue", lw=2, ls=':', marker=marker)
axes_settings(fig, axes[2], "markers", len(markers))

# marker size and color
markersizecolors = [(4, "white"), (8, "red"), (12, "yellow"), (16, "lightgreen")]
for n, (markersize, markerfacecolor) in enumerate(markersizecolors):
    axes[3].plot(x, y + n, color="blue", lw=1, ls='-',
                 marker='o', markersize=markersize,
                 markerfacecolor=markerfacecolor, markeredgewidth=2)
axes_settings(fig, axes[3], "marker size/color", len(markersizecolors))

plt.show()
```
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65255648-388e9d00-db39-11e9-8bb9-6ab95b74a035.png)

<hr class='division3'>
</details>
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (15)](https://user-images.githubusercontent.com/52376448/65261831-0d5d7b00-db44-11e9-939f-113b438e8bf1.png)
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

**Using custom colors for scatter plots**
<div class='jb-medium'>
<strong>Common color for all the dots</strong>: If the color parameter is a valid matplotlib color definition, then all the dots will appear in that color.<br>
<strong>Individual color for each dot</strong>: If the color parameter is a sequence of a valid matplotlib color definition, the ith dot will appear in the ith color. Of course, we have to give the required colors for each dot.<br>
</div>
```python
import numpy as np
import matplotlib.pyplot as plt

A = np.random.standard_normal((100, 2))
A += np.array((-1, -1)) # Center the distrib. at <-1, -1>

B = np.random.standard_normal((100, 2))
B += np.array((1, 1)) # Center the distrib. at <1, 1>

plt.scatter(A[:,0], A[:,1], color = '.25')
plt.scatter(B[:,0], B[:,1], color = '.75')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/65246096-d9289100-db28-11e9-9c49-4becaeddee8e.png)
<hr class='division3'>
</details>

```python
import numpy as np
import matplotlib.pyplot as plt

label_set = (
 b'Iris-setosa',
 b'Iris-versicolor',
 b'Iris-virginica',
)

def read_label(label):
     return label_set.index(label)

data = np.loadtxt('iris.data.txt',
                  delimiter = ',',
                  converters = { 4 : read_label })

color_set = ('.00', '.50', '.75')
color_list = [color_set[int(label)] for label in data[:,4]]

plt.scatter(data[:,0], data[:,1], color = color_list)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65246097-d9289100-db28-11e9-8a5e-c863cbb317fd.png)
<hr class='division3'>
</details>

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.standard_normal((100, 2))

plt.scatter(data[:,0], data[:,1], color = '1.0', edgecolor='0.0')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65246098-d9289100-db28-11e9-89b6-8ce6c1bf24e3.png)
<hr class='division3'>
</details>
<br><br><br>


**Using colormaps for scatter plots**
```python
%matplotlib inline
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

N = 256
angle = np.linspace(0, 8 * 2 * np.pi, N)
radius = np.linspace(.5, 1., N)

X = radius * np.cos(angle)
Y = radius * np.sin(angle)

plt.scatter(X, Y, c = angle, cmap = cm.hsv)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (18)](https://user-images.githubusercontent.com/52376448/65245812-425bd480-db28-11e9-8c5c-f9268073c446.png)
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


**Using custom colors for bar charts**
```python
import numpy as np
import matplotlib.pyplot as plt

women_pop = np.array([5., 30., 45., 22.])
men_pop = np.array([5., 25., 50., 20.])
X = np.arange(4)

plt.barh(X, women_pop, color = '.25')
plt.barh(X, -men_pop, color = '.75')
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65248179-b5674a00-db2c-11e9-940a-d431275e15f4.png)
<hr class='division3'>
</details>
```python
import numpy as np
import matplotlib.pyplot as plt

values = np.random.random_integers(99, size = 50)

color_set = ('.00', '.25', '.50', '.75')
color_list = [color_set[(len(color_set) * val) // 100] for val in values]

plt.bar(np.arange(len(values)), values, color = color_list)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65248181-b5674a00-db2c-11e9-99c9-120122ea0f9f.png)
<hr class='division3'>
</details>
<br><br><br>

**Using colormaps for bar charts**
```python
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.pyplot as plt

values = np.random.random_integers(99, size = 50)

cmap = cm.ScalarMappable(col.Normalize(0, 99), cm.binary)

plt.bar(np.arange(len(values)), values, color = cmap.to_rgba(values))
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/65248691-8dc4b180-db2d-11e9-8390-c720e20eec5a.png)
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

**Using custom colors for pie charts**

```python
import numpy as np
import matplotlib.pyplot as plt

values = np.random.rand(8)
color_set = ('.00', '.25', '.50', '.75')

plt.pie(values, colors = color_set)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/65248862-da0ff180-db2d-11e9-8e65-e999f470cae7.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Some more plots

```python
%matplotlib inline
import matplotlib.pyplot as plt

x = np.linspace(-5, 2, 20)
y =  x**3 + 5*x**2 + 10

fig, axes = plt.subplots(3,2)
axes[0, 0].step(x, y)
axes[0, 1].bar(x, y)
axes[1, 0].fill_between(x, y)
axes[1, 1].scatter(x, y)
axes[2, 0].quiver(x, y)
axes[2, 1].errorbar(x, y)

plt.tight_layout()
plt.show()
```
`SUPPLEMENT` : <span class="jb-medium">Refer to [here](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-Matplotlib.html#axes) about exes</span>
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드](https://user-images.githubusercontent.com/52376448/65252887-af756700-db34-11e9-8091-888e27546f61.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***Working with Figures***

#### Multiple plot, all at once

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
# linewidth       : lw       : 선 굵기
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


---

### ***Working with a File Output***


<hr class="division2">

## **Axes**

```python
%matplotlib inline
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=2)
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/65250814-26a8fc00-db31-11e9-9305-95ad79512ce5.png)
<hr class='division3'>
</details>
<br><br><br>

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
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65256122-fade4400-db39-11e9-936d-c17e751340ca.png)
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


```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 50, 500)
y = np.sin(x) * np.exp(-x/10)

fig, ax = plt.subplots(figsize=(8, 2), subplot_kw={'facecolor': "#ebf5ff"})

ax.plot(x, y, lw=2)

ax.set_xlabel ("x", labelpad=5, fontsize=18, fontname='serif', color="blue")
ax.set_ylabel ("f(x)", labelpad=15, fontsize=18, fontname='serif', color="blue")
ax.set_title("axis labels and title example", fontsize=16, fontname='serif', color="blue")
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65258127-53631080-db3d-11e9-91b0-950a29c79f52.png)
<hr class='division3'>
</details>
<br><br><br>

---

#### Axis range

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([10, 20, 30, 40], [1, 4, 9, 16])
plt.xlim(-20, 70)
plt.ylim(-10, 30)
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (11)](https://user-images.githubusercontent.com/52376448/65261631-a6d85d00-db43-11e9-86cd-28c7b5399d1f.png)
<hr class='division3'>
</details>
<br><br><br>

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 30, 500)
y = np.sin(x) * np.exp(-x/10)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={'facecolor': "#ebf5ff"})

axes[0].plot(x, y, lw=2)
axes[0].set_xlim(-5, 35)
axes[0].set_ylim(-1, 1)
axes[0].set_title("set_xlim / set_y_lim")

axes[1].plot(x, y, lw=2)
axes[1].axis('tight')
axes[1].set_title("axis('tight')")

axes[2].plot(x, y, lw=2)
axes[2].axis('equal')
axes[2].set_title("axis('equal')")

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65258517-fa47ac80-db3d-11e9-895f-d5637963b14b.png)
<hr class='division3'>
</details>

<br><br><br>

---


#### Axis ticks, tick labels, and grids

**Common ticks**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.plot(x, y)
plt.xticks([-3.14, -3.14/2, 0, 3.14/2, 3.14])
plt.yticks([-1, 0, +1])
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (17)](https://user-images.githubusercontent.com/52376448/65263585-e5701680-db47-11e9-88a0-32fe87068c4f.png)
<hr class='division3'>
</details>
<br><br><br>

**Ticks that corresponds to the numeric values**
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.plot(x, y)
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (18)](https://user-images.githubusercontent.com/52376448/65263932-878ffe80-db48-11e9-9838-4a478f625b4c.png)
<hr class='division3'>
</details>
<br><br><br>


```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
y = np.sin(x) * np.exp(-x**2/20)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].plot(x, y, lw=2)
axes[0].set_title("default ticks")

axes[1].plot(x, y, lw=2)
axes[1].set_title("set_xticks")
axes[1].set_yticks([-1, 0, 1])
axes[1].set_xticks([-5, 0, 5])

axes[2].plot(x, y, lw=2)
axes[2].set_title("set_major_locator")
axes[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
axes[2].yaxis.set_major_locator(mpl.ticker.FixedLocator([-1, 0, 1]))
axes[2].xaxis.set_minor_locator(mpl.ticker.MaxNLocator(8))
axes[2].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(8))

axes[3].plot(x, y, lw=2)
axes[3].set_title("set_xticklabels")
axes[3].set_yticks([-1, 0, 1])
axes[3].set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
axes[3].set_xticklabels([r'$-2\pi$', r'$-\pi$', 0, r'$\pi$',   r'$2\pi$'])
x_minor_ticker = mpl.ticker.FixedLocator([-3 * np.pi / 2,  -np.pi / 2, 0,
                                          
np.pi / 2, 3 * np.pi / 2])
axes[3].xaxis.set_minor_locator(x_minor_ticker)
axes[3].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(4))

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/65258741-62968e00-db3e-11e9-8e82-5deec42e8f4f.png)
<hr class='division3'>
</details>

<br><br><br>


```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

x_major_ticker = mpl.ticker.MultipleLocator(4)
x_minor_ticker = mpl.ticker.MultipleLocator(1)
y_major_ticker = mpl.ticker.MultipleLocator(0.5)
y_minor_ticker = mpl.ticker.MultipleLocator(0.25)

for ax in axes:
    ax.plot(x, y, lw=2)
    ax.xaxis.set_major_locator(x_major_ticker)
    ax.yaxis.set_major_locator(y_major_ticker)
    ax.xaxis.set_minor_locator(x_minor_ticker)
    ax.yaxis.set_minor_locator(y_minor_ticker)

axes[0].set_title("default grid")
axes[0].grid()

axes[1].set_title("major/minor grid")
axes[1].grid(color="blue", which="both", linestyle=':', linewidth=0.5)

axes[2].set_title("individual x/y major/minor grid")
axes[2].grid(color="grey", which="major", axis='x', linestyle='-', linewidth=0.5)
axes[2].grid(color="grey", which="minor", axis='x', linestyle=':', linewidth=0.25)
axes[2].grid(color="grey", which="major", axis='y', linestyle='-', linewidth=0.5)

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/65258959-d042ba00-db3e-11e9-8fd1-e5be4ae044e9.png)
<hr class='division3'>
</details>

<br><br><br>


```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

x = np.linspace(0, 1e5, 100)
y = x ** 2

axes[0].plot(x, y, 'b.')
axes[0].set_title("default labels", loc='right')

axes[1].plot(x, y, 'b')
axes[1].set_title("scientific notation labels", loc='right')

formatter = mpl.ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
axes[1].xaxis.set_major_formatter(formatter)
axes[1].yaxis.set_major_formatter(formatter)

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (7)](https://user-images.githubusercontent.com/52376448/65258960-d20c7d80-db3e-11e9-9db4-05414e7a3349.png)
<hr class='division3'>
</details>

<br><br><br>

---


#### Log plots

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

x = np.linspace(0, 1e3, 100)
y1, y2 = x**3, x**4

axes[0].set_title('loglog')
axes[0].loglog(x, y1, 'b', x, y2, 'r')

axes[1].set_title('semilogy')
axes[1].semilogy(x, y1, 'b', x, y2, 'r')

axes[2].set_title('plot / set_xscale / set_yscale')
axes[2].plot(x, y1, 'b', x, y2, 'r')
axes[2].set_xscale('log')
axes[2].set_yscale('log')

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/65259794-3976fd00-db40-11e9-8d01-c833b1e290ad.png)
<hr class='division3'>
</details>


<br><br><br>

---


#### Twin axes

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(8, 4))

r = np.linspace(0, 5, 100)
a = 4 * np.pi * r ** 2        # area
v = (4 * np.pi / 3) * r ** 3  # volume

ax1.set_title("surface area and volume of a sphere", fontsize=16)
ax1.set_xlabel("radius [m]", fontsize=16)

ax1.plot(r, a, lw=2, color="blue")
ax1.set_ylabel(r"surface area ($m^2$)", fontsize=16, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")
    
ax2 = ax1.twinx()
ax2.plot(r, v, lw=2, color="red")
ax2.set_ylabel(r"volume ($m^3$)", fontsize=16, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/65259796-3976fd00-db40-11e9-980b-5b72f8ac5f90.png)
<hr class='division3'>
</details>


<br><br><br>

---


#### Spines

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
y = np.sin(x) / x

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, y, linewidth=2)

# remove top and right spines

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# remove top and right spine ticks
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# move bottom and left spine to x = 0 and y = 0
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.set_xticks([-10, -5, 5, 10])
ax.set_yticks([0.5, 1])

#  give each label a solid background of white, to not overlap with the plot line
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_bbox({'facecolor': 'white', 'edgecolor': 'white'})

plt.show()
```
<details open markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/65259797-3976fd00-db40-11e9-972b-0fc15b2a18bb.png)
<hr class='division3'>
</details>


<br><br><br>

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



