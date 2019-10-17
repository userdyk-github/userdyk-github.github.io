---
layout : post
title : MATH05, Probability
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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

## **Discrete : Uni-variate random variable**
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/13827ff57e01b13cceff9cf50cd9542cd4b7db70" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:36.004ex; height:2.843ex;" alt="{\displaystyle \operatorname {P} (X\in S)=\operatorname {P} (\{\omega \in \Omega \mid X(\omega )\in S\})}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f81c05aba576a12b4e05ee3f4cba709dd16139c7" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:19.165ex; height:2.843ex;" alt="F_{X}(x)=\operatorname {P} (X\leq x)">
<br><br><br>
<hr class="division2">

## **Continuous: Uni-variate random variable**
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/13827ff57e01b13cceff9cf50cd9542cd4b7db70" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:36.004ex; height:2.843ex;" alt="{\displaystyle \operatorname {P} (X\in S)=\operatorname {P} (\{\omega \in \Omega \mid X(\omega )\in S\})}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f81c05aba576a12b4e05ee3f4cba709dd16139c7" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:19.165ex; height:2.843ex;" alt="F_{X}(x)=\operatorname {P} (X\leq x)">
<br><br><br>
### ***Normal distribution***
```python
from scipy import stats

X = stats.norm(1,.5)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Descript statistic</summary>
<hr class='division3'>
```python
X.rvs(10)
```
```
array([0.5903325 , 1.29429924, 1.00703009, 1.21073729, 1.51287354,
       0.76989052, 0.96913931, 2.03268324, 0.65025789, 0.15307278])
```
<br>
```python
X.mean()
```
```
1
```
<br>
```python
X.median() 
```
```
1.0
```
<br>
```python
X.std()
```
```
0.5
```

<br>
```python
X.var()
```
```
0.25
```

<br>
```python
[X.moment(n) for n in range(5)] 
```
```
[1.0, 1.0, 1.25, 1.75, 2.6875]
```

<br>
```python
X.stats()
# stats.norm.stats(loc=1, scale=0.5) 
# stats.norm(loc=1, scale=0.5).stats()
```
```
(array(1.), array(0.25))
```

<br>
```python
X.pdf([0, 1, 2]) 
```
```
array([0.10798193, 0.79788456, 0.10798193])
```

<br>
```python
X.cdf([0, 1, 2]) 
```
```
array([0.02275013, 0.5       , 0.97724987])
```

<br>
```python
X.interval(0.95)
```
```
(0.020018007729972975, 1.979981992270027)
```

<br>
```python
X.interval(0.99) 
```
```
(-0.2879146517744502, 2.28791465177445)
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_rv_distribution(X, axes=None):   
    """Plot the PDF or PMF, CDF, SF and PPF of a given random variable"""   
    if axes is None:   
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))   
        
    x_min_999, x_max_999 = X.interval(0.999)  
    x999 = np.linspace(x_min_999, x_max_999, 1000)  
    x_min_95, x_max_95 = X.interval(0.95)    
    x95 = np.linspace(x_min_95, x_max_95, 1000)    
    
    if hasattr(X.dist, "pdf"):    
        axes[0].plot(x999, X.pdf(x999), label="PDF")   
        axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)
    else:    
        # discrete random variables do not have a pdf method, instead we use pmf:  
        x999_int = np.unique(x999.astype(int))  
        axes[0].bar(x999_int, X.pmf(x999_int), label="PMF")  
    axes[1].plot(x999, X.cdf(x999), label="CDF")   
    axes[1].plot(x999, X.sf(x999), label="SF")   
    axes[2].plot(x999, X.ppf(x999), label="PPF")  
                                                                        
    for ax in axes:
        ax.legend()
        
        
fig, axes = plt.subplots(3, 3, figsize=(12, 9))   

X = stats.norm()   
plot_rv_distribution(X, axes=axes[0, :]) 
axes[0, 0].set_ylabel("Normal dist.") 

X = stats.f(2, 50)    
plot_rv_distribution(X, axes=axes[1, :]) 
axes[1, 0].set_ylabel("F dist.")  

X = stats.poisson(5)  
plot_rv_distribution(X, axes=axes[2, :])
axes[2, 0].set_ylabel("Poisson dist.")        
```
![download (20)](https://user-images.githubusercontent.com/52376448/66984484-c25c6680-f0f5-11e9-8eca-0730f046e00e.png)

<br><br><br>

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_dist_samples(X, X_samples, title=None, ax=None):
    """ Plot the PDF and histogram of samples of a continuous random variable"""  
    if ax is None:    
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
     
    x_lim = X.interval(.99) 
    x = np.linspace(*x_lim, num=100)
     
    ax.plot(x, X.pdf(x), label="PDF", lw=3)    
    ax.hist(X_samples, label="samples", normed=1, bins=75)  
    ax.set_xlim(*x_lim)  
    ax.legend()   
    
    if title:   
        ax.set_title(title) 
    return ax

fig, axes = plt.subplots(1, 3, figsize=(12, 3))  
N = 2000  
# Student's t distribution  
X = stats.t(7.0)  
plot_dist_samples(X, X.rvs(N), "Student's t dist.", ax=axes[0])

# The chisquared distribution  
X = stats.chi2(5.0)    
plot_dist_samples(X, X.rvs(N), r"$\chi^2$ dist.", ax=axes[1])

# The exponential distribution   
X = stats.expon(0.5)   
plot_dist_samples(X, X.rvs(N), "exponential dist.", ax=axes[2]) 
```
![download (21)](https://user-images.githubusercontent.com/52376448/66984899-9db4be80-f0f6-11e9-9aa4-a8739d2f40f6.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Chi-square distribution***
```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

X = stats.chi2(df=5) 
X_samples = X.rvs(500) 
df, loc, scale = stats.chi2.fit(X_samples)
Y = stats.chi2(df=df, loc=loc, scale=scale)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))  
x_lim = X.interval(.99)   
x = np.linspace(*x_lim, num=100)  

axes[0].plot(x, X.pdf(x), label="original")  
axes[0].plot(x, Y.pdf(x), label="recreated")    
axes[0].legend()   

axes[1].plot(x, X.pdf(x) - Y.pdf(x), label="error")    
axes[1].legend()
```
![download (22)](https://user-images.githubusercontent.com/52376448/66985061-f84e1a80-f0f6-11e9-8707-960c9cc6edca.png)
<br><br><br>
<hr class="division2">


## **Mass : Multi-variate random variable**

### ***joint probability mass function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8203262cf269dbc408cef23390b9a658a4cc4141" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -1.005ex; margin-left: -0.089ex; width:33.766ex; height:3.009ex;" alt="{\displaystyle p_{X,Y}(x,y)=\mathrm {P} (X=x\ \mathrm {and} \ Y=y)}"></div>
<br>
<span class="frame3">Dataset</span>
```python
import pandas as pd

grades = ["A", "B", "C", "D", "E", "F"]
scores = pd.DataFrame(
    [[1, 2, 1, 0, 0, 0],
     [0, 2, 3, 1, 0, 0],
     [0, 4, 7, 4, 1, 0],
     [0, 1, 4, 5, 4, 0],
     [0, 0, 1, 3, 2, 0],
     [0, 0, 0, 1, 2, 1]], 
    columns=grades, index=grades)
scores.index.name = "Y"
scores.columns.name = "X"
scores
```
```
X	A	B	C	D	E	F
Y						
A	1	2	1	0	0	0
B	0	2	3	1	0	0
C	0	4	7	4	1	0
D	0	1	4	5	4	0
E	0	0	1	3	2	0
F	0	0	0	1	2	1
```
<br>
<span class="frame3">joint probability mass function</span>
```python
pmf = scores / scores.values.sum()
pmf
```
```
X	A   	B   	C	    D	    E   	F
Y						
A	0.02	0.04	0.02	0.00	0.00	0.00
B	0.00	0.04	0.06	0.02	0.00	0.00
C	0.00	0.08	0.14	0.08	0.02	0.00
D	0.00	0.02	0.08	0.10	0.08	0.00
E	0.00	0.00	0.02	0.06	0.04	0.00
F	0.00	0.00	0.00	0.02	0.04	0.02
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.heatmap(pmf, cmap=mpl.cm.bone_r, annot=True,
            xticklabels=['A', 'B', 'C', 'D', 'E', 'F'],
            yticklabels=['A', 'B', 'C', 'D', 'E', 'F'])
plt.title("joint probability density function p(x,y)")
plt.tight_layout()
plt.show()
```
![download](https://user-images.githubusercontent.com/52376448/66946102-344b9600-f08b-11e9-9df4-e57393387530.png)
<hr class='division3'>
</details>

<br><br><br>

---

### ***marginal probability mass function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1139c2f18bfaccfd669eaafb58cacec22bbec926" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.338ex; width:72.207ex; height:5.843ex;" alt="{\displaystyle \Pr(X=x)=\sum _{y}\Pr(X=x,Y=y)=\sum _{y}\Pr(X=x\mid Y=y)\Pr(Y=y),}"></div>
<br>
<span class="frame3">Dataset</span>
```python
import pandas as pd
import numpy as np

grades = ["A", "B", "C", "D", "E", "F"]
scores = pd.DataFrame(
    [[1, 2, 1, 0, 0, 0],
     [0, 2, 3, 1, 0, 0],
     [0, 4, 7, 4, 1, 0],
     [0, 1, 4, 5, 4, 0],
     [0, 0, 1, 3, 2, 0],
     [0, 0, 0, 1, 2, 1]], 
    columns=grades, index=grades)
scores.index.name = "Y"
scores.columns.name = "X"
scores
```
```
X	A	B	C	D	E	F
Y						
A	1	2	1	0	0	0
B	0	2	3	1	0	0
C	0	4	7	4	1	0
D	0	1	4	5	4	0
E	0	0	1	3	2	0
F	0	0	0	1	2	1
```
<br>
<span class="frame3">marginal probability mass function</span>
```python
pmf = scores / scores.values.sum()
pmf_marginal_x = pmf.sum(axis=0)
pmf_marginal_y = pmf.sum(axis=1)
```
```python
pmf_marginal_x
#pmf_marginal_x[np.newaxis, :]
```
```
X
A    0.02
B    0.18
C    0.32
D    0.28
E    0.18
F    0.02
dtype: float64
```
```python
pmf_marginal_y
#pmf_marginal_y[:, np.newaxis]
```
```
Y
A    0.08
B    0.12
C    0.32
D    0.28
E    0.12
F    0.08
dtype: float64
```
<br><br><br>

---

### ***conditional probability mass function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9a1bf9c7af083e400a87dbbd646c508bf5de6ec0" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; margin-left: -0.089ex; width:59.017ex; height:6.509ex;" alt="{\displaystyle p_{Y|X}(y\mid x)\triangleq P(Y=y\mid X=x)={\frac {P(\{X=x\}\cap \{Y=y\})}{P(X=x)}}}"></div>
<br>
<span class="frame3">Dataset</span>
```python
import pandas as pd
import numpy as np

grades = ["A", "B", "C", "D", "E", "F"]
scores = pd.DataFrame(
    [[1, 2, 1, 0, 0, 0],
     [0, 2, 3, 1, 0, 0],
     [0, 4, 7, 4, 1, 0],
     [0, 1, 4, 5, 4, 0],
     [0, 0, 1, 3, 2, 0],
     [0, 0, 0, 1, 2, 1]], 
    columns=grades, index=grades)
scores.index.name = "Y"
scores.columns.name = "X"
scores
```
```
X	A	B	C	D	E	F
Y						
A	1	2	1	0	0	0
B	0	2	3	1	0	0
C	0	4	7	4	1	0
D	0	1	4	5	4	0
E	0	0	1	3	2	0
F	0	0	0	1	2	1
```
<br>
<span class="frame3">conditional probability mass function</span>
```python
pmf = scores / scores.values.sum()
pmf_marginal_x = pmf.sum(axis=0)
pmf_marginal_y = pmf.sum(axis=1)

def conditional_x(y):
    return pmf.iloc[y-1, :]/pmf_marginal_y[y-1]
def conditional_y(x):
    return pmf.iloc[:, x-1]/pmf_marginal_x[x-1]
```
```python
for i in range(1, pmf.shape[0]+1):
    print("conditional_x(y=%d)\n"%(i),conditional_x(i), "\n")
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
conditional_x(y=1)
 X
A    0.25
B    0.50
C    0.25
D    0.00
E    0.00
F    0.00
Name: A, dtype: float64 

conditional_x(y=2)
 X
A    0.000000
B    0.333333
C    0.500000
D    0.166667
E    0.000000
F    0.000000
Name: B, dtype: float64 

conditional_x(y=3)
 X
A    0.0000
B    0.2500
C    0.4375
D    0.2500
E    0.0625
F    0.0000
Name: C, dtype: float64 

conditional_x(y=4)
 X
A    0.000000
B    0.071429
C    0.285714
D    0.357143
E    0.285714
F    0.000000
Name: D, dtype: float64 

conditional_x(y=5)
 X
A    0.000000
B    0.000000
C    0.166667
D    0.500000
E    0.333333
F    0.000000
Name: E, dtype: float64 

conditional_x(y=6)
 X
A    0.00
B    0.00
C    0.00
D    0.25
E    0.50
F    0.25
Name: F, dtype: float64 
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<span class="frame3">given y, cross section of joint probability mass function</span>
```python
import string
import matplotlib.pyplot as plt

pmf = scores / scores.values.sum()

x = np.arange(6)
for i, y in enumerate(string.ascii_uppercase[:6]):
    ax = plt.subplot(6, 1, i + 1)
    ax.tick_params(labelleft=False)
    plt.bar(x, conditional_x(i+1))
    plt.ylabel("p(x, y=%s)/p(x)"%y, rotation=0, labelpad=40)
    plt.ylim(0, 1)
    plt.xticks(range(6), ['A', 'B', 'C', 'D', 'E', 'F'])

plt.suptitle("given y and $p(x)=\sum_{y} p(x,y)$, conditional probability mass function(x)", x=0.55 ,y=1.09)
plt.tight_layout()

plt.show()
```
![download (7)](https://user-images.githubusercontent.com/52376448/66974513-43a40100-f0d6-11e9-8f86-4d0b30305561.png)
<hr class='division3'>
</details>
<br>

```python
for i in range(1, pmf.shape[1]+1):
    print("conditional_y(x=%d)\n"%(i),conditional_y(i), "\n")
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
conditional_y(x=1)
 Y
A    1.0
B    0.0
C    0.0
D    0.0
E    0.0
F    0.0
Name: A, dtype: float64 

conditional_y(x=2)
 Y
A    0.222222
B    0.222222
C    0.444444
D    0.111111
E    0.000000
F    0.000000
Name: B, dtype: float64 

conditional_y(x=3)
 Y
A    0.0625
B    0.1875
C    0.4375
D    0.2500
E    0.0625
F    0.0000
Name: C, dtype: float64 

conditional_y(x=4)
 Y
A    0.000000
B    0.071429
C    0.285714
D    0.357143
E    0.214286
F    0.071429
Name: D, dtype: float64 

conditional_y(x=5)
 Y
A    0.000000
B    0.000000
C    0.111111
D    0.444444
E    0.222222
F    0.222222
Name: E, dtype: float64 

conditional_y(x=6)
 Y
A    0.0
B    0.0
C    0.0
D    0.0
E    0.0
F    1.0
Name: F, dtype: float64 
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<span class="frame3">given x, cross section of joint probability mass function</span>
```python
import string
import matplotlib.pyplot as plt

pmf = scores / scores.values.sum()

x = np.arange(6)
for i, y in enumerate(string.ascii_uppercase[:6]):
    ax = plt.subplot(6, 1, i + 1)
    ax.tick_params(labelleft=False)
    plt.bar(x, conditional_y(i+1))
    plt.ylabel("p(x=%s, y)/p(y)"%y, rotation=0, labelpad=40)
    plt.ylim(0, 1)
    plt.xticks(range(6), ['A', 'B', 'C', 'D', 'E', 'F'])

plt.suptitle("given x and $p(y)=\sum_{x} p(x,y)$, conditional probability mass function(y)", x=0.55 ,y=1.09)
plt.tight_layout()

plt.show()
```
![download (8)](https://user-images.githubusercontent.com/52376448/66974514-43a40100-f0d6-11e9-951a-e794949e0fbd.png)
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Density : Multi-variate random variable**

### ***joint probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/58f7f825cb219d7e826edc68dd99f75de9f626d0" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.505ex; width:26.31ex; height:6.509ex;" alt="{\displaystyle f_{X,Y}(x,y)={\frac {\partial ^{2}F_{X,Y}(x,y)}{\partial x\partial y}}}"></div>
```python
from scipy import stats 
import matplotlib.pyplot as plt

# x:weight, y:height
mu = [70, 170]
cov = [[150, 140], [140, 300]]
rv = stats.multivariate_normal(mu, cov)

xx = np.linspace(20, 120, 100)
yy = np.linspace(100, 250, 100)
XX, YY = np.meshgrid(xx, yy)
ZZ = rv.pdf(np.dstack([XX, YY]))

plt.contour(XX, YY, ZZ)
plt.xlabel("x")
plt.ylabel("y")
plt.title("joint probability density function p(x,y)")
plt.show()
```
![download (9)](https://user-images.githubusercontent.com/52376448/66976410-d34cae00-f0dc-11e9-9553-4c4cfb49523d.png)
<br><br><br>

---

### ***marginal probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/243911724de0d94b5b041482401c4c1e067cdf3e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; margin-left: -0.089ex; width:50.596ex; height:6.009ex;" alt="{\displaystyle p_{X}(x)=\int _{y}p_{X,Y}(x,y)\,\mathrm {d} y=\int _{y}p_{X\mid Y}(x\mid y)\,p_{Y}(y)\,\mathrm {d} y,}"></div>
```python
from matplotlib.ticker import NullFormatter
from matplotlib import transforms
from scipy.integrate import simps  # 심슨법칙(Simpson's rule)을 사용한 적분 계산

xx = np.linspace(20, 120, 100)
yy = np.linspace(100, 250, 100)
XX, YY = np.meshgrid(xx, yy)
ZZ = rv.pdf(np.dstack([XX, YY]))
fx = [simps(Z, yy) for Z in ZZ.T]
fy = [simps(Z, xx) for Z in ZZ]

plt.figure(figsize=(6, 6))

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.05

rect1 = [left, bottom, width, height]
rect2 = [left, bottom_h, width, 0.2]
rect3 = [left_h, bottom, 0.2, height]

ax1 = plt.axes(rect1)
ax2 = plt.axes(rect2)
ax3 = plt.axes(rect3)

ax2.xaxis.set_major_formatter(NullFormatter())
ax3.yaxis.set_major_formatter(NullFormatter())

ax1.contour(XX, YY, ZZ)
ax1.set_title("joint probability density function $p_{XY}(x, y)$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.plot(xx, fx)
ax2.set_title("marginal probability \n density function $p_X(x)$")

base = ax3.transData
rot = transforms.Affine2D().rotate_deg(-90)
plt.plot(-yy, fy, transform=rot + base)
plt.title("marginal probability \n density function $p_Y(y)$")

ax1.set_xlim(38, 102)
ax1.set_ylim(120, 220)
ax2.set_xlim(38, 102)
ax3.set_xlim(0, 0.025)
ax3.set_ylim(120, 220)

plt.show()
```
![download (10)](https://user-images.githubusercontent.com/52376448/66976438-e95a6e80-f0dc-11e9-94ca-9f0fc0167c3d.png)
<br><br><br>

---

### ***conditional probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b2e16052d580d418e683bb220a41c2c895227945" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; width:24.46ex; height:6.509ex;" alt="{\displaystyle f_{Y\mid X}(y\mid x)={\frac {f_{X,Y}(x,y)}{f_{X}(x)}}}"></div>
<span class="frame3">Cross section of joint probability density function</span>
```python
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(20, 120, 100)
yy = np.linspace(100, 250, 16)
XX, YY = np.meshgrid(xx, yy)
ZZ = rv.pdf(np.dstack([XX, YY]))

fig = plt.figure(dpi=150)
ax = fig.gca(projection='3d')

xs = np.hstack([0, xx, 0])
zs = np.zeros_like(xs)
verts = []
for i, y in enumerate(yy):
    zs[1:-1] = ZZ[i]
    verts.append(list(zip(xx, zs)))

poly = PolyCollection(verts)
poly.set_alpha(0.5)
ax.add_collection3d(poly, zs=yy, zdir='y')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(20, 120)
ax.set_ylim(100, 250)
ax.set_zlim3d(0, 0.0007)
ax.view_init(20, -50)
plt.title("cross section of joint probability density function")
plt.show()
```
![download (11)](https://user-images.githubusercontent.com/52376448/66976497-258dcf00-f0dd-11e9-9a54-b82207ca5a5d.png)
<br>
```python
from scipy.integrate import simps  # 심슨법칙(Simpson's rule)을 사용한 적분 계산
import matplotlib.pyplot as plt
import numpy as np

mag = 10 # 확대 비율
xx = np.linspace(20, 120, 100)
yy = np.linspace(100, 250, 16)
XX, YY = np.meshgrid(xx, yy)
ZZ = rv.pdf(np.dstack([XX, YY]))
plt.figure(figsize=(8, 6))
for i, j in enumerate(range(9, 4, -1)):
    ax = plt.subplot(5, 1, i + 1)
    ax.tick_params(labelleft=False)
    plt.plot(xx, ZZ[j, :] * mag, 'r--', lw=2, label="cross section of joint probability density function")
    marginal = simps(ZZ[j, :], xx)
    plt.plot(xx, ZZ[j, :] / marginal, 'b-', lw=2, label="conditional probability density function")
    plt.ylim(0, 0.05)
    ax.xaxis.set_ticklabels([])
    plt.ylabel("p(x, y={:.0f})".format(yy[j]), rotation=0, labelpad=40)
    if i == 0: 
        plt.legend(loc=2)
plt.xlabel("x")
plt.tight_layout()
plt.show()
```
![download (12)](https://user-images.githubusercontent.com/52376448/66976569-51a95000-f0dd-11e9-996b-90bb39db06f5.png)

<br><br><br>

<hr class="division2">

## **Independent**
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c3fee81720676c2887e6304414377aecb51e5579" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:22.872ex; height:2.843ex;" alt="\mathrm{P}(A \cap B) = \mathrm{P}(A)\mathrm{P}(B)"></div>
<span class="frame3">independent two variable</span>
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pmf1 = np.array([[1, 2,  4, 2, 1],
                 [2, 4,  8, 4, 2],
                 [4, 8, 16, 8, 4],
                 [2, 4,  8, 4, 2],
                 [1, 2,  4, 2, 1]])
pmf1 = pmf1/pmf1.sum()

pmf1_marginal_x = np.round(pmf1.sum(axis=0), 2)
pmf1_marginal_y = np.round(pmf1.sum(axis=1), 2)
pmf1x = pmf1_marginal_x * pmf1_marginal_y[:, np.newaxis]

plt.subplot(121)
sns.heatmap(pmf1, cmap=mpl.cm.bone_r, annot=True, square=True, linewidth=1, linecolor="k",
            cbar=False, xticklabels=pmf1_marginal_x, yticklabels=pmf1_marginal_y)
plt.title("independent two variable - \n joint probability mass function")

plt.subplot(122)
pmf1x = pmf1_marginal_x * pmf1_marginal_y[:, np.newaxis]
sns.heatmap(pmf1x, cmap=mpl.cm.bone_r, annot=True, square=True, linewidth=1, linecolor="k",
            cbar=False, xticklabels=pmf1_marginal_x, yticklabels=pmf1_marginal_y)
plt.title("two variable - the product of \n joint probability mass function")
plt.tight_layout()
plt.show()
```
![download (13)](https://user-images.githubusercontent.com/52376448/66979066-23c90900-f0e7-11e9-82ec-5fefae6027ac.png)
<br><br><br>
<span class="frame3">dependent two variable</span>
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pmf2 = np.array([[0, 0,  0, 5, 5],
                 [0, 5,  5, 5, 5],
                 [0, 5, 30, 5, 0],
                 [5, 5,  5, 5, 0],
                 [5, 5,  0, 0, 0]])
pmf2 = pmf2/pmf2.sum()

pmf2_marginal_x = np.round(pmf2.sum(axis=0), 2)
pmf2_marginal_y = np.round(pmf2.sum(axis=1), 2)

plt.subplot(121)
sns.heatmap(pmf2, cmap=mpl.cm.bone_r, annot=True, square=True, linewidth=1, linecolor="k",
            cbar=False, xticklabels=pmf2_marginal_x, yticklabels=pmf2_marginal_y)
plt.title("dependent two variable - \n joint probability mass function")

plt.subplot(122)
pmf2x = pmf2_marginal_x * pmf2_marginal_y[:, np.newaxis]
sns.heatmap(pmf2x, cmap=mpl.cm.bone_r, annot=True, square=True, linewidth=1, linecolor="k",
            cbar=False, xticklabels=pmf2_marginal_x, yticklabels=pmf2_marginal_y)
plt.title("two variable - the product of \n joint probability mass function")
plt.tight_layout()
plt.show()
```
![download (14)](https://user-images.githubusercontent.com/52376448/66979137-67237780-f0e7-11e9-8809-7ce8456989f4.png)
<br><br><br>

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/e5c379559a4a4fe9a9d8eeace69da425/' target="_blank">multi-variate random number</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

