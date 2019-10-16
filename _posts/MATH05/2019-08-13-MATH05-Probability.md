---
layout : post
title : MATH05, Probability
categories: [MATH05]
comments : true
tags : [MATH05]
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

## **Mass : Multi-variate random variable**

### ***joint probability mass function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8203262cf269dbc408cef23390b9a658a4cc4141" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -1.005ex; margin-left: -0.089ex; width:33.766ex; height:3.009ex;" alt="{\displaystyle p_{X,Y}(x,y)=\mathrm {P} (X=x\ \mathrm {and} \ Y=y)}"></div>
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
<br><br><br>

---

### ***conditional probability mass function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9a1bf9c7af083e400a87dbbd646c508bf5de6ec0" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; margin-left: -0.089ex; width:59.017ex; height:6.509ex;" alt="{\displaystyle p_{Y|X}(y\mid x)\triangleq P(Y=y\mid X=x)={\frac {P(\{X=x\}\cap \{Y=y\})}{P(X=x)}}}"></div>
<br><br><br>

<hr class="division2">

## **Density : Multi-variate random variable**

### ***joint probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/58f7f825cb219d7e826edc68dd99f75de9f626d0" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.505ex; width:26.31ex; height:6.509ex;" alt="{\displaystyle f_{X,Y}(x,y)={\frac {\partial ^{2}F_{X,Y}(x,y)}{\partial x\partial y}}}"></div>
<br><br><br>

---

### ***marginal probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/243911724de0d94b5b041482401c4c1e067cdf3e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; margin-left: -0.089ex; width:50.596ex; height:6.009ex;" alt="{\displaystyle p_{X}(x)=\int _{y}p_{X,Y}(x,y)\,\mathrm {d} y=\int _{y}p_{X\mid Y}(x\mid y)\,p_{Y}(y)\,\mathrm {d} y,}"></div>
<br><br><br>

---

### ***conditional probability density function***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b2e16052d580d418e683bb220a41c2c895227945" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; width:24.46ex; height:6.509ex;" alt="{\displaystyle f_{Y\mid X}(y\mid x)={\frac {f_{X,Y}(x,y)}{f_{X}(x)}}}"></div>
<br><br><br>

<hr class="division2">

## **Independent**
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c3fee81720676c2887e6304414377aecb51e5579" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:22.872ex; height:2.843ex;" alt="\mathrm{P}(A \cap B) = \mathrm{P}(A)\mathrm{P}(B)"></div>
<br><br><br>


<hr class="division2">


## title3

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

