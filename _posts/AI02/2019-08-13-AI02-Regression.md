---
layout : post
title : AI02, Regression
categories: [AI02]
comments : true
tags : [AI02]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[Meachine learning](https://userdyk-github.github.io/ai02/AI02-Contents.html)<br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Python Libraries</a>


---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Simple linear regression**
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/bf2c1cac7c1e6c9a426d92e9adad6ff4d8b4152e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:11.89ex; height:2.509ex;" alt="y=\alpha +\beta x,">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/968be557dd22b1a2e536b8d22369cfdb37f58703" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:18.197ex; height:2.509ex;" alt=" y_i = \alpha + \beta x_i + \varepsilon_i.">
</div>	
<details markdown="1">
<summary class='jb-small' style="color:blue">DERIVING</summary>
<hr class='division3'>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/91acaa000421826a90ecc40210890584d2d303ea" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:18.47ex; height:2.676ex;" alt="{\displaystyle {\widehat {\varepsilon }}_{i}=y_{i}-\alpha -\beta x_{i}.}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/5b3f79816862a7a29272130d3d75e87ed5cfeaa4" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:64.456ex; height:6.843ex;" alt="{\displaystyle {\text{Find }}\min _{\alpha ,\,\beta }Q(\alpha ,\beta ),\quad {\text{for }}Q(\alpha ,\beta )=\sum _{i=1}^{n}{\widehat {\varepsilon }}_{i}^{\,2}=\sum _{i=1}^{n}(y_{i}-\alpha -\beta x_{i})^{2}\ .}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/944e96221f03e99dbd57290c328b205b0f04c803" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -12.171ex; width:27.556ex; height:25.509ex;" alt="{\displaystyle {\begin{aligned}{\widehat {\alpha }}&amp;={\bar {y}}-{\widehat {\beta }}\,{\bar {x}},\\[5pt]{\widehat {\beta }}&amp;={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]&amp;={\frac {s_{x,y}}{s_{x}^{2}}}\\[5pt]&amp;=r_{xy}{\frac {s_{y}}{s_{x}}}.\\[6pt]\end{aligned}}}">
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">RESULTS</summary>
<hr class='division3'>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/944e96221f03e99dbd57290c328b205b0f04c803" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -12.171ex; width:27.556ex; height:25.509ex;" alt="{\displaystyle {\begin{aligned}{\widehat {\alpha }}&amp;={\bar {y}}-{\widehat {\beta }}\,{\bar {x}},\\[5pt]{\widehat {\beta }}&amp;={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]&amp;={\frac {s_{x,y}}{s_{x}^{2}}}\\[5pt]&amp;=r_{xy}{\frac {s_{y}}{s_{x}}}.\\[6pt]\end{aligned}}}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1f21324fcf481023ee5fafea8f4846f602c9c8f9" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -6.005ex; width:31.131ex; height:9.676ex;" alt="{\displaystyle r_{xy}={\frac {{\overline {xy}}-{\bar {x}}{\bar {y}}}{\sqrt {\left({\overline {x^{2}}}-{\bar {x}}^{2}\right)\left({\overline {y^{2}}}-{\bar {y}}^{2}\right)}}}.}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/3d570a1ea8bb90927e92f82d3864e4ade0447024" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:16.784ex; height:6.843ex;" alt="{\displaystyle {\overline {xy}}={\frac {1}{n}}\sum _{i=1}^{n}x_{i}y_{i}.}">
<hr class='division3'>
</details>


<br><br><br>
<hr class="division2">

## **Multiple linear regression**
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e8fe92790a76066af5556c62f5230bcc0bdf9f38" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -1.005ex; width:58.966ex; height:3.176ex;" alt="{\displaystyle y_{i}=\beta _{1}x_{i1}+\cdots +\beta _{p}x_{ip}+\varepsilon _{i}=\mathbf {x} _{i}^{\rm {T}}{\boldsymbol {\beta }}+\varepsilon _{i},\qquad i=1,\ldots ,n,}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8119b3ed1259aa8ff15166488548104b50a0f92e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:13.128ex; height:2.509ex;" alt="{\displaystyle \mathbf {y} =X{\boldsymbol {\beta }}+{\boldsymbol {\varepsilon }},\,}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c4b08d888086d1e0948f9019afec8ecec4a83151" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -6.838ex; width:80.728ex; height:14.843ex;" alt="{\displaystyle \mathbf {y} ={\begin{pmatrix}y_{1}\\y_{2}\\\vdots \\y_{n}\end{pmatrix}},\quad \mathbf {X} ={\begin{pmatrix}\mathbf {x} _{1}^{\rm {T}}\\\mathbf {x} _{2}^{\rm {T}}\\\vdots \\\mathbf {x} _{n}^{\rm {T}}\end{pmatrix}}={\begin{pmatrix}x_{11}&amp;\cdots &amp;x_{1p}\\x_{21}&amp;\cdots &amp;x_{2p}\\\vdots &amp;\ddots &amp;\vdots \\x_{n1}&amp;\cdots &amp;x_{np}\end{pmatrix}},\quad {\boldsymbol {\beta }}={\begin{pmatrix}\beta _{1}\\\beta _{2}\\\vdots \\\beta _{p}\end{pmatrix}},\quad {\boldsymbol {\varepsilon }}={\begin{pmatrix}\varepsilon _{1}\\\varepsilon _{2}\\\vdots \\\varepsilon _{n}\end{pmatrix}}.}">
</div>
<details markdown="1">
<summary class='jb-small' style="color:blue">DERIVING</summary>
<hr class='division3'>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/bf227a8af979e716d08f7c82dd95b17440e33a15" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.838ex; width:18.045ex; height:5.343ex;" alt="{\hat {\boldsymbol {\beta }}}={\underset {\boldsymbol {\beta }}{\operatorname {arg\,min} }}\,S({\boldsymbol {\beta }}),">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f607606c60752eff21a8fd43c12ced695975b61e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.338ex; width:43.109ex; height:7.343ex;" alt="{\displaystyle S({\boldsymbol {\beta }})=\sum _{i=1}^{n}{\bigl |}y_{i}-\sum _{j=1}^{p}X_{ij}\beta _{j}{\bigr |}^{2}={\bigl \|}\mathbf {y} -\mathbf {X} {\boldsymbol {\beta }}{\bigr \|}^{2}.}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/d2a206e40a6dfbda414d6faf32c3c6e2d2165a64" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:17.432ex; height:3.343ex;" alt="(\mathbf {X} ^{\rm {T}}\mathbf {X} ){\hat {\boldsymbol {\beta }}}=\mathbf {X} ^{\rm {T}}\mathbf {y} .">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/46cf247a57b181c36165a0b6ae5ede6bdc1a24a3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:19.765ex; height:3.343ex;" alt="{\displaystyle {\hat {\boldsymbol {\beta }}}=(\mathbf {X} ^{\rm {T}}\mathbf {X} )^{-1}\mathbf {X} ^{\rm {T}}\mathbf {y} .}">
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">RESULTS</summary>
<hr class='division3'>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/46cf247a57b181c36165a0b6ae5ede6bdc1a24a3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:19.765ex; height:3.343ex;" alt="{\displaystyle {\hat {\boldsymbol {\beta }}}=(\mathbf {X} ^{\rm {T}}\mathbf {X} )^{-1}\mathbf {X} ^{\rm {T}}\mathbf {y} .}">
<hr class='division3'>
</details>

<br><br><br>

### ***Multicollinearity***


Multicollinearity refers to a situation in which two or more explanatory variables in a multiple regression model are highly linearly related. <br>

**Detection of multicollinearity** : Variance inflation factor (VIF)
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9ece9d349f82af412b94fcdd2e81c1c6de926936" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:15.656ex; height:6.343ex;" alt="{\displaystyle \mathrm {VIF} _{i}={\frac {1}{1-R_{i}^{2}}}}">


<br>

**Way to relieve multicollinearity**

- with eliminatation of any variables(Feature Selection)
	- considering correlation coefficient
	- Lasso
	- Stepwise
	- etc
	
- without eliminatation of any variables
	- AutoEncoder
	- PCA
	- Ridge

<br><br><br>
<hr class="division2">


## **Logistic regression model**

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">
penalty


## **Penalty of regression model**

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">



## **Implementation with a variety of library**

### ***Regression with statsmodel***

<span class="frame2">Simple linear regression</span>

`Data preprocessing`
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def f(x,a,b):
    return a*x + b

x = np.random.random(1000)
a = 3
b = 5

target = f(x,a,b)
df_input = pd.DataFrame(x)
df_target = pd.DataFrame(target)
df = pd.concat([df_input, df_target], axis=1)
df.columns = ['input','target']
Input = df['input']
Target = df['target']
constant_input = sm.add_constant(Input, has_constant='add')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Input</summary>
<hr class='division3'>
```python
Input.head()
```
`OUTPUT`
```
0    0.830166
1    0.542949
2    0.357683
3    0.688297
4    0.645634
Name: input, dtype: float64
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Constant_input</summary>
<hr class='division3'>
```python
constant_input.head()
```
`OUTPUT`
```
	const	input
0	1.0	0.830166
1	1.0	0.542949
2	1.0	0.357683
3	1.0	0.688297
4	1.0	0.645634
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Target</summary>
<hr class='division3'>
```python
Target.head()
```
`OUTPUT`
```
0    7.490499
1    6.628847
2    6.073050
3    7.064890
4    6.936902
Name: target, dtype: float64
```
<hr class='division3'>
</details>
<br>
`Regression analysis`
```python
model = sm.OLS(Target, constant_input)
fitted_model = model.fit()
fitted_model.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT : Regression coefficients etc</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/65604433-7597db00-dfe2-11e9-8141-dc5126370fb1.JPG)
<br>
```python
# Regression coefficients
fitted_model.params
```
`OUTPUT`
```
const    5.0
input    3.0
dtype: float64
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Estimated values v.s. Original values for target</summary>
<hr class='division3'>
`Estimated values` : 
<span style="font-size: 70%;">$$ \hat{y} = \hat{a}x + \hat{b} \to A\vec{X}$$</span>
```python
np.dot(constant_input, fitted_model.params)
```
```
array([7.49049949, 6.62884716, 6.07305033, 7.0648904 , 6.93690197,
       6.04064573, 6.5576149 , 6.74231639, 6.73183572, 7.07796106,
       ...
       5.74719815, 6.58978836, 6.25943715, 5.88547536, 7.40743629,
       5.77773424, 5.99074449, 6.12113732, 6.13392177, 6.92979226])
```
`Original values` : 
<span style="font-size: 70%;">$$ y = ax + b$$</span>
```python
f(x,a,b)
```
```
array([7.49049949, 6.62884716, 6.07305033, 7.0648904 , 6.93690197,
       6.04064573, 6.5576149 , 6.74231639, 6.73183572, 7.07796106,
       ...
       5.74719815, 6.58978836, 6.25943715, 5.88547536, 7.40743629,
       5.77773424, 5.99074449, 6.12113732, 6.13392177, 6.92979226])
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Residual analysis</summary>
<hr class='division3'>
`Residue`
```python
fitted_model.resid
```
```
0     -1.776357e-15
1     -2.664535e-15
2     -2.664535e-15
...
...
998   -3.552714e-15
999   -1.776357e-15
Length: 1000, dtype: float64
```
`Residual summation`
```python
np.sum(fitted_model.resid)
```
```
-2.652988939644274e-12
```
`Visualization for residue`
```python
fitted_model.resid.plot()
```
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65611092-4a66b900-dfed-11e9-8b83-44d5737fef70.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Prediction for new dataset</summary>
<hr class='division3'>
`Prediction`
```python
sample = np.random.random(10)
constant_sample = sm.add_constant(sample, has_constant='add')
fitted_model.predict(constant_sample)
```
```
array([5.20371122, 6.07617745, 7.77126507, 5.35615965, 7.44019585,
       5.94592521, 5.94306959, 6.56256376, 6.09420242, 6.39866773])
```
`Verification`
```python
f(sample,a,b)
```
```
array([5.20371122, 6.07617745, 7.77126507, 5.35615965, 7.44019585,
       5.94592521, 5.94306959, 6.56256376, 6.09420242, 6.39866773])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Curve fitting</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

plt.plot(x, f(x,a,b), 'x', lw=0, label="data")
plt.plot(x, 3*x + 5, label='result')            # from fitted_model.params
plt.ylim(0,10)
plt.legend()
plt.show()
```
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65609675-24d8b000-dfeb-11e9-85a8-6313e236843b.png)
<hr class='division3'>
</details>
<br><br><br>





<span class="frame2">Multiple linear regression</span>

`Data preprocessing`
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def f(x,y,z,a,b,c,r):
    return a*x + b*y + c*z + r

x = np.random.random(100)
y = np.random.random(100)
z = np.random.random(100)
a = 20
b = 50
c = 7
r = 3

target = f(x,y,z,a,b,c,r)
df_input1 = pd.DataFrame(x)
df_input2 = pd.DataFrame(y)
df_input3 = pd.DataFrame(z)
df_target = pd.DataFrame(target)
df = pd.concat([df_input1, df_input2, df_input3, df_target], axis=1)
df.columns = ['input1', 'input2', 'input3', 'target']
Input = df[['input1', 'input2', 'input3']]
Target = df['target']
constant_input = sm.add_constant(Input, has_constant='add')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Input</summary>
<hr class='division3'>
```python
Input.head()
```
`OUTPUT`
```
	input1		input2		input3
0	0.957632	0.276408	0.345041
1	0.821460	0.653252	0.549964
2	0.506590	0.261659	0.393543
3	0.500052	0.056861	0.041176
4	0.267245	0.639603	0.769945
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Constant_input</summary>
<hr class='division3'>
```python
constant_input.head()
```
`OUTPUT`
```
	const	input1		input2		input3
0	1.0	0.957632	0.276408	0.345041
1	1.0	0.821460	0.653252	0.549964
2	1.0	0.506590	0.261659	0.393543
3	1.0	0.500052	0.056861	0.041176
4	1.0	0.267245	0.639603	0.769945
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Target</summary>
<hr class='division3'>
```python
Target.head()
```
`OUTPUT`
```
0    38.388309
1    55.941549
2    28.969561
3    16.132320
4    45.714644
Name: target, dtype: float64
```
<hr class='division3'>
</details>
<br>
`Regression analysis`
```python
model = sm.OLS(Target, constant_input)
fitted_model = model.fit()
fitted_model.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT : Regression coefficients etc</summary>
<hr class='division3'>
![캡처](https://user-images.githubusercontent.com/52376448/65622909-a0ddf280-e001-11e9-8824-2113afe162bd.JPG)
```python
# Regression coefficients
fitted_model.params
```
`OUTPUT`
```
const      3.0
input1    20.0
input2    50.0
input3     7.0
dtype: float64
```
`Verification`

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/46cf247a57b181c36165a0b6ae5ede6bdc1a24a3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:19.765ex; height:3.343ex;" alt="{\displaystyle {\hat {\boldsymbol {\beta }}}=(\mathbf {X} ^{\rm {T}}\mathbf {X} )^{-1}\mathbf {X} ^{\rm {T}}\mathbf {y} .}">

```python
from numpy import linalg

B = linalg.inv(np.dot(constant_input.T, constant_input))
np.dot(np.dot(B, constant_input.T),target)
```
```
array([ 3., 20., 50.,  7.])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Estimated values v.s. Original values for target</summary>
<hr class='division3'>
`Estimated values` : <span style="font-size: 70%;">$$ \hat{s} = \hat{a}x + \hat{b}y + \hat{c}z + \hat{r} \to \hat{S}=\hat{A}X$$</span>
```python
np.dot(constant_input, fitted_model.params)
```
```
array([38.38830915, 55.94154925, 28.96956111, 16.13232006, 45.71464433,
       35.66915115, 54.48721376, 35.3255576 , 17.57414208, 12.2024595 ,
       40.89621614, 33.05053896, 14.50158372, 38.67065445, 53.48709859,
       42.59911466, 54.53748705, 46.69193071, 18.38867267, 45.87908774,
       40.6693773 , 36.01122162, 11.68815215, 44.31558167, 41.80645497,
       49.37841447, 47.09113841, 53.96541726, 36.77556825, 23.52950327,
       38.64777777, 34.16965497, 50.26840963, 40.02741955, 44.16716928,
       42.3150182 , 25.99497711, 41.40530879, 27.36066677, 47.86915385,
       25.70932186, 24.86294199, 55.0745327 , 22.98417126, 32.50294778,
       17.8420005 , 61.35284467, 36.43911886, 49.76839721, 50.56165004,
       40.71292581, 36.41847389, 23.38460759, 59.30680731, 39.40085223,
       25.87053451, 40.11977913, 24.80379252, 53.38541514, 60.33980335,
       45.01501126, 51.37600515, 48.30658941, 30.00273352, 42.44824437,
       52.17219373, 21.72628098, 74.51174471, 47.41694199, 16.47748332,
       16.18670621, 26.77202999, 67.7470938 , 46.24996358, 41.99306012,
       35.44894821, 28.65531671, 29.65139668, 53.31971577, 22.99141254,
       51.20655459, 50.54080656, 66.4153275 , 39.5569899 , 39.35911854,
       39.014512  , 34.51325153, 35.5253818 , 50.8264082 , 18.76223046,
       66.14916028, 37.23867282, 28.3269569 , 53.50468595, 55.85972521,
       54.48370671, 61.87997791, 24.69145197, 47.79432371, 41.2612825 ])
```
`Original values` : <span style="font-size: 70%;">$$ s = ax + by + cz + r$$</span>
```python
f(x,y,z,a,b,c,r)
```
```
array([38.38830915, 55.94154925, 28.96956111, 16.13232006, 45.71464433,
       35.66915115, 54.48721376, 35.3255576 , 17.57414208, 12.2024595 ,
       40.89621614, 33.05053896, 14.50158372, 38.67065445, 53.48709859,
       42.59911466, 54.53748705, 46.69193071, 18.38867267, 45.87908774,
       40.6693773 , 36.01122162, 11.68815215, 44.31558167, 41.80645497,
       49.37841447, 47.09113841, 53.96541726, 36.77556825, 23.52950327,
       38.64777777, 34.16965497, 50.26840963, 40.02741955, 44.16716928,
       42.3150182 , 25.99497711, 41.40530879, 27.36066677, 47.86915385,
       25.70932186, 24.86294199, 55.0745327 , 22.98417126, 32.50294778,
       17.8420005 , 61.35284467, 36.43911886, 49.76839721, 50.56165004,
       40.71292581, 36.41847389, 23.38460759, 59.30680731, 39.40085223,
       25.87053451, 40.11977913, 24.80379252, 53.38541514, 60.33980335,
       45.01501126, 51.37600515, 48.30658941, 30.00273352, 42.44824437,
       52.17219373, 21.72628098, 74.51174471, 47.41694199, 16.47748332,
       16.18670621, 26.77202999, 67.7470938 , 46.24996358, 41.99306012,
       35.44894821, 28.65531671, 29.65139668, 53.31971577, 22.99141254,
       51.20655459, 50.54080656, 66.4153275 , 39.5569899 , 39.35911854,
       39.014512  , 34.51325153, 35.5253818 , 50.8264082 , 18.76223046,
       66.14916028, 37.23867282, 28.3269569 , 53.50468595, 55.85972521,
       54.48370671, 61.87997791, 24.69145197, 47.79432371, 41.2612825 ])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Residual analysis</summary>
<hr class='division3'>
`Residue`
```python
fitted_model.resid
```
```
0     0.000000e+00
1    -7.105427e-15
2     1.776357e-14
3     2.486900e-14
4     7.105427e-15
5     7.105427e-15
          ...     
95    0.000000e+00
96   -1.421085e-14
97    2.486900e-14
98    1.421085e-14
99    7.105427e-15
Length: 100, dtype: float64
```
`Visualization for residue`
```python
import matplotlib.pyplot as plt

fitted_model.resid.plot()
plt.show()
```
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65626087-ed2c3100-e007-11e9-9073-7983f8d3984c.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Prediction for new dataset</summary>
<hr class='division3'>
`Prediction`
```python
sample = np.random.random((10,3))
constant_sample = sm.add_constant(sample, has_constant='add')
fitted_model.predict(constant_sample)
```
```
array([47.18460385, 29.42685672, 45.34542694, 21.18367219, 60.83667819,
       31.51219742, 16.92413439, 31.70573065, 19.8877936 , 47.38519353])
```
`Verification`
```python
f(sample[:,0],sample[:,1],sample[:,2],a,b,c,r)
```
```
array([47.18460385, 29.42685672, 45.34542694, 21.18367219, 60.83667819,
       31.51219742, 16.92413439, 31.70573065, 19.8877936 , 47.38519353])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Curve fitting</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>





---

### ***Regression with sklearn***

```python
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model 
import matplotlib.pyplot as plt 
import numpy as np

X_all, y_all = datasets.make_regression(n_samples=50, n_features=50, n_informative=10)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, train_size=0.5)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

def sse(resid):
    return np.sum(resid**2) 
    
resid_train = y_train - model.predict(X_train) 
sse_train = sse(resid_train)   
sse_train

resid_test = y_test - model.predict(X_test)  
sse_test = sse(resid_test)   
sse_test 

# R-squared score 
model.score(X_train, y_train) 
model.score(X_test, y_test) 

def plot_residuals_and_coeff(resid_train, resid_test, coeff): 
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))  
    axes[0].bar(np.arange(len(resid_train)), resid_train) 
    axes[0].set_xlabel("sample number")  
    axes[0].set_ylabel("residual")    
    axes[0].set_title("training data")   
    axes[1].bar(np.arange(len(resid_test)), resid_test) 
    axes[1].set_xlabel("sample number")  
    axes[1].set_ylabel("residual")   
    axes[1].set_title("testing data")  
    axes[2].bar(np.arange(len(coeff)), coeff)  
    axes[2].set_xlabel("coefficient number")
    axes[2].set_ylabel("coefficient")   
    fig.tight_layout()   
    return fig, axes
    
fig, ax = plot_residuals_and_coeff(resid_train, resid_test,  model.coef_)
```
![Figure_1](https://user-images.githubusercontent.com/52376448/65573279-5cbd0480-dfa5-11e9-8b36-c315e0f9653f.png)

<br><br><br>

---


### ***Regression with tensorflow***

<br><br><br>

---

### ***Regression with pytorch***

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
