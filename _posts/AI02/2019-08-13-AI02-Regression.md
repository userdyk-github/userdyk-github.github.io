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

### ***Model performance indicators for training dataset***
- <a href='https://en.wikipedia.org/wiki/Mean_squared_error' target="_blank" class="frame2">Mean squared error (MSE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:24.729ex; height:6.843ex;" alt="{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}">
- <a href='https://en.wikipedia.org/wiki/Mean_absolute_percentage_error' target="_blank" class="frame2">Mean absolute percentage error (MAPE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4cf2158513b0345211300fe585cc88a05488b451" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:26.512ex; height:6.843ex;" alt="{\displaystyle {\mbox{M}}={\frac {100\%}{n}}\sum _{t=1}^{n}\left|{\frac {A_{t}-F_{t}}{A_{t}}}\right|,}">
- <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2" target="_blank" class="frame2">Adjusted R-squared</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/16a082d105dfbb4339e40cf7898950ce748743e8" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.338ex; width:28.793ex; height:5.676ex;" alt="{\displaystyle {\bar {R}}^{2}=1-(1-R^{2}){n-1 \over n-p-1}}">
- <a href="https://en.wikipedia.org/wiki/Akaike_information_criterion" target="_blank" class="frame2">Akaike information criterion (AIC)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fe67d436d9064a370cbe800b24b05ee8a68d491b" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:20.229ex; height:3.343ex;" alt="{\displaystyle \mathrm {AIC} \,=\,2k-2\ln({\hat {L}})}">
- <a href="https://en.wikipedia.org/wiki/Bayesian_information_criterion" target="_blank" class="frame2">Bayesian information criterion (BIC)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ed4064ea6babc6e374a149840c19f0ae7396d7d3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:24.952ex; height:3.343ex;" alt="{\displaystyle \mathrm {BIC} =\ln(n)k-2\ln({\widehat {L}}).\ }">
<br><br><br>

### ***Diagnosis for regression***

- Residuals Scatter plot
- Normal Q-Q Plot
- Residual vs Fitted plot

<br><br><br>
<hr class="division2">





## **Multivariate linear regression**
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

### ***Model performance indicators for training dataset***
- <a href='https://en.wikipedia.org/wiki/Mean_squared_error' target="_blank" class="frame2">Mean squared error (MSE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:24.729ex; height:6.843ex;" alt="{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}">
- <a href='https://en.wikipedia.org/wiki/Mean_absolute_percentage_error' target="_blank" class="frame2">Mean absolute percentage error (MAPE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4cf2158513b0345211300fe585cc88a05488b451" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:26.512ex; height:6.843ex;" alt="{\displaystyle {\mbox{M}}={\frac {100\%}{n}}\sum _{t=1}^{n}\left|{\frac {A_{t}-F_{t}}{A_{t}}}\right|,}">
- <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2" target="_blank" class="frame2">Adjusted R-squared</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/16a082d105dfbb4339e40cf7898950ce748743e8" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.338ex; width:28.793ex; height:5.676ex;" alt="{\displaystyle {\bar {R}}^{2}=1-(1-R^{2}){n-1 \over n-p-1}}">
- <a href="https://en.wikipedia.org/wiki/Akaike_information_criterion" target="_blank" class="frame2">Akaike information criterion (AIC)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fe67d436d9064a370cbe800b24b05ee8a68d491b" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:20.229ex; height:3.343ex;" alt="{\displaystyle \mathrm {AIC} \,=\,2k-2\ln({\hat {L}})}">
- <a href="https://en.wikipedia.org/wiki/Bayesian_information_criterion" target="_blank" class="frame2">Bayesian information criterion (BIC)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ed4064ea6babc6e374a149840c19f0ae7396d7d3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:24.952ex; height:3.343ex;" alt="{\displaystyle \mathrm {BIC} =\ln(n)k-2\ln({\widehat {L}}).\ }">

<br><br><br>


### ***Diagnosis for regression***

- Residuals Scatter plot
- Normal Q-Q Plot
- Residual vs Fitted plot

<br><br><br>


#### Multicollinearity


Multicollinearity refers to a situation in which two or more explanatory variables in a multiple regression model are highly linearly related. <br>

**Detection of multicollinearity** : Variance inflation factor(<a href='https://en.wikipedia.org/wiki/Variance_inflation_factor#Calculation_and_analysis' target="_blank">VIF</a>)
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9ece9d349f82af412b94fcdd2e81c1c6de926936" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:15.656ex; height:6.343ex;" alt="{\displaystyle \mathrm {VIF} _{i}={\frac {1}{1-R_{i}^{2}}}}">


<br>

**Way to relieve multicollinearity**

- with eliminatation of any variables(Feature Selection)
	- Variables selection
		- Feedforward selection
		- Backward selection
		- Stepwise
	- Correlation coefficient
	- Lasso
	- etc
	
- without eliminatation of any variables
	- <a href="https://userdyk-github.github.io/ai03/AI03-Restricted-boltzmann-machines-and-auto-encoders.html" target="_blank">AutoEncoder</a>
	- [PCA](https://userdyk-github.github.io/ai02/AI02-Dimensionality-reduction.html)
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



### ***Model performance indicators***

- <a href='https://en.wikipedia.org/wiki/Mean_squared_error' target="_blank" class="frame2">Mean squared error (MSE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:24.729ex; height:6.843ex;" alt="{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}">
- <a href='https://en.wikipedia.org/wiki/Mean_absolute_percentage_error' target="_blank" class="frame2">Mean absolute percentage error (MAPE)</a>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4cf2158513b0345211300fe585cc88a05488b451" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:26.512ex; height:6.843ex;" alt="{\displaystyle {\mbox{M}}={\frac {100\%}{n}}\sum _{t=1}^{n}\left|{\frac {A_{t}-F_{t}}{A_{t}}}\right|,}">
- <a href="https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank" class="frame2">Confusion matrix</a>
- <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" target="_blank" class="frame2">Receiver operating characteristic curve, or ROC curve</a>
- <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve" target="_blank" class="frame2">Area under the curve, or AUC</a>



<br><br><br>

### ***Diagnosis for regression***

- Residuals Scatter plot
- Normal Q-Q Plot
- Residual vs Fitted plot

<br><br><br>

<hr class="division2">

## **Nonlinear regression**
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7e1ed763761f312a68f66dd0c7c37cbdf0ca9d65" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:11.576ex; height:2.843ex;" alt="{\displaystyle \mathbf {y} \sim f(\mathbf {x} ,{\boldsymbol {\beta }})}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/d2d3e51114f731925c32dd44daa47103bff0e17f" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.338ex; width:29.873ex; height:5.843ex;" alt="{\displaystyle f(x_{i},{\boldsymbol {\beta }})\approx f(x_{i},0)+\sum _{j}J_{ij}\beta _{j}}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/32e16aa7490c3f8c5885d467ddeeaded77f1f032" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.671ex; width:15.805ex; height:6.509ex;" alt="{\displaystyle J_{ij}={\frac {\partial f(x_{i},{\boldsymbol {\beta }})}{\partial \beta _{j}}}}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b9a506569aaa2fdb4c0e6d690a6b5ccc1cac8c5c" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; width:18.688ex; height:3.343ex;" alt="{\hat  {{\boldsymbol  {\beta }}}}\approx {\mathbf  {(J^{T}J)^{{-1}}J^{T}y}}.">
</div>
<br><br><br>

### ***Linearization***
<div class="frame1">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/88162961396e4f8edc14ac210d5303c405d03a22" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; margin-right: -0.387ex; width:10.615ex; height:3.009ex;" alt="y=ae^{{bx}}U\,\!">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e44ab255511d24d3076303dad10ed53f1f7979d5" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.838ex; margin-right: -0.387ex; width:24.127ex; height:2.843ex;" alt="\ln {(y)}=\ln {(a)}+bx+u,\,\!">
</div>

<br><br><br>
<hr class="division2">



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

#### Simple linear regression about artificial dataset

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
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65604433-7597db00-dfe2-11e9-8141-dc5126370fb1.JPG)
<hr class='division3_1'>
</details>

<br>
```python
# Regression coefficients
fitted_model.params
```
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
<summary class='jb-small' style="color:blue">Model diagnosis</summary>
<hr class='division3'>
`Residual`
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
<summary class='jb-small' style="color:blue">Model prediction</summary>
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





#### Multivariate linear regression about artificial dataset

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
<br>
```python
constant_input.head()
```
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
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65622909-a0ddf280-e001-11e9-8824-2113afe162bd.JPG)
<hr class='division3_1'>
</details>
<br>
```python
# Regression coefficients
fitted_model.params
```
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
<summary class='jb-small' style="color:blue">Model diagnosis</summary>
<hr class='division3'>
`Residual`
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
<summary class='jb-small' style="color:blue">Model prediction</summary>
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


#### Multivariate linear regression about dataset on real world

[Dataset download][1] ｜ <a href="https://www.kaggle.com/prasadperera/the-boston-housing-dataset/data" target="_blank">URL</a>

<details markdown="1">
<summary class='jb-small' style="color:blue">Dataset Description</summary>
<hr class='division3'>
- CRIM - per capita crime rate by town
- ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS - proportion of non-retail business acres per town.
- CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX - nitric oxides concentration (parts per 10 million)
- RM - average number of rooms per dwelling
- AGE - proportion of owner-occupied units built prior to 1940
- DIS - weighted distances to five Boston employment centres
- RAD - index of accessibility to radial highways
- TAX - full-value property-tax rate per $10,000
- PTRATIO - pupil-teacher ratio by town
- B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT - % lower status of the population
- MEDV - Median value of owner-occupied homes in $1000's
<hr class='division3'>
</details>

`Data preprocessing`
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

boston = pd.read_csv(r'C:\Users\userd\Desktop\dataset\boston_house.csv')
Input_s = boston[['CRIM', 'RM', 'LSTAT']]
Input_L = boston[['CRIM', 'RM', 'LSTAT', 'B', 'TAX', 'AGE', 'ZN', 'NOX', 'INDUS']]
Target = boston['MEDV']
constant_Input_s = sm.add_constant(Input_s, has_constant='add')
constant_Input_L = sm.add_constant(Input_L, has_constant='add')
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Input</summary>
<hr class='division3'>
```python
constant_Input_s.head()
```
```
	const	CRIM	RM	LSTAT
0	1.0	0.00632	6.575	4.98
1	1.0	0.02731	6.421	9.14
2	1.0	0.02729	7.185	4.03
3	1.0	0.03237	6.998	2.94
4	1.0	0.06905	7.147	5.33
```
<br>
```python
constant_Input_L.head()
```
```
	const	CRIM	RM	LSTAT	B	TAX	AGE	ZN	NOX	INDUS
0	1.0	0.00632	6.575	4.98	396.90	296	65.2	18.0	0.538	2.31
1	1.0	0.02731	6.421	9.14	396.90	242	78.9	0.0	0.469	7.07
2	1.0	0.02729	7.185	4.03	392.83	242	61.1	0.0	0.469	7.07
3	1.0	0.03237	6.998	2.94	394.63	222	45.8	0.0	0.458	2.18
4	1.0	0.06905	7.147	5.33	396.90	222	54.2	0.0	0.458	2.18
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Data diagnosis</summary>
<hr class='division3'>
`Multicollinearity` : Variance inflation factor(VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(Input_L.values, i) for i in range(Input_L.shape[1])]
vif['features'] = Input_L.columns
vif
```
```
	VIF Factor	features
0	1.917332	CRIM
1	46.535369	RM
2	8.844137	LSTAT
3	16.856737	B
4	19.923044	TAX
5	18.457503	AGE
6	2.086502	ZN
7	72.439753	NOX
8	12.642137	INDUS
```
<br>
`Multicollinearity` : Correlation coefficient
```python
Input_L.corr()
```
```
	CRIM		RM		LSTAT		B		TAX		AGE		ZN		NOX		INDUS
CRIM	1.000000	-0.219247	0.455621	-0.385064	0.582764	0.352734	-0.200469	0.420972	0.406583
RM	-0.219247	1.000000	-0.613808	0.128069	-0.292048	-0.240265	0.311991	-0.302188	-0.391676
LSTAT	0.455621	-0.613808	1.000000	-0.366087	0.543993	0.602339	-0.412995	0.590879	0.603800
B	-0.385064	0.128069	-0.366087	1.000000	-0.441808	-0.273534	0.175520	-0.380051	-0.356977
TAX	0.582764	-0.292048	0.543993	-0.441808	1.000000	0.506456	-0.314563	0.668023	0.720760
AGE	0.352734	-0.240265	0.602339	-0.273534	0.506456	1.000000	-0.569537	0.731470	0.644779
ZN	-0.200469	0.311991	-0.412995	0.175520	-0.314563	-0.569537	1.000000	-0.516604	-0.533828
NOX	0.420972	-0.302188	0.590879	-0.380051	0.668023	0.731470	-0.516604	1.000000	0.763651
INDUS	0.406583	-0.391676	0.603800	-0.356977	0.720760	0.644779	-0.533828	0.763651	1.000000
```
```python
import seaborn as sns
cmap = sns.light_palette('darkgray', as_cmap=True)
sns.heatmap(Input_L.corr(), annot=True, cmap=cmap)
plt.show()
```
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65659538-f993a680-e066-11e9-87b2-7a2b2ab1069c.png)
```python
sns.pairplot(Input_L)
plt.show()
```
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65659569-129c5780-e067-11e9-98de-68bb15197a09.png)
<hr class='division3'>
</details>


<br>
`Regression analysis`
```python
model_s = sm.OLS(Target, constant_Input_s)
model_L = sm.OLS(Target, constant_Input_L)
fitted_model_s = model_s.fit()
fitted_model_L = model_L.fit()
```
```python
fitted_model_s.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65658719-d3203c00-e063-11e9-8fc5-4cf3ad12393d.JPG)
<hr class='division3_1'>
</details>



<br>
```python
fitted_model_s.params
```
```
const   -2.562251
CRIM    -0.102941
RM       5.216955
LSTAT   -0.578486
dtype: float64
```
<hr class='division3'>
</details>
<br>
```python
fitted_model_L.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65658743-ed5a1a00-e063-11e9-9238-7473a57981cd.JPG)
<hr class='division3_1'>
</details>



<br>
```python
fitted_model_L.params
```
```
const   -7.108827
CRIM    -0.045293
RM       5.092238
LSTAT   -0.565133
B        0.008974
TAX     -0.006025
AGE      0.023619
ZN       0.029377
NOX      3.483832
INDUS    0.029270
dtype: float64
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Model diagnosis</summary>
<hr class='division3'>
`Residual analysis`
```python
import matplotlib.pyplot as plt

fitted_model_s.resid.plot(label="base")
fitted_model_L.resid.plot(label="full")
plt.legend()
plt.show()
```
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65658872-68bbcb80-e064-11e9-82ca-b8c0f7dc0f69.png)
<hr class='division3'>
</details>
<br>
`Modify regression model(based on backward elimination)`
```python
from sklearn.model_selection import train_test_split

# Data preprocessing
Input1 = Input_L.drop('NOX', axis=1)
Input2 = Input_L.drop(['NOX','RM'], axis=1)
constant_Input1 = sm.add_constant(Input1, has_constant='add')
constant_Input2 = sm.add_constant(Input2, has_constant='add')
X = constant_Input_L
X1 = constant_Input1
X2 = constant_Input2
y = Target

train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state = 1)
train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y, train_size=0.7, test_size=0.3, random_state = 1)
train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y, train_size=0.7, test_size=0.3, random_state = 1)

# Regression analysis
model = sm.OLS(train_y, train_x)
model1 = sm.OLS(train_y1, train_x1)
model2 = sm.OLS(train_y2, train_x2)

fitted_model = model.fit()
fitted_model1 = model1.fit()
fitted_model2 = model2.fit()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Data : Input</summary>
<hr class='division3'>
```python
constant_Input1.head()
```
```
	const	CRIM	RM	LSTAT	B	TAX	AGE	ZN	INDUS
0	1.0	0.00632	6.575	4.98	396.90	296	65.2	18.0	2.31
1	1.0	0.02731	6.421	9.14	396.90	242	78.9	0.0	7.07
2	1.0	0.02729	7.185	4.03	392.83	242	61.1	0.0	7.07
3	1.0	0.03237	6.998	2.94	394.63	222	45.8	0.0	2.18
4	1.0	0.06905	7.147	5.33	396.90	222	54.2	0.0	2.18
```
<br>
```python
constant_Input2.head()
```
```
	const	CRIM	LSTAT	B	TAX	AGE	ZN	INDUS
0	1.0	0.00632	4.98	396.90	296	65.2	18.0	2.31
1	1.0	0.02731	9.14	396.90	242	78.9	0.0	7.07
2	1.0	0.02729	4.03	392.83	242	61.1	0.0	7.07
3	1.0	0.03237	2.94	394.63	222	45.8	0.0	2.18
4	1.0	0.06905	5.33	396.90	222	54.2	0.0	2.18
```
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Data diagnosis</summary>
<hr class='division3'>
`Multicollinearity` : Variance inflation factor(VIF)
```python
vif1 = pd.DataFrame()
vif2 = pd.DataFrame()
vif1['VIF1 Factor'] = [variance_inflation_factor(Input1.values, i) for i in range(Input1.shape[1])]
vif2['VIF2 Factor'] = [variance_inflation_factor(Input2.values, i) for i in range(Input2.shape[1])]
vif1['features1'] = Input1.columns
vif2['features2'] = Input2.columns
pd.concat([vif,vif1,vif2], axis=1)
```
```
	VIF Factor	features	VIF1 Factor	features1	VIF2 Factor	features2
0	1.917332	CRIM		1.916648	CRIM		1.907517	CRIM
1	46.535369	RM		30.806301	RM		7.933529	LSTAT
2	8.844137	LSTAT		8.171214	LSTAT		7.442569	B
3	16.856737	B		16.735751	B		16.233237	TAX
4	19.923044	TAX		18.727105	TAX		13.765377	AGE
5	18.457503	AGE		16.339792	AGE		1.820070	ZN
6	2.086502	ZN		2.074500	ZN		11.116823	INDUS
7	72.439753	NOX		11.217461	INDUS		NaN		NaN
8	12.642137	INDUS		NaN		NaN		NaN		NaN
```

<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
```python
fitted_model.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65662631-b12cb680-e06f-11e9-9d53-ab8bcf12aad6.JPG)
<hr class='division3_1'>
</details>

<br>
```python
fitted_model1.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65662654-c275c300-e06f-11e9-9536-b19450689907.JPG)
<hr class='division3_1'>
</details>

<br>
```python
fitted_model2.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/65662687-d91c1a00-e06f-11e9-9e74-0f03b0f1df1b.JPG)
<hr class='division3_1'>
</details>

<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Model prediction</summary>
<hr class='division3'>
```python
plt.plot(np.array(fitted_model.predict(test_x)), label="model with full variables")
plt.plot(np.array(fitted_model1.predict(test_x1)), label="model1 eliminated 1 variable")
plt.plot(np.array(fitted_model2.predict(test_x2)), label="model2 eliminated 2 variables")
plt.plot(np.array(test_y), label="true")
plt.legend()
plt.show()
```
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65662871-4a5bcd00-e070-11e9-934e-72160199bde1.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Model diagnosis</summary>
<hr class='division3'>
`Residual analysis`
```python
plt.plot(np.array(test_y.values-fitted_model.predict(test_x)),label='residual of model')
plt.plot(np.array(test_y.values-fitted_model1.predict(test_x1)),label='residual of model1')
plt.plot(np.array(test_y.values-fitted_model2.predict(test_x2)),label='residual; of model2')
plt.legend()
plt.show()
```
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/65662921-6e1f1300-e070-11e9-9d2c-f6d38eb8a57d.png)

<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Model performance</summary>
<hr class='division3'>
```python
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_true=test_y.values, y_pred=fitted_model.predict(test_x)))
print(mean_squared_error(y_true=test_y.values, y_pred=fitted_model1.predict(test_x1)))
print(mean_squared_error(y_true=test_y.values, y_pred=fitted_model2.predict(test_x2)))
```
```
26.148631468819843
26.14006260984654
38.788453179128304
```
<hr class='division3'>
</details>
<br><br><br>


#### Multivariate nonlinear regression about dataset on real world

[Dataset download][2] ｜ <a href="https://www.kaggle.com/klkwak/toyotacorollacsv/version/1#ToyotaCorolla.csv" target="_blank">URL</a>

`Data preprocessing`
```python
## [0] : Load libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


## [1] : Load dataset
corolla = pd.read_csv(r'C:\Users\userd\Desktop\dataset\ToyotaCorolla.csv')
nCar = corolla.shape[0]
nVar = corolla.shape[1]


## [2] : categorical data-type > binary data-type
# Create dummy variables
dummy_p = np.repeat(0,nCar)
dummy_d = np.repeat(0,nCar)
dummy_c = np.repeat(0,nCar)

# Save index for 'Fuel_Type'
p_idx = np.array(corolla.Fuel_Type == "Petrol")
d_idx = np.array(corolla.Fuel_Type == "Diesel")
c_idx = np.array(corolla.Fuel_Type == "CNG")

# Substitute binary = 1 after slicing
dummy_p[p_idx] = 1  # Petrol
dummy_d[d_idx] = 1  # Diesel
dummy_c[c_idx] = 1  # CNG


## [3] : Eliminate unnecessary variables and add dummy variables
Fuel = pd.DataFrame({'Petrol': dummy_p, 'Diesel': dummy_d, 'CNG': dummy_c})
corolla_ = corolla.dropna().drop(['Id','Model','Fuel_Type'], axis=1, inplace=False)
mlr_data = pd.concat((corolla_, Fuel), 1)


## [4] : Add bias
mlr_data = sm.add_constant(mlr_data, has_constant='add')


## [5] : Divide into input data and output data
feature_columns = list(mlr_data.columns.difference(['Price']))
X = mlr_data[feature_columns]
y = mlr_data.Price
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">[1] Data : Input</summary>
<hr class='division3'>
```python
corolla.head()
```
```
	Id	Model						Price	Age_08_04	Mfg_Month	Mfg_Year	KM		Fuel_Type	HP	Met_Color	...	Central_Lock	Powered_Windows	Power_Steering	Radio	Mistlamps		Sport_Model	Backseat_Divider	Metallic_Rim	Radio_cassette	Tow_Bar
0	1	TOYOTA Corolla 2.0 D4D HATCHB TERRA 2/3-Doors	13500	23		10		2002		46986		Diesel		90	1		...	1		1		1		0	0			0		1			0		0		0
1	2	TOYOTA Corolla 2.0 D4D HATCHB TERRA 2/3-Doors	13750	23		10		2002		72937		Diesel		90	1		...	1		0		1		0	0			0		1			0		0		0
2	3	?TOYOTA Corolla 2.0 D4D HATCHB TERRA 2/3-Doors	13950	24		9		2002		41711		Diesel		90	1		...	0		0		1		0	0			0		1			0		0		0
3	4	TOYOTA Corolla 2.0 D4D HATCHB TERRA 2/3-Doors	14950	26		7		2002		48000		Diesel		90	0		...	0		0		1		0	0			0		1			0		0		0
4	5	TOYOTA Corolla 2.0 D4D HATCHB SOL 2/3-Doors	13750	30		3		2002		38500		Diesel		90	0		...	1		1		1		0	1			0		1			0		0		0
5 rows × 37 columns
```

<br>
```python
print('nCar: %d' % nCar, 'nVar: %d' % nVar )
```
```
nCar: 1436 nVar: 37
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">[2] Data : Input</summary>
<hr class='division3'>
```python
dummy_p
```
```
array([0, 0, 0, ..., 0, 0, 0])
```
```python
dummy_d
```
```
array([0, 0, 0, ..., 0, 0, 0])
```
```python
dummy_c
```
```
array([0, 0, 0, ..., 0, 0, 0])
```
<br>
```python
p_idx
```
```
array([False, False, False, ...,  True,  True,  True])
```
```python
d_idx
```
```
array([ True,  True,  True, ..., False, False, False])
```
```python
c_idx
```
```
array([False, False, False, ..., False, False, False])
```
<br>

```python
dummy_p
```
```
array([0, 0, 0, ..., 1, 1, 1])
```
```python
dummy_d
```
```
array([1, 1, 1, ..., 0, 0, 0])
```
```python
dummy_c
```
```
array([0, 0, 0, ..., 0, 0, 0])
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">[3] Data : Input</summary>
<hr class='division3'>
```python
Fuel.head()
```
```
	Petrol	Diesel	CNG
0	0	1	0
1	0	1	0
2	0	1	0
3	0	1	0
4	0	1	0
```
```python
Fuel.shape
```
```
(1436, 3)
```
<br>
```python
corolla_.head()
```
```
	Price	Age_08_04	Mfg_Month	Mfg_Year	KM	HP	Met_Color	Automatic	cc	Doors	...	Central_Lock	Powered_Windows	Power_Steering	Radio	Mistlamps	Sport_Model	Backseat_Divider	Metallic_Rim	Radio_cassette	Tow_Bar
0	13500	23		10		2002		46986	90	1		0		2000	3	...	1		1		1		0	0		0		1			0		0		0
1	13750	23		10		2002		72937	90	1		0		2000	3	...	1		0		1		0	0		0		1			0		0		0
2	13950	24		9		2002		41711	90	1		0		2000	3	...	0		0		1		0	0		0		1			0		0		0
3	14950	26		7		2002		48000	90	0		0		2000	3	...	0		0		1		0	0		0		1			0		0		0
4	13750	30		3		2002		38500	90	0		0		2000	3	...	1		1		1		0	1		0		1			0		0		0
5 rows × 34 columns
```
```python
corolla_.shape
```
```
(1436, 34)
```
<br>
```python
mlr_data.head()
```
```
	Price	Age_08_04	Mfg_Month	Mfg_Year	KM	HP	Met_Color	Automatic	cc	Doors	...	Radio	Mistlamps	Sport_Model	Backseat_Divider	Metallic_Rim	Radio_cassette	Tow_Bar	Petrol	Diesel	CNG
0	13500	23		10		2002		46986	90	1		0		2000	3	...	0	0		0		1			0		0		0	0	1	0
1	13750	23		10		2002		72937	90	1		0		2000	3	...	0	0		0		1			0		0		0	0	1	0
2	13950	24		9		2002		41711	90	1		0		2000	3	...	0	0		0		1			0		0		0	0	1	0
3	14950	26		7		2002		48000	90	0		0		2000	3	...	0	0		0		1			0		0		0	0	1	0
4	13750	30		3		2002		38500	90	0		0		2000	3	...	0	1		0		1			0		0		0	0	1	0
5 rows × 37 columns
```
```python
mlr_data.shape
```
```
(1436, 37)
```
<hr class='division3'>
</details>
<br>

`Regression analysis`
```python
# Train the MLR(fitting regression model)
full_model = sm.OLS(train_y, train_x)
fitted_full_model = full_model.fit()
fitted_full_model.summary()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Model performance</summary>
<hr class='division3'>
<span class="jb-medium">R2 is high, a majority of variables is meaningful</span>
![1](https://user-images.githubusercontent.com/52376448/66271015-5461a500-e894-11e9-8101-4138ae77a0cb.JPG)
![2](https://user-images.githubusercontent.com/52376448/66271016-5461a500-e894-11e9-93e5-e5afa748c16b.JPG)
![3](https://user-images.githubusercontent.com/52376448/66271017-5461a500-e894-11e9-86c7-3b2b03013b85.JPG)
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

[1]:{{ site.url }}/download/AI02/boston_house.csv
[2]:{{ site.url }}/download/AI02/ToyotaCorolla.csv
