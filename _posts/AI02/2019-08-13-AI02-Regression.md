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

|STEP|INPUT|PROCESS|OUTPUT|
|:-|:-|:-|:-|
|1|csv file|Data preprocessing|train dataset, test dataset|
|2|train dataset, test dataset|Regression analysis|full model|
|3|full model|Modify regression model|forward, backward, stepwise|



<br><br><br>

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
constant_mlr_data = sm.add_constant(mlr_data, has_constant='add')


## [5] : Divide into input data and output data
feature_columns = list(constant_mlr_data.columns.difference(['Price']))
X = constant_mlr_data[feature_columns]
y = constant_mlr_data.Price
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
<details markdown="1">
<summary class='jb-small' style="color:blue">[4] Data : Input</summary>
<hr class='division3'>
```python
constant_mlr_data.head()
```
```
	const	Price	Age_08_04	Mfg_Month	Mfg_Year	KM	HP	Met_Color	Automatic	cc	...	Radio	Mistlamps	Sport_Model	Backseat_Divider	Metallic_Rim	Radio_cassette	Tow_Bar	Petrol	Diesel	CNG
0	1.0	13500	23		10		2002		46986	90	1		0		2000	...	0	0		0		1			0		0		0	0	1	0
1	1.0	13750	23		10		2002		72937	90	1		0		2000	...	0	0		0		1			0		0		0	0	1	0
2	1.0	13950	24		9		2002		41711	90	1		0		2000	...	0	0		0		1			0		0		0	0	1	0
3	1.0	14950	26		7		2002		48000	90	0		0		2000	...	0	0		0		1			0		0		0	0	1	0
4	1.0	13750	30		3		2002		38500	90	0		0		2000	...	0	1		0		1			0		0		0	0	1	0
5 rows × 38 columns
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">[5] Data : Input</summary>
<hr class='division3'>
```python
mlr_data.columns.difference(['Price'])
```
```
Index(['ABS', 'Age_08_04', 'Airbag_1', 'Airbag_2', 'Airco', 'Automatic',
       'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider',
       'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders',
       'Diesel', 'Doors', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Met_Color',
       'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Mistlamps',
       'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio',
       'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const'],
      dtype='object')
```
<br>
```python
X.head()
```
```
	ABS	Age_08_04	Airbag_1	Airbag_2	Airco	Automatic	Automatic_airco	BOVAG_Guarantee			Backseat_Divider	Boardcomputer	...	Power_Steering	Powered_Windows	Quarterly_Tax	Radio	Radio_cassette		Sport_Model	Tow_Bar	Weight	cc	const
0	1	23		1		1		0	0		0		1				1			1		...	1		1		210		0	0			0		0	1165	2000	1.0
1	1	23		1		1		1	0		0		1				1			1		...	1		0		210		0	0			0		0	1165	2000	1.0
2	1	24		1		1		0	0		0		1				1			1		...	1		0		210		0	0			0		0	1165	2000	1.0
3	1	26		1		1		0	0		0		1				1			1		...	1		0		210		0	0			0		0	1165	2000	1.0
4	1	30		1		1		1	0		0		1				1			1		...	1		1		210		0	0			0		0	1170	2000	1.0
5 rows × 37 columns
```
```python
y.head()
```
```
0    13500
1    13750
2    13950
3    14950
4    13750
Name: Price, dtype: int64
```

<br>
```
>>> print(X.shape, y.shape)
(1436, 37), (1436,)

>>> print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
(1005, 37) (431, 37) (1005,) (431,)
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
vif["VIF Factor"] = [variance_inflation_factor(mlr_data.values, i) for i in range(mlr_data.shape[1])]
vif["features"] = mlr_data.columns
vif
```
```
	VIF Factor	features
0	10.953474	Price
1	inf		Age_08_04
2	inf		Mfg_Month
3	inf		Mfg_Year
4	2.400334	KM
5	2.621514	HP
6	1.143778	Met_Color
7	1.121303	Automatic
8	1.258641	cc
9	1.352288	Doors
10	0.000000	Cylinders
11	1.271814	Gears
12	5.496805	Quarterly_Tax
13	4.487491	Weight
14	1.210815	Mfr_Guarantee
15	1.392485	BOVAG_Guarantee
16	1.573026	Guarantee_Period
17	2.276617	ABS
18	1.612758	Airbag_1
19	3.106933	Airbag_2
20	1.846429	Airco
21	2.009866	Automatic_airco
22	2.647036	Boardcomputer
23	1.564446	CD_Player
24	4.593157	Central_Lock
25	4.676311	Powered_Windows
26	1.582829	Power_Steering
27	62.344621	Radio
28	2.076846	Mistlamps
29	1.510131	Sport_Model
30	2.702141	Backseat_Divider
31	1.349642	Metallic_Rim
32	62.172860	Radio_cassette
33	1.153760	Tow_Bar
34	inf		Petrol
35	inf		Diesel
36	inf		CNG
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
<summary class='jb-small' style="color:blue">OUTPUT : Model results</summary>
<hr class='division3'>
<span class="jb-medium">R2 is high, a majority of variables is meaningful</span>
![1](https://user-images.githubusercontent.com/52376448/66271015-5461a500-e894-11e9-8101-4138ae77a0cb.JPG)
![2](https://user-images.githubusercontent.com/52376448/66271016-5461a500-e894-11e9-93e5-e5afa748c16b.JPG)
![3](https://user-images.githubusercontent.com/52376448/66271017-5461a500-e894-11e9-86c7-3b2b03013b85.JPG)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Model diagnosis</summary>
<hr class='division3'>
`Normal Q-Q Plot`
```python
import matplotlib.pyplot as plt

# checking residual
res = fitted_full_model.resid  # residual

# q-q plot
fig = sm.qqplot(res, fit=True, line='45')
```
![다운로드](https://user-images.githubusercontent.com/52376448/66272429-91ce2e80-e8a4-11e9-91ca-4589b85342d6.png)
<br><br><br>


`Residual vs Fitted plot`
```python
import matplotlib.pyplot as plt

pred_y=fitted_full_model.predict(train_x)
res = fitted_full_model.resid  # residual

fig = plt.scatter(pred_y,res, s=4)
plt.xlim(4000,30000)
plt.xlim(4000,30000)
plt.xlabel('Fitted values')
plt.ylabel('Residual')
```
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/66272430-91ce2e80-e8a4-11e9-9939-f1a54811d148.png)

<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Model prediction</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

pred_y = fitted_full_model.predict(test_x) ## 검증 데이터에 대한 예측 
plt.plot(np.array(test_y-pred_y),label="pred_full")
plt.legend()
plt.show()
```
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/66272471-2a64ae80-e8a5-11e9-9166-01b2c6bb326c.png)

<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Model performance</summary>
<hr class='division3'>
```python
from sklearn.metrics import mean_squared_error

pred_y = fitted_full_model.predict(test_x)
mean_squared_error(y_true= test_y, y_pred= pred_y)
```
<span class='jb-medium'>1441488.811437499</span>

<hr class='division3'>
</details>
<br>

`Modify regression model(Variables selection)`
```python
# [0]
import time
import itertools

# [1]
def processSubset(X,y, feature_set):
            model = sm.OLS(y,X[list(feature_set)]) # Modeling
            regr = model.fit() # 모델 학습
            AIC = regr.aic # 모델의 AIC
            return {"model":regr, "AIC":AIC}

# [2] getBest: 가장 낮은 AIC를 가지는 모델 선택 및 저장
def getBest(X,y,k):
    tic = time.time() # 시작시간
    results = [] # 결과 저장공간
    for combo in itertools.combinations(X.columns.difference(['const']), k): # 각 변수조합을 고려한 경우의 수
        combo=(list(combo)+['const'])
        
        results.append(processSubset(X,y,feature_set=combo))  # 모델링된 것들을 저장
    models = pd.DataFrame(results) # 데이터 프레임으로 변환
    # 가장 낮은 AIC를 가지는 모델 선택 및 저장
    best_model = models.loc[models['AIC'].argmin()] # index
    toc = time.time() # 종료시간
    print("Processed ", models.shape[0], "models on", k, "predictors in", (toc - tic),
          "seconds.")
    return best_model

print(getBest(X=train_x, y=train_y,k=2))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Processed  630 models on 2 predictors in 1.8201320171356201 seconds.
AIC                                                17516.6
model    <statsmodels.regression.linear_model.Regressio...
Name: 211, dtype: object
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT [1]</summary>
<hr class='division3'>
```python
processSubset(X=train_x, y=train_y, feature_set = feature_columns)
```
```
{'model': <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1fbccd16080>,
 'AIC': 16970.52868834004}
```
<br>
```python
processSubset(X=train_x, y=train_y, feature_set = feature_columns[0:5])
```
```
{'model': <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x000001FBCCDEB358>, 'AIC': 19176.91230693121}
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT [2]</summary>
<hr class='division3'>
```python
for combo in itertools.combinations(X.columns.difference(['const']), 2):
    print((list(combo)+['const']))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
['ABS', 'Age_08_04', 'const']
['ABS', 'Airbag_1', 'const']
['ABS', 'Airbag_2', 'const']
['ABS', 'Airco', 'const']
['ABS', 'Automatic', 'const']
['ABS', 'Automatic_airco', 'const']
['ABS', 'BOVAG_Guarantee', 'const']
['ABS', 'Backseat_Divider', 'const']
['ABS', 'Boardcomputer', 'const']
['ABS', 'CD_Player', 'const']
['ABS', 'CNG', 'const']
['ABS', 'Central_Lock', 'const']
['ABS', 'Cylinders', 'const']
['ABS', 'Diesel', 'const']
['ABS', 'Doors', 'const']
['ABS', 'Gears', 'const']
['ABS', 'Guarantee_Period', 'const']
['ABS', 'HP', 'const']
['ABS', 'KM', 'const']
['ABS', 'Met_Color', 'const']
['ABS', 'Metallic_Rim', 'const']
['ABS', 'Mfg_Month', 'const']
['ABS', 'Mfg_Year', 'const']
['ABS', 'Mfr_Guarantee', 'const']
['ABS', 'Mistlamps', 'const']
['ABS', 'Petrol', 'const']
['ABS', 'Power_Steering', 'const']
['ABS', 'Powered_Windows', 'const']
['ABS', 'Quarterly_Tax', 'const']
['ABS', 'Radio', 'const']
['ABS', 'Radio_cassette', 'const']
['ABS', 'Sport_Model', 'const']
['ABS', 'Tow_Bar', 'const']
['ABS', 'Weight', 'const']
['ABS', 'cc', 'const']
['Age_08_04', 'Airbag_1', 'const']
['Age_08_04', 'Airbag_2', 'const']
['Age_08_04', 'Airco', 'const']
['Age_08_04', 'Automatic', 'const']
['Age_08_04', 'Automatic_airco', 'const']
['Age_08_04', 'BOVAG_Guarantee', 'const']
['Age_08_04', 'Backseat_Divider', 'const']
['Age_08_04', 'Boardcomputer', 'const']
['Age_08_04', 'CD_Player', 'const']
['Age_08_04', 'CNG', 'const']
['Age_08_04', 'Central_Lock', 'const']
['Age_08_04', 'Cylinders', 'const']
['Age_08_04', 'Diesel', 'const']
['Age_08_04', 'Doors', 'const']
['Age_08_04', 'Gears', 'const']
['Age_08_04', 'Guarantee_Period', 'const']
['Age_08_04', 'HP', 'const']
['Age_08_04', 'KM', 'const']
['Age_08_04', 'Met_Color', 'const']
['Age_08_04', 'Metallic_Rim', 'const']
['Age_08_04', 'Mfg_Month', 'const']
['Age_08_04', 'Mfg_Year', 'const']
['Age_08_04', 'Mfr_Guarantee', 'const']
['Age_08_04', 'Mistlamps', 'const']
['Age_08_04', 'Petrol', 'const']
['Age_08_04', 'Power_Steering', 'const']
['Age_08_04', 'Powered_Windows', 'const']
['Age_08_04', 'Quarterly_Tax', 'const']
['Age_08_04', 'Radio', 'const']
['Age_08_04', 'Radio_cassette', 'const']
['Age_08_04', 'Sport_Model', 'const']
['Age_08_04', 'Tow_Bar', 'const']
['Age_08_04', 'Weight', 'const']
['Age_08_04', 'cc', 'const']
['Airbag_1', 'Airbag_2', 'const']
['Airbag_1', 'Airco', 'const']
['Airbag_1', 'Automatic', 'const']
['Airbag_1', 'Automatic_airco', 'const']
['Airbag_1', 'BOVAG_Guarantee', 'const']
['Airbag_1', 'Backseat_Divider', 'const']
['Airbag_1', 'Boardcomputer', 'const']
['Airbag_1', 'CD_Player', 'const']
['Airbag_1', 'CNG', 'const']
['Airbag_1', 'Central_Lock', 'const']
['Airbag_1', 'Cylinders', 'const']
['Airbag_1', 'Diesel', 'const']
['Airbag_1', 'Doors', 'const']
['Airbag_1', 'Gears', 'const']
['Airbag_1', 'Guarantee_Period', 'const']
['Airbag_1', 'HP', 'const']
['Airbag_1', 'KM', 'const']
['Airbag_1', 'Met_Color', 'const']
['Airbag_1', 'Metallic_Rim', 'const']
['Airbag_1', 'Mfg_Month', 'const']
['Airbag_1', 'Mfg_Year', 'const']
['Airbag_1', 'Mfr_Guarantee', 'const']
['Airbag_1', 'Mistlamps', 'const']
['Airbag_1', 'Petrol', 'const']
['Airbag_1', 'Power_Steering', 'const']
['Airbag_1', 'Powered_Windows', 'const']
['Airbag_1', 'Quarterly_Tax', 'const']
['Airbag_1', 'Radio', 'const']
['Airbag_1', 'Radio_cassette', 'const']
['Airbag_1', 'Sport_Model', 'const']
['Airbag_1', 'Tow_Bar', 'const']
['Airbag_1', 'Weight', 'const']
['Airbag_1', 'cc', 'const']
['Airbag_2', 'Airco', 'const']
['Airbag_2', 'Automatic', 'const']
['Airbag_2', 'Automatic_airco', 'const']
['Airbag_2', 'BOVAG_Guarantee', 'const']
['Airbag_2', 'Backseat_Divider', 'const']
['Airbag_2', 'Boardcomputer', 'const']
['Airbag_2', 'CD_Player', 'const']
['Airbag_2', 'CNG', 'const']
['Airbag_2', 'Central_Lock', 'const']
['Airbag_2', 'Cylinders', 'const']
['Airbag_2', 'Diesel', 'const']
['Airbag_2', 'Doors', 'const']
['Airbag_2', 'Gears', 'const']
['Airbag_2', 'Guarantee_Period', 'const']
['Airbag_2', 'HP', 'const']
['Airbag_2', 'KM', 'const']
['Airbag_2', 'Met_Color', 'const']
['Airbag_2', 'Metallic_Rim', 'const']
['Airbag_2', 'Mfg_Month', 'const']
['Airbag_2', 'Mfg_Year', 'const']
['Airbag_2', 'Mfr_Guarantee', 'const']
['Airbag_2', 'Mistlamps', 'const']
['Airbag_2', 'Petrol', 'const']
['Airbag_2', 'Power_Steering', 'const']
['Airbag_2', 'Powered_Windows', 'const']
['Airbag_2', 'Quarterly_Tax', 'const']
['Airbag_2', 'Radio', 'const']
['Airbag_2', 'Radio_cassette', 'const']
['Airbag_2', 'Sport_Model', 'const']
['Airbag_2', 'Tow_Bar', 'const']
['Airbag_2', 'Weight', 'const']
['Airbag_2', 'cc', 'const']
['Airco', 'Automatic', 'const']
['Airco', 'Automatic_airco', 'const']
['Airco', 'BOVAG_Guarantee', 'const']
['Airco', 'Backseat_Divider', 'const']
['Airco', 'Boardcomputer', 'const']
['Airco', 'CD_Player', 'const']
['Airco', 'CNG', 'const']
['Airco', 'Central_Lock', 'const']
['Airco', 'Cylinders', 'const']
['Airco', 'Diesel', 'const']
['Airco', 'Doors', 'const']
['Airco', 'Gears', 'const']
['Airco', 'Guarantee_Period', 'const']
['Airco', 'HP', 'const']
['Airco', 'KM', 'const']
['Airco', 'Met_Color', 'const']
['Airco', 'Metallic_Rim', 'const']
['Airco', 'Mfg_Month', 'const']
['Airco', 'Mfg_Year', 'const']
['Airco', 'Mfr_Guarantee', 'const']
['Airco', 'Mistlamps', 'const']
['Airco', 'Petrol', 'const']
['Airco', 'Power_Steering', 'const']
['Airco', 'Powered_Windows', 'const']
['Airco', 'Quarterly_Tax', 'const']
['Airco', 'Radio', 'const']
['Airco', 'Radio_cassette', 'const']
['Airco', 'Sport_Model', 'const']
['Airco', 'Tow_Bar', 'const']
['Airco', 'Weight', 'const']
['Airco', 'cc', 'const']
['Automatic', 'Automatic_airco', 'const']
['Automatic', 'BOVAG_Guarantee', 'const']
['Automatic', 'Backseat_Divider', 'const']
['Automatic', 'Boardcomputer', 'const']
['Automatic', 'CD_Player', 'const']
['Automatic', 'CNG', 'const']
['Automatic', 'Central_Lock', 'const']
['Automatic', 'Cylinders', 'const']
['Automatic', 'Diesel', 'const']
['Automatic', 'Doors', 'const']
['Automatic', 'Gears', 'const']
['Automatic', 'Guarantee_Period', 'const']
['Automatic', 'HP', 'const']
['Automatic', 'KM', 'const']
['Automatic', 'Met_Color', 'const']
['Automatic', 'Metallic_Rim', 'const']
['Automatic', 'Mfg_Month', 'const']
['Automatic', 'Mfg_Year', 'const']
['Automatic', 'Mfr_Guarantee', 'const']
['Automatic', 'Mistlamps', 'const']
['Automatic', 'Petrol', 'const']
['Automatic', 'Power_Steering', 'const']
['Automatic', 'Powered_Windows', 'const']
['Automatic', 'Quarterly_Tax', 'const']
['Automatic', 'Radio', 'const']
['Automatic', 'Radio_cassette', 'const']
['Automatic', 'Sport_Model', 'const']
['Automatic', 'Tow_Bar', 'const']
['Automatic', 'Weight', 'const']
['Automatic', 'cc', 'const']
['Automatic_airco', 'BOVAG_Guarantee', 'const']
['Automatic_airco', 'Backseat_Divider', 'const']
['Automatic_airco', 'Boardcomputer', 'const']
['Automatic_airco', 'CD_Player', 'const']
['Automatic_airco', 'CNG', 'const']
['Automatic_airco', 'Central_Lock', 'const']
['Automatic_airco', 'Cylinders', 'const']
['Automatic_airco', 'Diesel', 'const']
['Automatic_airco', 'Doors', 'const']
['Automatic_airco', 'Gears', 'const']
['Automatic_airco', 'Guarantee_Period', 'const']
['Automatic_airco', 'HP', 'const']
['Automatic_airco', 'KM', 'const']
['Automatic_airco', 'Met_Color', 'const']
['Automatic_airco', 'Metallic_Rim', 'const']
['Automatic_airco', 'Mfg_Month', 'const']
['Automatic_airco', 'Mfg_Year', 'const']
['Automatic_airco', 'Mfr_Guarantee', 'const']
['Automatic_airco', 'Mistlamps', 'const']
['Automatic_airco', 'Petrol', 'const']
['Automatic_airco', 'Power_Steering', 'const']
['Automatic_airco', 'Powered_Windows', 'const']
['Automatic_airco', 'Quarterly_Tax', 'const']
['Automatic_airco', 'Radio', 'const']
['Automatic_airco', 'Radio_cassette', 'const']
['Automatic_airco', 'Sport_Model', 'const']
['Automatic_airco', 'Tow_Bar', 'const']
['Automatic_airco', 'Weight', 'const']
['Automatic_airco', 'cc', 'const']
['BOVAG_Guarantee', 'Backseat_Divider', 'const']
['BOVAG_Guarantee', 'Boardcomputer', 'const']
['BOVAG_Guarantee', 'CD_Player', 'const']
['BOVAG_Guarantee', 'CNG', 'const']
['BOVAG_Guarantee', 'Central_Lock', 'const']
['BOVAG_Guarantee', 'Cylinders', 'const']
['BOVAG_Guarantee', 'Diesel', 'const']
['BOVAG_Guarantee', 'Doors', 'const']
['BOVAG_Guarantee', 'Gears', 'const']
['BOVAG_Guarantee', 'Guarantee_Period', 'const']
['BOVAG_Guarantee', 'HP', 'const']
['BOVAG_Guarantee', 'KM', 'const']
['BOVAG_Guarantee', 'Met_Color', 'const']
['BOVAG_Guarantee', 'Metallic_Rim', 'const']
['BOVAG_Guarantee', 'Mfg_Month', 'const']
['BOVAG_Guarantee', 'Mfg_Year', 'const']
['BOVAG_Guarantee', 'Mfr_Guarantee', 'const']
['BOVAG_Guarantee', 'Mistlamps', 'const']
['BOVAG_Guarantee', 'Petrol', 'const']
['BOVAG_Guarantee', 'Power_Steering', 'const']
['BOVAG_Guarantee', 'Powered_Windows', 'const']
['BOVAG_Guarantee', 'Quarterly_Tax', 'const']
['BOVAG_Guarantee', 'Radio', 'const']
['BOVAG_Guarantee', 'Radio_cassette', 'const']
['BOVAG_Guarantee', 'Sport_Model', 'const']
['BOVAG_Guarantee', 'Tow_Bar', 'const']
['BOVAG_Guarantee', 'Weight', 'const']
['BOVAG_Guarantee', 'cc', 'const']
['Backseat_Divider', 'Boardcomputer', 'const']
['Backseat_Divider', 'CD_Player', 'const']
['Backseat_Divider', 'CNG', 'const']
['Backseat_Divider', 'Central_Lock', 'const']
['Backseat_Divider', 'Cylinders', 'const']
['Backseat_Divider', 'Diesel', 'const']
['Backseat_Divider', 'Doors', 'const']
['Backseat_Divider', 'Gears', 'const']
['Backseat_Divider', 'Guarantee_Period', 'const']
['Backseat_Divider', 'HP', 'const']
['Backseat_Divider', 'KM', 'const']
['Backseat_Divider', 'Met_Color', 'const']
['Backseat_Divider', 'Metallic_Rim', 'const']
['Backseat_Divider', 'Mfg_Month', 'const']
['Backseat_Divider', 'Mfg_Year', 'const']
['Backseat_Divider', 'Mfr_Guarantee', 'const']
['Backseat_Divider', 'Mistlamps', 'const']
['Backseat_Divider', 'Petrol', 'const']
['Backseat_Divider', 'Power_Steering', 'const']
['Backseat_Divider', 'Powered_Windows', 'const']
['Backseat_Divider', 'Quarterly_Tax', 'const']
['Backseat_Divider', 'Radio', 'const']
['Backseat_Divider', 'Radio_cassette', 'const']
['Backseat_Divider', 'Sport_Model', 'const']
['Backseat_Divider', 'Tow_Bar', 'const']
['Backseat_Divider', 'Weight', 'const']
['Backseat_Divider', 'cc', 'const']
['Boardcomputer', 'CD_Player', 'const']
['Boardcomputer', 'CNG', 'const']
['Boardcomputer', 'Central_Lock', 'const']
['Boardcomputer', 'Cylinders', 'const']
['Boardcomputer', 'Diesel', 'const']
['Boardcomputer', 'Doors', 'const']
['Boardcomputer', 'Gears', 'const']
['Boardcomputer', 'Guarantee_Period', 'const']
['Boardcomputer', 'HP', 'const']
['Boardcomputer', 'KM', 'const']
['Boardcomputer', 'Met_Color', 'const']
['Boardcomputer', 'Metallic_Rim', 'const']
['Boardcomputer', 'Mfg_Month', 'const']
['Boardcomputer', 'Mfg_Year', 'const']
['Boardcomputer', 'Mfr_Guarantee', 'const']
['Boardcomputer', 'Mistlamps', 'const']
['Boardcomputer', 'Petrol', 'const']
['Boardcomputer', 'Power_Steering', 'const']
['Boardcomputer', 'Powered_Windows', 'const']
['Boardcomputer', 'Quarterly_Tax', 'const']
['Boardcomputer', 'Radio', 'const']
['Boardcomputer', 'Radio_cassette', 'const']
['Boardcomputer', 'Sport_Model', 'const']
['Boardcomputer', 'Tow_Bar', 'const']
['Boardcomputer', 'Weight', 'const']
['Boardcomputer', 'cc', 'const']
['CD_Player', 'CNG', 'const']
['CD_Player', 'Central_Lock', 'const']
['CD_Player', 'Cylinders', 'const']
['CD_Player', 'Diesel', 'const']
['CD_Player', 'Doors', 'const']
['CD_Player', 'Gears', 'const']
['CD_Player', 'Guarantee_Period', 'const']
['CD_Player', 'HP', 'const']
['CD_Player', 'KM', 'const']
['CD_Player', 'Met_Color', 'const']
['CD_Player', 'Metallic_Rim', 'const']
['CD_Player', 'Mfg_Month', 'const']
['CD_Player', 'Mfg_Year', 'const']
['CD_Player', 'Mfr_Guarantee', 'const']
['CD_Player', 'Mistlamps', 'const']
['CD_Player', 'Petrol', 'const']
['CD_Player', 'Power_Steering', 'const']
['CD_Player', 'Powered_Windows', 'const']
['CD_Player', 'Quarterly_Tax', 'const']
['CD_Player', 'Radio', 'const']
['CD_Player', 'Radio_cassette', 'const']
['CD_Player', 'Sport_Model', 'const']
['CD_Player', 'Tow_Bar', 'const']
['CD_Player', 'Weight', 'const']
['CD_Player', 'cc', 'const']
['CNG', 'Central_Lock', 'const']
['CNG', 'Cylinders', 'const']
['CNG', 'Diesel', 'const']
['CNG', 'Doors', 'const']
['CNG', 'Gears', 'const']
['CNG', 'Guarantee_Period', 'const']
['CNG', 'HP', 'const']
['CNG', 'KM', 'const']
['CNG', 'Met_Color', 'const']
['CNG', 'Metallic_Rim', 'const']
['CNG', 'Mfg_Month', 'const']
['CNG', 'Mfg_Year', 'const']
['CNG', 'Mfr_Guarantee', 'const']
['CNG', 'Mistlamps', 'const']
['CNG', 'Petrol', 'const']
['CNG', 'Power_Steering', 'const']
['CNG', 'Powered_Windows', 'const']
['CNG', 'Quarterly_Tax', 'const']
['CNG', 'Radio', 'const']
['CNG', 'Radio_cassette', 'const']
['CNG', 'Sport_Model', 'const']
['CNG', 'Tow_Bar', 'const']
['CNG', 'Weight', 'const']
['CNG', 'cc', 'const']
['Central_Lock', 'Cylinders', 'const']
['Central_Lock', 'Diesel', 'const']
['Central_Lock', 'Doors', 'const']
['Central_Lock', 'Gears', 'const']
['Central_Lock', 'Guarantee_Period', 'const']
['Central_Lock', 'HP', 'const']
['Central_Lock', 'KM', 'const']
['Central_Lock', 'Met_Color', 'const']
['Central_Lock', 'Metallic_Rim', 'const']
['Central_Lock', 'Mfg_Month', 'const']
['Central_Lock', 'Mfg_Year', 'const']
['Central_Lock', 'Mfr_Guarantee', 'const']
['Central_Lock', 'Mistlamps', 'const']
['Central_Lock', 'Petrol', 'const']
['Central_Lock', 'Power_Steering', 'const']
['Central_Lock', 'Powered_Windows', 'const']
['Central_Lock', 'Quarterly_Tax', 'const']
['Central_Lock', 'Radio', 'const']
['Central_Lock', 'Radio_cassette', 'const']
['Central_Lock', 'Sport_Model', 'const']
['Central_Lock', 'Tow_Bar', 'const']
['Central_Lock', 'Weight', 'const']
['Central_Lock', 'cc', 'const']
['Cylinders', 'Diesel', 'const']
['Cylinders', 'Doors', 'const']
['Cylinders', 'Gears', 'const']
['Cylinders', 'Guarantee_Period', 'const']
['Cylinders', 'HP', 'const']
['Cylinders', 'KM', 'const']
['Cylinders', 'Met_Color', 'const']
['Cylinders', 'Metallic_Rim', 'const']
['Cylinders', 'Mfg_Month', 'const']
['Cylinders', 'Mfg_Year', 'const']
['Cylinders', 'Mfr_Guarantee', 'const']
['Cylinders', 'Mistlamps', 'const']
['Cylinders', 'Petrol', 'const']
['Cylinders', 'Power_Steering', 'const']
['Cylinders', 'Powered_Windows', 'const']
['Cylinders', 'Quarterly_Tax', 'const']
['Cylinders', 'Radio', 'const']
['Cylinders', 'Radio_cassette', 'const']
['Cylinders', 'Sport_Model', 'const']
['Cylinders', 'Tow_Bar', 'const']
['Cylinders', 'Weight', 'const']
['Cylinders', 'cc', 'const']
['Diesel', 'Doors', 'const']
['Diesel', 'Gears', 'const']
['Diesel', 'Guarantee_Period', 'const']
['Diesel', 'HP', 'const']
['Diesel', 'KM', 'const']
['Diesel', 'Met_Color', 'const']
['Diesel', 'Metallic_Rim', 'const']
['Diesel', 'Mfg_Month', 'const']
['Diesel', 'Mfg_Year', 'const']
['Diesel', 'Mfr_Guarantee', 'const']
['Diesel', 'Mistlamps', 'const']
['Diesel', 'Petrol', 'const']
['Diesel', 'Power_Steering', 'const']
['Diesel', 'Powered_Windows', 'const']
['Diesel', 'Quarterly_Tax', 'const']
['Diesel', 'Radio', 'const']
['Diesel', 'Radio_cassette', 'const']
['Diesel', 'Sport_Model', 'const']
['Diesel', 'Tow_Bar', 'const']
['Diesel', 'Weight', 'const']
['Diesel', 'cc', 'const']
['Doors', 'Gears', 'const']
['Doors', 'Guarantee_Period', 'const']
['Doors', 'HP', 'const']
['Doors', 'KM', 'const']
['Doors', 'Met_Color', 'const']
['Doors', 'Metallic_Rim', 'const']
['Doors', 'Mfg_Month', 'const']
['Doors', 'Mfg_Year', 'const']
['Doors', 'Mfr_Guarantee', 'const']
['Doors', 'Mistlamps', 'const']
['Doors', 'Petrol', 'const']
['Doors', 'Power_Steering', 'const']
['Doors', 'Powered_Windows', 'const']
['Doors', 'Quarterly_Tax', 'const']
['Doors', 'Radio', 'const']
['Doors', 'Radio_cassette', 'const']
['Doors', 'Sport_Model', 'const']
['Doors', 'Tow_Bar', 'const']
['Doors', 'Weight', 'const']
['Doors', 'cc', 'const']
['Gears', 'Guarantee_Period', 'const']
['Gears', 'HP', 'const']
['Gears', 'KM', 'const']
['Gears', 'Met_Color', 'const']
['Gears', 'Metallic_Rim', 'const']
['Gears', 'Mfg_Month', 'const']
['Gears', 'Mfg_Year', 'const']
['Gears', 'Mfr_Guarantee', 'const']
['Gears', 'Mistlamps', 'const']
['Gears', 'Petrol', 'const']
['Gears', 'Power_Steering', 'const']
['Gears', 'Powered_Windows', 'const']
['Gears', 'Quarterly_Tax', 'const']
['Gears', 'Radio', 'const']
['Gears', 'Radio_cassette', 'const']
['Gears', 'Sport_Model', 'const']
['Gears', 'Tow_Bar', 'const']
['Gears', 'Weight', 'const']
['Gears', 'cc', 'const']
['Guarantee_Period', 'HP', 'const']
['Guarantee_Period', 'KM', 'const']
['Guarantee_Period', 'Met_Color', 'const']
['Guarantee_Period', 'Metallic_Rim', 'const']
['Guarantee_Period', 'Mfg_Month', 'const']
['Guarantee_Period', 'Mfg_Year', 'const']
['Guarantee_Period', 'Mfr_Guarantee', 'const']
['Guarantee_Period', 'Mistlamps', 'const']
['Guarantee_Period', 'Petrol', 'const']
['Guarantee_Period', 'Power_Steering', 'const']
['Guarantee_Period', 'Powered_Windows', 'const']
['Guarantee_Period', 'Quarterly_Tax', 'const']
['Guarantee_Period', 'Radio', 'const']
['Guarantee_Period', 'Radio_cassette', 'const']
['Guarantee_Period', 'Sport_Model', 'const']
['Guarantee_Period', 'Tow_Bar', 'const']
['Guarantee_Period', 'Weight', 'const']
['Guarantee_Period', 'cc', 'const']
['HP', 'KM', 'const']
['HP', 'Met_Color', 'const']
['HP', 'Metallic_Rim', 'const']
['HP', 'Mfg_Month', 'const']
['HP', 'Mfg_Year', 'const']
['HP', 'Mfr_Guarantee', 'const']
['HP', 'Mistlamps', 'const']
['HP', 'Petrol', 'const']
['HP', 'Power_Steering', 'const']
['HP', 'Powered_Windows', 'const']
['HP', 'Quarterly_Tax', 'const']
['HP', 'Radio', 'const']
['HP', 'Radio_cassette', 'const']
['HP', 'Sport_Model', 'const']
['HP', 'Tow_Bar', 'const']
['HP', 'Weight', 'const']
['HP', 'cc', 'const']
['KM', 'Met_Color', 'const']
['KM', 'Metallic_Rim', 'const']
['KM', 'Mfg_Month', 'const']
['KM', 'Mfg_Year', 'const']
['KM', 'Mfr_Guarantee', 'const']
['KM', 'Mistlamps', 'const']
['KM', 'Petrol', 'const']
['KM', 'Power_Steering', 'const']
['KM', 'Powered_Windows', 'const']
['KM', 'Quarterly_Tax', 'const']
['KM', 'Radio', 'const']
['KM', 'Radio_cassette', 'const']
['KM', 'Sport_Model', 'const']
['KM', 'Tow_Bar', 'const']
['KM', 'Weight', 'const']
['KM', 'cc', 'const']
['Met_Color', 'Metallic_Rim', 'const']
['Met_Color', 'Mfg_Month', 'const']
['Met_Color', 'Mfg_Year', 'const']
['Met_Color', 'Mfr_Guarantee', 'const']
['Met_Color', 'Mistlamps', 'const']
['Met_Color', 'Petrol', 'const']
['Met_Color', 'Power_Steering', 'const']
['Met_Color', 'Powered_Windows', 'const']
['Met_Color', 'Quarterly_Tax', 'const']
['Met_Color', 'Radio', 'const']
['Met_Color', 'Radio_cassette', 'const']
['Met_Color', 'Sport_Model', 'const']
['Met_Color', 'Tow_Bar', 'const']
['Met_Color', 'Weight', 'const']
['Met_Color', 'cc', 'const']
['Metallic_Rim', 'Mfg_Month', 'const']
['Metallic_Rim', 'Mfg_Year', 'const']
['Metallic_Rim', 'Mfr_Guarantee', 'const']
['Metallic_Rim', 'Mistlamps', 'const']
['Metallic_Rim', 'Petrol', 'const']
['Metallic_Rim', 'Power_Steering', 'const']
['Metallic_Rim', 'Powered_Windows', 'const']
['Metallic_Rim', 'Quarterly_Tax', 'const']
['Metallic_Rim', 'Radio', 'const']
['Metallic_Rim', 'Radio_cassette', 'const']
['Metallic_Rim', 'Sport_Model', 'const']
['Metallic_Rim', 'Tow_Bar', 'const']
['Metallic_Rim', 'Weight', 'const']
['Metallic_Rim', 'cc', 'const']
['Mfg_Month', 'Mfg_Year', 'const']
['Mfg_Month', 'Mfr_Guarantee', 'const']
['Mfg_Month', 'Mistlamps', 'const']
['Mfg_Month', 'Petrol', 'const']
['Mfg_Month', 'Power_Steering', 'const']
['Mfg_Month', 'Powered_Windows', 'const']
['Mfg_Month', 'Quarterly_Tax', 'const']
['Mfg_Month', 'Radio', 'const']
['Mfg_Month', 'Radio_cassette', 'const']
['Mfg_Month', 'Sport_Model', 'const']
['Mfg_Month', 'Tow_Bar', 'const']
['Mfg_Month', 'Weight', 'const']
['Mfg_Month', 'cc', 'const']
['Mfg_Year', 'Mfr_Guarantee', 'const']
['Mfg_Year', 'Mistlamps', 'const']
['Mfg_Year', 'Petrol', 'const']
['Mfg_Year', 'Power_Steering', 'const']
['Mfg_Year', 'Powered_Windows', 'const']
['Mfg_Year', 'Quarterly_Tax', 'const']
['Mfg_Year', 'Radio', 'const']
['Mfg_Year', 'Radio_cassette', 'const']
['Mfg_Year', 'Sport_Model', 'const']
['Mfg_Year', 'Tow_Bar', 'const']
['Mfg_Year', 'Weight', 'const']
['Mfg_Year', 'cc', 'const']
['Mfr_Guarantee', 'Mistlamps', 'const']
['Mfr_Guarantee', 'Petrol', 'const']
['Mfr_Guarantee', 'Power_Steering', 'const']
['Mfr_Guarantee', 'Powered_Windows', 'const']
['Mfr_Guarantee', 'Quarterly_Tax', 'const']
['Mfr_Guarantee', 'Radio', 'const']
['Mfr_Guarantee', 'Radio_cassette', 'const']
['Mfr_Guarantee', 'Sport_Model', 'const']
['Mfr_Guarantee', 'Tow_Bar', 'const']
['Mfr_Guarantee', 'Weight', 'const']
['Mfr_Guarantee', 'cc', 'const']
['Mistlamps', 'Petrol', 'const']
['Mistlamps', 'Power_Steering', 'const']
['Mistlamps', 'Powered_Windows', 'const']
['Mistlamps', 'Quarterly_Tax', 'const']
['Mistlamps', 'Radio', 'const']
['Mistlamps', 'Radio_cassette', 'const']
['Mistlamps', 'Sport_Model', 'const']
['Mistlamps', 'Tow_Bar', 'const']
['Mistlamps', 'Weight', 'const']
['Mistlamps', 'cc', 'const']
['Petrol', 'Power_Steering', 'const']
['Petrol', 'Powered_Windows', 'const']
['Petrol', 'Quarterly_Tax', 'const']
['Petrol', 'Radio', 'const']
['Petrol', 'Radio_cassette', 'const']
['Petrol', 'Sport_Model', 'const']
['Petrol', 'Tow_Bar', 'const']
['Petrol', 'Weight', 'const']
['Petrol', 'cc', 'const']
['Power_Steering', 'Powered_Windows', 'const']
['Power_Steering', 'Quarterly_Tax', 'const']
['Power_Steering', 'Radio', 'const']
['Power_Steering', 'Radio_cassette', 'const']
['Power_Steering', 'Sport_Model', 'const']
['Power_Steering', 'Tow_Bar', 'const']
['Power_Steering', 'Weight', 'const']
['Power_Steering', 'cc', 'const']
['Powered_Windows', 'Quarterly_Tax', 'const']
['Powered_Windows', 'Radio', 'const']
['Powered_Windows', 'Radio_cassette', 'const']
['Powered_Windows', 'Sport_Model', 'const']
['Powered_Windows', 'Tow_Bar', 'const']
['Powered_Windows', 'Weight', 'const']
['Powered_Windows', 'cc', 'const']
['Quarterly_Tax', 'Radio', 'const']
['Quarterly_Tax', 'Radio_cassette', 'const']
['Quarterly_Tax', 'Sport_Model', 'const']
['Quarterly_Tax', 'Tow_Bar', 'const']
['Quarterly_Tax', 'Weight', 'const']
['Quarterly_Tax', 'cc', 'const']
['Radio', 'Radio_cassette', 'const']
['Radio', 'Sport_Model', 'const']
['Radio', 'Tow_Bar', 'const']
['Radio', 'Weight', 'const']
['Radio', 'cc', 'const']
['Radio_cassette', 'Sport_Model', 'const']
['Radio_cassette', 'Tow_Bar', 'const']
['Radio_cassette', 'Weight', 'const']
['Radio_cassette', 'cc', 'const']
['Sport_Model', 'Tow_Bar', 'const']
['Sport_Model', 'Weight', 'const']
['Sport_Model', 'cc', 'const']
['Tow_Bar', 'Weight', 'const']
['Tow_Bar', 'cc', 'const']
['Weight', 'cc', 'const']
```
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Measure training time</summary>
<hr class='division3'>
```python
# 변수 선택에 따른 학습시간과 저장
models = pd.DataFrame(columns=["AIC", "model"])
tic = time.time()
for i in range(1,4):
    models.loc[i] = getBest(X=train_x,y=train_y,k=i)
toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
```

```
Processed  36 models on 1 predictors in 0.09873557090759277 seconds.
Processed  630 models on 2 predictors in 1.3473966121673584 seconds.
Processed  7140 models on 3 predictors in 17.01948356628418 seconds.
Total elapsed time: 18.805707454681396 seconds.
```

<br>

```python
models
```
```
	AIC		model
1	17824.309811	<statsmodels.regression.linear_model.Regressio...
2	17579.120147	<statsmodels.regression.linear_model.Regressio...
3	17351.640619	<statsmodels.regression.linear_model.Regressio...
```

<br>

```python
models.loc[3, "model"].summary()
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![캡처](https://user-images.githubusercontent.com/52376448/66407947-07630780-ea29-11e9-9169-4e8736394d67.JPG)
<hr class='division3_1'>
</details>

<br>

```python
# 모든 변수들 모델링 한것과 비교 
print("full model Rsquared: ","{:.5f}".format(fitted_full_model.rsquared))
print("full model AIC: ","{:.5f}".format(fitted_full_model.aic))
print("full model MSE: ","{:.5f}".format(fitted_full_model.mse_total))
print("selected model Rsquared: ","{:.5f}".format(models.loc[3, "model"].rsquared))
print("selected model AIC: ","{:.5f}".format(models.loc[3, "model"].aic))
print("selected model MSE: ","{:.5f}".format(models.loc[3, "model"].mse_total))
```
```
full model Rsquared:  0.91141
full model AIC:  16960.68542
full model MSE:  13196639.65991
selected model Rsquared:  0.86124
selected model AIC:  17351.64062
selected model MSE:  13196639.65991
```

<br>

```python
# Plot the result
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

## Mallow Cp
plt.subplot(2, 2, 1)
Cp= models.apply(lambda row: (row[1].params.shape[0]+(row[1].mse_total-
                               fitted_full_model.mse_total)*(train_x.shape[0]-
                                row[1].params.shape[0])/fitted_full_model.mse_total
                               ), axis=1)
plt.plot(Cp)
plt.plot(Cp.argmin(), Cp.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('Cp')

# adj-rsquared plot
# adj-rsquared = Explained variation / Total variation
adj_rsquared = models.apply(lambda row: row[1].rsquared_adj, axis=1)
plt.subplot(2, 2, 2)
plt.plot(adj_rsquared)
plt.plot(adj_rsquared.argmax(), adj_rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# aic
aic = models.apply(lambda row: row[1].aic, axis=1)
plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

# bic
bic = models.apply(lambda row: row[1].bic, axis=1)
plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel(' # Predictors')
plt.ylabel('BIC')
```

<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![다운로드](https://user-images.githubusercontent.com/52376448/66408285-aa1b8600-ea29-11e9-9cbf-dbcc71d6d81a.png)
<hr class='division3_1'>
</details>

<hr class='division3'>
</details>

<br>

`Modify regression model(Feedforward selection)`
```python
########전진선택법(step=1)

def forward(X, y, predictors):
    # 데이터 변수들이 미리정의된 predictors에 있는지 없는지 확인 및 분류
    remaining_predictors = [p for p in X.columns.difference(['const']) if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X=X, y= y, feature_set=predictors+[p]+['const']))
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)

    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()] # index
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model[0] )
    return best_model


#### 전진선택법 모델

def forward_model(X,y):
    Fmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X,y=y,predictors=predictors)
        if i > 1:
            if Forward_result['AIC'] > Fmodel_before:
                break
        Fmodels.loc[i] = Forward_result
        predictors = Fmodels.loc[i]["model"].model.exog_names
        Fmodel_before = Fmodels.loc[i]["AIC"]
        predictors = [ k for k in predictors if k != 'const']
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return(Fmodels['model'][len(Fmodels['model'])])
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
Forward_best_model = forward_model(X=train_x, y= train_y)
```
```
Processed  36 models on 1 predictors in 0.08973240852355957
Selected predictors: ['Mfg_Year', 'const']  AIC: 17755.072760646137
Processed  35 models on 2 predictors in 0.09027957916259766
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'const']  AIC: 17504.57948159159
Processed  34 models on 3 predictors in 0.06283736228942871
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'const']  AIC: 17398.182235131313
Processed  33 models on 4 predictors in 0.06283116340637207
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'const']  AIC: 17150.1641103143
Processed  32 models on 5 predictors in 0.07981634140014648
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'const']  AIC: 17091.096715621316
Processed  31 models on 6 predictors in 0.0840911865234375
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'const']  AIC: 17055.57896394218
Processed  30 models on 7 predictors in 0.0738370418548584
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'const']  AIC: 17033.36951099978
Processed  29 models on 8 predictors in 0.06878113746643066
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'const']  AIC: 17019.85679678918
Processed  28 models on 9 predictors in 0.09375500679016113
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'const']  AIC: 16995.322287055787
Processed  27 models on 10 predictors in 0.10174226760864258
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'const']  AIC: 16983.818299485778
Processed  26 models on 11 predictors in 0.10377311706542969
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'const']  AIC: 16964.290655626864
Processed  25 models on 12 predictors in 0.11771559715270996
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'const']  AIC: 16928.537083027266
Processed  24 models on 13 predictors in 0.12260055541992188
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'const']  AIC: 16921.374043681804
Processed  23 models on 14 predictors in 0.12865686416625977
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'const']  AIC: 16918.48093923768
Processed  22 models on 15 predictors in 0.16057229042053223
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'const']  AIC: 16916.04018485048
Processed  21 models on 16 predictors in 0.18660974502563477
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'const']  AIC: 16912.806529494097
Processed  20 models on 17 predictors in 0.11269783973693848
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'const']  AIC: 16909.805620763276
Processed  19 models on 18 predictors in 0.10549688339233398
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'const']  AIC: 16907.82736115733
Processed  18 models on 19 predictors in 0.10871052742004395
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'const']  AIC: 16907.14151076706
Processed  17 models on 20 predictors in 0.11475992202758789
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'const']  AIC: 16906.91814803349
Processed  16 models on 21 predictors in 0.1306447982788086
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'Gears', 'const']  AIC: 16906.641600994546
Processed  15 models on 22 predictors in 0.11366558074951172
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994557
Total elapsed time: 2.4412221908569336 seconds.
```
<br>

```python
Forward_best_model.aic
```
```
16906.641600994546
```
<hr class='division3'>
</details>




<br>

`Modify regression model(Backward selection)`
```python
######## 후진선택법(step=1)
def backward(X,y,predictors):
    tic = time.time()
    results = []
    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(X=X, y= y,feature_set=list(combo)+['const']))
    models = pd.DataFrame(results)
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in",
          (toc - tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model[0] )
    return best_model
    

# 후진 소거법 모델
def backward_model(X, y):
    Bmodels = pd.DataFrame(columns=["AIC", "model"], index = range(1,len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X,y,predictors)['AIC']
    while (len(predictors) > 1):
        Backward_result = backward(X=train_x, y= train_y, predictors = predictors)
        if Backward_result['AIC'] > Bmodel_before:
            break
        Bmodels.loc[len(predictors) - 1] = Backward_result
        predictors = Bmodels.loc[len(predictors) - 1]["model"].model.exog_names
        Bmodel_before = Backward_result['AIC']
        predictors = [ k for k in predictors if k != 'const']

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Bmodels['model'].dropna().iloc[0])
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
Backward_best_model = backward_model(X=train_x,y=train_y)
```
```
Processed  36 models on 35 predictors in 0.5307836532592773
Selected predictors: ['ABS', 'Age_08_04', 'Airbag_1', 'Airbag_2', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Doors', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Met_Color', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16919.554953086037
Processed  35 models on 34 predictors in 0.5086104869842529
Selected predictors: ['ABS', 'Age_08_04', 'Airbag_1', 'Airbag_2', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Doors', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16917.56065836032
Processed  34 models on 33 predictors in 0.47121691703796387
Selected predictors: ['ABS', 'Age_08_04', 'Airbag_2', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Doors', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16915.573733838028
Processed  33 models on 32 predictors in 0.3795206546783447
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Doors', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16913.747808225216
Processed  32 models on 31 predictors in 0.33935022354125977
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16912.053646583932
Processed  31 models on 30 predictors in 0.29421567916870117
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'cc', 'const']  AIC: 16910.726801088837
Processed  30 models on 29 predictors in 0.29419445991516113
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Power_Steering', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16909.60778490872
Processed  29 models on 28 predictors in 0.25033020973205566
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Metallic_Rim', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16908.55343667602
Processed  28 models on 27 predictors in 0.2254021167755127
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16907.502655808014
Processed  27 models on 26 predictors in 0.20220327377319336
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Central_Lock', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.70136854976
Processed  26 models on 25 predictors in 0.20789861679077148
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CD_Player', 'CNG', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.676844492846
Processed  25 models on 24 predictors in 0.18823885917663574
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CNG', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Petrol', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.641600994557
Processed  24 models on 23 predictors in 0.1715404987335205
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CNG', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfg_Year', 'Mfr_Guarantee', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.641600994557
Processed  23 models on 22 predictors in 0.15358972549438477
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CNG', 'Cylinders', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfr_Guarantee', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.641600994557
Processed  22 models on 21 predictors in 0.1326441764831543
Selected predictors: ['ABS', 'Age_08_04', 'Airco', 'Automatic_airco', 'BOVAG_Guarantee', 'Backseat_Divider', 'Boardcomputer', 'CNG', 'Diesel', 'Gears', 'Guarantee_Period', 'HP', 'KM', 'Mfg_Month', 'Mfr_Guarantee', 'Powered_Windows', 'Quarterly_Tax', 'Radio_cassette', 'Sport_Model', 'Tow_Bar', 'Weight', 'const']  AIC: 16906.64160099456
Total elapsed time: 4.432608604431152 seconds.
```
<br>

```python
Backward_best_model.aic
```
```
16906.641600994557
```
<hr class='division3'>
</details>


<br>
`Modify regression model(Stepwise)`
```python
def Stepwise_model(X,y):
    Stepmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []
    Smodel_before = processSubset(X,y,predictors+['const'])['AIC']
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors) # constant added
        print('forward')
        Stepmodels.loc[i] = Forward_result
        predictors = Stepmodels.loc[i]["model"].model.exog_names
        predictors = [ k for k in predictors if k != 'const']
        Backward_result = backward(X=X, y=y, predictors=predictors)
        if Backward_result['AIC']< Forward_result['AIC']:
            Stepmodels.loc[i] = Backward_result
            predictors = Stepmodels.loc[i]["model"].model.exog_names
            Smodel_before = Stepmodels.loc[i]["AIC"]
            predictors = [ k for k in predictors if k != 'const']
            print('backward')
        if Stepmodels.loc[i]['AIC']> Smodel_before:
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Stepmodels['model'][len(Stepmodels['model'])])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
Stepwise_best_model=Stepwise_model(X=train_x,y=train_y)
```
```
Processed  36 models on 1 predictors in 0.09873390197753906
Selected predictors: ['Mfg_Year', 'const']  AIC: 17755.072760646137
forward
Processed  1 models on 0 predictors in 0.009046554565429688
Selected predictors: ['const']  AIC: 19355.08856819785
Processed  35 models on 2 predictors in 0.130143404006958
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'const']  AIC: 17504.57948159159
forward
Processed  2 models on 1 predictors in 0.015958309173583984
Selected predictors: ['Mfg_Year', 'const']  AIC: 17755.072760646137
Processed  34 models on 3 predictors in 0.1465761661529541
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'const']  AIC: 17398.182235131313
forward
Processed  3 models on 2 predictors in 0.016946792602539062
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'const']  AIC: 17504.57948159159
Processed  33 models on 4 predictors in 0.1317136287689209
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'const']  AIC: 17150.1641103143
forward
Processed  4 models on 3 predictors in 0.015963077545166016
Selected predictors: ['Mfg_Year', 'Weight', 'KM', 'const']  AIC: 17306.79774531549
Processed  32 models on 5 predictors in 0.08627820014953613
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'const']  AIC: 17091.096715621316
forward
Processed  5 models on 4 predictors in 0.011969327926635742
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'const']  AIC: 17150.1641103143
Processed  31 models on 6 predictors in 0.07229804992675781
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'const']  AIC: 17055.57896394218
forward
Processed  6 models on 5 predictors in 0.016991615295410156
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'const']  AIC: 17091.096715621316
Processed  30 models on 7 predictors in 0.05830645561218262
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'const']  AIC: 17033.36951099978
forward
Processed  7 models on 6 predictors in 0.01599907875061035
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'const']  AIC: 17055.57896394218
Processed  29 models on 8 predictors in 0.06846237182617188
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'const']  AIC: 17019.85679678918
forward
Processed  8 models on 7 predictors in 0.017005205154418945
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'const']  AIC: 17033.36951099978
Processed  28 models on 9 predictors in 0.11175131797790527
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'const']  AIC: 16995.322287055787
forward
Processed  9 models on 8 predictors in 0.01898479461669922
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Guarantee_Period', 'BOVAG_Guarantee', 'const']  AIC: 17012.519514899912
Processed  27 models on 10 predictors in 0.1047210693359375
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'const']  AIC: 16983.818299485778
forward
Processed  10 models on 9 predictors in 0.03191518783569336
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'const']  AIC: 16995.322287055787
Processed  26 models on 11 predictors in 0.10965585708618164
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'const']  AIC: 16964.290655626864
forward
Processed  11 models on 10 predictors in 0.04288458824157715
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'CNG', 'Quarterly_Tax', 'const']  AIC: 16978.68338783714
Processed  25 models on 12 predictors in 0.15957117080688477
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'const']  AIC: 16928.537083027266
forward
Processed  12 models on 11 predictors in 0.08481073379516602
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'Quarterly_Tax', 'Petrol', 'const']  AIC: 16932.104261902947
Processed  24 models on 13 predictors in 0.17156600952148438
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'const']  AIC: 16921.374043681804
forward
Processed  13 models on 12 predictors in 0.09979891777038574
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'const']  AIC: 16924.75355369365
Processed  23 models on 14 predictors in 0.17253684997558594
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'const']  AIC: 16918.48093923768
forward
Processed  14 models on 13 predictors in 0.08875823020935059
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'const']  AIC: 16921.374043681804
Processed  22 models on 15 predictors in 0.15457653999328613
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'const']  AIC: 16916.04018485048
forward
Processed  15 models on 14 predictors in 0.10401105880737305
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'const']  AIC: 16918.48093923768
Processed  21 models on 16 predictors in 0.15857505798339844
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'const']  AIC: 16912.806529494097
forward
Processed  16 models on 15 predictors in 0.11768555641174316
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'const']  AIC: 16916.04018485048
Processed  20 models on 17 predictors in 0.13663506507873535
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'const']  AIC: 16909.805620763276
forward
Processed  17 models on 16 predictors in 0.08477330207824707
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Airco', 'ABS', 'Sport_Model', 'const']  AIC: 16912.187005800086
Processed  19 models on 18 predictors in 0.10272526741027832
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'const']  AIC: 16907.82736115733
forward
Processed  18 models on 17 predictors in 0.1127007007598877
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'const']  AIC: 16908.531987499395
Processed  18 models on 19 predictors in 0.11521244049072266
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'const']  AIC: 16907.14151076706
forward
Processed  19 models on 18 predictors in 0.15088891983032227
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'const']  AIC: 16907.82736115733
Processed  17 models on 20 predictors in 0.16663289070129395
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'const']  AIC: 16906.91814803349
forward
Processed  20 models on 19 predictors in 0.2127993106842041
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'const']  AIC: 16907.14151076706
Processed  16 models on 21 predictors in 0.10770010948181152
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'Gears', 'const']  AIC: 16906.641600994546
forward
Processed  21 models on 20 predictors in 0.1256864070892334
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'const']  AIC: 16906.91814803349
Processed  15 models on 22 predictors in 0.10097765922546387
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Mfg_Month', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994557
forward
Processed  22 models on 21 predictors in 0.17354369163513184
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.15059447288513184
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.19049072265625
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.1495981216430664
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.13814973831176758
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.11270356178283691
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.15808415412902832
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.09469938278198242
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.14464545249938965
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.1326456069946289
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.1595752239227295
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.11668825149536133
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.13965892791748047
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.17457914352416992
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.19448089599609375
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.10567355155944824
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.15602421760559082
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.09773826599121094
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.1266651153564453
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.0937490463256836
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.12469983100891113
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.11266231536865234
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.14760518074035645
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.10172867774963379
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.19098138809204102
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.14887738227844238
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.15437889099121094
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Processed  15 models on 22 predictors in 0.10134077072143555
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'Diesel', 'const']  AIC: 16906.64160099451
forward
Processed  22 models on 21 predictors in 0.14920735359191895
Selected predictors: ['Mfg_Year', 'Automatic_airco', 'Weight', 'KM', 'Powered_Windows', 'HP', 'Mfr_Guarantee', 'Guarantee_Period', 'BOVAG_Guarantee', 'CNG', 'Quarterly_Tax', 'Petrol', 'Tow_Bar', 'Boardcomputer', 'Airco', 'ABS', 'Sport_Model', 'Backseat_Divider', 'Radio_cassette', 'Gears', 'Age_08_04', 'const']  AIC: 16906.641600994506
backward
Total elapsed time: 8.44080114364624 seconds.
```
<br>

```python
Stepwise_best_model.aic
```
```
16906.641600994506
```
<hr class='division3'>
</details>
<br>

`Model performance`
```python
# 모델에 의해 예측된/추정된 값 <->  test_y
pred_y_full = fitted_full_model.predict(test_x)
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names])
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])

perf_mat = pd.DataFrame(columns=["ALL", "FORWARD", "BACKWARD", "STEPWISE"],
                        index =['MSE', 'RMSE','MAE', 'MAPE'])
			
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn import metrics

# 성능지표
perf_mat.loc['MSE']['ALL'] = metrics.mean_squared_error(test_y,pred_y_full)
perf_mat.loc['MSE']['FORWARD'] = metrics.mean_squared_error(test_y,pred_y_forward)
perf_mat.loc['MSE']['BACKWARD'] = metrics.mean_squared_error(test_y,pred_y_backward)
perf_mat.loc['MSE']['STEPWISE'] = metrics.mean_squared_error(test_y,pred_y_stepwise)

perf_mat.loc['RMSE']['ALL'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_full))
perf_mat.loc['RMSE']['FORWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_forward))
perf_mat.loc['RMSE']['BACKWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_backward))
perf_mat.loc['RMSE']['STEPWISE'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_stepwise))

perf_mat.loc['MAE']['ALL'] = metrics.mean_absolute_error(test_y, pred_y_full)
perf_mat.loc['MAE']['FORWARD'] = metrics.mean_absolute_error(test_y, pred_y_forward)
perf_mat.loc['MAE']['BACKWARD'] = metrics.mean_absolute_error(test_y, pred_y_backward)
perf_mat.loc['MAE']['STEPWISE'] = metrics.mean_absolute_error(test_y, pred_y_stepwise)

perf_mat.loc['MAPE']['ALL'] = mean_absolute_percentage_error(test_y, pred_y_full)
perf_mat.loc['MAPE']['FORWARD'] = mean_absolute_percentage_error(test_y, pred_y_forward)
perf_mat.loc['MAPE']['BACKWARD'] = mean_absolute_percentage_error(test_y, pred_y_backward)
perf_mat.loc['MAPE']['STEPWISE'] = mean_absolute_percentage_error(test_y, pred_y_stepwise)

print(perf_mat)
```
```
              ALL      FORWARD     BACKWARD     STEPWISE
MSE   1.44149e+06  1.46142e+06  1.46142e+06  1.46142e+06
RMSE      1200.62      1208.89      1208.89      1208.89
MAE       853.494      863.524      863.524      863.524
MAPE      8.48549      8.59054      8.59054      8.59054
```
<details markdown="1">
<summary class='jb-small' style="color:blue">The number of params</summary>
<hr class='division3'>
```python
print(Forward_best_model.params.shape, Backward_best_model.params.shape, Stepwise_best_model.params.shape)
```
```
(24,) (24,) (24,)
```

<br>
```python
print(len(fitted_full_model.params))
print(len(Forward_best_model.params))
print(len(Backward_best_model.params))
print(len(Stepwise_best_model.params))
```
```
37
24
24
24
```

<hr class='division3'>
</details>



<br><br><br>


#### Logistic regression about dataset on real world


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
