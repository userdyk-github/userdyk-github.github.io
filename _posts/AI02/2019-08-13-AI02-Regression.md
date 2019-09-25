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

## **Basic linear regression**

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Multiple linear regression**

```python
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

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
<summary class='jb-small' style="color:blue">Estimated values v.s. Original values</summary>
<hr class='division3'>
`Estimated values`
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
`Original values`
```python
target
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
