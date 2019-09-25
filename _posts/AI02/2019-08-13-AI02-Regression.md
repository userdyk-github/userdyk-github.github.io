---
layout : post
title : AI02, Regression
categories: [AI02]
comments : true
tags : [AI02]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[Meachine learning](https://userdyk-github.github.io/ai02/AI02-Contents.html)<br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Libraries</a>


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
