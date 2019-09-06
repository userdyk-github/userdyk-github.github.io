---
layout : post
title : AI02, Classification
categories: [AI02]
comments : true
tags : [AI02]
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

## **A wide variety of alternative algorithms for classification**

- Logistic regression
- **KNN**, k-nearest neighbor methods
- **SVM**, support vector machines
- Decision trees
- Random forest methods

<br><br><br>

---

## **Implement with sklearn**

### ***Through logistic regression***

```python
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model 
from sklearn import metrics

# loading dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, train_size=0.7)

# classification for loaded dataset
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# result
print(metrics.classification_report(y_test, y_test_pred))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      0.93      0.97        15
           2       0.94      1.00      0.97        16

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Through k-nearest neighbor methods***

```python
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree 

# loading dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, train_size=0.7)

# classification for loaded dataset
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# result
print(metrics.classification_report(y_test, y_test_pred))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        15
           1       0.92      0.92      0.92        13
           2       0.94      0.94      0.94        17

    accuracy                           0.96        45
   macro avg       0.95      0.95      0.95        45
weighted avg       0.96      0.96      0.96        45
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Through support vector machines***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Through decision trees***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>

---

### ***Through Random forest methods***

```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>


<br><br><br>

<hr class="division2">

## **Implement with tensorflow**

<br><br><br>

<hr class="division2">

## **Implement with pytorch**

<br><br><br>

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
