---
layout : post
title : AI02, Dimensionality reduction
categories: [AI02]
comments : true
tags : [AI02]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜[Meachine learning](https://userdyk-github.github.io/ai02/AI02-Contents.html)<br>
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

## **Feature selection**

<hr class="division2">

## **Feature extraction**

### ***Principal Component Analysis***

`Data preprocessing`
```python
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris=datasets.load_iris()
X=iris.data[:,[0,2]]
y=iris.target
feature_names=[iris.feature_names[0],iris.feature_names[2]]
df_X=pd.DataFrame(X)
df_Y=pd.DataFrame(y)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>

`The meaning of output through PCA`
```python
pca=PCA(n_components=2)
pca.fit(X)
PCscore=pca.transform(X)
eigens_v=pca.components_.transpose()

mX=np.matrix(X)
for i in range(X.shape[1]):
    mX[:,i]=mX[:,i]-np.mean(X[:,i])
dfmX=pd.DataFrame(mX)

plt.scatter(dfmX[0],dfmX[1])
origin = [0], [0] # origin point
plt.quiver(*origin, eigens_v[0,:], eigens_v[1,:], color=['r','b'], scale=3)
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br>

`Regression with PC scores`
```python
X2 = iris.data
pca2 = PCA(n_components=4)
pca2.fit(X2)
pca2.explained_variance_
PCs=pca2.transform(X2)[:,0:2]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

clf2 = LogisticRegression(solver="sag",multi_class="multinomial").fit(PCs,y)
confusion_matrix(y,clf2.predict(PCs))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
    

---

### ***Multilinear Principal Component Analysis***

---


### ***Kernel PCA***

---


### ***Graph-based kernel PCA***

---


### ***Singular Value Decomposition***

---

### ***Non-negative matrix factorization (NMF)***

---


### ***Linear discriminant analysis (LDA)***

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

clf=LinearDiscriminantAnalysis()
clf.fit(X,y)
print(clf.predict([[-0.8, -1]]))
```
<span class="jb-medium">[1]</span>

<details markdown="1">
<summary class='jb-small' style="color:blue">Performance</summary>
<hr class='division3'>
```python
from sklearn.metrics import confusion_matrix

y_pred=clf.predict(X)
confusion_matrix(y,y_pred) 
```
```
array([[3, 0],
       [0, 3]], dtype=int64)
```
<hr class='division3'>
</details>
<br><br><br>


---

### ***Quadratic discriminant analysis (QDA)***

```python
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

clf2=QuadraticDiscriminantAnalysis()
clf2.fit(X,y)
print(clf2.predict([[-0.8, -1]]))
```
<span class="jb-medium">[1]</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Performance</summary>
<hr class='division3'>
```python
from sklearn.metrics import confusion_matrix

y_pred2=clf2.predict(X)
confusion_matrix(y,y_pred2)  
```
```
array([[3, 0],
       [0, 3]], dtype=int64)
```
<hr class='division3'>
</details>
<br><br><br>

---

### ***Generalized discriminant analysis (GDA)***

---

### ***Autoencoder***


<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference

- <a href='https://en.wikipedia.org/wiki/Feature_selection' target="_blank">Feature selection</a>
- <a href='https://en.wikipedia.org/wiki/Feature_extraction' target="_blank">Feature extraction</a>
- <a href='https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/' target="_blank">PCA</a>
- <a href='https://datascienceschool.net/view-notebook/f10aad8a34a4489697933f77c5d58e3a/' target="_blank">PCA</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
