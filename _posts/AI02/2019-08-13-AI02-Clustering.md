---
layout : post
title : AI02, Clustering
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

## Clustering with sklearn

```python
# [0] : importing modules
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import numpy as np

# [1] : loading dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# [2] : clustering
clustering = cluster.KMeans(n_clusters=3)
clustering.fit(X)
y_pred = clustering.predict(X)

# [3] : correction assigned different integer values to the groups
idx_0, idx_1, idx_2 = (np.where(y_pred == n) for n in range(3))
y_pred[idx_0], y_pred[idx_1], y_pred[idx_2] = 2, 0, 1

# [4] : summarize the overlaps between the supervised and unsupervised classification
metrics.confusion_matrix(y, y_pred)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([[50,  0,  0],
       [ 0, 48,  2],
       [ 0, 14, 36]], dtype=int64)
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Details code[1]</summary>
<hr class='division3'>
**iris dataset**
```python
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
iris.feature_names.append('target_names')

df1 = pd.DataFrame(iris.data)
df2 = pd.DataFrame(iris.target)
df = pd.concat([df1,df2], axis=1)
df.columns = iris.feature_names

print(df)
```
```
     s.length (cm)  s.width (cm)  ...  p.width (cm)  target_names
0              5.1           3.5  ...           0.2             0
1              4.9           3.0  ...           0.2             0
2              4.7           3.2  ...           0.2             0
3              4.6           3.1  ...           0.2             0
4              5.0           3.6  ...           0.2             0
..             ...           ...  ...           ...           ...
145            6.7           3.0  ...           2.3             2
146            6.3           2.5  ...           1.9             2
147            6.5           3.0  ...           2.0             2
148            6.2           3.4  ...           2.3             2
149            5.9           3.0  ...           1.8             2

[150 rows x 5 columns]
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Details code[2]</summary>
<hr class='division3'>
`OUTPUT`
```python
# [0] : importing modules
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import numpy as np

# [1] : loading dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# [2] : clustering
clustering = cluster.KMeans(n_clusters=3)
clustering.fit(X)
```
```
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Details code[3]</summary>
<hr class='division3'>
```python
# [0] : importing modules
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import numpy as np

# [1] : loading dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# [2] : clustering
clustering = cluster.KMeans(n_clusters=3)
clustering.fit(X)
y_pred = clustering.predict(X)
```
```python
y_pred[::8]
```
`OUTPUT` : array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0])
<br>
```python
y[::8]
```
`OUTPUT` : array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Details code[4]</summary>
<hr class='division3'>
On the above confusion matrix matrix, **the diagonals** correspond to the number of samples that are correctly classified for each level of the category variable, and **the off-diagonal elements** are the number of incorrectly classified samples. More specifically, the element of the confusion matrix C is the number of samples of category i that were categorized as j. 
<hr class='division3'>
</details>

<hr class="division2">

## title2

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

