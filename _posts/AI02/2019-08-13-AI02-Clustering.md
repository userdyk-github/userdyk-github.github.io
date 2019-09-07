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
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import numpy as np

# loading dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# clustering
clustering = cluster.KMeans(n_clusters=3)
clustering.fit(X)
y_pred = clustering.predict(X)

# correction assigned different integer values to the groups
idx_0, idx_1, idx_2 = (np.where(y_pred == n) for n in range(3))
y_pred[idx_0], y_pred[idx_1], y_pred[idx_2] = 2, 0, 1

# summarize the overlaps between the supervised and unsupervised classification
print(metrics.confusion_matrix(y, y_pred))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[[ 0  0 50]
 [48  2  0]
 [14 36  0]]
```
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

