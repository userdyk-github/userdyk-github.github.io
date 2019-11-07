---
layout : post
title : AI01, Datasets for research
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
<a href='https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research' target="_blank">List of datasets for machine-learning research</a>

<hr class="division1">

- <a href="www.kaggle.com/" target="_blank">Kaggle</a>
- <a href="www.data.gov/" target="_blank">US Government Open Data</a>
- <a href="https://data.gov.in/" target="_blank">Indian Government Open Data</a>
- <a href="https://registry.opendata.aws/" target="_blank">Amazon Web Service Datasets</a>
- <a href="https://toolbox.google.com/datasetsearch" target="_blank">Google Dataset Search</a>
- <a href="https://archive.ics.uci.edu/ml/" target="_blank">UCI ML Repository</a>
- <a href="https://www.data.go.kr/" target="_blank">Open Data potal</a>
- <a href="http://kosis.kr/index/index.do" target="_blank">kosis</a>

<hr class="division2">

## **From sklearn**

```python
from sklearn.datasets import load_diabetes
loaded_dataset = load_diabetes()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(442, 10)
(442,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_boston
load_boston = load_boston()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(506, 13)
(506,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_breast_cancer
loaded_dataset = load_breast_cancer()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(569, 30)
(569,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_digits
loaded_dataset = load_digits()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(1797, 64)
(1797,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_iris
loaded_dataset = load_iris()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(150, 4)
(150,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_linnerud
loaded_dataset = load_linnerud()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(20, 3)
(20, 3)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_wine
loaded_dataset = load_wine()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(178, 13)
(178,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>


<br><br><br>
<hr class="division1">

Reference

- <a href='https://lionbridge.ai/datasets/20-best-image-datasets-for-computer-vision/' target="_blank">20 Free Image Datasets for Computer Vision</a>

---
