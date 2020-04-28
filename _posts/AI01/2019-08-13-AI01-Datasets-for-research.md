---
layout : post
title : AI01, Datasets for research
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Datasets-for-research.md" target="_blank">page management</a><br>
<a href='https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research' target="_blank">List of datasets for machine-learning research</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">


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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```
<hr class='division3'>
</details>


<br>
```python
from sklearn.datasets import load_boston
loaded_dataset = load_boston()

print(loaded_dataset.data.shape)
print(loaded_dataset.target.shape)
```
```
(506, 13)
(506,)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3,3, figsize=(10,10))
for i in range(3):
    for j in range(3):
        axes[i, j].scatter(loaded_dataset.data[:,3*i+j], loaded_dataset.target)
        axes[i, j].set_title("%d"%(3*i+j))
plt.tight_layout()
plt.show()
```
![download](https://user-images.githubusercontent.com/52376448/68390612-000a5780-01a9-11ea-9ff1-edf2bc853663.png)
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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

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
<summary class='jb-small' style="color:blue">Split dataset</summary>
<hr class='division3'>
```python
from sklearn.model_selection import train_test_split

x = loaded_dataset.data
y = loaded_dataset.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - x_train_mean)/x_train_std

x_val_mean = np.mean(x_val, axis=0)
x_val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - x_val_mean)/x_val_std

x_test_mean = np.mean(x_test, axis=0)
x_test_std = np.std(x_test, axis=0)
x_test_scaled = (x_test - x_test_mean)/x_test_std
```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

<hr class='division3'>
</details>


<br><br><br>

## **From keras**
```python
from keras.datasets import mnist
```
<br><br><br>

<hr class="division1">

Reference

- <a href='https://lionbridge.ai/datasets/20-best-image-datasets-for-computer-vision/' target="_blank">20 Free Image Datasets for Computer Vision</a>

---

- <a href="www.kaggle.com/" target="_blank">Kaggle</a>
- <a href="https://www.data.gov/" target="_blank">US Government Open Data</a>
- <a href="https://data.gov.in/" target="_blank">Indian Government Open Data</a>
- <a href="https://registry.opendata.aws/" target="_blank">Amazon Web Service Datasets</a>
- <a href="https://toolbox.google.com/datasetsearch" target="_blank">Google Dataset Search</a>
- <a href="https://archive.ics.uci.edu/ml/" target="_blank">UCI ML Repository</a>
- <a href="https://www.data.go.kr/" target="_blank">Open Data potal</a>
- <a href="http://kosis.kr/index/index.do" target="_blank">kosis</a>

---
