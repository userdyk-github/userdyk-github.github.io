---
layout : post
title : PL03-Topic02, TensorFlow_version2
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) <br>
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

## **Basic Framework**

### ***Create Tensor***

```
>>> import numpy as np
>>> import tensorflow as tf

# list -> tensor
>>> tf.constant([1, 2, 3])

# tuple -> tensor
>>> tf.constant((1,2,3), (1,2,3))

# array -> tensor
>>> tf.constant(np.array([1,2,3]))

```
<br><br><br>

---

### ***Confirm the information about tensor***

```
>>> import numpy as np
>>> import tensorflow as tf

# check shape
>>> tensor = tf.constant(np.array([1,2,3]))
>>> tensor.shape

# check data type
>>> tensor.dtype

# define data type
>>> tf.constant([1,2,3], dtype=tf.float32)

# convert data type
>>> tensor = tf.constant([1,2,3], dtype=tf.float32)
>>> tf.cast(tensor, dtype=tf.unit8)

# (1) : tensor -> numpy
>>> tensor.numpy()

# (2) : tensor -> numpy
>>> np.array(tensor)

```
<br><br><br>

---

### ***Generate random numbers***

```
>>> import tensorflow as tf

# normal distribution
>>> tf.random.normal([3,3])

# uniform distribution
>>> tf.random.uniform([4,4])

```

<br><br><br>

<hr class="division2">

## **Load Dataset**

<hr class="division2">

## **Layer and Parameters**

<hr class="division2">

## **Optimizaer and Training**

<hr class="division2">

## **Evaluating and Predicting**

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Santanu Pattanayak, Pro Deep Learning with TensorFlow, 2017
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>
