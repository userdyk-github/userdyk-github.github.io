---
layout : post
title : PL03-Topic02, TensorFlow
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

## **Mathematical Foundations**

### ***Linear Algebra***

- Vector
- Scalar
- Matrix
- Tensor
- Matrix Operations and Manipulations
- Linear Independence of Vectors
- Rank of a Matrix
- Identity Matrix or Operator
- Determinant of a Matrix
- Inverse of a Matrix
- Norm of a Vector
- Pseudo Inverse of a Matrix
- Unit Vector in the Direction of a Speciﬁc Vector
- Projection of a Vector in the Direction of Another Vector
- Eigen Vectors

---

### ***Calculus***

- Differentiation
- Gradient of a Function
- Successive Partial Derivatives
- Hessian Matrix of a Function
- Maxima and Minima of Functions
- Local Minima and Global Minima
- Positive Semi-Deﬁnite and Positive Deﬁnite
- Convex Set
- Convex Function
- Non-convex Function
- Multivariate Convex and Non-convex Functions Examples
- Taylor Series

---

### ***Probability***

- Unions, Intersection, and Conditional Probability
- Chain Rule of Probability for Intersection of Event
- Mutually Exclusive Events
- Independence of Events
- Conditional Independence of Event
- Bayes Rule
- Probability Mass Function
- Probability Density Function
- Expectation of a Random Variable
- Variance of a Random Variable
- Skewness and Kurtosis
- Covariance
- Correlation Coefﬁcient
- Some Common Probability Distribution
- Likelihood Function
- Maximum Likelihood Estimate
- Hypothesis Testing and p Value

---

### ***Formulation of Machine-Learning Algorithm and Optimization Techniques***

- Supervised Learning
  - Linear Regression as a Supervised Learning Method
  - Linear Regression Through Vector Space Approach
  - Hyperplanes and Linear Classifiers
- Unsupervised Learning
- Optimization Techniques for Machine Learning
  - Gradient Descent
  - Gradient Descent for a Multivariate Cost Function
  - Steepest Descent
  - Stochastic Gradient Descent
  - Newton’s Method
- Constrained Optimization Problem
  
---

### ***A Few Important Topics in Machine Learning***

- Dimensionality Reduction Methods
  - Principal Component Analysis
  - Singular Value Decomposition  
- Regularization
- Regularization Viewed as a Constraint Optimization Problem


<hr class="division2">

## **Deep-Learning Concepts**

### ***Deep Learning and Its Evolution***

```python

```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

---

### ***Perceptrons and Perceptron Learning Algorithm***

- Geometrical Interpretation of Perceptron Learning
- Limitations of Perceptron Learning
- Need for Non-linearity 
- Hidden Layer Perceptrons’ Activation Function for Non-linearity
- Different Activation Functions for a Neuron/Perceptron
- Learning Rule for Multi-Layer Perceptrons Network
- Backpropagation for Gradient Computation
- Generalizing the Backpropagation Method for Gradient Computation

<br><br><br>

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

---

### ***TensorFlow***

#### TensorFlow(1.*) Basics for Development

<br><br><br>

#### TensorFlow(2.0) Basics for Development

<span class="frame2">Create list</span>

```
>>> l1 = [1,2,3]
>>> l1
[1,2,3]

>>> l2 = [[1,2,3], [4,5,6]]
>>> l2
[[1,2,3], [4,5,6]]
```

<br><br><br>

<span class="frame2">Create array</span>

```
>>> import numpy as np

# convert dtype
>>> arr1 = np.array([1,2,3])
>>> arr1
array([1, 2, 3])
>>> arr1.shape
(3,)
>>> arr1.astype(np.uint8)
array([1, 2, 3], dtype=uint8)

# with dtype format : ''
>>> arr2 = np.array([1,2,3], dtype='float32')
>>> arr2
array([1, 2, 3], dtype=float32)

# with dtype format : np.
>>> arr3 = np.array([1,2,3], dtype=np.float32)
>>> arr3
array([1, 2, 3], dtype=float32)
>>> arr3.shape
(3,)

>>> arr4 = np.array([[1,2,3],[4,5,6]])
>>> arr4
array([[1, 2, 3],
       [4, 5, 6]])
>>> arr4.shape
(2, 3)
```

<br><br><br>


<span class="frame2">Create tensor</span>

```
>>> import numpy as np
>>> import tensorflow as tf

# list > tensor
>>> tf.constant([1,2,3])
<tf.Tensor 'Const:0' shape=(3,) dtype=int32>

>>> tf.constant([1,2,3], dtype="float32")
<tf.Tensor 'Const_1:0' shape=(3,) dtype=float32>

>>> tf.constant([1,2,3], dtype=tf.float32)
<tf.Tensor 'Const_2:0' shape=(3,) dtype=float32>

>>> tensor = tf.constant([1,2,3], dtype="float32")
>>> tf.cast(tensor, dtype="uint8")
<tf.Tensor 'Cast:0' shape=(3,) dtype=uint8>


# tuple > tensor
>>> tf.constant((1,2,3))
<tf.Tensor 'Const_3:0' shape=(3,) dtype=int32>


# array > tensor
>>> tf.constant(np.array([1,2,3]))
<tf.Tensor 'Const_4:0' shape=(3,) dtype=int64>


# format1 : tensor > array
>>> tensor = tf.constant(np.array([1,2,3]))
>>> tensor.numpy()
array([1, 2, 3])

# format2 : tensor > array
>>> tensor = tf.constant(np.array([1,2,3]))
>>> np.array(tensor)
array([1, 2, 3])
```

<br><br><br>



<span class="frame2">Generate random numbers</span>

```python
```

<br><br><br>



#### Optimizers in TensorFlow

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>


#### XOR Implementation Using TensorFlow

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>


#### Linear Regression in TensorFlow

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

#### Multi-class Classification with SoftMax Function Using Full-Batch Gradient Descent

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

#### Multi-class Classification with SoftMax Function Using Stochastic Gradient Descent

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>


---

### ***GPU***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">



## **Advanced Neural Networks**

### ***Image Segmentation***

#### Otsu’s Method

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

#### Watershed Algorithm for Image Segmentation

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

#### Image Segmentation Using K-means Clustering

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

#### Semantic Segmentation in TensorFlow with Fully Connected Neural Networks

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>



---

### ***Image Classification and Localization Network***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

---

### ***Object Detection***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>


---

### ***TensorFlow Models’ Deployment in Production***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>



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



