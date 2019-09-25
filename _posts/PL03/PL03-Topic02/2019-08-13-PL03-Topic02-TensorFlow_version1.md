---
layout : post
title : PL03-Topic02, TensorFlow_version1
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

```
# Import TensorFlow and Numpy Library
>>> import tensorflow as tf 
>>> import numpy as np

# Activate a TensorFlow Interactive Session
>>> tf.InteractiveSession()

# Define Tensors
>>> a = tf.zeros((2,2)); 
>>> b = tf.ones((2,2))

# Sum the Elements of the Matrix (2D Tensor) Across the Horizontal Axis
>>> tf.reduce_sum(b,reduction_indices = 1).eval()
array([2., 2.], dtype=float32)

# verify tensor values
>>> sess = tf.Session()
...    print(sess.run(b))
...    print(sess.run(tf.reduce_sum(b)))
[[1. 1.]
 [1. 1.]]
4.0

# Check the Shape of the Tensor
>>> a.get_shape()
TensorShape([Dimension(2), Dimension(2)])

# Reshape a Tensor
>>> tf.reshape(a,(1,4)).eval()
array([[0., 0., 0., 0.]], dtype=float32)


# Explicit Evaluation in TensorFlow and Difference with Numpy
>>> ta = tf.zeros((2,2))
>>> print(ta)
Tensor("zeros_3:0", shape=(2, 2), dtype=float32)

>>> print(ta.eval())
[[0. 0.]
 [0. 0.]]


>>> a = np.zeros((2,2))
>>> print(a)
[[0. 0.]
 [0. 0.]]

# Define TensorFlow Constants
>>> a = tf.constant(1) 
>>> b = tf.constant(5) 
>>> c= a*b
>>> with tf.Session() as sess:   
...    print(c.eval())  
...    print(sess.run(c))


# Define TensorFlow Variables
>>> w = tf.Variable(tf.ones(2,2),name='weights')

# Initialize the Variables After Invoking the Session
>>> with tf.Session() as sess: 
...    sess.run(tf.global_variables_initializer())  
...    print(sess.run(w))
[1. 1.]


# Define the TensorFlow Variable with Random Initial Values from Standard Normal Distribution
>>> rw = tf.Variable(tf.random_normal((2,2)),name='random_weights')

# Invoke Session and Display the Initial State of the Variable
>>> with tf.Session() as sess:   
...    sess.run(tf.global_variables_initializer()) 
...    print(sess.run(rw))
[[-1.0602931  -0.20061749]
 [-1.1879984   2.0883346 ]]


# TensorFlow Variable State Update
>>> var_1 = tf.Variable(0,name='var_1')
>>> add_op = tf.add(var_1,tf.constant(1)) 
>>> upd_op = tf.assign(var_1,add_op) 
>>> with tf.Session() as sess:  
...        sess.run(tf.global_variables_initializer())   
...        for i in range(5):      
...            print(sess.run(upd_op))
1
2
3
4
5


# Display the TensorFlow Variable State
>>> x = tf.constant(1) 
>>> y = tf.constant(5) 
>>> z = tf.constant(7)

>>> mul_x_y = x*y
>>> final_op = mul_x_y + z

>>> with tf.Session() as sess:  
...     print(sess.run([mul_x_y,final_op]))
[5, 12]


# Convert a Numpy Array to Tensor
>>>a = np.ones((3,3)) 
>>>b = tf.convert_to_tensor(a)
>>>with tf.Session() as sess:  
...     print(sess.run(b))
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]


# Placeholders and Feed Dictionary
>>> inp1 = tf.placeholder(tf.float32,shape=(1,2)) 
>>> inp2 = tf.placeholder(tf.float32,shape=(2,1)) 
>>> output = tf.matmul(inp1,inp2) 

>>> with tf.Session() as sess:
...     print(sess.run([output],feed_dict={inp1:[[1.,3.]],inp2:[[1],[3]]}))
[array([[10.]], dtype=float32)]
```
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
# XOR Implementation with Hidden Layers That Have Sigmoid Activation Functions
#XOR  implementation in Tensorflow with hidden layers being sigmoid to # introduce Non-Linearity 
import tensorflow as tf 

# Create placeholders for training input and output labels 
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input") 
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input") 

#Define the weights to the hidden and output layer respectively.
w1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Weights1")
w2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Weights2") 

# Define the bias to the hidden and output layers respectively 
b1 = tf.Variable(tf.zeros([2]), name="Bias1") 
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Define the final output through forward pass
z2 = tf.sigmoid(tf.matmul(x_, w1) + b1)
pred = tf.sigmoid(tf.matmul(z2,w2) + b2)

# Define the Cross-entropy/Log-loss Cost function based on the output label y and 
# the predicted probability by the forward pass 
cost = tf.reduce_mean(( (y_ * tf.log(pred)) + ((1 - y_) * tf.log(1.0 - pred)) ) * -1) 
learning_rate = 0.01 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

#Now that we have all that we need set up we will start the training 
XOR_X = [[0,0],[0,1],[1,0],[1,1]] 
XOR_Y = [[0],[1],[1],[0]]

# Initialize the variables 
init = tf.initialize_all_variables()
sess = tf.Session() 
writer = tf.summary.FileWriter("./Downloads/XOR_logs", sess.graph_def)
sess.run(init) 
for i in range(100000):     
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
print('Final Prediction', sess.run(pred, feed_dict={x_: XOR_X, y_: XOR_Y})) 
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Final Prediction [[0.04043783]
 [0.9463439 ]
 [0.94631964]
 [0.08983095]]
```
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



