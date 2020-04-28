---
layout : post
title : PL03-Topic02, TensorFlow_version1
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html)｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/2019-08-13-PL03-Topic02-TensorFlow_version1.md" target="_blank">page management</a> <br>
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

## **Installation**
### ***For linux***
```bash
$ 
```
<br><br><br>

### ***For windows***
```dos

```
<br><br><br>

### ***Version Control***
```python
import tensorflow as tf

print(tf.__version__)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">How to use 1.x version on 2.x version</summary>
<hr class='division3'>
```python
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
```
<hr class='division3'>
</details>


<br><br><br>


<hr class="division2">

## **Tutorials**
### ***Dataset : loader***

<br><br><br>

---

### ***Neural net : Custom layers***
#### Sequential 

<br><br><br>
#### Module 

<br><br><br>

---

### ***Optimization : Training***

<br><br><br>

---

### ***Evaluation : Predicting***

<br><br><br>
<hr class="division2">

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
# Linear Regression Implementation in TensorFlow
# Importing TensorFlow, Numpy, and the Boston Housing price dataset
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston

# Function to load the Boston data set
def read_infile():
    data = load_boston()
    features = np.array(data.data)
    target = np.array(data.target)
    return features,target

# Normalize the features by Z scaling; i.e., subtract from each feature value its mean and then divide by its
#standard deviation. Accelerates gradient descent.
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std

# Append the feature for the bias term.
def append_bias(features,target):
    n_samples = features.shape[0]
    n_features = features.shape[1]
    intercept_feature  = np.ones((n_samples,1))
    X = np.concatenate((features,intercept_feature),axis=1)
    X = np.reshape(X,[n_samples,n_features +1])
    Y = np.reshape(target,[n_samples,1])
    return X,Y

#  Execute the functions to read, normalize, and add append bias term to the data
features,target = read_infile()
z_features = feature_normalize(features)
X_input,Y_input = append_bias(z_features,target)
num_features = X_input.shape[1]

# Create TensorFlow ops for placeholders, weights, and weight initialization
X = tf.placeholder(tf.float32,[None,num_features])
Y = tf.placeholder(tf.float32,[None,1])
w = tf.Variable(tf.random_normal((num_features,1)),name='weights')
init = tf.global_variables_initializer()

# Define the different TensorFlow ops and input parameters for Cost and Optimization.
learning_rate = 0.01
num_epochs = 1000
cost_trace = []
pred = tf.matmul(X,w)
error = pred - Y
cost = tf.reduce_mean(tf.square(error))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Execute the gradient-descent learning
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epochs):
        sess.run(train_op,feed_dict={X:X_input,Y:Y_input})
        cost_trace.append(sess.run(cost,feed_dict={X:X_input,Y:Y_input}))
        error_ = sess.run(error,{X:X_input,Y:Y_input})
        pred_ = sess.run(pred,{X:X_input})
print('MSE in training:',cost_trace[-1])
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
MSE in training: 21.928276
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization of Cost</summary>
<hr class='division3'>
```python
# Linear Regression Cost Plot over Epochs or Iterations
# Plot the reduction in cost over iterations or epochs
import matplotlib.pyplot as plt
plt.plot(cost_trace)
plt.show()
```
`OUTPUT`
![Figure_1](https://user-images.githubusercontent.com/52376448/65562411-4c932e00-df81-11e9-9530-6a2b4b79c294.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization of Prediction</summary>
<hr class='division3'>
```python
# Linear Regression Actual House Price Versus Predicted House Price
# Plot the Predicted House Prices vs the Actual House Prices
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.scatter(Y_input,pred_)
ax.set_xlabel('Actual House price')
ax.set_ylabel('Predicted House price')
plt.show()
```
`OUTPUT`
![Figure_1](https://user-images.githubusercontent.com/52376448/65562640-0c807b00-df82-11e9-8360-b96213765971.png)
<hr class='division3'>
</details>

<br><br><br>

#### Multi-class Classification with SoftMax Function Using Full-Batch Gradient Descent

```python
# Multi-class Classification with Softmax Function Using Full-Batch Gradient Descent
# Import the required libraries 
import tensorflow as tf
import numpy as np
from sklearn import datasets 
from tensorflow.examples.tutorials.mnist import input_data

# Function to read the MNIST dataset along with the labels
def read_infile():  
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
    train_X, train_Y,test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels   
    return train_X, train_Y,test_X, test_Y

#  Define the weights and biases for the neural network 
def weights_biases_placeholder(n_dim,n_classes):  
    X = tf.placeholder(tf.float32,[None,n_dim])    
    Y = tf.placeholder(tf.float32,[None,n_classes])  
    w = tf.Variable(tf.random_normal([n_dim,n_classes],stddev=0.01),name='weights') 
    b = tf.Variable(tf.random_normal([n_classes]),name='weights')  
    return X,Y,w,b

# Define the forward pass
def forward_pass(w,b,X):  
    out = tf.matmul(X,w) + b
    return out

# Define the cost function for the SoftMax unit
def multiclass_cost(out,Y):  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y)) 
    return cost

# Define the initialization op
def init():
    return tf.global_variables_initializer()

# Define the training op
def train_op(learning_rate,cost): 
    op_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)   
    return op_train
train_X, train_Y,test_X, test_Y = read_infile()
X,Y,w,b = weights_biases_placeholder(train_X.shape[1],train_Y.shape[1])
out = forward_pass(w,b,X)
cost = multiclass_cost(out,Y)
learning_rate,epochs = 0.01,1000 
op_train = train_op(learning_rate,cost) 
init = init() 
loss_trace = [] 
accuracy_trace = []

# Activate the TensorFlow session and execute the stochastic gradient descent 
with tf.Session() as sess:   
    sess.run(init)
    for i in range(epochs):    
        sess.run(op_train,feed_dict={X:train_X,Y:train_Y})  
        loss_ = sess.run(cost,feed_dict={X:train_X,Y:train_Y})  
        accuracy_ = np.mean(np.argmax(sess.run(out,feed_dict={X:train_X,Y:train_Y}),axis=1) == np.argmax(train_Y,axis=1))       
        loss_trace.append(loss_)      
        accuracy_trace.append(accuracy_)   
        if (((i+1) >= 100) and ((i+1) % 100 == 0 )) :   
            print('Epoch:',(i+1),'loss:',loss_,'accuracy:',accuracy_)
            
    print('Final training result:','loss:',loss_,'accuracy:',accuracy_)   
    loss_test = sess.run(cost,feed_dict={X:test_X,Y:test_Y})  
    test_pred = np.argmax(sess.run(out,feed_dict={X:test_X,Y:test_Y}),axis=1)  
    accuracy_test = np.mean(test_pred == np.argmax(test_Y,axis=1))  
    print('Results on test dataset:','loss:',loss_test,'accuracy:',accuracy_test)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">Training</summary>
<hr class='division3'>
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Epoch: 100 loss: 1.5383416 accuracy: 0.7381636363636364
Epoch: 200 loss: 1.184439 accuracy: 0.7911454545454546
Epoch: 300 loss: 0.9936375 accuracy: 0.8137818181818182
Epoch: 400 loss: 0.8766257 accuracy: 0.8260363636363637
Epoch: 500 loss: 0.7975932 accuracy: 0.8344363636363636
Epoch: 600 loss: 0.74041194 accuracy: 0.8407818181818182
Epoch: 700 loss: 0.6969224 accuracy: 0.8461818181818181
Epoch: 800 loss: 0.6625869 accuracy: 0.8504363636363637
Epoch: 900 loss: 0.6346859 accuracy: 0.8542909090909091
Epoch: 1000 loss: 0.6114902 accuracy: 0.8569272727272728
Final training result: loss: 0.6114902 accuracy: 0.8569272727272728
Results on test dataset: loss: 0.584958 accuracy: 0.869
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Prediction</summary>
<hr class='division3'>
```python
# Display the Actual Digits Versus the Predicted Digits Along with the Images of the Actual Digits
import matplotlib.pyplot as plt 
%matplotlib inline

f, a = plt.subplots(1, 10, figsize=(10, 2))
print('Actual digits:   ', np.argmax(test_Y[0:10],axis=1))
print('Predicted digits:',test_pred[0:10])
print('Actual images of the digits follow:')

for i in range(10):      
    a[i].imshow(np.reshape(test_X[i],(28, 28)))
```
`OUTPUT`
```
Actual digits:    [7 2 1 0 4 1 4 9 5 9]
Predicted digits: [7 2 1 0 4 1 4 9 2 9]
Actual images of the digits follow:
```
![다운로드](https://user-images.githubusercontent.com/52376448/65567285-99cacc00-df90-11e9-8baa-79cf95760a71.png)
<hr class='division3'>
</details>
<br><br><br>

#### Multi-class Classification with SoftMax Function Using Stochastic Gradient Descent

```python
import tensorflow as tf
import numpy as np
from sklearn import datasets 
from tensorflow.examples.tutorials.mnist import input_data

# Multi-class Classification with Softmax Function Using Stochastic Gradient Descent
def read_infile():   
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
    train_X, train_Y,test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels   
    return train_X, train_Y,test_X, test_Y

def weights_biases_placeholder(n_dim,n_classes): 
    X = tf.placeholder(tf.float32,[None,n_dim])   
    Y = tf.placeholder(tf.float32,[None,n_classes])   
    w = tf.Variable(tf.random_normal([n_dim,n_classes],stddev=0.01),name='weights') 
    b = tf.Variable(tf.random_normal([n_classes]),name='weights')   
    return X,Y,w,b

def forward_pass(w,b,X):  
    out = tf.matmul(X,w) + b  
    return out

def multiclass_cost(out,Y):   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))  
    return cost

def init(): 
    return tf.global_variables_initializer()

def train_op(learning_rate,cost): 
    op_train = tf.train.AdamOptimizer(learning_rate).minimize(cost) 
    return op_train

train_X, train_Y,test_X, test_Y = read_infile()
X,Y,w,b = weights_biases_placeholder(train_X.shape[1],train_Y.shape[1]) 
out = forward_pass(w,b,X) 
cost = multiclass_cost(out,Y) 
learning_rate,epochs,batch_size = 0.01,1000,1000 
num_batches = int(train_X.shape[0]/batch_size)
op_train = train_op(learning_rate,cost) 
init = init() 
epoch_cost_trace = []
epoch_accuracy_trace = []

with tf.Session() as sess:  
    sess.run(init)
    for i in range(epochs):  
        epoch_cost, epoch_accuracy = 0, 0
        for j in range(num_batches):         
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size]})      
            actual_batch_size = train_X[j*batch_size:(j+1)*batch_size].shape[0]         
            epoch_cost += actual_batch_size*sess.run(cost,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size]})
            
        epoch_cost = epoch_cost/float(train_X.shape[0])     
        epoch_accuracy = np.mean(np.argmax(sess.run(out,feed_dict={X:train_X,Y:train_Y}), axis=1) == np.argmax(train_Y,axis=1))    
        epoch_cost_trace.append(epoch_cost)     
        epoch_accuracy_trace.append(epoch_accuracy)
        if (((i +1) >= 100) and ((i+1) % 100 == 0 )) :  
            print('Epoch:',(i+1),'Average loss:',epoch_cost,'accuracy:',epoch_accuracy)
    print('Final epoch training results:','Average loss:',epoch_cost,'accuracy:',epoch_accuracy)   
    loss_test = sess.run(cost,feed_dict={X:test_X,Y:test_Y})    
    test_pred = np.argmax(sess.run(out,feed_dict={X:test_X,Y:test_Y}),axis=1) 
    accuracy_test = np.mean(test_pred == np.argmax(test_Y,axis=1)) 
    print('Results on test dataset:','Average loss:',loss_test,'accuracy:',accuracy_test)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">Training</summary>
<hr class='division3'>
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Epoch: 100 Average loss: 0.21720419471914118 accuracy: 0.9389636363636363
Epoch: 200 Average loss: 0.21221693090417168 accuracy: 0.9397454545454546
Epoch: 300 Average loss: 0.21042687581344086 accuracy: 0.9401272727272727
Epoch: 400 Average loss: 0.20955939401279797 accuracy: 0.9402363636363636
Epoch: 500 Average loss: 0.2090753911571069 accuracy: 0.9405636363636364
Epoch: 600 Average loss: 0.208774649690498 accuracy: 0.9406545454545454
Epoch: 700 Average loss: 0.20857206650755622 accuracy: 0.9406181818181818
Epoch: 800 Average loss: 0.20842633586038242 accuracy: 0.9406545454545454
Epoch: 900 Average loss: 0.20831616845997897 accuracy: 0.9408
Epoch: 1000 Average loss: 0.20822952132333408 accuracy: 0.9408727272727273
Final epoch training results: Average loss: 0.20822952132333408 accuracy: 0.9408727272727273
Results on test dataset: Average loss: 0.4592456 accuracy: 0.9155
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Prediction</summary>
<hr class='division3'>
```python
# Actual Digits Versus Predicted Digits for SoftMax Classification Through Stochastic Gradient Descent
import matplotlib.pyplot as plt

f, a = plt.subplots(1, 10, figsize=(10, 2))
print('Actual digits:   ', np.argmax(test_Y[0:10],axis=1) )
print('Predicted digits:',test_pred[0:10] )
print('Actual images of the digits follow:' )
for i in range(10):      
    a[i].imshow(np.reshape(test_X[i],(28, 28)))
```
`OUTPUT`
```
Actual digits:    [7 2 1 0 4 1 4 9 5 9]
Predicted digits: [7 2 1 0 4 1 4 9 6 9]
Actual images of the digits follow:
```
![다운로드 (1)](https://user-images.githubusercontent.com/52376448/65567386-0940bb80-df91-11e9-9e2f-f5ce47df3b0c.png)
<hr class='division3'>
</details>
<br><br><br>



<hr class="division2">

## **Example**
### ***XOR problem***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---



### ***Simple linear regression***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Multi-variate linear regression***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Logistic regression : binary class***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Softmax regression : multi-class***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---


### ***Neural network***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Convolutional neural network : Digit Recognition***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Recurrent neural network : Next-Word Prediction***
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>








<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Santanu Pattanayak, Pro Deep Learning with TensorFlow, 2017
- <a href="https://github.com/deeplearningzerotoall/TensorFlow" target="_blank">github : zerotoall, tensorflow</a>
- <a href="https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm" target="_blank">lecture : zerotoall, tensorflow</a>
- <a href='https://tensorflowkorea.gitbooks.io/tensorflow-kr/' target="_blank">tensorflow-kr</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>


