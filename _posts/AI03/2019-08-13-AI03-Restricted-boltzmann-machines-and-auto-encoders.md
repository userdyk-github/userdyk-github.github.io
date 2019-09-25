---
layout : post
title : AI03, Restricted boltzmann machines and auto-encoders
categories: [AI03]
comments : true
tags : [AI03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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


## **Boltzmann Distribution**

```python

```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">


## **Bayesian Inference: Likelihood, Priors, and Posterior Probability Distribution**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Markov Chain Monte Carlo Methods for Sampling**

```python
import numpy as np 
number_sample = 100000
inner_area,outer_area = 0,0
for i in range(number_sample): 
    x = np.random.uniform(0,1) 
    y = np.random.uniform(0,1)  
    if (x**2 + y**2) < 1 :   
        inner_area += 1  
    outer_area += 1
print("The computed value of Pi:",4*(inner_area/float(outer_area)))
```
The computed value of Pi: 3.1358
<br><br><br>

### ***Metropolis Algorithm***

```python
import numpy as np
import matplotlib.pyplot as plt

#Now let’s generate this with one of the Markov Chain Monte Carlo methods called Metropolis Hastings algorithm 
# Our assumed transition probabilities would follow normal distribution X2 ~ N(X1,Covariance= [[0.2 , 0],[0,0.2]])
import time 
start_time = time.time()

# Set up constants and initial variable conditions
num_samples=100000
prob_density = 0 

## Plan is to sample from a Bivariate Gaussian Distribution with mean (0,0) and covariance of
## 0.7 between the two variables
mean = np.array([0,0]) 
cov = np.array([[1,0.7],[0.7,1]]) 
cov1 = np.matrix(cov)
mean1 = np.matrix(mean) 
x_list,y_list = [],[] 
accepted_samples_count = 0

## Normalizer of the Probability distibution 
## This is not actually required since we are taking ratio of probabilities for inference 
normalizer = np.sqrt( ((2*np.pi)**2)*np.linalg.det(cov))

## Start wtih initial Point (0,0) 
x_initial, y_initial = 0,0
x1,y1 = x_initial, y_initial

for i in range(num_samples):  
    ## Set up the Conditional Probability distribution, taking the existing point  
    ## as the mean and a small variance = 0.2 so that points near the existing point  
    ## have a high chance of getting sampled.  
    mean_trans = np.array([x1,y1])  
    cov_trans = np.array([[0.2,0],[0,0.2]])  
    x2,y2 = np.random.multivariate_normal(mean_trans,cov_trans).T 
    X = np.array([x2,y2])  
    X2 = np.matrix(X)  
    X1 = np.matrix(mean_trans) 
    
    ## Compute the probability density of the existing point and the new sampled   
    ## point  
    mahalnobis_dist2 = (X2 - mean1)*np.linalg.inv(cov)*(X2 - mean1).T  
    prob_density2 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist2)  
    mahalnobis_dist1 = (X1 - mean1)*np.linalg.inv(cov)*(X1 - mean1).T   
    prob_density1 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist1)   
    
    ##  This is the heart of the algorithm. Comparing the ratio of probability density  of the new  
    ##  point and the existing point(acceptance_ratio) and selecting the new point if it is to have more probability 
    ##  density. If it has less probability it is randomly selected with the probability  of getting  
    ## selected being proportional to the ratio of the acceptance ratio   
    acceptance_ratio = prob_density2[0,0] / float(prob_density1[0,0])
    
    if (acceptance_ratio >= 1) | ((acceptance_ratio < 1) and (acceptance_ratio >= np.random.uniform(0,1)) ):      
        x_list.append(x2)    
        y_list.append(y2)    
        x1 = x2   
        y1 = y2   
        accepted_samples_count += 1
            
end_time = time.time()
print ('Time taken to sample ' + str(accepted_samples_count) + ' points ==> ' + str(end_time -  start_time) + ' seconds' )
print ('Acceptance ratio ===> ' , accepted_samples_count/float(100000) )

## Time to display the samples generated
plt.xlabel('X') 
plt.ylabel('Y') 
plt.scatter(x_list,y_list,color='black') 
print ("Mean of the Sampled Points" )
print (np.mean(x_list),np.mean(y_list))
print ("Covariance matrix of the Sampled Points" )
print (np.cov(x_list,y_list))
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Time taken to sample 71400 points ==> 50.548850536346436 seconds
Acceptance ratio ===>  0.714
Mean of the Sampled Points
-0.006802046013446434 -0.028115488125035174
Covariance matrix of the Sampled Points
[[0.97409116 0.6805006 ]
 [0.6805006  0.96396597]]
```
![다운로드 (6)](https://user-images.githubusercontent.com/52376448/65573055-d7395480-dfa4-11e9-837b-7750b033b684.png)
<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Restricted Boltzmann Machines**

### ***A Restricted Boltzmann Implementation in TensorFlow***

```python

```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Deep Belief Networks (DBNs)***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>


<hr class="division2">

## **Auto-encoders**

### ***Sparse Auto-Encoder Implementation in TensorFlow***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Denoising Auto-Encoder***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **PCA and ZCA Whitening**

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
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
