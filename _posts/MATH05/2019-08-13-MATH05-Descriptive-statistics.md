---
layout : post
title : MATH05, Descriptive statistics
categories: [MATH05]
comments : true
tags : [MATH05]
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

## **Review : Descriptive statistics with Excel**

![descriptive_statistics](https://user-images.githubusercontent.com/52376448/66703405-4f499d80-ed4d-11e9-8cc7-c33dbbfec77c.JPG)
![covariance_analysis](https://user-images.githubusercontent.com/52376448/66703416-51136100-ed4d-11e9-85c1-1861caf05537.JPG)
![correlation_analysis](https://user-images.githubusercontent.com/52376448/66703414-51136100-ed4d-11e9-8acd-093dec33923f.JPG)

<br><br><br>
<hr class="division2">

## **count**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
len(x) 
```
<span class="jb-medium">26</span>
<br><br><br>
<hr class="division2">

## **mean, average**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.mean(x)
```
<span class="jb-medium">4.8076923076923075</span>
<br><br><br>
<hr class="division2">

## **variance**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.var(x), np.var(x, ddof=1)
```
<span class="jb-medium">(115.23224852071006, 119.84153846153846)</span>
<br><br><br>
<hr class="division2">

## **standard deviation**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.std(x)
```
<span class="jb-medium">10.734628476137871</span>

<br><br><br>
<hr class="division2">

## **maximum, minimum**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.max(x), np.min(x)
```
<span class="jb-medium">(23, -24)</span>

<br><br><br>
<hr class="division2">

## **median**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.median(x)
```
<span class="jb-medium">5.0</span>

<br><br><br>
<hr class="division2">

## **quartile**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])

print(np.percentile(x, 0),
      np.percentile(x, 25),
      np.percentile(x, 50),
      np.percentile(x, 75),
      np.percentile(x, 100))
```
```
-24.0
0.0
5.0
10.0
23.0
```
<br><br><br>

<hr class="division2">

## **Describe all at once**

```python
import numpy as np
from scipy.stats import describe

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
describe(x)
```
<span class="jb-medium">DescribeResult(nobs=26, minmax=(-24, 23), mean=4.8076923076923075, variance=119.84153846153846, skewness=-0.4762339485461929, kurtosis=0.37443381660038977)</span>

<br><br><br>
<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://datascienceschool.net/view-notebook/8c39a71b453e43d9bd1000f38638e937/' target="_blank">deterministic and stochastic data, random variable</a>
- <a href='https://datascienceschool.net/view-notebook/dd6a7633d69f401bb00409b9ae8806e8/' target="_blank">expectation and transform of random variable</a>
- <a href='https://datascienceschool.net/view-notebook/b9dcd289a49546ffacfdc5f5bc9a2fc0/' target="_blank">(standard) variance</a>
- <a href='https://datascienceschool.net/view-notebook/e5c379559a4a4fe9a9d8eeace69da425/' target="_blank">multivariate random variables</a>
- <a href='https://datascienceschool.net/view-notebook/4cab41c0d9cd4eafaff8a45f590592c5/' target="_blank">covariance and correlation</a>
- <a href='https://datascienceschool.net/view-notebook/88867007c4cf4f5b96e3d07afa83206f/' target="_blank">conditional expectation, prediction</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>
