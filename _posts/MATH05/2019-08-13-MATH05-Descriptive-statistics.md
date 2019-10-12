---
layout : post
title : MATH05, Descriptive statistics
categories: [MATH05]
comments : true
tags : [MATH05]
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

## **Review : Descriptive statistics with Excel**

![descriptive_statistics](https://user-images.githubusercontent.com/52376448/66703405-4f499d80-ed4d-11e9-8cc7-c33dbbfec77c.JPG)
![covariance_analysis](https://user-images.githubusercontent.com/52376448/66703416-51136100-ed4d-11e9-85c1-1861caf05537.JPG)
![correlation_analysis](https://user-images.githubusercontent.com/52376448/66703414-51136100-ed4d-11e9-8acd-093dec33923f.JPG)

<br><br><br>
<hr class="division2">

## **Count**

<details markdown="1">
<summary class='jb-small' style="color:blue">Generating random number</summary>
<hr class='division3'>
```python
import numpy as np

np.random.seed(2019)
rv = np.random.RandomState(2019)
np.round(rv.normal(168,15,(100,)))
```
```
array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
       163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
       120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
       171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
       175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
       173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
       158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
       170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
       152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
       148.])
```
<hr class='division3'>
</details>

```python
import numpy as np

x = np.array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
              163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
              120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
              171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
              175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
              173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
              158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
              170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
              152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
              148.])
len(x)
```
<span class="jb-medium">100</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
              163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
              120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
              171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
              175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
              173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
              158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
              170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
              152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
              148.])

sns.set();
plt.hist(x)
```
![download (2)](https://user-images.githubusercontent.com/52376448/66706005-c93b5000-ed68-11e9-8490-3e4dbdd16a17.png)
<details markdown="1">
<summary class='jb-small' style="color:red">Another method</summary>
<hr class='division3_1'>
```python
import seaborn as sns
import numpy as np

x = np.array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
              163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
              120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
              171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
              175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
              173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
              158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
              170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
              152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
              148.])

sns.set();
sns.distplot(x, bins=10, kde=False)
```
![download](https://user-images.githubusercontent.com/52376448/66705895-93e23280-ed67-11e9-8d22-e2b2285e740f.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<br><br><br>

<hr class="division2">

## **Mean(average), median**

```python
import numpy as np

x = np.array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
              163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
              120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
              171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
              175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
              173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
              158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
              170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
              152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
              148.])
np.mean(x), np.median(x)
```
<span class="jb-medium">(165.76, 165.0)</span>
```python
x.mean()
```
<span class="jb-medium">165.76</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([165., 180., 190., 188., 163., 178., 177., 172., 164., 182., 143.,
              163., 168., 160., 172., 165., 208., 175., 181., 160., 154., 169.,
              120., 184., 180., 175., 174., 175., 160., 155., 156., 161., 184.,
              171., 150., 154., 153., 177., 184., 172., 156., 153., 145., 150.,
              175., 165., 190., 156., 196., 161., 185., 159., 153., 155., 173.,
              173., 191., 162., 152., 158., 190., 136., 171., 173., 146., 158.,
              158., 159., 169., 145., 193., 178., 160., 153., 142., 143., 172.,
              170., 130., 165., 177., 190., 164., 167., 172., 160., 184., 158.,
              152., 175., 158., 156., 171., 164., 165., 160., 162., 140., 172.,
              148.])

sns.set();
plt.hist(x)
plt.plot([165.76, 165.76], [0, 30], label="mean")
plt.plot([165.0, 165.0], [0, 30], label="median")
plt.legend()
```

<hr class='division3'>
</details>
<br><br><br>
<hr class="division2">

## **Variance**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.var(x), np.var(x, ddof=1)
```
<span class="jb-medium">(115.23224852071006, 119.84153846153846)</span>
```python
x.var(), x.var(ddof=1)
```
<span class="jb-medium">(115.23224852071006, 119.84153846153846)</span>
<br><br><br>
<hr class="division2">

## **Standard deviation**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.std(x), np.std(x, ddof=1)
```
<span class="jb-medium">(10.734628476137871, 10.947216014199157)</span>
```python
x.std(), x.std(ddof=1) 
```
<span class="jb-medium">(10.734628476137871, 10.947216014199157)</span>
<br><br><br>
<hr class="division2">

## **Maximum, minimum**

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
np.max(x), np.min(x)
```
<span class="jb-medium">(23, -24)</span>
```python
x.max(), x.min()
```
<span class="jb-medium">(23, -24)</span>
<br><br><br>
<hr class="division2">


## **Quartile**

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
