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
sns.distplot(x, rug=True, bins=10, kde=False)
```
![download (6)](https://user-images.githubusercontent.com/52376448/66740682-b9c61f00-eeae-11e9-863a-98d5efe53b54.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<br><br><br>

<hr class="division2">

## **Arithmetic mean(average), median**

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4e3313161244f8ab61d897fb6e5fbf6647e1d5f5" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:39.094ex; height:7.509ex;" alt="{\displaystyle {\bar {x}}={\frac {1}{n}}\left(\sum _{i=1}^{n}{x_{i}}\right)={\frac {x_{1}+x_{2}+\cdots +x_{n}}{n}}}">

> If the distribution is symmetrical with respect to the sample mean, the sample median is equal to the sample mean.<br>
> If the distribution is symmetrical and has only one extreme value(uni-modal), the sample peak value is the same as the sample mean.<br>
> When data is added that makes the symmetric distribution asymmetrical, the sample mean is most affected and the sample mode is least affected.<br>



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
plt.axvline(x=165.76, ls="--", c="r", linewidth=2, label="sample mean")
plt.axvline(x=165, ls="--", c="y", linewidth=2, label="sample median")
plt.legend()
```
![download (5)](https://user-images.githubusercontent.com/52376448/66740445-33114200-eeae-11e9-80a8-70dffe9ed062.png)
<hr class='division3'>
</details>


<br><br><br>
<hr class="division2">

## **Variance and Standard deviation**
<span class="frame3">Variance</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2577f2b00102ca127d8867a756b85e17d97eab5f" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:27.977ex; height:6.843ex;" alt="\operatorname {Var} (X)=\sum _{i=1}^{n}p_{i}\cdot (x_{i}-\mu )^{2},">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/bd2e4161a60cffb12e219f479b2bbbb2ebfab48f" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:69.479ex; height:7.509ex;" alt="{\displaystyle s^{2}={\frac {n}{n-1}}\sigma _{Y}^{2}={\frac {n}{n-1}}\left({\frac {1}{n}}\sum _{i=1}^{n}\left(Y_{i}-{\overline {Y}}\right)^{2}\right)={\frac {1}{n-1}}\sum _{i=1}^{n}\left(Y_{i}-{\overline {Y}}\right)^{2}}">
```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
              
# np.var(x)         : for population variance(biased sample variance)
# np.var(x, ddof=1) : for unbiased sample variance
np.var(x), np.var(x, ddof=1)
```
<span class="jb-medium">(115.23224852071006, 119.84153846153846)</span>
```python
# x.var()       : for population variance(biased sample variance)
# x.var(ddof=1) : for unbiased sample variance
x.var(), x.var(ddof=1)
```
<span class="jb-medium">(115.23224852071006, 119.84153846153846)</span>
<br><br><br><br>

<span class="frame3">Standard deviation</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4c98cfcd7dc201f65aa452ed555666f1b23bf477" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:44.119ex; height:8.009ex;" alt="\sigma ={\sqrt {\sum _{i=1}^{N}p_{i}(x_{i}-\mu )^{2}}},{\rm {\ \ where\ \ }}\mu =\sum _{i=1}^{N}p_{i}x_{i}.">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/32b79f6c8b96d3c5f2178f0fb759b77c851d3500" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.005ex; width:34.698ex; height:7.676ex;" alt="{\displaystyle s=\pm {\sqrt {\frac {\Sigma w(x-{\overline {x}})^{2}}{n-1}}}=\pm {\sqrt {\frac {\Sigma w\nu ^{2}}{n-1}}}}">

```python
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
              
# np.std(x)         : for population standard deviation(biased sample standard deviation)
# np.std(x, ddof=1) : for unbiased sample standard deviation
np.std(x), np.std(x, ddof=1)
```
<span class="jb-medium">(10.734628476137871, 10.947216014199157)</span>
```python
# x.std()       : for population standard deviation(biased sample standard deviation)
# x.std(ddof=1) : for unbiased sample standard deviation
x.std(), x.std(ddof=1)
```
<span class="jb-medium">(10.734628476137871, 10.947216014199157)</span><br>
<br><br><br><br>

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

x.mean(), x.var(), x.std()
```
```
(165.76, 224.16240000000002, 14.972053967308561)
```
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

plt.scatter(range(len(x)), x)
plt.plot([0, 100], [165.76, 165.76], c ="k", label="mean")
plt.annotate('',
             xytext = (50, 165.76),
             xy = (50, 180.73205396730856),
             arrowprops = {'facecolor' : 'red', 
                           'edgecolor' : 'red', 
                           'shrink' : 0.05})
plt.annotate('',
             xytext = (50, 165.76),
             xy = (50, 150.78794603269142),
             arrowprops = {'facecolor' : 'red',
                           'edgecolor' : 'red',
                           'shrink' : 0.05 })
plt.text(55, 170, 'Standard Deviation', c='red')
plt.legend()
```
![download (12)](https://user-images.githubusercontent.com/52376448/66714305-29290980-edf0-11e9-8362-df630c5a1a5d.png)
<hr class='division3'>
</details>

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

## **Skewness and Kurtosis**
<span class="frame3">Pearson's moment coefficient of skewness</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7c090ae95df78e4fe8d3984b5a67c0238fc95491" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.505ex; width:55.95ex; height:7.843ex;" alt="{\displaystyle \gamma _{1}=\operatorname {E} \left[\left({\frac {X-\mu }{\sigma }}\right)^{3}\right]={\frac {\mu _{3}}{\sigma ^{3}}}={\frac {\operatorname {E} \left[(X-\mu )^{3}\right]}{(\operatorname {E} \left[(X-\mu )^{2}\right])^{3/2}}}={\frac {\kappa _{3}}{\kappa _{2}^{3/2}}}}">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9fa2a6ebc4d719d3f4f3e3cad52120d578551ce1" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -12.505ex; width:41.59ex; height:26.176ex;" alt="{\displaystyle {\begin{aligned}\gamma _{1}&amp;=\operatorname {E} \left[\left({\frac {X-\mu }{\sigma }}\right)^{3}\right]\\&amp;={\frac {\operatorname {E} [X^{3}]-3\mu \operatorname {E} [X^{2}]+3\mu ^{2}\operatorname {E} [X]-\mu ^{3}}{\sigma ^{3}}}\\&amp;={\frac {\operatorname {E} [X^{3}]-3\mu (\operatorname {E} [X^{2}]-\mu \operatorname {E} [X])-\mu ^{3}}{\sigma ^{3}}}\\&amp;={\frac {\operatorname {E} [X^{3}]-3\mu \sigma ^{2}-\mu ^{3}}{\sigma ^{3}}}.\end{aligned}}}">
```python

```
<span class="jb-medium"></span>
<br><br><br>



<span class="frame3">Sample skewness</span>
![캡처](https://user-images.githubusercontent.com/52376448/66758607-8fd62200-eed9-11e9-9791-79b1ca73dd30.JPG)
```python
from scipy import stats
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
stats.skew(x)
```
<span class="jb-medium">-0.4762339485461929</span>



<br><br><br>
<span class="frame3">Kurtosis</span>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/89bc1d05929bb9c2c62cb88e895eda2733a7b2d6" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:52.914ex; height:7.509ex;" alt="{\displaystyle \operatorname {Kurt} [X]=\operatorname {E} \left[\left({\frac {X-\mu }{\sigma }}\right)^{4}\right]={\frac {\mu _{4}}{\sigma ^{4}}}={\frac {\operatorname {E} [(X-\mu )^{4}]}{(\operatorname {E} [(X-\mu )^{2}])^{2}}},}">
```python
from scipy import stats
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
stats.kurtosis(x)
```
<span class="jb-medium">0.37443381660038977</span>



<br><br><br>
<hr class="division2">

## **Moment**

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c1bbc8f08ad7d8f9e00b3bbc27767cdae4d622d7" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.505ex; width:26.831ex; height:6.009ex;" alt="{\displaystyle \mu _{n}=\int _{-\infty }^{\infty }(x-c)^{n}\,f(x)\,\mathrm {d} x.}">
```python
from scipy import stats
import numpy as np

x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
              2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
stats.moment(x, 1), stats.moment(x, 2), stats.moment(x, 3), stats.moment(x, 4)
```
<span class="jb-medium">(0.0, 115.23224852071006, -589.0896677287208, 44807.32190968453)</span>


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
