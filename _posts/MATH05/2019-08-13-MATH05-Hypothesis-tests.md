---
layout : post
title : MATH05, Hypothesis tests
categories: [MATH05]
comments : true
tags : [MATH05]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)  ｜ <a href="https://userdyk-github.github.io/math05/MATH05-Contents.html" target="_blank">Statistics</a><br>
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
![download](https://user-images.githubusercontent.com/52376448/66982315-ba4df800-f0f0-11e9-8157-89b03fda950b.jpg)

<br><br><br>
## **Binomial Test**

```python
from scipy import stats
import numpy as np

N, mu_0 = 10, 0.5
np.random.seed(0)
x = stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)
stats.binom_test(n, N)
```
<span class="jb-medoum">0.3437499999999999</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>
```python
from scipy import stats
import numpy as np

N, mu_0 = 100, 0.5
np.random.seed(0)
x = stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)
stats.binom_test(n, N)
```
<span class="jb-medoum">0.9204107626128206</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Normality Test**
### ***Shapiro–Wilk test***
### ***Anderson–Darling test***

```python
from scipy import stats
import numpy as np


np.random.seed(0)
N1, N2 = 50, 100

x1 = stats.norm(0, 1).rvs(N1)
x2 = stats.norm(0.5, 1.5).rvs(N2)

stats.ks_2samp(x1, x2)
```
<span class="jb-medoum">Ks_2sampResult(statistic=0.23000000000000004, pvalue=0.049516112814422863)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
N1, N2 = 50, 100

x1 = stats.norm(0, 1).rvs(N1)
x2 = stats.norm(0.5, 1.5).rvs(N2)

ax = sns.distplot(x1, kde=False, fit=stats.norm, label="1st dataset")
ax = sns.distplot(x2, kde=False, fit=stats.norm, label="2rd dataset")
ax.lines[0].set_linestyle(":")
plt.legend()
plt.show()
```
![download (4)](https://user-images.githubusercontent.com/52376448/66739719-40c5c800-eeac-11e9-9912-ac6206d67193.png)
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">


## **Chi-squared Test**
<span class="frame3">goodness of fit test</span>
```python
from scipy import stats
import numpy as np

N, K = 10, 4
mu_0 = np.ones(K)/K
np.random.seed(0)
x = np.random.choice(K, N, p=mu_0)
n = np.bincount(x, minlength=K)
stats.chisquare(n)
```
<span class="jb-medoum">Power_divergenceResult(statistic=5.199999999999999, pvalue=0.157724450396663)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>
<br><br><br>
<span class="frame3">test of independence</span>
```python
from scipy import stats
import numpy as np

obs = np.array([[5, 15], [10, 20]])
stats.chi2_contingency(obs)
```
<span class="jb-medoum">(0.0992063492063492, 0.7527841326498471, 1, array([[ 6., 14.], [ 9., 21.]]))</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">


## **One-sample z-Test**
![test_z](https://user-images.githubusercontent.com/52376448/66703410-507aca80-ed4d-11e9-9438-0e2d0024579c.JPG)
```python
from scipy import stats
import numpy as np

def ztest_1samp(x, sigma2=1, mu=0):
    z = (x.mean() - mu) / np.sqrt(sigma2/len(x))
    return z, 2 * stats.norm().sf(np.abs(z))

N, mu_0 = 10, 0
np.random.seed(0)
x = stats.norm(mu_0).rvs(N)
ztest_1samp(x)
```
<span class="jb-medoum">(2.3338341854824276, 0.019604406021683538)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **One-sample t-Test**
```python
from scipy import stats
import numpy as np

N, mu_0 = 10, 0
np.random.seed(0)
x = stats.norm(mu_0).rvs(N)
stats.ttest_1samp(x, popmean=0)
```
<span class="jb-medoum">Ttest_1sampResult(statistic=2.28943967238967, pvalue=0.04781846490857058)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Independent two-sample t-Test**

<span class="frame3">Equal sample sizes, equal variance</span>
![test_t_for_mean_under_unknown_different_variances_of_two_group_with_large_samples](https://user-images.githubusercontent.com/52376448/66703408-4fe23400-ed4d-11e9-9c49-c07d5dd666f4.JPG)
![test_t_for_mean_under_unknown_equivalent_variances_of_two_group_with_small_samples](https://user-images.githubusercontent.com/52376448/66703409-4fe23400-ed4d-11e9-86d9-37a1e2d54ee7.JPG)
```python
from scipy import stats
np.random.seed(12345678)

rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)

stats.ttest_ind(rvs1,rvs2)
stats.ttest_ind(rvs1,rvs2, equal_var = False)
```
```
(0.26833823296239279, 0.78849443369564776)
(0.26833823296239279, 0.78849452749500748)
```
<span class="jb-medoum"></span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python

```

<hr class='division3'>
</details>
<br><br><br>
<span class="frame3">Equal or unequal sample sizes, equal variance</span>
```python
from scipy import stats
import numpy as np

N_1, mu_1, sigma_1 = 50, 0, 1
N_2, mu_2, sigma_2 = 100, 0.5, 1

np.random.seed(0)
x1 = stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = stats.norm(mu_2, sigma_2).rvs(N_2)
stats.ttest_ind(x1, x2, equal_var=True)
```
<span class="jb-medoum">Ttest_indResult(statistic=-2.6826951236616963, pvalue=0.008133970915722658)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">	Equal or unequal sample sizes, unequal variances</span>
```python
from scipy import stats
import numpy as np

N_1, mu_1, sigma_1 = 10, 0, 1
N_2, mu_2, sigma_2 = 10, 0.5, 1

np.random.seed(0)
x1 = stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = stats.norm(mu_2, sigma_2).rvs(N_2)
stats.ttest_ind(x1, x2, equal_var=False)
```
<span class="jb-medoum">Ttest_indResult(statistic=-0.4139968526988655, pvalue=0.6843504889824326)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N_1, mu_1, sigma_1 = 10, 0, 1
N_2, mu_2, sigma_2 = 10, 0.5, 1

np.random.seed(0)
x1 = stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = stats.norm(mu_2, sigma_2).rvs(N_2)

ax = sns.distplot(x1, kde=False, fit=stats.norm, label="1st dataset")
ax = sns.distplot(x2, kde=False, fit=stats.norm, label="2nd dataset")
ax.lines[0].set_linestyle(":")
plt.legend()
plt.show()
```
![download](https://user-images.githubusercontent.com/52376448/66738651-f4798880-eea9-11e9-8f26-b1194565dd8c.png)
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Paired two-sample t-Test**
![test_t_for_mean_of_paired_sample](https://user-images.githubusercontent.com/52376448/66703407-4fe23400-ed4d-11e9-96d5-3155b3e80002.JPG)
```python
from scipy import stats
import numpy as np

N = 5
mu_1, mu_2 = 0, 0.4

np.random.seed(1)
x1 = stats.norm(mu_1).rvs(N)
x2 = x1 + stats.norm(mu_2, 0.1).rvs(N)

stats.ttest_rel(x1, x2)
```
<span class="jb-medoum">Ttest_relResult(statistic=-5.662482449248929, pvalue=0.0047953456833781305)</span>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N = 5
mu_1, mu_2 = 0, 0.4

np.random.seed(1)
x1 = stats.norm(mu_1).rvs(N)
x2 = x1 + stats.norm(mu_2, 0.1).rvs(N)

ax = sns.distplot(x1, kde=False, fit=stats.norm, label="1st dataset")
ax = sns.distplot(x2, kde=False, fit=stats.norm, label="2nd dataset")
ax.lines[0].set_linestyle(":")
plt.legend()
plt.show()
```
![download (1)](https://user-images.githubusercontent.com/52376448/66739508-cdbc5180-eeab-11e9-8bae-a028ea23c369.png)
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **Equal-variance Test**
![test_F_for_variance_rate_of_two_group](https://user-images.githubusercontent.com/52376448/66703406-4fe23400-ed4d-11e9-99f6-cb0906916be8.JPG)
```python
from scipy import stats
import numpy as np

N1, sigma_1 = 100, 1
N2, sigma_2 = 100, 1.2

np.random.seed(0)
x1 = stats.norm(0, sigma_1).rvs(N1)
x2 = stats.norm(0, sigma_2).rvs(N2)

print(stats.bartlett(x1, x2))
print(stats.fligner(x1, x2))
print(stats.levene(x1, x2))
```
<div class="jb-medoum">BartlettResult(statistic=4.253473837232266, pvalue=0.039170128783651344)<br>
FlignerResult(statistic=7.224841990409457, pvalue=0.007190150106748367)<br>
LeveneResult(statistic=7.680708947679437, pvalue=0.0061135154970207925)<br></div>
<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
```python
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N1, sigma_1 = 100, 1
N2, sigma_2 = 100, 1.2

np.random.seed(0)
x1 = stats.norm(0, sigma_1).rvs(N1)
x2 = stats.norm(0, sigma_2).rvs(N2)

ax = sns.distplot(x1, kde=False, fit=stats.norm, label="1st dataset")
ax = sns.distplot(x2, kde=False, fit=stats.norm, label="2rd dataset")
ax.lines[0].set_linestyle(":")
plt.legend()
plt.show()
```
![download (2)](https://user-images.githubusercontent.com/52376448/66739569-fa706900-eeab-11e9-8d9a-188c1c4acb7b.png)

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
- [basic hypothesis tests with Excel][1]
- <a href='https://datascienceschool.net/view-notebook/37a330dfc8de45e9ba475cbbd201ab53/' target="_blank">statistical hypothesis testing and p-value</a>
- <a href='https://datascienceschool.net/view-notebook/14bde0cc05514b2cae2088805ef9ed52/' target="_blank">parameter testing</a>

---

[1]:{{ site.url }}/download/MATH05/test_with_excel.zip


<details markdown="1">
<summary class='jb-small' style="color:blue">Visualization</summary>
<hr class='division3'>
<hr class='division3'>
</details>
