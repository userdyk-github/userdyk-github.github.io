---
layout : post
title : MATH05, Distribution
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

<div style="color:black; font-size: 80%; text-align: center;">
  $$Parameters(loc,\ scale)\ :\ y=\frac{x-loc}{scale}$$
  $$Parameter(df)\ :\ a\ shape\ parameter$$
</div>

---

## **Discrete distribution**

> Objective : 
>
> When something is important enough, you do it even if the odds are not in your favor.

---

### Geometric

---

### Poisson

---

### Logarithmic

---

### NegativeBinomial

---

### YuleSimon

---

### Zeta




<hr class="division2">

## **Continous distribution**

> Objective : 
>
> When something is important enough, you do it even if the odds are not in your favor.

---

### Beta

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}{\Gamma(a) \Gamma(b)}\qquad for\ 0\le x \le 1,\ \ \ \ a>0,\ b>0$$
</div>

`Main code`
```python
# [] : 
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

# [] : 
X = stats.beta(a=5, b=0.1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100)

# [] : 
fig,ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.beta, kde=False, ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download](https://user-images.githubusercontent.com/52376448/66644314-9bb4b080-ec5b-11e9-8e18-6b1ec4a3348e.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.beta(a=5, b=0.1,loc=1, scale=1)
sns.distplot(X.rvs(100), fit=stats.beta, kde=False)
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download](https://user-images.githubusercontent.com/52376448/66646479-32d03700-ec61-11e9-9b60-a3e9b6dc8964.png)
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.beta(a=5, b=0.1,loc=1, scale=1)
sns.distplot(X.rvs(100))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (1)](https://user-images.githubusercontent.com/52376448/66646501-48ddf780-ec61-11e9-809b-f356675eae4f.png)
<hr class='division3_1'>
</details>
<br>
`PDF`
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

X = stats.beta(a=5, b=0.1,loc=1, scale=1)
x = np.linspace(*X.interval(0.999), num=100)

plt.plot(x, X.pdf(x))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (2)](https://user-images.githubusercontent.com/52376448/66646528-5d21f480-ec61-11e9-9c71-1ff932e3d536.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>


  
<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | a, b, loc, scale      | $$\xrightarrow{stats.beta}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |



<br><br><br>

---

### ChiSquared

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}x^{k/2-1} \exp \left( -x/2 \right)\qquad for\ x>0\ and\ k>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.chi2(df=1,loc=1, scale=1)      # df : parameter k
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.chi2, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (1)](https://user-images.githubusercontent.com/52376448/66644359-b555f800-ec5b-11e9-83a9-1e9023d324ff.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.chi2(df=1,loc=1, scale=1)      # df : parameter k
sns.distplot(X.rvs(100), fit=stats.chi2, kde=False)
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download](https://user-images.githubusercontent.com/52376448/66646937-898a4080-ec62-11e9-897c-571bb0231320.png)
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.chi2(df=1,loc=1, scale=1)      # df : parameter k
sns.distplot(X.rvs(100))  
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (1)](https://user-images.githubusercontent.com/52376448/66646939-8a22d700-ec62-11e9-886b-6d5e5fe0726f.png)
<hr class='division3_1'>
</details>
<br>
`PDF`
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

X = stats.chi2(df=1,loc=1, scale=1)      # df : parameter k
x = np.linspace(*X.interval(0.999), num=100) 

plt.plot(x, X.pdf(x))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (2)](https://user-images.githubusercontent.com/52376448/66646940-8a22d700-ec62-11e9-8670-b8c58db34e91.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | df, loc, scale        | $$\xrightarrow{stats.chi2}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |


<br><br><br>

---


### Exponential


<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \lambda*\exp(-\lambda*x)\qquad for\ x\ge0$$
  $$scale\ =\ 1/\lambda$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.expon(scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig, ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.expon, kde=False, ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (2)](https://user-images.githubusercontent.com/52376448/66644397-cdc61280-ec5b-11e9-8542-a2758b4d07a6.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.expon(scale=1) 
sns.distplot(X.rvs(100), fit=stats.expon, kde=False)
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download](https://user-images.githubusercontent.com/52376448/66647183-2b119200-ec63-11e9-92f6-5f48f4e846d7.png)
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.expon(scale=1) 
sns.distplot(X.rvs(100))  
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (1)](https://user-images.githubusercontent.com/52376448/66647184-2baa2880-ec63-11e9-844b-b142c98f123e.png)
<hr class='division3_1'>
</details>
<br>
`PDF`
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

X = stats.expon(scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

plt.plot(x, X.pdf(x))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (2)](https://user-images.githubusercontent.com/52376448/66647185-2baa2880-ec63-11e9-880d-a9c531d569de.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | scale                 | $$\xrightarrow{stats.expon}$$  | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |



<br><br><br>

---

### FDistribution

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}\qquad for\ x>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.f(dfn=1, dfd=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.f, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (3)](https://user-images.githubusercontent.com/52376448/66644437-e6362d00-ec5b-11e9-8056-ee5151681803.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.f(dfn=1, dfd=1) 
sns.distplot(X.rvs(100), fit=stats.f, kde=False)
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download](https://user-images.githubusercontent.com/52376448/66648063-8b093800-ec65-11e9-90b6-cbb18b8b4ecd.png)
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.f(dfn=1, dfd=1) 
sns.distplot(X.rvs(100))  
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (1)](https://user-images.githubusercontent.com/52376448/66648064-8b093800-ec65-11e9-8254-928b09e9d6e7.png)
<hr class='division3_1'>
</details>
<br>
`PDF`
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

X = stats.f(dfn=1, dfd=1) 
x = np.linspace(*X.interval(0.999), num=100) 

plt.plot(x, X.pdf(x))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (2)](https://user-images.githubusercontent.com/52376448/66648066-8ba1ce80-ec65-11e9-88cb-39aa70136738.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | dfn, dfd              | $$\xrightarrow{stats.f}$$      | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |




<br><br><br>

---

### FisherZ

<br><br><br>

---

### Gamma

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a) = \frac{x^{a-1} \exp(-x)}{\Gamma(a)}\qquad for\ x\ge0,\ a>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.gamma(a=1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.gamma, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (4)](https://user-images.githubusercontent.com/52376448/66644491-0534bf00-ec5c-11e9-9b78-3a9fab9b7676.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.gamma(a=1,loc=1, scale=1) 
sns.distplot(X.rvs(100), fit=stats.gamma, kde=False)
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download](https://user-images.githubusercontent.com/52376448/66648200-e509fd80-ec65-11e9-8e20-ea0dcba3ac89.png)
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python
from scipy import stats
import seaborn as sns

X = stats.gamma(a=1,loc=1, scale=1) 
sns.distplot(X.rvs(100))  
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (1)](https://user-images.githubusercontent.com/52376448/66648202-e509fd80-ec65-11e9-99f8-6e069e1337b9.png)
<hr class='division3_1'>
</details>
<br>
`PDF`
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

X = stats.gamma(a=1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

plt.plot(x, X.pdf(x))
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
![download (2)](https://user-images.githubusercontent.com/52376448/66648203-e5a29400-ec65-11e9-933c-e313fb22f0b5.png)
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | a, loc, scale         | $$\xrightarrow{stats.gamma}$$  | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |


<br><br><br>

### Laplace

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{1}{2} \exp(-|x|)\qquad for\ a\ real\ number\ x$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.laplace() 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.laplace, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (5)](https://user-images.githubusercontent.com/52376448/66644532-1da4d980-ec5c-11e9-8705-a00ff7def32a.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                        | OUTPUT           |
|:---|:----------------------|:--------------------------------|:-----------------|
| 1  |                       | $$\xrightarrow{stats.laplace}$$ | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$         | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$       | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$  | graph : fitting  |





<br><br><br>

---

### Logistic

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{\exp(-x)}{(1+\exp(-x))^2}$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.logistic(loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.logistic, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (6)](https://user-images.githubusercontent.com/52376448/66644555-331a0380-ec5c-11e9-8d5d-7d455be77a51.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                           | OUTPUT           |
|:---|:----------------------|:-----------------------------------|:-----------------|
| 1  | loc, scale            | $$\xrightarrow{stats.logistic}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$            | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$          | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$     | graph : fitting  |



<br><br><br>

---

### LogNormal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; s) = \frac{1}{s x \sqrt{2\pi}}\exp\left(-\frac{\log^2(x)}{2s^2}\right)\qquad for\ x>0,\ s>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.lognorm(s=1, loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1, figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.lognorm, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (7)](https://user-images.githubusercontent.com/52376448/66644599-49c05a80-ec5c-11e9-930c-d5785220f0ff.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                          | OUTPUT           |
|:---|:----------------------|:----------------------------------|:-----------------|
| 1  | s, loc, scale         | $$\xrightarrow{stats.lognorm}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$           | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$         | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$    | graph : fitting  |




<br><br><br>

---

### Normal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x;\mu,\sigma) = \frac{\exp(-(x-\mu)^2/2)}{\sqrt{2\pi\sigma}}\qquad for\ a\ real\ number\ x$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.norm(loc=100, scale=10) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.norm, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (8)](https://user-images.githubusercontent.com/52376448/66644638-6197de80-ec5c-11e9-85d9-b6ebdd9de800.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | loc, scale            | $$\xrightarrow{stats.norm}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |




<br><br><br>

---

### StudentT

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu)}(1+x^2/\nu)^{-(\nu+1)/2}$$
  $$ where\ x\ is\ a\ real\ number\ and\ degrees\ of\ freedom\ parameter\ \nu>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.t(df=4,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig,ax= plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.t, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (9)](https://user-images.githubusercontent.com/52376448/66644670-783e3580-ec5c-11e9-9ca1-e5d58dda59c3.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                       | OUTPUT           |
|:---|:----------------------|:-------------------------------|:-----------------|
| 1  | df, loc, scale        | $$\xrightarrow{stats.t}$$      | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$        | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$      | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$ | graph : fitting  |



<br><br><br>

---

### Uniform

<div style="color:black; font-size: 80%; text-align: center;">
  $$ [𝑙𝑜𝑐,𝑙𝑜𝑐 + 𝑠𝑐𝑎𝑙𝑒] $$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.uniform(loc=2, scale=10) 
x = np.linspace(*X.interval(0.999), num=100) 

fig, ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.uniform, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (10)](https://user-images.githubusercontent.com/52376448/66644705-91df7d00-ec5c-11e9-988f-d1d8c10ec0a9.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                          | OUTPUT           |
|:---|:----------------------|:----------------------------------|:-----------------|
| 1  | loc, scale            | $$\xrightarrow{stats.uniform}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$           | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$         | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$    | graph : fitting  |




<br><br><br>

---

### Weibull

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ min\ :\ f(x; c) = c x^{c-1} \exp(-x^c)\qquad for\ x\ge0,\ c>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.weibull_min(c=1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig, ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.weibull_min, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (11)](https://user-images.githubusercontent.com/52376448/66644739-a885d400-ec5c-11e9-89db-8792deacb1b2.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                               | OUTPUT           |
|:---|:----------------------|:---------------------------------------|:-----------------|
| 1  | c, loc, scale         | $$\xrightarrow{stats.weibull\_min}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$                | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$              | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$         | graph : fitting  |




<br><br><br>

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ max\ :\ f(x; c) = c (-x)^{c-1} \exp(-(-x)^c)\qquad for\ x<0,\ c>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.weibull_max(c=1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig, ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.weibull_max, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (12)](https://user-images.githubusercontent.com/52376448/66644780-c05d5800-ec5c-11e9-8652-11c53f8fc38c.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                               | OUTPUT           |
|:---|:----------------------|:---------------------------------------|:-----------------|
| 1  | c, loc, scale         | $$\xrightarrow{stats.weibull\_max}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$                | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$              | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$         | graph : fitting  |




<br><br><br>

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ double\ :\ f(x; c) = c / 2 |x|^{c-1} \exp(-|x|^c)\qquad for\ a\ real\ numbers,\ x\ and\ c>0$$
</div>

`Main code`
```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

X = stats.dweibull(c=1,loc=1, scale=1) 
x = np.linspace(*X.interval(0.999), num=100) 

fig, ax = plt.subplots(3,1,figsize=(10, 8))

sns.distplot(X.rvs(100), fit=stats.dweibull, kde=False,ax=ax[0])
sns.distplot(X.rvs(100),ax=ax[1])  
ax[2].plot(x, X.pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![download (13)](https://user-images.githubusercontent.com/52376448/66644816-d79c4580-ec5c-11e9-88ff-b6b7991dc19f.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
`model fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`data fitting`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<br>
`PDF`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
<hr class='division3_1'>
</details>
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">Sub-code</summary>
<hr class='division3'>
`Sub-code`
```python
# Random variables
print(X.rvs(size=10, random_state=None))
```

```python
# Probability density function
x = np.linspace(*X.interval(0.999), num=5) 
print(X.pdf(x))
```

```python
# Log of the probability density function.
print(X.logpdf(x))
```

```python
# Cumulative distribution function.
print(X.cdf(x))
```

```python
# Log of the cumulative distribution function.
print(X.logcdf(x))
```

```python
# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
print(X.sf(x))
```

```python
# Log of the survival function.
print(X.logsf(x))
```

```python
# Percent point function (inverse of cdf — percentiles).
q = np.linspace(0.01,0.99, num=5) 
print(X.ppf(q))
```

```python
# Inverse survival function (inverse of sf).
print(X.isf(q))
```

```python
# Non-central moment of order n
for n in [1,2]:
    print(X.moment(n))    
```

```python
# Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
print(X.stats(moments='mvsk'))
```

```python
# (Differential) entropy of the RV.
print(X.entropy())
```

```python
# Parameter estimates for generic data.
data = X.rvs(size=10, random_state=None)

# loc : mean, scale : standard deviation
print(stats.beta.fit(data, 1, 2, loc=0, scale=1))
```

```python
# Median
print(X.median())
```

```python
# Mean
print(X.mean())
```

```python
# Variance
print(X.var())
```

```python
# Standard deviation
print(X.std())
```

```python
# Interval
print(X.interval(0.05))
```
<hr class='division3'>
</details>

<div>
$$ random\ variable\ :\ X \xrightarrow{function} distribution\ :\ X.pdf(x) $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                 | FUNCTION                           | OUTPUT           |
|:---|:----------------------|:-----------------------------------|:-----------------|
| 1  | c, loc, scale         | $$\xrightarrow{stats.dweibull}$$   | X                |
| 2  | x                     | $$\xrightarrow{X.pdf}$$            | X.pdf(x)         |
| 3  | x, X.pdf(x)           | $$\xrightarrow{ax.plot}$$          | graph : X.pdf(x) |
| 4  | X.rvs(#)              | $$\xrightarrow{sns.distplot}$$     | graph : fitting  |




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

