---
layout : post
title : MATH05, Distribution
categories: [MATH05]
comments : true
tags : [MATH05]
---

List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/'>post1</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

---

<div style="color:black; font-size: 80%; text-align: center;">
  $$Parameters(loc,\ scale)\ :\ y=\frac{x-loc}{scale}$$
  $$Parameter(df)\ :\ a\ shape\ parameter$$
</div>

---

## **Discrete distribution**

> Objective : 
>
> When something is important enough, you do it even if the odds are not in your favor.

### Geometric

### Poisson

### Logarithmic

### NegativeBinomial

### YuleSimon

### Zeta




---

## **Continous distribution**

> Objective : 
>
> When something is important enough, you do it even if the odds are not in your favor.

### Beta

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}{\Gamma(a) \Gamma(b)}\qquad for\ 0\le x \le 1,\ \ \ \ a>0,\ b>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

<span class='"jb-medium"'>
  $$ hello $$
</span>
  
<div align="left" style="color:black; font-size: 80%;">
$$ random\ variable\ X \xrightarrow{function} distribution $$
<div class='frame2'> </div>
$$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
$$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function\ :\ X.pdf(x) $$
$$ x,\ X.pdf(x) \xrightarrow{ax.plot} visuallization\ for\ X.pdf(x)\ :\ ax.plot(x,X.pdf(x)) $$
$$ sample\ data\ fitting :\ X.rvs(the\ number),\ fit=stats.beta \xrightarrow{sns.distplot} visuallization\ for\ fit\ curve $$
</div>



<br><br><br>

### ChiSquared

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}x^{k/2-1} \exp \left( -x/2 \right)\qquad for\ x>0\ and\ k>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ df,\ loc,\ scale\ \xrightarrow{stats.chi2} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>

<br><br><br>


### Exponential


<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \lambda*\exp(-\lambda*x)\qquad for\ x\ge0$$
  $$scale\ =\ 1/\lambda$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### FDistribution

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}\qquad for\ x>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### FisherZ

<br><br><br>

### Gamma

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a) = \frac{x^{a-1} \exp(-x)}{\Gamma(a)}\qquad for\ x\ge0,\ a>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>

<br><br><br>

### Laplace

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{1}{2} \exp(-|x|)\qquad for\ a\ real\ number\ x$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>




<br><br><br>

### Logistic

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{\exp(-x)}{(1+\exp(-x))^2}$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### LogNormal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; s) = \frac{1}{s x \sqrt{2\pi}}\exp\left(-\frac{\log^2(x)}{2s^2}\right)\qquad for\ x>0,\ s>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### Normal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x;\mu,\sigma) = \frac{\exp(-(x-\mu)^2/2)}{\sqrt{2\pi\sigma}}\qquad for\ a\ real\ number\ x$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### StudentT

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu)}(1+x^2/\nu)^{-(\nu+1)/2}$$
  $$ where\ x\ is\ a\ real\ number\ and\ degrees\ of\ freedom\ parameter\ \nu>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### Uniform

<div style="color:black; font-size: 80%; text-align: center;">
  $$ [𝑙𝑜𝑐,𝑙𝑜𝑐 + 𝑠𝑐𝑎𝑙𝑒] $$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

### Weibull

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ min\ :\ f(x; c) = c x^{c-1} \exp(-x^c)\qquad for\ x\ge0,\ c>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ max\ :\ f(x; c) = c (-x)^{c-1} \exp(-(-x)^c)\qquad for\ x<0,\ c>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


<br><br><br>

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ double\ :\ f(x; c) = c / 2 |x|^{c-1} \exp(-|x|^c)\qquad for\ a\ real\ numbers,\ x\ and\ c>0$$
</div>

<div class='frame1'>
  Main code
</div>

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

> <div style="color:black; font-size: 80%; text-align: center;">
>   $$ random\ variable\ X \xrightarrow{function} distribution $$
>   <div class='frame2'> </div>
>   $$ parameters\ :\ a,\ b,\ loc,\ scale\ \xrightarrow{stats.beta} random\ variable\ :\ X $$
>   $$ x\ axis\ range : x \xrightarrow{X.pdf} distribution\ function $$
>   $$ sample\ :\ X.rvs\ stats.beta \xrightarrow{sns.distplot} visuallization $$
> </div>


---

<div class='frame1'>Sub-code</div>

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



<br><br><br>

---

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
