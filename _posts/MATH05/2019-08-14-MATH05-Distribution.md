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
$$Parameters(loc,\ scale)\ :\ y=\frac{x-loc}{scale}$<br>$Parameter(df)\ :\ a\ shape\ parameter$$

---

## Discrete distribution

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

## Continous distribution

> Objective : 
>
> When something is important enough, you do it even if the odds are not in your favor.

### Beta

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}{\Gamma(a) \Gamma(b)}\qquad for\ 0\le x \le 1,\ \ \ \ a>0,\ b>0$$
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
ax[2].plot(x, stats.beta(a=5,b=0.1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### ChiSquared

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}x^{k/2-1} \exp \left( -x/2 \right)\qquad for\ x>0\ and\ k>0$$
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
ax[2].plot(x, stats.chi2(df=1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Exponential

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \lambda*\exp(-\lambda*x)\qquad for\ x\ge0$$
  $$scale\ =\ 1/\lambda$$
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
ax[2].plot(x, stats.expon(scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### FDistribution

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}\qquad for\ x>0$$
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
ax[2].plot(x, stats.f(dfn=1,dfd=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### FisherZ

### Gamma

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; a) = \frac{x^{a-1} \exp(-x)}{\Gamma(a)}\qquad for\ x\ge0,\ a>0$$
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
ax[2].plot(x, stats.gamma(a=1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Laplace

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{1}{2} \exp(-|x|)\qquad for\ a\ real\ number\ x$$
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
ax[2].plot(x, stats.laplace().pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Logistic

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x) = \frac{\exp(-x)}{(1+\exp(-x))^2}$$
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
ax[2].plot(x, stats.logistic(loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### LogNormal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; s) = \frac{1}{s x \sqrt{2\pi}}\exp\left(-\frac{\log^2(x)}{2s^2}\right)\qquad for\ x>0,\ s>0$$
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
ax[2].plot(x, stats.lognorm(s=1, loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Normal

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x;\mu,\sigma) = \frac{\exp(-(x-\mu)^2/2)}{\sqrt{2\pi\sigma}}\qquad for\ a\ real\ number\ x$$
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
ax[2].plot(x, stats.norm(loc=100, scale=10).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### StudentT

<div style="color:black; font-size: 80%; text-align: center;">
  $$f(x; \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu)}(1+x^2/\nu)^{-(\nu+1)/2}$$
  $$ where\ x\ is\ a\ real\ number\ and\ degrees\ of\ freedom\ parameter\ \nu>0$$
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
ax[2].plot(x, stats.t(df=4,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Uniform

<div style="color:black; font-size: 80%; text-align: center;">
  $$ [𝑙𝑜𝑐,𝑙𝑜𝑐 + 𝑠𝑐𝑎𝑙𝑒] $$
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
ax[2].plot(x, stats.uniform(loc=2, scale=10).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

### Weibull

<div style="color:black; font-size: 80%; text-align: center;">
  $$Weibull\ min\ :\ f(x; c) = c x^{c-1} \exp(-x^c)\qquad for\ x\ge0,\ c>0$$
  $$Weibull\ max\ :\ f(x; c) = c (-x)^{c-1} \exp(-(-x)^c)\qquad for\ x<0,\ c>0$$
  $$Weibull\ double\ :\ f(x; c) = c / 2 |x|^{c-1} \exp(-|x|^c)\qquad for\ a\ real\ numbers,\ x\ and\ c>0$$
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
ax[2].plot(x, stats.weibull_min(c=1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

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
ax[2].plot(x, stats.weibull_max(c=1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

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
ax[2].plot(x, stats.dweibull(c=1,loc=1, scale=1).pdf(x))

ax[0].set_title("model fitting")
ax[1].set_title("data fitting")
ax[2].set_title("PDF")

plt.tight_layout()
plt.show()
```

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
