---
layout : post
title : MATH05, Curve fitting
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

## **Expoential model fitting**

![fig](https://user-images.githubusercontent.com/52376448/63042334-b80aca00-bf04-11e9-8ca9-4c2923b720df.png)

I will introduce data fitting techniques. The goal is to answer which mathematical graph are best suited to data given to you, as shown in the figure on the left. ***As a result, as an answer to this question, if you run the below following main code, you can get the above graph on the right.*** 

<dl>
<dt class='frame1'>Main code</dt>
</dl>

```python
# [0] : Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# [1] : Mathematical representation of the graph you want to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# [2] : Input Data
np.random.seed(1729)
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=xdata.size)
plt.scatter(xdata, ydata, marker='.', label='data')

# [3] : Input target function
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='better fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# [4] : 
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--', label='best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# [5] : Output
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

<br><br><br>





You might find it difficult to understand only the above code. To simplify the problem, the mathematical graph you are going to fit on will be confined to models that can be mathematically represented in the form of $$ a e^{-bx} + c $$ rather than in any form. In other word, the following sub-code [1] below is the description of this as a programming language. Instead of thinking about the complex number of cases, you should approach simple thing first. ***And then in order to finally obtain the graph you want to fit, you must find the value of the constant $$ a, b, c $$.***

<dl>
<dt class='frame2'>Sub-code [1] : Mathematical representation of the graph you want to fit</dt>
</dl>

```python
# [1] : Mathematical representation of the graph you want to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
```

<br><br><br>





![fig](https://user-images.githubusercontent.com/52376448/63051109-7c2d3000-bf17-11e9-84ae-612bb7646240.png)

In reality, actual data is used. However, for convenience here, random data will be generated through sub-code [2] and used as input data.

<dl>
<dt class='frame2'>Sub-code [2] : Input Data</dt>
</dl>

```python
# [2] : Input Data
np.random.seed(1729)
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=xdata.size)
plt.scatter(xdata, ydata, marker='.', label='data')
```

The data generated in this way will be as follows. <br>

$$ xdata = [0.00000000 \quad 0.08163265 \quad 0.16326531 \quad ... \quad 4.00000000] $$ <br>
$$ ydata = [2.86253211 \quad 2.58408736 \quad 2.85238869 \quad ... \quad 0.55991963] $$ <br>

<br><br><br>



$$ popt = \begin{pmatrix} a & b & c \end{pmatrix} $$ <br>
Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized <br>
$$ pcov =
\begin{pmatrix}
\sigma_{x_{a}y_{a}} & \sigma_{x_{a}y_{b}} & \sigma_{x_{a}y_{c}}\\
\sigma_{x_{b}y_{a}} & \sigma_{x_{b}y_{b}} & \sigma_{x_{b}y_{c}}\\
\sigma_{x_{c}y_{a}} & \sigma_{x_{c}y_{b}} & \sigma_{x_{c}y_{c}}
\end{pmatrix} $$ <br>
The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)). How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above. If the Jacobian matrix at the solution doesn’t have a full rank, then ‘lm’ method returns a matrix filled with np.inf, on the other hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to compute the covariance matrix. <br>

<dl>
<dt class='frame2'>Sub-code [3] : Input target function</dt>
</dl>

```python
# [3] : Input target function
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='better fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

```
$$ popt = \begin{pmatrix} 2.55423706 & 1.35190947 & 0.47450618 \end{pmatrix} $$ <br>
$$ pcov =
\begin{pmatrix}
0.0158905 & 0.00681778 & -0.0007614\\
0.00681778 & 0.02019919 & 0.00541905\\
-0.0007614 & 0.00541905 & 0.00282595
\end{pmatrix} $$ <br>

<br><br><br>





<dl>
<dt class='frame2'>Sub-code [4]</dt>
</dl>

```python
# [4] : 
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--', label='best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
```

<br><br><br>

<dl>
<dt class='frame2'>Sub-code [5]</dt>
</dl>

```python
# [5] : Output
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

<br><br><br>

 


---

## **Gaussian model fitting**

---

## **Two gaussian model fitting**


---

## **Curve fit with seaborn based on specific distribution**

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
