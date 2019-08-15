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

# [3] : Output through input target function(1)
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='better fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# [4] : Output through input target function(2)
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--', label='best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# [5] : Visualization
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```


<div>
$$ data \xrightarrow{curve\ fitting} y = a e^{-bx}+c $$ </div>
<div class='frame2'> </div>

|STEP| INPUT                      | FUNCTION                       | OUTPUT                        |      |
|:---|:---------------------------|:-------------------------------|:------------------------------|:------|
| 1  | func, xdata, ydata         | $$\xrightarrow{curve\_fit}$$   | popt(= a, b, c), pcov         |  popt = <br> [a, b, c]  |
| 2  | xdata, func(xdata, *popt)  | $$\xrightarrow{X.pdf}$$        | graph : $$ y = a e^{-bx}+c $$ |    |



<br><br><br>





It might be difficult to understand the meaning by looking at only the above code. To simplify the problem, the mathematical graph you are going to fit on will be confined to models that can be mathematically represented in the form of $$ a e^{-bx} + c $$ rather than in any form. In other word, the following details-code [1] below is the description of this as a programming language. Instead of thinking about the complex number of cases, you should approach simple thing first. ***And then in order to finally obtain the curve you want to fit, you must find the value of the constant $$ a, b, c $$.***

<dl>
<dt class='frame2'>Details-code [1] : Mathematical representation of the curve you want to fit</dt>
</dl>

```python
# [1] : Mathematical representation of the graph you want to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
```

<br><br><br>



Before you find the values of a, b, and c defined above, let's look at the data. In reality, actual data is used. However, for convenience here, random data will be generated through details-code [2] and used as input data.



<dl>
<dt class='frame2'>Details-code [2] : Input Data</dt>
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

![fig](https://user-images.githubusercontent.com/52376448/63051109-7c2d3000-bf17-11e9-84ae-612bb7646240.png){: width="500" height="500"){: .center}

<div style="color:black; font-size: 80%; text-align: center;">
$$ xdata = [0.00000000 \quad 0.08163265 \quad 0.16326531 \quad ... \quad 4.00000000] $$ 
$$ ydata = [2.86253211 \quad 2.58408736 \quad 2.85238869 \quad ... \quad 0.55991963] $$ 
</div>
<br><br><br>


***Now, most of the process from now on will be a series of steps to find the values of $$ a, b, c $$.*** It should not be forgotten that finding a value of $$ a,b,c $$ is equivalent to obtaining a fitting curve. Estimated optimal values for the parameters so that the sum of the squared residuals of $$ func(xdata, *popt) - ydata $$ is minimized. <br>
<div style="font-size: 100%; text-align: center;">$$ popt = \begin{pmatrix} \hat{a} & \hat{b} & \hat{c} \end{pmatrix} $$</div>
Next, let us look at the estimated covariance of popt below, pocv. The diagonals provide the variance of the parameter estimate. To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)). How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above. If the Jacobian matrix at the solution doesn’t have a full rank, then ‘lm’ method returns a matrix filled with np.inf, on the other hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to compute the covariance matrix. <br>
<div style="color:black; font-size: 80%; text-align: center;">
$$ pcov =
\begin{pmatrix}
\sigma_{x_{a}y_{a}} & \sigma_{x_{a}y_{b}} & \sigma_{x_{a}y_{c}}\\
\sigma_{x_{b}y_{a}} & \sigma_{x_{b}y_{b}} & \sigma_{x_{b}y_{c}}\\
\sigma_{x_{c}y_{a}} & \sigma_{x_{c}y_{b}} & \sigma_{x_{c}y_{c}}
\end{pmatrix} $$ <br>
</div>

<dl>
<dt class='frame2'>Details-code [3] : Output through input target function(1)</dt>
</dl>

```python
# [3] : Output through input target function(1)
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='better fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
```

![Figure_1](https://user-images.githubusercontent.com/52376448/63059979-d46e2d00-bf2b-11e9-91e5-6d82858fb574.png){: width="500" height="500"){: .center}

<div style="color:black; font-size: 80%; text-align: center;">
$$ popt = \begin{pmatrix} 2.554 & 1.352 & 0.475 \end{pmatrix} $$
$$ pcov =
\begin{pmatrix}
0.0158905 & 0.00681778 & -0.0007614\\
0.00681778 & 0.02019919 & 0.00541905\\
-0.0007614 & 0.00541905 & 0.00282595
\end{pmatrix} $$ 
</div>
<br><br><br>





<dl>
<dt class='frame2'>Details-code [4] : Output through input target function(2)</dt>
</dl>

```python
# [4] : Output through input target function(2)
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--', label='best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
```

![Figure_1](https://user-images.githubusercontent.com/52376448/63060885-6aa35280-bf2e-11e9-88ee-fd20844eee4f.png){: width="500" height="500"){: .center}


<div style="color:black; font-size: 80%; text-align: center;">
$$ popt = \begin{pmatrix} 2.43708905 & 1. & 0.35015434 \end{pmatrix} $$
$$ pcov =
\begin{pmatrix}
0.01521864 & 0.00291009 & -0.00223465\\
0.00291009 & 0.01677755 & 0.00839441\\
-0.00223465 & 0.00839441 & 0.00615306
\end{pmatrix} $$ 
</div>
<br><br><br>

<dl>
<dt class='frame2'>Details-code [5]</dt>
</dl>

```python
# [5] : Visualization
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/52376448/63061205-50b63f80-bf2f-11e9-989f-e705b7811169.png){: width="500" height="500"){: .center}

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
