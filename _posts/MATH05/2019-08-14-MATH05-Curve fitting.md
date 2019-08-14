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

# [2] : Input
np.random.seed(1729)
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=xdata.size)
plt.scatter(xdata, ydata, marker='.', label='data')

# [3] : 
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



You might find it difficult to understand only the above code. To simplify the problem, the mathematical graph you are going to fit on will be confined to models that can be mathematically represented in the form of $$ a e^{-bx} + c $$ rather than in any form. In other word, the following sub-code [1] below is the description of this as a programming language. Instead of thinking about the complex number of cases, you should approach simple thing first. ***In order to finally obtain the graph you want to fit, you must find the value of the constant $$ a, b, c $$.***

<dl>
<dt class='frame2'>Sub-code [1]</dt>
</dl>

```python
# [1] : Mathematical representation of the graph you want to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
```

<br><br><br>



![fig](https://user-images.githubusercontent.com/52376448/63051109-7c2d3000-bf17-11e9-84ae-612bb7646240.png)



<dl>
<dt class='frame2'>Sub-code [2] : Input</dt>
</dl>

```python
# [2] : Input
np.random.seed(1729)
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=xdata.size)
plt.scatter(xdata, ydata, marker='.', label='data')
```

xdata = <br>
[0.         0.08163265 0.16326531 0.24489796 0.32653061 0.40816327
 0.48979592 0.57142857 0.65306122 0.73469388 0.81632653 0.89795918
 0.97959184 1.06122449 1.14285714 1.2244898  1.30612245 1.3877551
 1.46938776 1.55102041 1.63265306 1.71428571 1.79591837 1.87755102
 1.95918367 2.04081633 2.12244898 2.20408163 2.28571429 2.36734694
 2.44897959 2.53061224 2.6122449  2.69387755 2.7755102  2.85714286
 2.93877551 3.02040816 3.10204082 3.18367347 3.26530612 3.34693878
 3.42857143 3.51020408 3.59183673 3.67346939 3.75510204 3.83673469
 3.91836735 4.        ] <br>
 
 ydata = <br>
 [2.86253211 2.58408736 2.85238869 2.20328028 2.35505211 2.15580098
 1.62385977 1.51773747 1.58460558 1.56780331 1.38926991 1.23309392
 0.88831179 1.14039132 1.09814583 0.58195014 0.97957583 1.00016637
 0.94937867 0.88281865 1.05904456 0.51311865 0.54700714 0.66390699
 0.47465497 0.65048962 0.53479316 0.44458898 0.05609672 0.45631037
 0.54242354 0.93328116 0.45668558 0.57625432 0.79441476 0.58926037
 0.97314073 0.29536801 0.39975342 0.53597812 0.49927828 0.30842097
 0.38933359 0.9474868  0.5229339  0.60288646 0.5756889  0.1270815
 0.59254164 0.55991963] <br>

<br><br><br>

popt : Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
pcov : The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)). How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above. If the Jacobian matrix at the solution doesn’t have a full rank, then ‘lm’ method returns a matrix filled with np.inf, on the other hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to compute the covariance matrix.

<dl>
<dt class='frame2'>Sub-code [3]</dt>
</dl>

```python
# [3] : 
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='better fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

```

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

Text can be **bold**, _italic_, ~~strikethrough~~ or `keyword`.

[Link to another page](another-page).

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

* * *

*   Item foo
*   Item bar
*   Item baz
*   Item zip


1.  Item one
1.  Item two
1.  Item three
1.  Item four

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>


![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)
![](https://guides.github.com/activities/hello-world/branching.png)

