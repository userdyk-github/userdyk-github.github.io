---
layout : post
title : PHY03, Heat flow via time stepping
categories: [PHY03]
comments : true
tags : [PHY03]
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

---

## Heat Flow via Time-Stepping (Leapfrog)

---

## The Parabolic Heat Equation (Theory)

<div class='frame1'>Main code</div>

```python
# EqHeat.py : solves heat equation via finite differences , 3D plot
from numpy import *
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D

Nx = 101; Nt = 3000; Dx = 0.03; Dt = 0.9
KAPPA = 210.; SPH = 900.; RHO = 2700.                           # Conductivity , specf heat , density
T = zeros((Nx, 2), float); Tpl = zeros((Nx, 31), float)

print("Working , wait for figure after count to 10")

for ix in range(1, Nx - 1):
    T[ix, 0] = 100.0;                                           # Initial T
T[0, 0] = 0.0; T[0, 1] = 0.                                     # 1st & last T = 0
T [Nx-1, 0] = 0.; T[Nx-1, 1] = 0.0
cons = KAPPA / (SPH*RHO)*Dt / (Dx*Dx);                          # constant
m = 1                                                           # counter

for t in range(1, Nt):
    for ix in range(1 , Nx-1):
        T[ix, 1] = T[ix, 0] + cons * (T[ix+1, 0] + T[ix-1, 0] - 2.*T[ix, 0])
    if t %300 == 0 or t == 1:                                   # Every 300 steps
        for ix in range(1, Nx-1, 2):
            Tpl[ix, m] = T[ix, 1]
        print(m)
        m = m + 1
    for ix in range(1 , Nx-1) :
        T[ix , 0] = T[ix , 1]
x = list(range(1, Nx-1, 2))                                     # Plot alternate pts
y = list(range(1, 30))
X, Y = p.meshgrid(x, y)

def functz (Tpl):
    z = Tpl[X, Y]
    return z

Z = functz (Tpl )
fig = p.figure()                                               # Create figure
ax = Axes3D(fig)
ax . plot_wireframe(X, Y, Z, color='r')
ax . set_xlabel('Position')
ax . set_ylabel('time')
ax . set_zlabel('Temperature')
p . show( )
print ('finished')
```

---

##  Assessment and Visualization

---

##  Improved Heat Flow: Crank–Nicolson Method

---
List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Rubin H. Landau, ManuelJ. Páez, Cristian C. Bordeianu - Computational Physics, Problem Solving with Python, 3rd completely revised edition
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
