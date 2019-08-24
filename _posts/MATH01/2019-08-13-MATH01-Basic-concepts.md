---
layout : post
title : MATH01, Basic concepts
categories: [MATH01]
comments : true
tags : [MATH01]
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

## **Functions and Methods for Operating**

### Columns vector

`INPUT`
```python
import sympy
#sympy.init_printing()

sympy.Matrix([1, 2])
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}1\\2\end{matrix}\right]$$</span>
```python
Matrix([
[1],
[2]])
```

<br><br>

### Row vector

`INPUT`
```python
import sympy
#sympy.init_printing()

sympy.Matrix([[1, 2]])
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}1 & 2\end{matrix}\right]$$</span>
```python
Matrix([[1, 2]])
```
<br><br>

### Value of a corresponding element

`INPUT`
```python
import sympy
#sympy.init_printing()

sympy.Matrix(3, 4, lambda m, n: 10 * m + n)
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}0 & 1 & 2 & 3\\10 & 11 & 12 & 13\\20 & 21 & 22 & 23\end{matrix}\right]$$</span>
```python
Matrix([
[ 0,  1,  2,  3],
[10, 11, 12, 13],
[20, 21, 22, 23]])
```
<br><br>

### Matrix multiplication

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

x_1, x_2 = sympy.symbols("x_1, x_2")
x = sympy.Matrix([x_1, x_2])

M * x
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}a x_{1} + b x_{2}\\c x_{1} + d x_{2}\end{matrix}\right]$$</span>
```python
Matrix([
[a*x_1 + b*x_2],
[c*x_1 + d*x_2]])
```
<br><br>

### Transpose of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.T
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}a & c\\b & d\end{matrix}\right]$$</span>
```python
Matrix([
[a, c],
[b, d]])
```
<br><br>

### Adjoint of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.H
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}\overline{a} & \overline{c}\\\overline{b} & \overline{d}\end{matrix}\right]$$</span>
```python
Matrix([
[conjugate(a), conjugate(c)],
[conjugate(b), conjugate(d)]])
```
<br><br>

### Trace (sum of diagonal elements) of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.trace()
```

`OUTPUT` : <span class='jb-small'>$$a + d$$</span>
```python
a + d
```
<br><br>

### Determinant of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.det()
```

`OUTPUT` : <span class='jb-small'>$$a d - b c$$</span>
```python
a*d - b*c
```
<br><br>

### Inverse of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.inv()
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}\frac{d}{a d - b c} & - \frac{b}{a d - b c}\\- \frac{c}{a d - b c} & \frac{a}{a d - b c}\end{matrix}\right]$$</span>
```python
Matrix([
[ d/(a*d - b*c), -b/(a*d - b*c)],
[-c/(a*d - b*c),  a/(a*d - b*c)]])
```
<br><br>

### LU decomposition of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.LUdecomposition()
```

`OUTPUT` : <span class='jb-small'>$$\left ( \left[\begin{matrix}1 & 0\\\frac{c}{a} & 1\end{matrix}\right], \quad \left[\begin{matrix}a & b\\0 & d - \frac{b c}{a}\end{matrix}\right], \quad \left [ \right ]\right )$$</span>
```python
(Matrix([
[  1, 0],
[c/a, 1]]), Matrix([
[a,         b],
[0, d - b*c/a]]), [])
```
<br><br>

### Linear system of equations in the form Mx = v, using LU factorization

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

v_1, v_2 = sympy.symbols("v_1, v_2")
v = sympy.Matrix([v_1, v_2])

M.LUsolve(v)
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}\frac{1}{a} \left(- \frac{b \left(v_{2} - \frac{c v_{1}}{a}\right)}{d - \frac{b c}{a}} + v_{1}\right)\\\frac{v_{2} - \frac{c v_{1}}{a}}{d - \frac{b c}{a}}\end{matrix}\right]$$</span>
```python
Matrix([
[(-b*(v_2 - c*v_1/a)/(d - b*c/a) + v_1)/a],
[             (v_2 - c*v_1/a)/(d - b*c/a)]])
```
<br><br>

### QR decomposition of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.QRdecomposition()
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left( \left[\begin{matrix}\frac{a}{\sqrt{a^{2} + c^{2}}} & \frac{- \frac{a \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} + b}{\sqrt{\left(\frac{a \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - b\right)^{2} + \left(\frac{c \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - d\right)^{2}}}\\\frac{c}{\sqrt{a^{2} + c^{2}}} & \frac{- \frac{c \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} + d}{\sqrt{\left(\frac{a \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - b\right)^{2} + \left(\frac{c \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - d\right)^{2}}}\end{matrix}\right], \  \left[\begin{matrix}\sqrt{a^{2} + c^{2}} & \frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\\0 & \sqrt{\left(\frac{a \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - b\right)^{2} + \left(\frac{c \left(\frac{a b}{\sqrt{a^{2} + c^{2}}} + \frac{c d}{\sqrt{a^{2} + c^{2}}}\right)}{\sqrt{a^{2} + c^{2}}} - d\right)^{2}}\end{matrix}\right]\right)$$</span>
```python
(Matrix([
[a/sqrt(Abs(a)**2 + Abs(c)**2), (-a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + b)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)],
[c/sqrt(Abs(a)**2 + Abs(c)**2), (-c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + d)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)]]), Matrix([
[sqrt(Abs(a)**2 + Abs(c)**2),                                                                                                                                                                   a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2)],
[                          0, sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)]]))
```
<br><br>

### Linear system of equations in the form Mx = v, using QR factorization

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

v_1, v_2 = sympy.symbols("v_1, v_2")
v = sympy.Matrix([v_1, v_2])

M.QRsolve(v)
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left[\begin{matrix}\frac{\frac{a v_{1}}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c v_{2}}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - \frac{\left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right) \left(\frac{v_{1} \left(- \frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + b\right)}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}} + \frac{v_{2} \left(- \frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + d\right)}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}}\right)}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}}}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\\\frac{\frac{v_{1} \left(- \frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + b\right)}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}} + \frac{v_{2} \left(- \frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + d\right)}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}}}{\sqrt{\left|{\frac{a \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - b}\right|^{2} + \left|{\frac{c \left(\frac{a b}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} + \frac{c d}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}}\right)}{\sqrt{\left|{a}\right|^{2} + \left|{c}\right|^{2}}} - d}\right|^{2}}}\end{matrix}\right]$$</span>
```python
Matrix([
[(a*v_1/sqrt(Abs(a)**2 + Abs(c)**2) + c*v_2/sqrt(Abs(a)**2 + Abs(c)**2) - (a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))*(v_1*(-a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + b)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2) + v_2*(-c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + d)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2))/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2))/sqrt(Abs(a)**2 + Abs(c)**2)],
[                                                                                                                                                                          (v_1*(-a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + b)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2) + v_2*(-c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + d)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2))/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)]])
```
<br><br>

### Diagonalization of a matrix M

`INPUT`
```python
import sympy
#sympy.init_printing()

M = sympy.Matrix([[1, 2], [3, 4]])

M.diagonalize()
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left( \left[\begin{matrix}- \frac{\sqrt{33}}{6} - \frac{1}{2} & - \frac{1}{2} + \frac{\sqrt{33}}{6}\\1 & 1\end{matrix}\right], \  \left[\begin{matrix}\frac{5}{2} - \frac{\sqrt{33}}{2} & 0\\0 & \frac{5}{2} + \frac{\sqrt{33}}{2}\end{matrix}\right]\right)$$</span>
```python
(Matrix([
[-sqrt(33)/6 - 1/2, -1/2 + sqrt(33)/6],
[                1,                 1]]), Matrix([
[5/2 - sqrt(33)/2,                0],
[               0, 5/2 + sqrt(33)/2]]))
```
<br><br>

### Norm of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.norm()
```

`OUTPUT` : <span class='jb-small'>$$\sqrt{\left|{a}\right|^{2} + \left|{b}\right|^{2} + \left|{c}\right|^{2} + \left|{d}\right|^{2}}$$</span>
```python
sqrt(Abs(a)**2 + Abs(b)**2 + Abs(c)**2 + Abs(d)**2)
```
<br><br>

### Set of vectors that span the null space of a Matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b = sympy.symbols("a, b")
M = sympy.Matrix([[a, b], [a, b]])

M.nullspace()
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left[ \left[\begin{matrix}- \frac{b}{a}\\1\end{matrix}\right]\right]$$</span>
```python
[Matrix([
[-b/a],
[   1]])]
```
<br><br>

### Rank of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.rank()
```

`OUTPUT` : <span class='jb-small'>$$2$$</span>
```python
2
```
<br><br>

### Singular values of a matrix

`INPUT`
```python
import sympy
#sympy.init_printing()

M = sympy.Matrix([[1, 2], [3, 4]])

M.singular_values()
```

`OUTPUT` : <span class='jb-small'>$$\displaystyle \left[ \sqrt{\sqrt{221} + 15}, \  \sqrt{15 - \sqrt{221}}\right]$$</span>
```python
[sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
```
<br><br>

### Linear system of equations in the form Mx = v

`INPUT`
```python
import sympy
#sympy.init_printing()

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.solve(v)
```

`OUTPUT` : <span class='jb-small'>$$\left[\begin{matrix}- \frac{b v_{2}}{a d - b c} + \frac{d v_{1}}{a d - b c}\\\frac{a v_{2}}{a d - b c} - \frac{c v_{1}}{a d - b c}\end{matrix}\right]$$</span>
```python
Matrix([
[(-b*(a*v_2 - c*v_1) + v_1*(a*d - b*c))/(a*(a*d - b*c))],
[                           (a*v_2 - c*v_1)/(a*d - b*c)]])
```
<br><br>

---

## title2

---

## title3

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

