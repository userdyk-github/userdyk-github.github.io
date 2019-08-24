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

## Functions and Methods for Operating

### Columns vector

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
sympy.Matrix([1, 2])
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[1],
[2]])
```
<br><br>

### Row vector

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
sympy.Matrix([[1, 2]])
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([[1, 2]])
```
<br><br>

### Value of a corresponding element

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
sympy.Matrix(3, 4, lambda m, n: 10 * m + n)
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[ 0,  1,  2,  3],
[10, 11, 12, 13],
[20, 21, 22, 23]])
```
<br><br>

### Matrix multiplication

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

x_1, x_2 = sympy.symbols("x_1, x_2")
x = sympy.Matrix([x_1, x_2])

M * x
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[a*x_1 + b*x_2],
[c*x_1 + d*x_2]])
```
<br><br>

### Transpose of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.T
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[a, c],
[b, d]])
```
<br><br>

### Adjoint of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.H
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[conjugate(a), conjugate(c)],
[conjugate(b), conjugate(d)]])
```
<br><br>

### Trace (sum of diagonal elements) of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.trace()
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
a + d
```
<br><br>

### Determinant of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.det()
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
a*d - b*c
```
<br><br>

### Inverse of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.inv()
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[ d/(a*d - b*c), -b/(a*d - b*c)],
[-c/(a*d - b*c),  a/(a*d - b*c)]])
```
<br><br>

### LU decomposition of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.LUdecomposition()
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
(Matrix([
[  1, 0],
[c/a, 1]]), Matrix([
[a,         b],
[0, d - b*c/a]]), [])
```
<br><br>

### Linear system of equations in the form Mx = v, using LU factorization

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

v_1, v_2 = sympy.symbols("v_1, v_2")
v = sympy.Matrix([v_1, v_2])

M.LUsolve(v)
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
Matrix([
[(-b*(v_2 - c*v_1/a)/(d - b*c/a) + v_1)/a],
[             (v_2 - c*v_1/a)/(d - b*c/a)]])
```
<br><br>

### QR decomposition of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy

a, b, c, d = sympy.symbols("a, b, c, d")
M = sympy.Matrix([[a, b], [c, d]])

M.QRdecomposition()
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
(Matrix([
[a/sqrt(Abs(a)**2 + Abs(c)**2), (-a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + b)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)],
[c/sqrt(Abs(a)**2 + Abs(c)**2), (-c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) + d)/sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)]]), Matrix([
[sqrt(Abs(a)**2 + Abs(c)**2),                                                                                                                                                                   a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2)],
[                          0, sqrt(Abs(a*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - b)**2 + Abs(c*(a*b/sqrt(Abs(a)**2 + Abs(c)**2) + c*d/sqrt(Abs(a)**2 + Abs(c)**2))/sqrt(Abs(a)**2 + Abs(c)**2) - d)**2)]]))
```
<br><br>

### Linear system of equations in the form Mx = v, using QR factorization

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Diagonalization of a matrix M

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Norm of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Set of vectors that span the null space of a Matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Rank of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Singular values of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Linear system of equations in the form Mx = v

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
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

