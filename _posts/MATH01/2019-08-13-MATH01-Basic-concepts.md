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

```
<br><br>

###  The value of a corresponding element

<span class='frame2 jb-small'>INPUT</span>
```python
import sympy
sympy.Matrix(3, 4, lambda m, n: 10 * m + n)
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The transpose of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The adjoint of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The trace (sum of diagonal elements) of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The determinant of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The inverse of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The LU decomposition of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### A linear system of equations in the form Mx = b, using LU factorization

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The QR decomposition of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### A linear system of equations in the form Mx = b, using QR factorization

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### Diagonalization of a matrix M

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The norm of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### A set of vectors that span the null space of a Matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The rank of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### The singular values of a matrix

<span class='frame2 jb-small'>INPUT</span>
```python
```

<span class='frame2 jb-small'>OUTPUT</span>
```python
```
<br><br>

### A linear system of equations in the form Mx = b

<span class='frame2 jb-small'>INPUT</span>
```python
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

