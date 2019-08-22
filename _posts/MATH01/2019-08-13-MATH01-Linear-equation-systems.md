---
layout : post
title : MATH01, Linear equation systems
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

## Square Systems : $$Ax=b$$

$$\begin{pmatrix}2 & 3 \\5 & 4\end{pmatrix}x=\begin{pmatrix}4 & 3 \end{pmatrix}$$

<div class='frame1'>Main code : symbolic</div>
```python
from sympy import Matrix

A = Matrix([[2,3],[5,4]])
b = Matrix([4,3])
x = A.solve(b)

print(x)
```

\begin{pmatrix}
x & y \\
z & v
\end{pmatrix}

<div class='frame1'>Main code : numerical</div>
```python
import numpy as np
from scipy import linalg as la

A = np.array([[2,3],[5,4]])
b = np.array([4,3])
x = la.solve(A,b)

print(x)
```



---

## Rectangular Systems

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
