---
layout : post
title : AI05, Basic image processing
categories: [AI05]
comments : true
tags : [AI05]
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

<hr class="division1">

## **Building 2d-image with numpy**

### ***dtype = np.uint8***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.uint8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint8<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1d7d630>
</p>
![다운로드 (8)](https://user-images.githubusercontent.com/52376448/63788477-86e0bf80-c930-11e9-80ff-3959fe9caf3d.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***dtype = np.int8***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.int8)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  int8<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1ddf208>
</p>
![다운로드 (9)](https://user-images.githubusercontent.com/52376448/63788479-86e0bf80-c930-11e9-905e-423fad817438.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***dtype = np.uint16***

```python
import numpy as np
from skimage import io

image = np.array([[255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255],
                  [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0],
                  [0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]], dtype=np.uint16)

print('dtype : ', image.dtype)
print('shape : ', image.shape)
io.imshow(image)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
  dtype :  uint16<br>
  shape :  (12, 16)<br>
  <matplotlib.image.AxesImage at 0x260c1e96208>
</p>
![다운로드 (10)](https://user-images.githubusercontent.com/52376448/63788476-86482900-c930-11e9-8fe8-5c9c48218b9d.png)
<hr class='division3'>
</details>
<br><br><br>



<hr class="division2">

## **Building 3d-image with numpy**

<hr class="division1">

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

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

