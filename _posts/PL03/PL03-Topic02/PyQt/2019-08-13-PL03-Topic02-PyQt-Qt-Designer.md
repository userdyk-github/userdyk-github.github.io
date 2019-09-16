---
layout : post
title : PL03-Topic02, PyQt, Qt Designer
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-PyQt.html) <br>
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

## Execute the ui-file on python

```python
from PyQt5 import QtWidgets, uic

app = QtWidgets.QApplication([])
dlg = uic.loadUi("test.ui")

dlg.show()
app.exec()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">FILE PATH</summary>
<hr class='division3'>
<div class='jb-medium'>when there exist the ui-file in parent folder,</div>
`dlg = uic.loadUi("../test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in same folder,</div>
`dlg = uic.loadUi("test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in sub-folder,</div>
`dlg = uic.loadUi("sub-folder/test.ui")`<br>

<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## title2

<hr class="division2">

## title3

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





