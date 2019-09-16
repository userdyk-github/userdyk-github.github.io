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

## Execute the ui-file by python

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
![그림1](https://user-images.githubusercontent.com/52376448/64966645-22cf6c80-d8da-11e9-910d-740977ac18ad.png)
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

## Convert

```python
from PyQt5 import QtWidgets, uic

def Convert():
    dlg.lineEdit_2.setText(str(float(dlg.lineEdit.text())*1.23))

app = QtWidgets.QApplication([])
dlg = uic.loadUi("test.ui")

dlg.pushButton.clicked.connect(Convert)

dlg.show()
app.exec()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<hr class="division2">

## title3

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://www.youtube.com/playlist?list=PLuTktZ8WcEGTdId-Kjbj6gsZTk65yudJh' target="_blank">Youtube Lecture</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---





