---
layout : post
title : PL03-Topic02, PyQt, Qt Designer
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-PyQt.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/PyQt/2019-08-13-PL03-Topic02-PyQt-Qt-Designer.md" target="_blank">page management</a><br>
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

## **Execute the ui-file by python**
### ***basic form***
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
### ***advanced form***
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("test.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```
<br><br><br>
<hr class="division2">

## **Notepad**
<br><br><br>
<hr class="division2">

## **Calculator**
<br><br><br>
<hr class="division2">

### Convert(1)

`ui file`
![캡처](https://user-images.githubusercontent.com/52376448/64971472-b73dcd00-d8e2-11e9-816d-82de1be7c8a9.JPG)

<br>

`python-code`
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
![캡처](https://user-images.githubusercontent.com/52376448/64976720-f8d37580-d8ec-11e9-9514-dba0530d65e8.JPG)
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

### Convert(2)

```python
from PyQt5 import QtWidgets, uic

def Convert():
    dlg.lineEdit_2.setText(str(float(dlg.lineEdit.text())*1.23))

app = QtWidgets.QApplication([])
dlg = uic.loadUi("test.ui")

dlg.lineEdit.setFocus()
dlg.lineEdit.setPlaceholderText("Insert")
dlg.pushButton.clicked.connect(Convert)

dlg.lineEdit.returnPressed.connect(Convert)
dlg.lineEdit_2.setReadOnly(True)

dlg.show()
app.exec()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![그림1](https://user-images.githubusercontent.com/52376448/64975105-3209e680-d8e9-11e9-9122-dbbaf018d92c.png)
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">

## **Building .exe file**

```
pyuic5 -x main.ui -o main.py
```
```
pyinstaller main.py
```

<br><br><br>

<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference

- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://www.tutorialspoint.com/pyqt/index.htm' target="_blank">PyQt Tutorial(official)</a>
- <a href="http://codetorial.net/" target="_blank">PyQt5 Tutorial</a>
- <a href='https://www.youtube.com/playlist?list=PLuTktZ8WcEGTdId-Kjbj6gsZTk65yudJh' target="_blank">Youtube Lecture about Qt designer</a>
- <a href='https://www.youtube.com/watch?v=qiPS70TSvBk' target="_blank">Building .exe file</a>

---





