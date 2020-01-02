---
layout : post
title : PL03-Topic02, PyQt, Qt Designer
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-PyQt.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/PyQt/2019-08-13-PL03-Topic02-PyQt-Qt-Designer.md" target="_blank">page management</a> ｜<a href="https://www.youtube.com/playlist?list=PLnIaYcDMsScwsKo1rQ18cLHvBdjou-kb5" target="_blank">Lecture</a><br>
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
### ***basic form without qtdesigner***
```python
import sys
from PyQt5.QtWidgets import QApplication, QDialog

app = QApplication(sys.argv)
mainDialog = QDialog()
mainDialog.show()
app.exec_()
```

<br><br><br>
### ***basic form with qtdesigner***
```python
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic

app = QApplication([])
mainDialog = uic.loadUi("test.ui")
mainDialog.show()
app.exec()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">FILE PATH</summary>
<hr class='division3'>
![그림1](https://user-images.githubusercontent.com/52376448/64966645-22cf6c80-d8da-11e9-910d-740977ac18ad.png)
<div class='jb-medium'>when there exist the ui-file in parent folder,</div>
`mainDialog = uic.loadUi("../test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in same folder,</div>
`mainDialog = uic.loadUi("test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in sub-folder,</div>
`mainDialog = uic.loadUi("sub-folder/test.ui")`<br>

<hr class='division3'>
</details>

<br><br><br>
### ***advanced form with qtdesigner***
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
### ***(1) Hello***
![image](https://user-images.githubusercontent.com/52376448/71642282-4d5bfe80-2cec-11ea-8a3e-a05940b1678d.png)

`Code`
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("notepad.ui")[0]

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

---

### ***(2) Layout***
![image](https://user-images.githubusercontent.com/52376448/71642292-75e3f880-2cec-11ea-8aac-a4f9e588a483.png)
`Code`
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("notepad.ui")[0]

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

---

### ***(3) Menubar***
![image](https://user-images.githubusercontent.com/52376448/71642306-b3e11c80-2cec-11ea-964f-bdab4e00a35b.png)

`Code`
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("notepad.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.action_open.triggered.connect(self.openFunction)
        self.action_save.triggered.connect(self.saveFunction)

    def openFunction(self):
        print("open!!")

    def saveFunction(self):
        print("save!!")

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```

<br><br><br>

---

### ***(4) Open/Save***
![image](https://user-images.githubusercontent.com/52376448/71642342-323dbe80-2ced-11ea-9b8f-d88ad7766aa0.png)

`Code`
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("notepad.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.action_open.triggered.connect(self.openFunction)
        self.action_save.triggered.connect(self.saveFunction)

    def openFunction(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0]:
            with open(fname[0], encoding='UTF8') as f:
                data = f.read()
            self.plainTextEdit.setPlainText(data)

            print("open {}!!".format(fname[0]))

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self)
        if fname[0]:
            data = self.plainTextEdit.toPlainText()

            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)

            print("save {}!!".format(fname[0]))

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```

<br><br><br>

---

### ***(5) SaveAs***
`Code`
```python

```

<br><br><br>

---

### ***(6) CloseEvent***
`Code`
```python

```

<br><br><br>

---

### ***(7) MessageBox***
`Code`
```python

```

<br><br><br>

---

### ***(8) PlainTextEdit***
`Code`
```python

```

<br><br><br>

---

### ***(9) Find***
`Code`
```python

```

<br><br><br>

---

### ***(10) KeyboardEvent***
`Code`
```python

```

<br><br><br>

---

### ***(11) SetCursor***
`Code`
```python

```

<br><br><br>

---

### ***(12) FindText***
`Code`
```python

```

<br><br><br>

---

### ***(13) IgnoreFlag***
`Code`
```python

```

<br><br><br>

---

### ***(14) Radiobutton***
`Code`
```python

```

<br><br><br>

---

### ***(15) SearchDirection***
`Code`
```python

```

<br><br><br>

---

### ***(16) reverseSearchDirection***
`Code`
```python

```

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





