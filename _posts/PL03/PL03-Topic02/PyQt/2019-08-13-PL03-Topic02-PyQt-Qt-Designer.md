---
layout : post
title : PL03-Topic02, PyQt, Qt Designer
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-PyQt.html) ｜<a href="https://doc.qt.io/qtforpython/api.html" target="_blank">API</a>｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/PyQt/2019-08-13-PL03-Topic02-PyQt-Qt-Designer.md" target="_blank">page management</a> ｜<a href="https://www.youtube.com/playlist?list=PLnIaYcDMsScwsKo1rQ18cLHvBdjou-kb5" target="_blank">Notepad Lecture</a> ｜<a href="https://www.youtube.com/playlist?list=PLh665u8WZRR1d1hhLuZQThLhZbaOE5Whf" target="_blank">Calculator Lecture</a><br>
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
<b>ctrl + r</b> : form-preview on Qt-designer

### ***load MainWindow from qtdesigner***
#### basic form without qtdesigner
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
mainwindow = QMainWindow()
mainwindow.show()
app.exec_()
```
<br><br><br>

#### basic form with qtdesigner
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic

app = QApplication(sys.argv)
mainwindow = QMainWindow()
mainwindow = uic.loadUi("mainwindow.ui")
mainwindow.show()
app.exec()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">FILE PATH</summary>
<hr class='division3'>
![그림1](https://user-images.githubusercontent.com/52376448/64966645-22cf6c80-d8da-11e9-910d-740977ac18ad.png)
<div class='jb-medium'>when there exist the ui-file in parent folder,</div>
`mainwindow = uic.loadUi("../test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in same folder,</div>
`mainwindow = uic.loadUi("test.ui")`<br>
<div class='jb-medium'>when there exist the ui-file in sub-folder,</div>
`mainwindow = uic.loadUi("sub-folder/test.ui")`<br>

<hr class='division3'>
</details>

<br><br><br>
#### advanced form with qtdesigner
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("mainwindow.ui")[0]
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

### ***load Dialog from qtdesigner***
#### basic form without qtdesigner
```python
import sys
from PyQt5.QtWidgets import QApplication, QDialog

app = QApplication(sys.argv)
mainDialog = QDialog()
mainDialog.show()
app.exec_()
```

<br><br><br>

#### basic form with qtdesigner
```python
import sys
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5 import uic

app = QApplication(sys.argv)
mainDialog = QDialog()
uic.loadUi('dialog.ui',mainDialog)
mainDialog.show()
app.exec_()
```
<br><br><br>
#### advanced form with qtdesigner
```python
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

class MainDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('dialog.ui', self)

app = QApplication(sys.argv)
mainDialog = QDialog()
mainDialog.show()
app.exec_()
```
<br><br><br>
<hr class="division2">

## **Qt-designer API**
### ***Layouts***
<a href="" target="_blank"></a> ｜<a href="" target="_blank"></a> ｜
<br><br><br>

---

### ***Spacers***
<a href="" target="_blank"></a> ｜<a href="" target="_blank"></a> ｜
<br><br><br>

---

### ***Buttons***
<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QPushButton.html" target="_blank">QPushButton</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QToolButton.html" target="_blank">QToolButton</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QRadioButton.html" target="_blank">QRadioButton</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QCheckBox.html" target="_blank">QCheckBox</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QCommandLinkButton.html" target="_blank">QCommandLinkButton</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDialogButtonBox.html" target="_blank">QDialogButtonBox</a> ｜
<br><br><br>

---

### ***item views(model-based)***
<a href="" target="_blank"></a> ｜<a href="" target="_blank"></a> ｜
<br><br><br>

---

### ***item widgets(item-based)***
<a href="" target="_blank"></a> ｜<a href="" target="_blank"></a> ｜
<br><br><br>

---

### ***containers***
<a href="" target="_blank"></a> ｜<a href="" target="_blank"></a> ｜
<br><br><br>

---

### ***Input widgets***
<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QComboBox.html" target="_blank">QComboBox</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QFontComboBox.html" target="_blank">QFontComboBox</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QLineEdit.html" target="_blank">QLineEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QTextEdit.html" target="_blank">QTextEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QPlainTextEdit.html" target="_blank">QPlainTextEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QSpinBox.html" target="_blank">QSpinBox</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDoubleSpinBox.html" target="_blank">QDoubleSpinBox</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QTimeEdit.html" target="_blank">QTimeEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDateEdit.html" target="_blank">QDateEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDateTimeEdit.html" target="_blank">QDateTimeEdit</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDial.html" target="_blank">QDial</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QAbstractScrollArea.html" target="_blank">QAbstractScrollArea</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QScrollBar.html" target="_blank">QScrollBar</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QAbstractSlider.html" target="_blank">QAbstractSlider</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QSlider.html" target="_blank">QSlider</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QKeySequenceEdit.html" target="_blank">QKeySequenceEdit</a> ｜
<br><br><br>

---

### ***Display widgets***
<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QLabel.html" target="_blank">QLabel</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QTextBrowser.html" target="_blank">QTextBrowser</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QGraphicsView.html" target="_blank">QGraphicsView</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QCalendarWidget.html" target="_blank">QCalendarWidget</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QLCDNumber.html" target="_blank">QLCDNumber</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QProgressBar.html" target="_blank">QProgressBar</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtCore/QLine.html" target="_blank">QLine</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QOpenGLWidget.html" target="_blank">QOpenGLWidget</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtQuickWidgets/QQuickWidget.html" target="_blank">QQuickWidget</a> ｜<a href="https://doc.qt.io/qtforpython/PySide2/QtWebEngineWidgets/QWebEngineView.html" target="_blank">QWebEngineView</a>｜
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
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

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
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

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
        print("open")

    def saveFunction(self):
        print("save")

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>

- <a href="https://doc.qt.io/qtforpython/PySide2/QtCore/Signal.html" target="_blank">QtCore/Signal</a>
- <a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QAction.html" target="_blank">QtWidgets/QAction</a>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/71648735-824b6e00-2d4b-11ea-8dbb-357781466bd8.png)

<hr class='division3'>
</details>
<br><br><br>

---

### ***(4) Open/SaveAs***
![image](https://user-images.githubusercontent.com/52376448/71668207-4e079a00-2dab-11ea-92da-afe0b79d5292.png)

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
        self.action_saveas.triggered.connect(self.saveasFunction)

    def openFunction(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0]:
            with open(fname[0], encoding='UTF8') as f:
                data = f.read()
            self.plainTextEdit.setPlainText(data)
            print("open, {}".format(fname[0]))

    def saveasFunction(self):
        fname = QFileDialog.getSaveFileName(self)
        if fname[0]:
            data = self.plainTextEdit.toPlainText()
            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)
            print("save, {}".format(fname[0]))

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="https://doc.qt.io/qtforpython/PySide2/QtWidgets/QPlainTextEdit.html?highlight=qplaintextedit" target="_blank">plainTextEdit</a>
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/71659937-b5622180-2d8c-11ea-848f-e5b64c0bb258.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***(5) Open/SaveAs/Save***
![image](https://user-images.githubusercontent.com/52376448/71668153-21ec1900-2dab-11ea-9837-e35e1de03bb8.png)

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
        self.action_saveas.triggered.connect(self.saveasFunction)

        self.opened = False
        self.opened_file_path = ''

    def openFunction(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0]:
            with open(fname[0], encoding='UTF8') as f:
                data = f.read()
            self.plainTextEdit.setPlainText(data)
            self.opened = True
            self.opened_file_path = fname
            print("open, {}".format(fname))

    def saveasFunction(self):
        fname = QFileDialog.getSaveFileName(self)
        if fname[0]:
            data = self.plainTextEdit.toPlainText()
            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)
            self.opened = True
            self.opened_file_path = fname
            print("saveas, {}".format(fname))

    def saveFunction(self):
        if self.opened:
            fname = self.opened_file_path
            data = self.plainTextEdit.toPlainText()
            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)
            self.opened = True
            self.opened_file_path = fname
            print("save, {}".format(fname))
        else:
            self.saveasFunction()


app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(6) CloseEvent with MessageBox***
![image](https://user-images.githubusercontent.com/52376448/71667811-fddc0800-2da9-11ea-83ca-c5cbb1b370f6.png)

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
        self.action_saveas.triggered.connect(self.saveasFunction)
        self.action_close.triggered.connect(self.close)  # self.close is called with self.closeEvent

        self.opened = False
        self.opened_file_path = ''

    def openFunction(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0]:
            with open(fname[0], encoding='UTF8') as f:
                data = f.read()
            self.plainTextEdit.setPlainText(data)
            self.opened = True
            self.opened_file_path = fname
            print("open, {}".format(fname))

    def saveasFunction(self):
        fname = QFileDialog.getSaveFileName(self)
        if fname[0]:
            data = self.plainTextEdit.toPlainText()
            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)
            self.opened = True
            self.opened_file_path = fname
            print("saveas, {}".format(fname))

    def saveFunction(self):
        if self.opened:
            fname = self.opened_file_path
            data = self.plainTextEdit.toPlainText()
            with open(fname[0], 'w', encoding='UTF8') as f:
                f.write(data)
            self.opened = True
            self.opened_file_path = fname
            print("save, {}".format(fname))
        else:
            self.saveasFunction()

    def closeEvent(self, event):
        current_data = ''
        def ischanged():
            nonlocal current_data
            if not self.opened:
                print('This file was not loaded')
                if self.plainTextEdit.toPlainText().strip():
                    return True
                return False

            current_data = self.plainTextEdit.toPlainText()
            with open(self.opened_file_path[0], encoding='UTF8') as f:
                file_data = f.read()
            if current_data == file_data:
                return False
            else:
                return True

        def save_changed_data():
            msgBox = QMessageBox()
            msgBox.setText('Do you want to changes to {}'.format(self.opened_file_path))
            msgBox.addButton('Save(S)', QMessageBox.YesRole) #0
            msgBox.addButton('Don\'t Save(N)', QMessageBox.NoRole) #1
            msgBox.addButton('Cancel', QMessageBox.RejectRole) #2
            ret = msgBox.exec_()
            if ret == 0:
                self.saveFunction()
                event.ignore()
            else:
                return ret

        if ischanged():
            ret = save_changed_data()
            if ret == 2:
                event.ignore()

app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(8) PlainTextEdit***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(9) Find***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(10) KeyboardEvent***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(11) SetCursor***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(12) FindText***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(13) IgnoreFlag***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(14) Radiobutton***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(15) SearchDirection***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

<br><br><br>

---

### ***(16) reverseSearchDirection***
`Code`
```python

```
<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

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


<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">API</summary>
<hr class='division3'>
- <a href="" target="_blank"></a>
<hr class='division3'>
</details>

