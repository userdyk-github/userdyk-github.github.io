---
layout : post
title : PL03-Topic01, Class and method
categories: [PL03-Topic01]
comments : true
tags : [PL03-Topic01]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Syntax-and-semantics.html) <br>
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

## **Class advanced**
### ***Basic class information***
```python
class info:
    pass

CLA = info()
print(CLA.__class__)
print(which.__class__(CLA))
```
<br><br><br>

---


### ***__init__, __del__***
```python
class Variable:
    count = 0
    def __init__(self,a):
        self.a = a
        Variable.count += 1
        
        
    def __del__(self):
        Variable.count -= 1

print(Variable.count)
CLA1=Variable(1)
CLA2=Variable(2)
CLA3=Variable(3)

print(Variable.count)
print(CLA1.count)
print(CLA2.count)

del CLA1
print(Variable.count)
```
<br><br><br>

---


### ***__dict__, dir***
```python
class DICT:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
print(DICT.__dict__)
print(dir(DICT))

CLA=DICT(1,2)
print(CLA.__dict__)
print(dir(CLA))
```
<br><br><br>

---

### ***Description***
```python
class Description():
    """
    Description Class
    Author : Lee
    Date : 2019.05.25
    """
    def __str__(self):
        return 'hello, str'
    def __repr__(self):
        return 'hello, repr'
    
CLA=Description()
print(CLA.__doc__)
print(CLA)
print(CLA.__repr__())
```
<br><br><br>

---

<hr class="division2">

## **Class, Instance, Static Method**
```python
class vec():
    __abc = 0
    
    def __init__(self,*args):
        if len(args) == 0 :
            self.__x, self.__y = 0,0
        else:
            self.__x, self.__y = args

    @classmethod
    def create(cls):
        print('Instance have been created')
        return cls()
    
    @classmethod
    def cls_variable(cls):
        cls.__abc += 1
        print('cls.abc : ',cls.__abc)
    
    @staticmethod
    def inst_variable(inst):
        inst.__x += 1
        print('inst.x : ',inst.__x)
            
a = vec.create()
vec.cls_variable()
vec.inst_variable(a)
```
<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference

- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>




