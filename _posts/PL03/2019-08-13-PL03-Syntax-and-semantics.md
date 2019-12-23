---
layout : post
title : PL03, Syntax and semantics
categories: [PL03]
comments : true
tags : [PL03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/2019-08-13-PL03-Syntax-and-semantics.md" target="_blank">page management</a>｜ <a href="https://userdyk-github.github.io/pl03/PL03-Contents.html">Contents</a> <br>
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

## **Basis of python programming, Data structure**

<a href="https://wikidocs.net/11" target="_blank">URL</a>

### ***Number***

#### Integer

`int`
```python
type(10)
```
<span class="jb-medium">int</span>
<br>
```python
type(10 * 5)
```
<span class="jb-medium">int</span>

<br><br><br>

#### floating point number

`float`
```python
type(10.0)
```
<span class="jb-medium">float</span>
<br>
```python
type(0.1)
```
<span class="jb-medium">float</span>
<br>
```python
type(10.0 * 5)
```
<span class="jb-medium">float</span>
<br>
```python
type(10 / 5)
```
<span class="jb-medium">float</span>
<br>
```python
type(123e2)    # 12300.0
```
<span class="jb-medium">float</span>
<br>
```python
type(123e-2)   # 1.23
```
<span class="jb-medium">float</span>
<br>
```python
type(123.456e-3)   # 0.123456
```
<span class="jb-medium">float</span>

<br><br>

`floating point error`
```python
0.1
```
<span class="jb-medium">0.1</span>
```python
# method1
%precision 55   
0.1
```
<span class="jb-medium">0.1000000000000000055511151231257827021181583404541015625</span>

<details markdown="1">
<summary class='jb-small' style="color:blue">back to origin</summary>
<hr class='division3'>
```python
%precision %r
0.1
```
<span class="jb-medium">0.1</span>
<hr class='division3'>
</details>
<br>
```python
# method2
'%.55f'%0.1
```
<span class="jb-medium">0.1000000000000000055511151231257827021181583404541015625</span>

<br><br>

`comparsion of floating point numbers`
```python
0.1 + 0.2 == 0.3
```
<span class="jb-medium">False</span>

<details markdown="1">
<summary class='jb-small' style="color:blue">CAUTION</summary>
<hr class='division3'>
```python
0.1 + 0.2
```
<span class="jb-medium">0.30000000000000004</span>
```python
0.3
```
<span class="jb-medium">0.3</span>
<hr class='division3'>
</details>
<br>
```python
round(0.1 + 0.2, 5) == round(0.3, 5)
```
<span class="jb-medium">True</span>

<br><br><br>

#### Cast
`float->int`
```python
int(1.0)
```
<span class="jb-medium">1</span>
```python
int(3.14)
```
<span class="jb-medium">3</span>
```python
int(3.9)
```
<span class="jb-medium">3</span>
```python
int(-3.9)
```
<span class="jb-medium">-3</span>

<br>
`int->float`
```python
float(1) 
```
<span class="jb-medium">1.0</span>


<br><br><br>


#### NaN and inf

```python
float("NaN")
```
<span class="jb-medium">nan</span>
<br>

```python
float("Inf")
```
<span class="jb-medium">inf</span>
<br>

```python
float("-Inf")
```
<span class="jb-medium">-inf</span>
<br>

---

### ***String***

<br><br><br>

---

### ***List***

<br><br><br>

---

### ***Tuple***

<br><br><br>

---

### ***Dictionary***

```python
numbers = dict(x=5, y=0)
print('numbers =', numbers)
print(type(numbers))

empty = dict()
print('empty =', empty)
print(type(empty))
```
```
numbers = {'y': 0, 'x': 5}
<class 'dict'>
empty = {}
<class 'dict'>
```
<br><br><br>

---

### ***Set***

```python
s = {1, 2, 3}
s |= {4}
s
```
<span class="jb-medium">{1, 2, 3, 4}</span><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">Equivalent code</summary>
<hr class='division3'>
```python
s = set([1, 2, 3])
s.add(4)
s
```
<span class="jb-medium">{1, 2, 3, 4}</span>
<hr class='division3'>
</details>
<br>


```python
s = {1, 2, 3}
s |= {4,5,6}
s
```
<span class="jb-medium">{1, 2, 3, 4, 5, 6}</span><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">Equivalent code</summary>
<hr class='division3'>
```python
s = set([1, 2, 3])
s.update([4, 5, 6])
s
```
<span class="jb-medium">{1, 2, 3, 4, 5, 6}</span>
<hr class='division3'>
</details>
<br>

```python
s = {1, 2, 3}
s.remove(2)
s
```
<span class="jb-medium">{1, 3}</span><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">Equivalent code</summary>
<hr class='division3'>
```python
s = set([1, 2, 3])
s.remove(2)
s
```
<span class="jb-medium">{1, 3}</span>
<hr class='division3'>
</details>

<br><br><br>

---

### ***Bool***

<br><br><br>

---

### ***Variable***


<br><br><br>
<hr class="division2">

## **Build the structure of the program! Control statement**

<a href="https://wikidocs.net/19" target="_blank">URL</a>


### ***if***

#### is, ==

|is|reference|
|==|value|

```python
a = 1

print(a is 1)
print(a == 1)
print(id(a))
print(id(1))
```
```
True
True
1531412592
1531412592
```
```python
a = 257

print(a is 257)
print(a == 257)
print(id(a))
print(id(257))
```
```
False
True
2396517385200
2396517385616
```
<br><br><br>

#### comprehension

```python
a = 10
b = {a > 0 : 1,
     a == 0 : 0}.get(True, -1)
b
```
```python
a = 10

if a > 0:
    b = 1
elif a == 0:
    b = 0
else:
    b = -1
    
b
```
```python
a = 10
b = 1 if a > 0 else ( 0 if a==0 else -1)
b
```


---

### ***while***

<br><br><br>

---

### ***for***

<br><br><br>
<hr class="division2">

## **How to do the input and output of the program**

<a href="https://wikidocs.net/23" target="_blank">URL</a>


### ***Function***
#### position arguments
```python
def hello(a,b,c):
    print(a)
    print(b)
    print(c)

x = [1,2,3]
y = (1,2,3)
hello(*x)   # hello(*[1,2,3])
hello(*y)   # hello(*(1,2,3))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
1
2
3
1
2
3
```
<hr class='division3'>
</details>
```python
def hello(*args):
    print(args)

x = [1,2,3]
y = (1,2,3)
hello(*x)   # hello(*[1,2,3])
hello(*y)   # hello(*(1,2,3))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
1
2
3
1
2
3
```
<hr class='division3'>
</details>

<br><br><br>

#### keyword arguments
```python
def hello(name,age,address):
    print('name',name)
    print('age',age)
    print('address',address)

x = {'name':'ailever', 'age':27, 'address':312321}
hello(*x)
hello(**x)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
name name
age age
address address
name ailever
age 27
address 312321
```
<hr class='division3'>
</details>
```python
def hello(**kwargs):
    print(kwargs)
    
x = {'name':'ailever', 'age':27, 'address':312321}
hello(**x)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
name ailever
age 27
address 312321
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***Input/Ouput***

<br><br><br>

---

### ***Read/Write***

<br><br><br>
<hr class="division2">

## **Advanced python**

<a href="https://wikidocs.net/27" target="_blank">URL</a>


### ***Class***

#### Declare Class

```python
class Person:
    pass

p1 = Person()
p2 = Person()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
print(type(p1))
print(type(p2))
```
```
<class '__main__.Person'>
<class '__main__.Person'>
```
<hr class='division3'>
</details>
<br><br><br>

#### Class Constructor
```python
class Person:
    def __init__(self):
        self.name = ""
        self.age = 0

p1 = Person()
p1.name = 'bob'
p1.age = 21

p2 = Person()
p2.name = 'cathy'
p2.age = 25
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
print(p1)
print(p1.name)
print(p1.age)

print(p2)
print(p2.name)
print(p2.age)
```
```
<__main__.Person object at 0x000001C35285EDD8>
bob
21

<__main__.Person object at 0x000001C35285ED30>
cathy
25
```
<hr class='division3'>
</details>
<br><br><br>

#### Setter method

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person('bob', 21)
p1.name
p1.age

p2 = Person('cathy', 25)
p2.name
p2.age
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```python
print(p1)
print(p1.name)
print(p1.age)

print(p2)
print(p2.name)
print(p2.age)
```
```
<__main__.Person object at 0x000001C352867978>
bob
21

<__main__.Person object at 0x000001C352867358>
cathy
25
```
<hr class='division3'>
</details>
<br><br><br>


<span class="frame3">hasattr(object, name)</span><br>

> <strong>object</strong> - object whose named attribute is to be checked<br>
> <strong>name</strong> - name of the attribute to be searched<br>
> <strong>return</strong><br>
>> <strong>True</strong>, if object has the given named attribute<br>
>> <strong>False</strong>, if object has no given named attribute<br>

```python
class Person:
    age = 23
    name = 'Adam'

person = Person()

print('Person has age?:', hasattr(person, 'age'))
print('Person has salary?:', hasattr(person, 'salary'))
```
```
Person has age?: True
Person has salary?: False
```
<br><br><br>

#### Class Inheritance and Inclusion
<span class="frame3">Inheritance</span><br>
```python
class Person:
    def greeting(self):
        print('hello')

class Student(Person):
    def study(self):
        print('study')

james = Student()
james.greeting()
james.study()
```
<p style="font-size: 70%;">
hello<br>
study
</p>
```python
class Person:
    def __init__(self):
        print('Person')
        self.hello = 'hello'
    
class Student(Person):
    def __init__(self):
        print('Student')
        super().__init__()
        self.school = 'school'
        
james = Student()
print(james.school)
print(james.hello)
```
<p style="font-size: 70%;">
Student<br>
Person<br>
school<br>
hello
</p><br>

<details markdown="1">
<summary class='jb-small' style="color:blue">Error</summary>
<hr class='division3'>
```python
class Person:
    def __init__(self):
        print('Person')
        self.hello = 'hello'
    
class Student(Person):
    def __init__(self):
        print('Student')
        self.school = 'school'
        
james = Student()
print(james.school)
print(james.hello)
```
<p style="font-size: 70%;">
---> 13 print(james.hello)<br>
AttributeError: 'Student' object has no attribute 'hello'
</p>
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Caution</summary>
<hr class='division3'>
```python
class Person:
    def __init__(self):
        print('Person')
        self.hello = 'hello'
    
class Student(Person):
    pass        

james = Student()
print(james.hello)
```
<p style="font-size: 70%;">
Person<br>
hello
</p>
<hr class='division3'>
</details>
<br><br><br>
<span class="frame3_1">Multiple Inheritance</span><br>
```python
class A:
    def greeting(self):
        print('hello, A')
        
class B(A):
    def greeting(self):
        print('hello, B')
        
class C(A):
    def greeting(self):
        print('hello, C')
        
class D(B,C):    # left side has a priority
    pass
        
x = D()
x.greeting()
```
<p style="font-size: 70%;">
hello, B
</p><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">MRO:Method Resolution Order</summary>
<hr class='division3'>
```python
D.mro()
```
<p style="font-size: 70%;">
[__main__.D, __main__.B, __main__.C, __main__.A, object]
</p>
<hr class='division3'>
</details>


<br><br><br>
<span class="frame3">Inclusion</span><br>
```python
class Person:
    def greeting(self):
        print('hello')

class PersonList():
    def __init__(self):
        self.person_list = []
        
    def append_person(self, person):
        self.person_list.append(person)

recode = PersonList()
james = Person()
recode.append_person(james)
recode.person_list
```
<p style="font-size: 70%;">
    [<__main__.Person at 0x7f2158a17e48>]
</p>

<br><br><br>

#### Overiding Class
```python
class Person:
    def greeting(self):
        print('hello, Person')

class Student(Person):
    def greeting(self):
        print('hello, Student')

james = Student()
james.greeting()
```
<p style="font-size: 70%;">
hello, Student
</p>

```python
class Person:
    def greeting(self):
        print('hello, Person')

class Student(Person):
    def greeting(self):
        super().greeting()
        print('hello, Student')

james = Student()
james.greeting()
```
<p style="font-size: 70%;">
hello, Person<br>
hello, Student
</p>
<br><br><br>

#### Abstract Class
```python
from abc import *   # abc : abstract base class

class StudentBase(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass
    
    @abstractmethod
    def go_to_school(self):
        pass
    
class Student(StudentBase):
    def study(self):
        print('study')

    def go_to_school(self):
        print('go to school')
        
james = Student()
james.study()
james.go_to_school()
```
<p style="font-size: 70%;">
    study<br>
go to school
</p><br>
<details markdown="1">
<summary class='jb-small' style="color:blue">Error</summary>
<hr class='division3'>
```python
from abc import *   # abc : abstract base class

class StudentBase(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass
    
    @abstractmethod
    def go_to_school(self):
        pass
    
class Student(StudentBase):
    def study(self):
        print('study')

james = Student()
```
<p style="font-size: 70%;">
---> 16 james = Student()<br>
TypeError: Can't instantiate abstract class Student with abstract methods go_to_school
</p>
<hr class='division3'>
</details>

<br><br><br>

#### Meta Class
<span class="frame3">Method 1</span><br>
```python
# class = type('name_of_class', (base_class), {property:method})

Hello = type('Hello',(),{})
h = Hello()
h
```

<p style="font-size: 70%;">
    <__main__.Hello at 0x7f21589b5080>
</p>
    
```python
def replace(self, old, new):
    while old in self:
        self[self.index(old)] = new

AdvancedList = type('AdvancedList', (list,), {'desc':'improved list', 'replace':replace})

x = AdvancedList([1,2,3,1,2,3,1,2,3])
x.replace(1,100)
print(x)
print(x.desc)
```

<p style="font-size: 70%;">
    [100, 2, 3, 100, 2, 3, 100, 2, 3]<br>
improved list
</p>

<br><br><br>
<span class="frame3">Method 2</span><br>
```python
class MakeCalc(type):
    def __new__(metacls, name, bases, namespace):
        namespace['desc'] = 'calc class'
        namespace['add'] = lambda self, a, b : a + b
        return type.__new__(metacls, name, bases, namespace)
    
Calc = MakeCalc('Calc', (), {})
c = Calc()
print(c.desc)
print(c.add(1,2))
```
<p style="font-size: 70%;">
    calc class<br>
3
</p>

```python
# Singleton

class Singleton(type):
    __instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]
    
class Hello(metaclass=Singleton):
    pass

a = Hello()
b = Hello()
print(a is b)
```
<p style="font-size: 70%;">
    True
</p>

<br><br><br>


---

### ***Module***

```python
if __name__ == "__main__":
    pass
```
<br><br><br>

```python
help(object)
```

<br><br><br>

---

### ***Package***

<br><br><br>

---

### ***Exception handling***

```python
try:
    x = int(input('enter number : '))
    y = 10 / x 
except ZeroDivisionError:    # if exception is occured
    print('cannot divide')
else:                        # if exception is not occured
    print(y)
finally:                     # always
    print('end')
```
<p style="font-size: 70%;">
enter number : 2<br>
5.0<br>
end

</p>
<br><br><br>
#### Python error hierarchy
<a href="https://docs.python.org/3/library/exceptions.html" target="_blank">URL</a>
![image](https://user-images.githubusercontent.com/52376448/69781536-db2a6280-11f1-11ea-8e93-86e166cf3425.png)

<br><br><br>

---

### ***Built-in function***
<span class="frame3">abs</span><br>
```python
abs(3)
```
```
3
```
<br>
```python
abs(-3)
```
```
3
```
<br>
```python
abs(-1.2)
```
```
1.2
```
<br>
<span class="frame3">all</span><br>
```python
all([1, 2, 3])
```
```
True
```
<br>
```python
all([1, 2, 3, 0])
```
```
False
```
<br>


<span class="frame3">any</span><br>
```python
any([1, 2, 3, 0])
```
```
True
```
<br>
```python
any([0, ""])
```
```
False
```
<br>

<span class="frame3">chr</span><br>

```python
chr(97)
```
```
'a'
```
```python
chr(48)
```
```
'0'
```
<br>

<span class="frame3">dir</span><br>

```python
dir([1, 2, 3])
```
```
['append', 'count', 'extend', 'index', 'insert', 'pop',...]
```
<br>
```python
dir({'1':'a'})
```
```
['clear', 'copy', 'get', 'has_key', 'items', 'keys',...]
```
<br>

<span class="frame3">divmod</span><br>

```python
divmod(7, 3)
```
```
(2, 1)
```
<br>
```
7 // 3, 7 % 3
```
```
(2, 1)
```
<br>

<span class="frame3">enumerate</span><br>

```python
for i, name in enumerate(['body', 'foo', 'bar']):
    print(i, name)
```
```
0 body
1 foo
2 bar
```
<br>

<span class="frame3">eval</span><br>

```python
eval('1+2')
```
```
3
```
<br>
```python
eval("'hi' + 'a'")
```
```
'hia'
```
<br>
```python
eval('divmod(4, 3)')
```
```
(1, 1)
```
<br>

<span class="frame3">filter</span><br>

```python

```
```

```
<br>

<span class="frame3">hex</span><br>

```python
hex(234)
```
```
'0xea'
```
<br>
```python
hex(3)
```
```
'0x3'
```
<br>


<span class="frame3">id</span><br>
```python
>>> a = 3
>>> id(3)
```
```
135072304
```
<br>
```python
id(a)
```
```
135072304
```
<br>
```python
b = a
id(b)
```
```
135072304
```
<br>

 
<span class="frame3">input</span><br>
```python
>>> a = input()
hi
>>> a
'hi'

>>> b = input("Enter: ")
Enter: hi
>>> b
'hi'
```
<br>


<span class="frame3">int</span><br>
```python
int('3')
```
```
3
```
<br>
```python
int(3.4)
```
```
3
```
<br>

<span class="frame3">isinstance</span><br>

```python
class Person: pass

a = Person()
isinstance(a, Person)
```
```
True
```
<br>
```python
b = 3
isinstance(b, Person)
```
```
False
```
<br>


<span class="frame3">len</span><br>

```python
len("python")
```
```
6
```
<br>
```python
len([1,2,3])
```
```
3
```
<br>
```python
len((1, 'a'))
```
```
2
```
<br>


<span class="frame3">list</span><br>


```python
list("python")
```
```
['p', 'y', 't', 'h', 'o', 'n']
```
<br>
```python
list((1,2,3))
```
```
[1, 2, 3]
```
<br>

<span class="frame3">map</span><br>

```python
# two_times.py
def two_times(numberList):
    result = [ ]
    for number in numberList:
        result.append(number*2)
    return result

result = two_times([1, 2, 3, 4])
print(result)
```
```
[2, 4, 6, 8]
```
<br>
```python
def two_times(x): 
    return x*2

list(map(two_times, [1, 2, 3, 4]))
```
```
[2, 4, 6, 8]
```
<span class="frame3">max</span><br>

```python
max([1, 2, 3])
```
```
3
```
<br>
```python
max("python")
```
```
'y'
```
<br>

<span class="frame3">min</span><br>

```python
min([1, 2, 3])
```
```
1
```
<br>
```python
min("python")
```
```
'h'
```
<br>

<span class="frame3">oct</span><br>

```python
oct(34)
```
```
'0o42'
```
<br>
```python
oct(12345)
```
```
'0o30071'
```
<br>

<span class="frame3">open</span><br>

|mode|description|method|
|:-|:-|:-|
|w|write||
|r|read|seek,read,readline,readlines|
|a|append||
|b|binary||

```python
f = open("binary_file", "rb")
f.close()

with open("binary_file","rb") as f:
    pass
```
```python
fread = open("read_mode.txt", 'r')
fread2 = open("read_mode.txt")
```
```python
fappend = open("append_mode.txt", 'a')
```
<br>

<span class="frame3">ord</span><br>

```python
ord('a')
```
```
97
```
<br>
```python
ord('0')
```
```
48
```
<br>

<span class="frame3">pow</span><br>
```python
pow(2, 4)
```
```
16
```
<br>
```python
pow(3, 3)
```
```
27
```
<br>

<span class="frame3">range</span><br>

```python
list(range(5))
```
```
[0, 1, 2, 3, 4]
```
<br>
```python
list(range(5, 10))
```
```
[5, 6, 7, 8, 9]
```
<br>
```python
list(range(1, 10, 2))
```
```
[1, 3, 5, 7, 9]
```
<br>
```python
list(range(0, -10, -1))
```
```
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
```
<br>

<span class="frame3">round</span><br>

```python
round(4.6)
```
```
5
```
<br>
```python
round(4.2)
```
```
4
```
<br>
```python
round(5.678, 2)
```
```
5.68
```
<br>
<span class="frame3">sorted</span><br>

```python
sorted([3, 1, 2])
```
```
[1, 2, 3]
```
<br>
```python
sorted(['a', 'c', 'b'])
```
```
['a', 'b', 'c']
```
<br>
```python
sorted("zero")
```
```
['e', 'o', 'r', 'z']
```
<br>
```python
sorted((3, 2, 1))
```
```
[1, 2, 3]
```
<br>

<span class="frame3">str</span><br>

```python
str(3)
```
```
'3'
```
<br>
```python
str('hi')
```
```
'hi'
```
<br>
```python
str('hi'.upper())
```
```
'HI'
```
<br>

<span class="frame3">sum</span><br>

```python
sum([1,2,3])
```
```
6
```
<br>
```python
sum((4,5,6))
```
```
15
```
<br>

<span class="frame3">tuple</span><br>

```python
tuple("abc")
```
```
('a', 'b', 'c')
```
<br>
```python
tuple([1, 2, 3])
```
```
(1, 2, 3)
```
<br>
```python
tuple((1, 2, 3))
```
```
(1, 2, 3)
```
<br>

<span class="frame3">type</span><br>

```python
type("abc")
```
```
<class 'str'>
```
<br>
```python
type([ ])
```
```
<class 'list'>
```
<br>
```python
type(open("test", 'w'))
```
```
<class '_io.TextIOWrapper'>
```
<br>
<span class="frame3">zip</span><br>

```python
list(zip([1, 2, 3], [4, 5, 6]))
```
```
[(1, 4), (2, 5), (3, 6)]
```
<br>
```python
list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9]))
```
```
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```
<br>
```python
list(zip("abc", "def"))
```
```
[('a', 'd'), ('b', 'e'), ('c', 'f')]
```


<br><br><br>

---

### ***External function***

#### sys
```python
import sys
print(sys.argv)
```
```dos
C:/doit/Mymod>python argv_test.py you need python
```
```
['argv_test.py', 'you', 'need', 'python']
```
<br>
```python
import sys

print('hello1')
sys.exit()
print('hello2')
```
```dos
C:/doit/Mymod>python argv_test.py you need python
```
```
hello1
['argv_test.py', 'you', 'need', 'python']
```
<br>
```python
import sys

print(sys.path)
```
```
['', 'C:\\Windows\\SYSTEM32\\python37.zip', 'c:\\Python37\\DLLs', 
'c:\\Python37\\lib', 'c:\\Python37', 'c:\\Python37\\lib\\site-packages']
```

<br>
```python
import sys

sys.path.append('C:/doit/mymod')
print(sys.path)
```
```
['', 'C:\\Windows\\SYSTEM32\\python37.zip', 'c:\\Python37\\DLLs', 
'c:\\Python37\\lib', 'c:\\Python37', 'c:\\Python37\\lib\\site-packages', 'C:/doit/mymod']
```

<br>

#### pickle
<span class="frame3">Save</span>
```python
import pickle

f = open("test.txt", 'wb')
data = {1: 'python', 2: 'you need'}
pickle.dump(data, f)
f.close()
```
<br>
<span class="frame3">Load</span>
```python
import pickle

f = open("test.txt", 'rb')
data = pickle.load(f)
print(data)
```
```
{2:'you need', 1:'python'}
```
<br>

#### os

|method|description|
|:--|:--|
|os.chdir('path')|change directory|
|os.getcwd()|get current working directory|
|os.listdir('path')|list of files on directory|


```python
import os

os.environ
```
```
environ({'PROGRAMFILES': 'C:\\Program Files', 'APPDATA': … 생략 …})
```
<br>
```python
import os

os.chdir("C:\WINDOWS")
```
<br>
```python
import os

os.getcwd()
```
```
'C:\WINDOWS'
```
<br>
#### shutil
```python
```
<br>

#### glob
```python
```
<br>

#### tempfile
```python
```
<br>

#### time
```python
```
<br>

#### calendar
```python
```
<br>

#### random
```python
```
<br>

#### webbrowser
```python
```

<br><br><br>

---

### ***Decorator***
#### Function Decorator
```python
def hello():
    print('hello start')
    print('hello')
    print('hello end')

def world():
    print('world start')
    print('world')
    print('world end')
    
hello()
world()
```
<p style="font-size: 70%;">
    hello start<br>
    hello<br>
    hello end<br>
    world start<br>
    world<br>
    world end
</p>

<br><br><br>

```python
def trace(func):
    def wrapper():
        print(func.__name__, 'start')
        func()
        print(func.__name__, 'end')
    return wrapper

def hello():
    print('hello')

def world():
    print('world')

trace_hello = trace(hello)
trace_hello()
trace_world = trace(world)
trace_world()
```
<p style="font-size: 70%;">
    hello start<br>
    hello<br>
    hello end<br>
    world start<br>
    world<br>
    world end
</p>

<br><br><br>

```python
def trace(func):
    def wrapper():
        print(func.__name__, 'start')
        func()
        print(func.__name__, 'end')
    return wrapper

@trace
def hello():
    print('hello')

@trace
def world():
    print('world')

hello()
world()
```
<p style="font-size: 70%;">
    hello start<br>
    hello<br>
    hello end<br>
    world start<br>
    world<br>
    world end
</p>

<br><br><br>
<span class="frame3">Decorator with arguments</span><br>
```python
def trace(func):
    def wrapper(*args, **kwargs):
        r = func(*args, **kwargs)
        print('{0}(args={1}, kwargs={2}) -> {3}'.format(func.__name__, args, kwargs, r))
        return r
    return wrapper

@trace
def get_max(*args):
    return max(args)

@trace
def get_min(**kwargs):
    return min(kwargs.values())

print(get_max(10,20))
print(get_min(x=10, y=20, z=30))
```
<p style="font-size: 70%;">
get_max(args=(10, 20), kwargs={}) -> 20<br>
20<br>
get_min(args=(), kwargs={'x': 10, 'y': 20, 'z': 30}) -> 10<br>
10
</p>

<details markdown="1">
<summary class='jb-small' style="color:blue">Example</summary>
<hr class='division3'>
```python
def trace(x):
    def decorator(func):
        def wrapper(a,b):
            r = func(a,b)
            if r % x == 0:
                print('returned value of {0} is multiple of {1}'.format(func.__name__, x))
            else:
                print('returned value of {0} is not multiple of {1}'.format(func.__name__, x))
            return r
        return wrapper
    return decorator

@trace(3)
def add(a,b):
    return a + b

print(add(10,20))
print(add(2,5))
```
<p style="font-size: 70%;">
returned value of add is multiple of 3<br>
30<br>
returned value of add is not multiple of 3<br>
7
</p>

<hr class='division3'>
</details>

<br><br><br>

#### Class Decorator
```python
class trace:
    def __init__(self, func):
        self.func = func
    
    def __call__(self):
        print(self.func.__name__, 'start')
        self.func()
        print(self.func.__name__, 'end')

def hello():
    print('hello')

@trace
def world():
    print('world')

trace_hello = trace(hello)
trace_hello()
world()
```
<p style="font-size: 70%;">
    hello start<br>
    hello<br>
    hello end<br>
    world start<br>
    world<br>
    world end
</p>

<br><br><br>
<span class="frame3">Decorator with arguments</span><br>
```python
class trace:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        r = self.func(*args, **kwargs)
        print('{0}(args={1}, kwargs={2}) -> {3}'.format(self.func.__name__, args, kwargs, r))
        return r
        
@trace
def add(a, b):
    return a + b

print(add(10,20))
print(add(a=10, b=20))
```
<p style="font-size: 70%;">
add(args=(10, 20), kwargs={}) -> 30<br>
30<br>
add(args=(), kwargs={'a': 10, 'b': 20}) -> 30<br>
30<br>
</p>

<details markdown="1">
<summary class='jb-small' style="color:blue">Example</summary>
<hr class='division3'>
```python
class trace:
    def __init__(self, x):
        self.x = x
    
    def __call__(self, func):
        def wrapper(a,b):
            r = func(a,b)
            if r % self.x == 0:
                print('returned value of {0} is mutiple of {1}'.format(func.__name__, self.x))
            else:
                print('returned value of {0} is not mutiple of {1}'.format(func.__name__, self.x))
            return r
        return wrapper
    
@trace(3)
def add(a, b):
    return a + b

print(add(10,20))
print(add(2, 5))
```
<p style="font-size:70%;">
returned value of add is mutiple of 3<br>
30<br>
returned value of add is not mutiple of 3<br>
7
</p>

<hr class='division3'>
</details>


<br><br><br>

---

### ***Iterator***

<br><br><br>

---

### ***Generator***

#### Yeild
```python
# 'yield(co)' is different 'return(sub)'
def number_generator():
    yield 0
    yield 1
    yield 2
    
for i in number_generator():
    print(i)
```
<p style="font-size: 70%;">
    0<br>
1<br>
2
</p>
```python
g = number_generator()
print(g.__next__())
print(g.__next__())
print(g.__next__())
print(g.__next__())
```
<p style="font-size: 70%;">
0<br>
1<br>
2<br>
      3 print(g.__next__())<br>
      4 print(g.__next__())<br>
----> 5 print(g.__next__())<br>
<br>
StopIteration: 
</p>

<br><br><br>
```python
def number_generator(stop):
    n = 0
    while n < stop :
        yield n
        n += 1
        
for i in number_generator(3):
    print(i)

for i in range(3):
    print(i)
```
<p style="font-size: 70%;">
0<br>
1<br>
2<br>
0<br>
1<br>
2
</p>
```python
g = number_generator(3)
print(next(g))
print(next(g))
print(next(g))
```
<p style="font-size: 70%;">
0<br>
1<br>
2
</p>
<br><br><br>
```python
def upper_generator(x):
    for i in x:
        yield i.upper()

fruits = ['apple', 'pear', 'grape']

for i in upper_generator(fruits):
    print(i)
    
    
for i in fruits:
    print(i.upper())
```
<p style="font-size: 70%;">
0<br>
1<br>
2
</p>

<br><br><br>
```python
def number_generator1():
    x = [1,2,3]
    for i in x:
        yield i

for i in number_generator1():
    print(i)
    
    
def number_generator2():
    x = [1,2,3]
    yield from x
    
for i in number_generator2():
    print(i)
```
<p style="font-size: 70%;">
0<br>
1<br>
2<br>
0<br>
1<br>
2

</p>

<br><br><br>
```python
def number_generator(stop):
    n = 0
    while n < stop:
        yield n
        n += 1
        
def three_generator():
    yield from number_generator(3)
    
for i in three_generator():
    print(i)
```
<p style="font-size: 70%;">
0<br>
1<br>
2

</p>

<br><br><br>


<hr class="division2">

## **Python programming, how do I get started?**

<a href="https://wikidocs.net/34" target="_blank">URL</a>

### ******

<br><br><br>

---

### ******

<br><br><br>

---

### ******

<br><br><br>

---

### ******

<br><br><br>

---

### ******

<br><br><br>

---

### ******

<br><br><br>

---

### ******

<br><br><br>
<hr class="division2">

## **Regular expression**

<a href="https://wikidocs.net/1669" target="_blank">URL</a>

### ***Explore regular expressions***

#### re module

<span class="frame3">Raw string</span><br>
```python
print(r'abcd/n')
```
```
abcd/n
```
<br><br><br>

<span class="frame3">Search method</span><br>
```python
import re

re.search(r'abc','abcdef')
```
```
<_sre.SRE_Match object; span=(0, 3), match='abc'>
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
m = re.search(r'abc','abcdef')
print(m.start())
print(m.end())
print(m.group())
```
```
0
3
abc
```
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3_1">Examples</span><br>
```python
import re

re.search(r'\d\d\d\w','efw2342efwefwef')
```
```
<_sre.SRE_Match object; span=(3, 7), match='2342'>
```
```python
re.search(r'..\w\w','efw@#$23$@')
```
```
<_sre.SRE_Match object; span=(4, 8), match='#$23'>
```
<br><br><br>


<span class="frame3_1">Metacharacter</span><br>

|Meta(1)|Expr|
|:--|:--|
|[abck]|a,b,c,k|
|[abc.^]|a,b,c,.,^|
|[a-d]|range|
|[0-9]|range|
|[a-z]|range|
|[A-Z]|range|
|[a-zA-Z0-9]|range|
|[^0-9]|not|
|.|all of things|


```python
import re

re.search(r'[cbm]at','cat')
```
```
<_sre.SRE_Match object; span=(0, 3), match='cat'>
```
```python
re.search(r'[0-9]haha','1hahah')
```
```
<_sre.SRE_Match object; span=(0, 5), match='1haha'>
```
```python
re.search(r'[abc.^]aron','caron')
```
```
<_sre.SRE_Match object; span=(0, 5), match='caron'>
```
```python
re.search(r'[^abc]aron','#caron')
```
```
<_sre.SRE_Match object; span=(0, 5), match='#aron'>
```
```python
re.search(r'p.g','pig')
```
```
<_sre.SRE_Match object; span=(0, 3), match='pig'>
```
<br><br><br>


|Meta(2)|Expr|
|:--|:--|
|\d|[0-9]|
|\D|[^0-9]|
|\s|space word|
|\S|non-space word|
|\w|[0-9a-zA-Z]|
|\W|[^0-9a-zA-Z]|

```python
import re

re.search(r'\sand','apple and banana')
```
```
<_sre.SRE_Match object; span=(5, 9), match=' and'>
```
```
re.search(r'\Sand','apple and banana')
```
```

```
```python
re.search(r'.and','pand')
```
```
<_sre.SRE_Match object; span=(0, 4), match='pand'>
```
```python
re.search(r'\.and','.and')
```
```
<_sre.SRE_Match object; span=(0, 4), match='.and'>
```
<br><br><br>


<span class="frame3_1">Recurrence pattern</span><br>

<table style="width:100%">
  <tr>
    <td>+</td>
    <td>more than 1</td>
  </tr>
  <tr>
    <td>*</td>
    <td>more than 0</td>
  </tr>
  <tr>
    <td>{n}</td>
    <td>n</td>
  </tr>
  <tr>
    <td>?</td>
    <td>regardless</td>
  </tr>
</table>


```python
import re

re.search(r'a[bcd]*b','abcbcbcbccb')
```
```
<_sre.SRE_Match object; span=(0, 11), match='abcbcbcbccb'>
```
```python
re.search(r'b\w+a','banana')
```
```
<_sre.SRE_Match object; span=(0, 6), match='banana'>
```
```python
re.search(r'i+','piigiii')
```
```
<_sre.SRE_Match object; span=(1, 3), match='ii'>
```
```python
re.search(r'pi+g','pig')
```
```
<_sre.SRE_Match object; span=(0, 3), match='pig'>
```
```python
re.search(r'pi*g','pig')
```
```
<_sre.SRE_Match object; span=(0, 3), match='pig'>
```
```python
re.search(r'pi+g','pg')
```
```

```
```python
re.search(r'pi*g','pg')
```
```
<_sre.SRE_Match object; span=(0, 2), match='pg'>
```
```python
re.search(r'pi{3}g','piiig')
```
```
<_sre.SRE_Match object; span=(0, 5), match='piiig'>
```
```python
re.search(r'pi{3,5}g','piiiig')
```
```
<_sre.SRE_Match object; span=(0, 6), match='piiiig'>
```
```python
re.search(r'pi{3,5}g','piiiiiig')
```
```

```
```python
re.search(r'httpf?','https://www.naver.com')
```
```
<_sre.SRE_Match object; span=(0, 4), match='http'>
```
<br><br><br>

<span class="frame3_1">Condition</span><br>

<table style="width:100%">
  <tr>
    <td>^</td>
    <td>start</td>
  </tr>
  <tr>
    <td>$</td>
    <td>end</td>
  </tr>
</table>

```python
import re

re.search(r'b\w+a','cabana')
```
```
<_sre.SRE_Match object; span=(2, 6), match='bana'>
```
```python
re.search(r'^b\w+a','cabana')
```
```

```
```python
re.search(r'^b\w+a','babana')
```
```
<_sre.SRE_Match object; span=(0, 6), match='babana'>
```
```python
re.search(r'b\w+a$','cabana')
```
```
<_sre.SRE_Match object; span=(2, 6), match='bana'>
```
```python
re.search(r'b\w+a$','cabanap')
```
```

```
<br><br><br>

<span class="frame3_1">Grouping</span><br>
```python
import re

m = re.search(r'(\w+)@(.+)','test@gmail.com')
print(m.group(1))
print(m.group(2))
print(m.group(0))
```
```
test
gmail.com
test@gmail.com
```


<br><br><br>

---

### ***Get started with regular expressions***

<br><br><br>

---

### ***Into a world of powerful regular expression***

<br><br><br>
<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-Class_and_method.html'>Class and method</a>
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-Data-model.html'>Data model</a>
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-Sequence.html'>Sequence</a>
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-First-class-functions.html'>First class functions</a>
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-Object-reference.html'>Object reference</a>
- <a href='https://userdyk-github.github.io/pl03-topic01/PL03-Topic01-Concurrency.html'>Concurrency</a>

---

Reference
- <a href="https://docs.python.org/3/contents.html" target="_blank">python document</a>
- <a href="https://www.youtube.com/playlist?list=PLa9dKeCAyr7iWPMclcDxbnlTjQ2vjdIDD" target="_blank">python lectures</a>
- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://github.com/TheAlgorithms/Python' target="_blank">TheAlgorithms</a>
- <a href='https://wikidocs.net/book/1' target="_blank">Jump to Python</a>
- <a href='https://suwoni-codelab.com/category/#/Python%20%EA%B8%B0%EB%B3%B8' target="_blank">suwoni codelab</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

