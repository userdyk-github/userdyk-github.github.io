---
layout : post
title : PL03, Syntax and semantics
categories: [PL03]
comments : true
tags : [PL03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/2019-08-13-PL03-Syntax-and-semantics.md" target="_blank">page management</a>｜ <a href="https://userdyk-github.github.io/pl03/PL03-Contents.html">Contents</a> ｜<a href="https://docs.python.org/3.6/reference/index.html#reference-index" target="_blank">official python docs</a>｜<a href="https://www.tutorialsteacher.com/python" target="_blank">python tutorials</a>｜<a href="https://www.youtube.com/playlist?list=PLa9dKeCAyr7iWPMclcDxbnlTjQ2vjdIDD" target="_blank">Lecture</a><br>
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

#### set
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


#### get
```python
a = {1:'korea',2:'USA',3:'china'}

print(a.get(1), a)
print(a[1], a)
print(a.pop(1), a)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
korea {1: 'korea', 2: 'USA', 3: 'china'}
korea {1: 'korea', 2: 'USA', 3: 'china'}
korea {2: 'USA', 3: 'china'}
```
<hr class='division3'>
</details>


```python
a = {1:'korea',2:'USA',3:'china'}

print(a.popitem(), a)
print(a.popitem(), a)
print(a.popitem(), a)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
(3, 'china') {1: 'korea', 2: 'USA'}
(2, 'USA') {1: 'korea'}
(1, 'korea') {}
```
<hr class='division3'>
</details>
<br><br><br>


#### change type to list
```python
a = {1:'korea',2:'USA',3:'china'}

print(list(a))
print(list(a.keys()))
print(list(a.values()))
print(list(a.items()))
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[1, 2, 3]
[1, 2, 3]
['korea', 'USA', 'china']
[(1, 'korea'), (2, 'USA'), (3, 'china')]
```
<hr class='division3'>
</details>
<br><br><br>

#### iteration
```python
dict = {}
dict['name'] = 'ailever'
dict['age'] = 19

for i, j in dict.items():
     print(i, j)
```
```
name ailever
age 19    
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
</details><br>

```python
s = {1,2,3}
q = {3,4,5}

print(s & q)  # {3}, s.intersection(q) 
print(s | q)  # {1,2,3,4,5}, s.union(q)
print(s - q)  # {1,2}, s.difference(q)
```
<div class="jb-medium">  
     {3} <br>
     {1,2,3,4,5} <br>
     {1,2}
</div>
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
#### comprehension
<span class="frame3">for</span>
```python
for i in range(10) : print(i)
```

<br><br><br>

---


### ***try/except***
<a herf="https://python.bakyeono.net/chapter-9-4.html#946-assert-%EB%AC%B8-%EB%8B%A8%EC%96%B8%ED%95%98%EA%B8%B0" target="_blank">URL</a>
![img-9-4](https://user-images.githubusercontent.com/52376448/71385391-b5df1780-262a-11ea-9b0e-33de023b4a59.png)
![image](https://user-images.githubusercontent.com/52376448/74032311-02be5800-49f7-11ea-80f6-7f367bf8b1f6.png)

```python
try:
    raise ZeroDivisionError()
except ZeroDivisionError:
    print('error')
```
```
error
```
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


### ***Class(1) : basic***

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

#### Access instance
```python
class Person:
   def lee(self, x):
            print(x)
            
m = Person()
m.lee(10)              # instance name space = bound
Person.lee(m, 10)      # class name sapce = unbound
```
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

#### Setter, getter method

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

```python
class Person:
    def __init__(self):
        self.__age = 0
 
    @property
    def age(self):           # getter
        return self.__age
 
    @age.setter
    def age(self, value):    # setter
        self.__age = value
 
james = Person()
james.age = 20      # 인스턴스.속성 형식으로 접근하여 값 저장
print(james.age)    # 인스턴스.속성 형식으로 값을 가져옴
```
20
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
#### static method
```python
class CLASS:
    @staticmethod
    def func(*args, **kwargs):
        pass

CLASS.func()
```
<br><br><br>

#### magic method
<a href="https://corikachu.github.io/articles/python/python-magic-method" target="_blank">corikachu URL</a><br>
<a href="http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-oop-part-6-%EB%A7%A4%EC%A7%81-%EB%A9%94%EC%86%8C%EB%93%9C-magic-method/" target="_blank">schoolofweb URL</a><br>

```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'x={}, y={}'.format(self.x, self.y)

    def __len__(self):
        return int(self.x**2 + self.y**2)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return -1

    def __add__(self, point):
        new_x = self.x + point.x
        new_y = self.y + point.y
        return (new_x, new_y)

    def __sub__(self, point):
        new_x = self.x - point.x
        new_y = self.y - point.y
        return (new_x, new_y)

    def __mul__(self, point):
        if type(point) == int:
            return Point(point*self.x, point*self.y)
        elif type(point) == Point:
            return self.x*point.x + self.y*point.y

    def __eq__(self, point):
        return self.x == point.x and self.y == point.y

    def __ne__(self, point):
        return not (self.x == point.x and self.y == point.y)


p1 = Point(1.1,2.2)
p2 = Point(4.4,5.5)

print('__str__ : ',p1, p2)
print('__len__ : ',len(p1),len(p2))
print('__getitem__ :', p1[0],p1[1],p1[2])
print('__add__ : ',p1 + p2)
print('__sub__ : ',p1 - p2)
print('__mul__ : ',p1 * 3)
print('__mul__ : ',p1 * p2)
print('__eq__ : ',p1 == p1, p1 == p2)
print('__ne__ : ',p1 != p1, p1 != p2)
print('__abs__: ', abs(p1))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">__init__(self) : p = Point()</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__str__(self) : object = CLASS()</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__del__(self) : del object</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__repr__(self) : </summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__len__(self) : len(object)</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__abs__(self) : abs(object)</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">__add__(self) : object1 + object2</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__sub__(self) : object1 - object2</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">__mul__(self) : object1 * object2</summary>
<hr class='division3'>
```python

```
```

```
<hr class='division3'>
</details>

<br><br><br>

---

### ***Class(2) : Inheritance***
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
<a href="https://wikidocs.net/21132" target="_blank">module import</a>
#### import
`parent path`
```python
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
```
<br><br><br>

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
#### raise
<a herf="https://python.bakyeono.net/chapter-9-4.html#946-assert-%EB%AC%B8-%EB%8B%A8%EC%96%B8%ED%95%98%EA%B8%B0" target="_blank">URL</a>
![img-9-4](https://user-images.githubusercontent.com/52376448/71385391-b5df1780-262a-11ea-9b0e-33de023b4a59.png)

```python
class DoorException(Exception):
    pass

class DoorOpenedException(DoorException):
    pass

class DoorClosedException(DoorException):
    pass


class Door:
    def __init__(self):
        self.is_opened = True
    
    def state(self):
        raise DoorException('being preparing')

    def open(self):
        if self.is_opened:
            raise DoorOpenedException('already opened')
        else:
            print('open')
            self.is_opened = True

    def close(self):
        if not self.is_opened:
            raise DoorClosedException('already closed')
        else:
            print('close')
            self.is_opened = False
            
door = Door()
door.close()
door.open()
door.state()
```
```
close
open
---------------------------------------------------------------------------
DoorException                             Traceback (most recent call last)
<ipython-input-797-ad6705ac378e> in <module>
     33 door.close()
     34 door.open()
---> 35 door.state()

<ipython-input-797-ad6705ac378e> in state(self)
     14 
     15     def state(self):
---> 16         raise DoorException('being preparing')
     17 
     18     def open(self):

DoorException: being preparing
```
#### try/except/else/finally
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
<a href="https://securityspecialist.tistory.com/90" target="_blank">r, b, a, +</a>

|mode|description|method|
|:-|:-|:-|
|w|write||
|r|read|seek,read,readline,readlines|
|a|append||
|b|binary||
|+|overwrite a part of doc||


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
#### argparse
<a href="https://brownbears.tistory.com/413" target="_blank">URL</a>
```python
import argparse

parser = argparse.ArgumentParser(description="set your environment")
parser.add_argument('--env1', required=False, help="env 1")
parser.add_argument('--env2', required=False, help="env 2")
args = parser.parse_args()

print(args.env1)
print(args.env2)
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
|os.getcwd() <br> os.path.dirname(\_\_file\_\_)|get current working directory|

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
<details markdown="1">
<summary class='jb-small' style="color:blue">another way</summary>
<hr class='division3'>
```python
import os

current_path = os.path.abspath(os.path.dirname(__file__))
print(__file__)
print(current_path)
```
```
1341.py
s:\workspace\2020-02-04
```
<hr class='division3'>
</details>


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

### ***Object copy***
<span class='frame3'>immutable</span>

|                |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--             |:-- |:-- |:-- |:-- |
|origin          |1   |1   |1   |1   |
|simple(b=100)   |1   |100 |1   |1   |
|shallow(c=100)  |1   |1   |100 |1   |
|deep(d=100)     |1   |1   |1   |100 |



```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d))
    print()    

a = 1
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)
```
```
a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496
```
<details markdown="1">
<summary class='jb-small' style="color:blue">simple</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d))
    print()    

a = 1
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b = 100
description(a,b,c,d)
```
```
a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496

a value: 1 : original(a) ,
 a id: 10914496
b value: 100 : simple(b = a) ,
 b id: 10917664
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">shallow</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d))
    print()    

a = 1
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c = 100
description(a,b,c,d)
```
```
a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496

a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 100 : shallow(c = copy.copy(a))  ,
 c id: 10917664
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">deep</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d))
    print()    

a = 1
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d = 100
description(a,b,c,d)
```
```
a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 1 : deep(d = copy.deepcopy(d)) ,
 d id: 10914496

a value: 1 : original(a) ,
 a id: 10914496
b value: 1 : simple(b = a) ,
 b id: 10914496
c value: 1 : shallow(c = copy.copy(a))  ,
 c id: 10914496
d value: 100 : deep(d = copy.deepcopy(d)) ,
 d id: 10917664
```
<hr class='division3'>
</details><br>

<span class='frame3'>mutable</span>

|                   |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                |:--   |:-- |:-- |:-- |
|origin             |[1]   |[1]   |[1]   |[1]   |
|simple(b[0]=100)   |[100] |[100] |[1]   |[1]   |
|simple(b=[100])    |[1]   |[100] |[1]   |[1]   |
|shallow(c[0]=100)  |[1]   |[1]   |[100] |[1]   |
|shallow(c=[100])   |[1]   |[1]   |[100] |[1]   |
|deep(d[0]=100)     |[1]   |[1]   |[1]   |[100] |
|deep(d=[100])      |[1]   |[1]   |[1]   |[100] |



```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282058829128 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282058829128 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282056137608 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055629320 , d[0] id: 10914496
 ```
 <details markdown="1">
<summary class='jb-small' style="color:blue">simple</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b[0] = 100
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282055757896 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282055757896 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055759112 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055758152 , d[0] id: 10914496

a value: [100] : original(a) ,
 a id: 140282055757896 , a[0] id: 10917664
b value: [100] : simple(b = a) ,
 b id: 140282055757896 , b[0] id: 10917664
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055759112 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055758152 , d[0] id: 10914496
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b = [100]
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282054944712 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282054944712 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055640840 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055757896 , d[0] id: 10914496

a value: [1] : original(a) ,
 a id: 140282054944712 , a[0] id: 10914496
b value: [100] : simple(b = a) ,
 b id: 140282055758152 , b[0] id: 10917664
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055640840 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055757896 , d[0] id: 10914496
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">shallow</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c[0] = 100
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282057630472 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282057630472 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282054944712 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055758152 , d[0] id: 10914496

a value: [1] : original(a) ,
 a id: 140282057630472 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282057630472 , b[0] id: 10914496
c value: [100] : shallow(c = copy.copy(a))  ,
 c id: 140282054944712 , c[0] id: 10917664
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055758152 , d[0] id: 10914496
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c = [100]
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282054999432 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282054999432 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282057631304 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057630472 , d[0] id: 10914496

a value: [1] : original(a) ,
 a id: 140282054999432 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282054999432 , b[0] id: 10914496
c value: [100] : shallow(c = copy.copy(a))  ,
 c id: 140282055758152 , c[0] id: 10917664
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057630472 , d[0] id: 10914496
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">deep</summary>
<hr class='division3'>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d[0] = 100
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282055758152 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282055758152 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282054946120 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057631304 , d[0] id: 10914496

a value: [1] : original(a) ,
 a id: 140282055758152 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282055758152 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282054946120 , c[0] id: 10914496
d value: [100] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057631304 , d[0] id: 10917664
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]))
    print()
    
a = [1]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d = [100]
description(a,b,c,d)
```
```
a value: [1] : original(a) ,
 a id: 140282055726216 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282055726216 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055640840 , c[0] id: 10914496
d value: [1] : deep(d = copy.deepcopy(d)) ,
 d id: 140282055758152 , d[0] id: 10914496

a value: [1] : original(a) ,
 a id: 140282055726216 , a[0] id: 10914496
b value: [1] : simple(b = a) ,
 b id: 140282055726216 , b[0] id: 10914496
c value: [1] : shallow(c = copy.copy(a))  ,
 c id: 140282055640840 , c[0] id: 10914496
d value: [100] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057631304 , d[0] id: 10917664
```
<hr class='division3'>
</details>
<br><br><br>

#### simple copy

|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                 |[1,[2]]     |[1,[2]]     |[1,[2]]     |[1,[2]]   |
|simple(b[0]=2)         |[2,[2]]     |[2,[2]]     |[1,[2]]     |[1,[2]]   |
|simple(b[1]=[3])       |[1,[3]]     |[1,[3]]     |[1,[2]]     |[1,[2]]   |
|simple(b[1][0]=3)      |[1,[3]]     |[1,[3]]     |[1,[3]]     |[1,[2]]   |
|simple(b[1].append(4)) |[1,[2,4]]   |[1,[2,4]]   |[1,[2,4]]   |[1,[2]]   |

```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()
    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b[0] = 2
description(a,b,c,d)

b[1] = [3]
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282049351496 , a[0] id: 10914496 , a[1] id: 140282049578824
b value: [1, [2]] : simple(b = a) ,
 b id: 140282049351496 , b[0] id: 10914496 , b[1] id: 140282049578824
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282049351560 , c[0] id: 10914496 , c[1] id: 140282049578824
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057881608 , d[0] id: 10914496 , d[1] id: 140282049410312

a value: [2, [2]] : original(a) ,
 a id: 140282049351496 , a[0] id: 10914528 , a[1] id: 140282049578824
b value: [2, [2]] : simple(b = a) ,
 b id: 140282049351496 , b[0] id: 10914528 , b[1] id: 140282049578824
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282049351560 , c[0] id: 10914496 , c[1] id: 140282049578824
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057881608 , d[0] id: 10914496 , d[1] id: 140282049410312

a value: [2, [3]] : original(a) ,
 a id: 140282049351496 , a[0] id: 10914528 , a[1] id: 140282060263624
b value: [2, [3]] : simple(b = a) ,
 b id: 140282049351496 , b[0] id: 10914528 , b[1] id: 140282060263624
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282049351560 , c[0] id: 10914496 , c[1] id: 140282049578824
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057881608 , d[0] id: 10914496 , d[1] id: 140282049410312
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b[0] = 2
description(a,b,c,d)

b[1][0] = 3
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282054943752 , a[0] id: 10914496 , a[1] id: 140282054945928
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054943752 , b[0] id: 10914496 , b[1] id: 140282054945928
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944328 , c[0] id: 10914496 , c[1] id: 140282054945928
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944072 , d[0] id: 10914496 , d[1] id: 140282054943048

a value: [2, [2]] : original(a) ,
 a id: 140282054943752 , a[0] id: 10914528 , a[1] id: 140282054945928
b value: [2, [2]] : simple(b = a) ,
 b id: 140282054943752 , b[0] id: 10914528 , b[1] id: 140282054945928
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944328 , c[0] id: 10914496 , c[1] id: 140282054945928
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944072 , d[0] id: 10914496 , d[1] id: 140282054943048

a value: [2, [3]] : original(a) ,
 a id: 140282054943752 , a[0] id: 10914528 , a[1] id: 140282054945928
b value: [2, [3]] : simple(b = a) ,
 b id: 140282054943752 , b[0] id: 10914528 , b[1] id: 140282054945928
c value: [1, [3]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944328 , c[0] id: 10914496 , c[1] id: 140282054945928
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944072 , d[0] id: 10914496 , d[1] id: 140282054943048
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

b[0] = 2
description(a,b,c,d)

b[1].append(4)
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282049958344 , a[0] id: 10914496 , a[1] id: 140282055016520
b value: [1, [2]] : simple(b = a) ,
 b id: 140282049958344 , b[0] id: 10914496 , b[1] id: 140282055016520
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282049923464 , c[0] id: 10914496 , c[1] id: 140282055016520
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282049958024 , d[0] id: 10914496 , d[1] id: 140282049959432

a value: [2, [2]] : original(a) ,
 a id: 140282049958344 , a[0] id: 10914528 , a[1] id: 140282055016520
b value: [2, [2]] : simple(b = a) ,
 b id: 140282049958344 , b[0] id: 10914528 , b[1] id: 140282055016520
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282049923464 , c[0] id: 10914496 , c[1] id: 140282055016520
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282049958024 , d[0] id: 10914496 , d[1] id: 140282049959432

a value: [2, [2, 4]] : original(a) ,
 a id: 140282049958344 , a[0] id: 10914528 , a[1] id: 140282055016520
b value: [2, [2, 4]] : simple(b = a) ,
 b id: 140282049958344 , b[0] id: 10914528 , b[1] id: 140282055016520
c value: [1, [2, 4]] : shallow(c = copy.copy(a))  ,
 c id: 140282049923464 , c[0] id: 10914496 , c[1] id: 140282055016520
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282049958024 , d[0] id: 10914496 , d[1] id: 140282049959432
```
<br><br><br>
#### Sallow copy

|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                  |[1,[2]]     |[1,[2]]     |[1,[2]]     |[1,[2]]   |
|shallow(c[0]=2)         |[1,[2]]     |[1,[2]]     |[2,[2]]     |[1,[2]]   |
|shallow(c[1]=[3])       |[1,[2]]     |[1,[2]]     |[1,[3]]     |[1,[2]]   |
|shallow(c[1][0]=3)      |[1,[3]]     |[1,[3]]     |[1,[3]]     |[1,[2]]   |
|shallow(c[1].append(4)) |[1,[2,4]]   |[1,[2,4]]   |[1,[2,4]]   |[1,[2]]   |

```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    
    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c[0] = 2
description(a,b,c,d)

c[1] = [3]
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282054999560 , a[0] id: 10914496 , a[1] id: 140282054999752
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054999560 , b[0] id: 10914496 , b[1] id: 140282054999752
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054848712 , c[0] id: 10914496 , c[1] id: 140282054999752
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999688 , d[0] id: 10914496 , d[1] id: 140282054848776

a value: [1, [2]] : original(a) ,
 a id: 140282054999560 , a[0] id: 10914496 , a[1] id: 140282054999752
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054999560 , b[0] id: 10914496 , b[1] id: 140282054999752
c value: [2, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054848712 , c[0] id: 10914528 , c[1] id: 140282054999752
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999688 , d[0] id: 10914496 , d[1] id: 140282054848776

a value: [1, [2]] : original(a) ,
 a id: 140282054999560 , a[0] id: 10914496 , a[1] id: 140282054999752
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054999560 , b[0] id: 10914496 , b[1] id: 140282054999752
c value: [2, [3]] : shallow(c = copy.copy(a))  ,
 c id: 140282054848712 , c[0] id: 10914528 , c[1] id: 140282054848648
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999688 , d[0] id: 10914496 , d[1] id: 140282054848776
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c[0] = 2
description(a,b,c,d)

c[1][0] = 3
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282054945672 , a[0] id: 10914496 , a[1] id: 140282054944136
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054945672 , b[0] id: 10914496 , b[1] id: 140282054944136
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944904 , c[0] id: 10914496 , c[1] id: 140282054944136
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054943752 , d[0] id: 10914496 , d[1] id: 140282054945928

a value: [1, [2]] : original(a) ,
 a id: 140282054945672 , a[0] id: 10914496 , a[1] id: 140282054944136
b value: [1, [2]] : simple(b = a) ,
 b id: 140282054945672 , b[0] id: 10914496 , b[1] id: 140282054944136
c value: [2, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944904 , c[0] id: 10914528 , c[1] id: 140282054944136
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054943752 , d[0] id: 10914496 , d[1] id: 140282054945928

a value: [1, [3]] : original(a) ,
 a id: 140282054945672 , a[0] id: 10914496 , a[1] id: 140282054944136
b value: [1, [3]] : simple(b = a) ,
 b id: 140282054945672 , b[0] id: 10914496 , b[1] id: 140282054944136
c value: [2, [3]] : shallow(c = copy.copy(a))  ,
 c id: 140282054944904 , c[0] id: 10914528 , c[1] id: 140282054944136
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054943752 , d[0] id: 10914496 , d[1] id: 140282054945928
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

c[0] = 2
description(a,b,c,d)

c[1].append(4)
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282055726856 , a[0] id: 10914496 , a[1] id: 140282054999048
b value: [1, [2]] : simple(b = a) ,
 b id: 140282055726856 , b[0] id: 10914496 , b[1] id: 140282054999048
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282055728712 , c[0] id: 10914496 , c[1] id: 140282054999048
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054991560 , d[0] id: 10914496 , d[1] id: 140282050117832

a value: [1, [2, 4]] : original(a) ,
 a id: 140282055726856 , a[0] id: 10914496 , a[1] id: 140282054999048
b value: [1, [2, 4]] : simple(b = a) ,
 b id: 140282055726856 , b[0] id: 10914496 , b[1] id: 140282054999048
c value: [1, [2, 4]] : shallow(c = copy.copy(a))  ,
 c id: 140282055728712 , c[0] id: 10914496 , c[1] id: 140282054999048
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054991560 , d[0] id: 10914496 , d[1] id: 140282050117832
```
<br><br><br>
#### deep copy

|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                 |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[2]]   |
|deep(d[0]=2)           |[1,[2]]   |[1,[2]]   |[1,[2]]   |[2,[2]]   |
|deep(d[1]=[3])         |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[3]]   |
|deep(d[1][0]=3)        |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[3]]   |
|deep(d[1].append(4))   |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[2,4]]   |

```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    
    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d[0] = 2
description(a,b,c,d)

d[1] = [3]
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282055725768 , a[0] id: 10914496 , a[1] id: 140282055726920
b value: [1, [2]] : simple(b = a) ,
 b id: 140282055725768 , b[0] id: 10914496 , b[1] id: 140282055726920
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282055725640 , c[0] id: 10914496 , c[1] id: 140282055726920
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999560 , d[0] id: 10914496 , d[1] id: 140282054848648

a value: [1, [2]] : original(a) ,
 a id: 140282055725768 , a[0] id: 10914496 , a[1] id: 140282055726920
b value: [1, [2]] : simple(b = a) ,
 b id: 140282055725768 , b[0] id: 10914496 , b[1] id: 140282055726920
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282055725640 , c[0] id: 10914496 , c[1] id: 140282055726920
d value: [2, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999560 , d[0] id: 10914528 , d[1] id: 140282054848648

a value: [1, [2]] : original(a) ,
 a id: 140282055725768 , a[0] id: 10914496 , a[1] id: 140282055726920
b value: [1, [2]] : simple(b = a) ,
 b id: 140282055725768 , b[0] id: 10914496 , b[1] id: 140282055726920
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282055725640 , c[0] id: 10914496 , c[1] id: 140282055726920
d value: [2, [3]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054999560 , d[0] id: 10914528 , d[1] id: 140282054999688
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d[0] = 2
description(a,b,c,d)

d[1][0] = 3
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282057630280 , a[0] id: 10914496 , a[1] id: 140282054944712
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630280 , b[0] id: 10914496 , b[1] id: 140282054944712
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057630856 , c[0] id: 10914496 , c[1] id: 140282054944712
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057630728 , d[0] id: 10914496 , d[1] id: 140282057630472

a value: [1, [2]] : original(a) ,
 a id: 140282057630280 , a[0] id: 10914496 , a[1] id: 140282054944712
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630280 , b[0] id: 10914496 , b[1] id: 140282054944712
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057630856 , c[0] id: 10914496 , c[1] id: 140282054944712
d value: [2, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057630728 , d[0] id: 10914528 , d[1] id: 140282057630472

a value: [1, [2]] : original(a) ,
 a id: 140282057630280 , a[0] id: 10914496 , a[1] id: 140282054944712
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630280 , b[0] id: 10914496 , b[1] id: 140282054944712
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057630856 , c[0] id: 10914496 , c[1] id: 140282054944712
d value: [2, [3]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282057630728 , d[0] id: 10914528 , d[1] id: 140282057630472
```
<br><br><br>
```python
import copy

def description(a,b,c,d):
    print('a value:',a,': original(a)',
          ',\n a id:',id(a),', a[0] id:',id(a[0]),', a[1] id:',id(a[1]))
    print('b value:',b,': simple(b = a)',
          ',\n b id:',id(b),', b[0] id:',id(b[0]),', b[1] id:',id(b[1]))
    print('c value:',c,': shallow(c = copy.copy(a)) ',
          ',\n c id:',id(c),', c[0] id:',id(c[0]),', c[1] id:',id(c[1]))
    print('d value:',d,': deep(d = copy.deepcopy(d))',
          ',\n d id:',id(d),', d[0] id:',id(d[0]),', d[1] id:',id(d[1]))
    print()    

    
a = [1, [2]]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
description(a,b,c,d)

d[0] = 2
description(a,b,c,d)

d[1].append(4)
description(a,b,c,d)
```
```
a value: [1, [2]] : original(a) ,
 a id: 140282057630728 , a[0] id: 10914496 , a[1] id: 140282057630472
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630728 , b[0] id: 10914496 , b[1] id: 140282057630472
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057631304 , c[0] id: 10914496 , c[1] id: 140282057630472
d value: [1, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944840 , d[0] id: 10914496 , d[1] id: 140282055630280

a value: [1, [2]] : original(a) ,
 a id: 140282057630728 , a[0] id: 10914496 , a[1] id: 140282057630472
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630728 , b[0] id: 10914496 , b[1] id: 140282057630472
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057631304 , c[0] id: 10914496 , c[1] id: 140282057630472
d value: [2, [2]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944840 , d[0] id: 10914528 , d[1] id: 140282055630280

a value: [1, [2]] : original(a) ,
 a id: 140282057630728 , a[0] id: 10914496 , a[1] id: 140282057630472
b value: [1, [2]] : simple(b = a) ,
 b id: 140282057630728 , b[0] id: 10914496 , b[1] id: 140282057630472
c value: [1, [2]] : shallow(c = copy.copy(a))  ,
 c id: 140282057631304 , c[0] id: 10914496 , c[1] id: 140282057630472
d value: [2, [2, 4]] : deep(d = copy.deepcopy(d)) ,
 d id: 140282054944840 , d[0] id: 10914528 , d[1] id: 140282055630280
```
<br><br><br>
#### summary

|                |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--             |:-- |:-- |:-- |:-- |
|origin          |1   |1   |1   |1   |
|b=100   |1   |100 |1   |1   |
|c=100  |1   |1   |100 |1   |
|d=100     |1   |1   |1   |100 |


|                   |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                |:--   |:-- |:-- |:-- |
|origin             |[1]   |[1]   |[1]   |[1]   |
|b[0]=100   |[100] |[100] |[1]   |[1]   |
|b=[100]    |[1]   |[100] |[1]   |[1]   |
|c[0]=100  |[1]   |[1]   |[100] |[1]   |
|c=[100]   |[1]   |[1]   |[100] |[1]   |
|d[0]=100     |[1]   |[1]   |[1]   |[100] |
|d=[100]      |[1]   |[1]   |[1]   |[100] |


|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                 |[1,[2]]     |[1,[2]]     |[1,[2]]     |[1,[2]]   |
|b[0]=2         |[2,[2]]     |[2,[2]]     |[1,[2]]     |[1,[2]]   |
|b[1]=[3]       |[1,[3]]     |[1,[3]]     |[1,[2]]     |[1,[2]]   |
|b[1][0]=3      |[1,[3]]     |[1,[3]]     |[1,[3]]     |[1,[2]]   |
|b[1].append(4) |[1,[2,4]]   |[1,[2,4]]   |[1,[2,4]]   |[1,[2]]   |


|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                  |[1,[2]]     |[1,[2]]     |[1,[2]]     |[1,[2]]   |
|c[0]=2         |[1,[2]]     |[1,[2]]     |[2,[2]]     |[1,[2]]   |
|c[1]=[3]       |[1,[2]]     |[1,[2]]     |[1,[3]]     |[1,[2]]   |
|c[1][0]=3      |[1,[3]]     |[1,[3]]     |[1,[3]]     |[1,[2]]   |
|c[1].append(4) |[1,[2,4]]   |[1,[2,4]]   |[1,[2,4]]   |[1,[2]]   |


|                       |a|b=a|c=copy.copy(a)|d=copy.deepcopy(a)|
|:--                    |:--   |:-- |:-- |:-- |
|origin                 |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[2]]   |
|d[0]=2           |[1,[2]]   |[1,[2]]   |[1,[2]]   |[2,[2]]   |
|d[1]=[3]         |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[3]]   |
|d[1][0]=3        |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[3]]   |
|d[1].append(4)   |[1,[2]]   |[1,[2]]   |[1,[2]]   |[1,[2,4]]   |


<br><br><br>

---

### ***Closure : func of func***
#### global vs nonlocal
```python
a = 3
def calc():
    a = 1
    b = 5
    total = 0
    def mul_add(x):
        global a
        nonlocal total
        total = total + a*x + b
        print(total)
    return mul_add

c = calc()
print(c(1))
```
```
8
```
<br><br><br>

#### lambda closure
```python
def calc():
    a = 3
    b = 5
    return lambda x : a*x + b

c = calc()
print(c(1))
```
```
8
```
<br><br><br>
#### decorator
<span class="frame3" target="_blank">function decorator</span>
```python
def trace(func):
    def wrapper():
        return func()
    return wrapper

@trace
def function():
    pass

function()
```
```python
def trace(func):
    def wrapper(*agrs,**kwagrs):
        return func(*agrs,**kwagrs)
    return wrapper

@trace
def function(*args, **kwargs):
    pass

function()
```
```python
def trace(x):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@trace(x=3)
def function(*args, **kwargs):
    pass

print(function())
```
<br><br><br>
<span class="frame3" target="_blank">class decorator</span>

```python
class trace:
    def __init__(self, func):
        self.func = func

    def __call__(self):
        return self.func()

@trace
def function():
    pass

function()
```
```python
class trace:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

@trace
def function(*args, **kwargs):
    pass

function()
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

### ***Iterator : next!***

```python
class Counter:
    def __init__(self, stop):
        self.current = 0
        self.stop = stop
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.stop:
            r = self.current
            self.current += 1
            return r
        else:
            raise StopIteration

for i in Counter(3): print(i)
```
```
0
1
2
```
<br><br><br>
```python
class Counter:
    def __init__(self, stop):
        self.stop = stop
    
    def __getitem__(self, index):
        if index < self.stop :
            return index
        else:
            raise IndexError
    
#print(Counter(3)[0],Counter(3)[1],Counter(3)[2])

for i in Counter(3): print(i)
```
```
0
1
2
```
<br><br><br>
#### iter(iterable_object)
```python
a = iter(range(3))
next(a)
next(a)
next(a)
```
```
2
```
<br><br><br>
#### iter(callable_object, sentinel)
```python
import random

for i in iter(lambda : random.randint(0,5), 2):
    print(i)
```
```
3
1
4
4
3
5
3
3
1
5
4
```
<br><br><br>
#### next(iterable_object, default_value)
```python
a = iter(range(3))
next(a,10)
next(a,10)
next(a,10)
next(a,10)
```
```
10
```
<br><br><br>

---

### ***Generator : yield!***

#### Yield
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


#### Yield from
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
#### generator expression
for list
```python
import time

def sleep_func(x):
    print("sleep...")
    time.sleep(1)
    return x

start = time.time()
l = [sleep_func(x) for x in range(5)]
for j,i in enumerate(l):print('step = %d, '%j,i)
end = time.time()
end-start
```
```
sleep...
sleep...
sleep...
sleep...
sleep...
step = 0,  0
step = 1,  1
step = 2,  2
step = 3,  3
step = 4,  4

5.009469032287598
```
for generator
```python
import time

def sleep_func(x):
    print("sleep...")
    time.sleep(1)
    return x

start = time.time()
g = (sleep_func(x) for x in range(5))
for j,i in enumerate(g): print('step = %d, '%j,i)
end = time.time()
end-start
```
```
sleep...
step = 0,  0
sleep...
step = 1,  1
sleep...
step = 2,  2
sleep...
step = 3,  3
sleep...
step = 4,  4

5.008758068084717
```

<span class="frame3">memory size</span>
```python
import sys

# for list
print(sys.getsizeof( [i for i in range(100) if i % 2]),'   for list, iter_num:100')  
print(sys.getsizeof( [i for i in range(1000) if i % 2]),'   for list, iter_num:1000')

# for generator
print(sys.getsizeof( (i for i in range(100) if i % 2)),'   for generator, iter_num:100')
print(sys.getsizeof( (i for i in range(1000) if i % 2)),'   for generator, iter_num:1000')
```
```
528    for list, iter_num:100
4272    for list, iter_num:1000
88    for generator, iter_num:100
88    for generator, iter_num:1000
```
<br><br><br>

---

### ***Coroutine : send!***
<span class="frame3"> Main routine and Sub routine</span>
```python
def sub_add(a,b):
    c = a + b
    print(c)
    print('sub add func')
def main_calc():
    sub_add(1,2)
    print('main calc func')
    
main_calc()
```
```
3
sub add func
main calc func
```

<br><br><br>

#### put signal
<span class="frame3">(yield)</span>
```python
def number_coroutine():
    while True:
        x = (yield)
        print(x)
co = number_coroutine()
next(co)
co.send(1)
co.send(2)
co.send(3)
```
```
1
2
3
```

<br><br><br>
#### get signal
<span class="frame3">(yield + variable)</span>
```python
def sum_coroutine():
    total = 0
    while True:
        x = (yield total)
        total += x
co = sum_coroutine()
print(next(co))
print(co.send(1))
print(co.send(2))
print(co.send(3))
```
```
0
1
3
6
```
<br><br><br>

#### exit
<span class="frame3">close</span>
```python
def number_coroutine():
    while True:
        x = (yield)
        print(x, end=' ')

co = number_coroutine()
next(co)

for i in range(20):
    co.send(i)

co.close()
```
```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

<br><br><br>

#### error handling
<span class="frame3">close</span>
```python
def number_coroutine():
    try:
        while True:
            x = (yield)
            print(x, end=' ')
    except GeneratorExit:
        print()
        print('coroutine exit')

co = number_coroutine()
next(co)

for i in range(20):
    co.send(i)

co.close()
```
```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
coroutine exit
```
<br><br><br>
<span class="frame3">throw</span>
```python
def sum_coroutine():
    try:
        total = 0
        while True:
            x = (yield)
            total += x
    except RuntimeError as e:
        print(e)
        yield total

co = sum_coroutine()
next(co)

for i in range(20):
    co.send(i)

print(co.throw(RuntimeError,'exit through error'))
```
```
exit through error
190
```
<br><br><br>
#### get return
```python
def accumulate():
    total = 0
    while True:
        x = (yield)
        if x is None:
            return total
        total += x

def sum_coroutine():
    while True:
        total = yield from accumulate()
        print(total)

co = sum_coroutine()
next(co)

for i in range(1,11):
    co.send(i)
co.send(None)

for i in range(1,101):
    co.send(i)
co.send(None)
```
```
55
5050
```
<br><br><br>
#### StopIteration
<span class="frame3">python - V : 3.6</span>
```pyhon
def accumulate():
    total = 0
    while True:
        x = (yield)
        if x is None:
            raise StopIteration(total)
        total += x

def sum_coroutine():
    while True:
        total = yield from accumulate()
        print(total)

co = sum_coroutine()
next(co)

for i in range(1,11):
    co.send(i)
co.send(None)

for i in range(1,101):
    co.send(i)
co.send(None)
```
```
55
5050
```
<span class="frame3">python - V : 3.7</span>
```python
def accumulate():
    total = 0
    while True:
        x = (yield)
        if x is None:
            return total
        total += x

def sum_coroutine():
    while True:
        total = yield from accumulate()
        print(total)

co = sum_coroutine()
next(co)

for i in range(1,11):
    co.send(i)
co.send(None)

for i in range(1,101):
    co.send(i)
co.send(None)
```
```
55
5050
```
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
re.search(r'[^abc]aron','#aron')
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
<hr class="division2">

## **Debugging**

### ***unittest***
`workspace`
```python
class Calculator:
    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

def Matmul(a, b):
    return a * b
```
`unittest`
```python
import unittest, time
from workspace import Calculator, Matmul

class test_class(unittest.TestCase):
    def test_module_add(self):
        self.calculator = Calculator()
        self.assertEqual(self.calculator.add(3, 4), 7)

    def test_module_mul(self):
        time.sleep(1)
        self.fail()

class test_function(unittest.TestCase):
    def test_module_matmul(self):
        time.sleep(1)
        self.assertEqual(Matmul(3,5), 15)

unittest.main()
```

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
- <a href="https://www.fun-coding.org/daveblog.html" target="_blank">fun coding</a>
- <a href="https://dojang.io/course/view.php?id=7" target="_blank">dojang</a>
- <a href='https://wikidocs.net/book/1' target="_blank">Jump to Python</a>
- <a href="https://deepwelloper.tistory.com/130" target="_blank">deepwelloper(python memory)</a>
- <a href="https://docs.python.org/3/contents.html" target="_blank">python document</a>
- <a href="https://www.youtube.com/playlist?list=PLa9dKeCAyr7iWPMclcDxbnlTjQ2vjdIDD" target="_blank">python lectures</a>
- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://github.com/TheAlgorithms/Python' target="_blank">TheAlgorithms</a>
- <a href='https://suwoni-codelab.com/category/#/Python%20%EA%B8%B0%EB%B3%B8' target="_blank">suwoni codelab</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

