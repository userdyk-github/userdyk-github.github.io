---
layout : post
title : PL03, Syntax and semantics
categories: [PL03]
comments : true
tags : [PL03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜ <a href="https://userdyk-github.github.io/pl03/PL03-Contents.html">Python</a> <br>
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

<br><br><br>

---

### ***Built-in function***

#### abs
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

#### all
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

#### any
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


#### chr
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


#### dir
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

#### divmod
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


#### enumerate
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


#### eval
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


#### filter
```python

```
```

```
<br>


#### hex
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


#### id
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

 
#### input
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


#### int
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


#### isinstance
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


#### len
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


#### list
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


#### map
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

#### max
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


#### min
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


#### oct
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


#### open

|mode|description|
|:-|:-|
|w|write|
|r|read|
|a|append|
|b|binary|

```python
f = open("binary_file", "rb")
```
```python
fread = open("read_mode.txt", 'r')
fread2 = open("read_mode.txt")
```
```python
fappend = open("append_mode.txt", 'a')
```
<br>


#### ord
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


#### pow
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


#### range
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


#### round
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

#### sorted
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


#### str
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


#### sum
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


#### tuple
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


#### type
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

#### zip
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
<br>

<span class="frame3_1">Examples</span><br>
```python
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
<br>


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

|Meta(2)|Expr|
|:--|:--|
|\d|[0-9]|
|\D|[^0-9]|
|\s|space word|
|\S|non-space word|
|\w|[0-9a-zA-Z]|
|\W|[^0-9a-zA-Z]|
|\.|.|
|\\|\|


```python
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
<br><br><br>


<br><br><br>

---

### ***Get started with regular expressions***

<br><br><br>

---

### ***Into a world of powerful regular expression***

<br><br><br>
<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference

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

