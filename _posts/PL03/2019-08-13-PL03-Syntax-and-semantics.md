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

### ***if***

<br><br><br>

---

### ***while***

<br><br><br>

---

### ***for***

<br><br><br>
<hr class="division2">

## **How to do the input and output of the program**

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

### ***Class***

#### hasattr(object, name)

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

```
```

```
<br>

#### all
```python

```
```

```
<br>


#### any
```python

```
```

```
<br>


#### chr
```python

```
```

```
<br>


#### dir
```python

```
```

```
<br>


#### divmod
```python

```
```

```
<br>


#### enumerate
```python

```
```

```
<br>


#### eval
```python

```
```

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

```
```

```
<br>


#### id
```python

```
```

```
<br>

 
#### input
```python

```
```

```
<br>


#### int
```python

```
```

```
<br>


#### isinstance
```python

```
```

```
<br>


#### len
```python

```
```

```
<br>


#### list
```python

```
```

```
<br>


#### map
```python

```
```

```
<br>


#### max
```python

```
```

```
<br>


#### min
```python

```
```

```
<br>


#### oct
```python

```
```

```
<br>


#### open
```python

```
```

```
<br>


#### ord
```python

```
```

```
<br>


#### pow
```python

```
```

```
<br>


#### range
```python

```
```

```
<br>


#### round
```python

```
```

```
<br>


####
```python

```
```

```
<br>


#### sorted
```python

```
```

```
<br>


#### str
```python

```
```

```
<br>


#### sum
```python

```
```

```
<br>


#### tuple
```python

```
```

```
<br>


#### type
```python

```
```

```
<br>

#### zip
```python
list(zip([1, 2, 3], [4, 5, 6]))
```
```
[(1, 4), (2, 5), (3, 6)]
```
```python
list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9]))
```
```
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```
```python
list(zip("abc", "def"))
```
```
[('a', 'd'), ('b', 'e'), ('c', 'f')]
```


<br><br><br>

---

### ***External function***

<br><br><br>
<hr class="division2">

## **Python programming, how do I get started?**

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

### ***Explore regular expressions***

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
- <a href='https://github.com/TheAlgorithms/Python' target="_blank">TheAlgorithms</a>
- <a href='https://wikidocs.net/book/1' target="_blank">Jump to Python</a>
- <a href='https://suwoni-codelab.com/category/#/Python%20%EA%B8%B0%EB%B3%B8' target="_blank">suwoni codelab</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

