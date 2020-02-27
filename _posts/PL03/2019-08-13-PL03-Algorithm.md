---
layout : post
title : PL03, Algorithm
categories: [PL03]
comments : true
tags : [PL03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/2019-08-13-PL03-Algorithm.md" target="_blank">page management</a>｜ <a href="https://userdyk-github.github.io/pl03/PL03-Contents.html">Contents</a>  <br>
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

## **Data structure**
### ***array***
#### 1d array
```python
data_list = [1, 2, 3, 4, 5]
```
<br><br><br>

#### 2d array
```python
data_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```
<br><br><br>

---

### ***Queue***
#### FIFO Queue
```python
import queue

data_queue = queue.Queue()
data_queue.put(0)
data_queue.put(1)
data_queue.put(2)
data_queue.put(3)
data_queue.put(4)
data_queue.get()
data_queue.get()
data_queue.get()
data_queue.get()
data_queue.get()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">.qsize()</summary>
<hr class='division3'>
```python
data_queue.qsize()
```
<hr class='division3'>
</details><br>

```python
class fifo_queue:
    def __init__(self):
        self.list = list()
    
    def enqueue(self,x):
        self.list.append(x)
        print(x, 'was enqueued at', id(self.list[len(self.list)-1]))
        for i,j in enumerate(self.list):
            print(j, id(j)) if i != len(self.list)-1 else print('---'*10)
            
    def dequeue(self):
        dv = self.list[0]
        print('It was dequeued and present state of queue is as follow')
        del self.list[0]
        for i in self.list:
            print(i, id(i)) if i != self.list[-1] else print('---'*10)
        return dv
    
Q = fifo_queue()
Q.enqueue(0)
Q.enqueue(1)
Q.enqueue(2)
Q.dequeue()
Q.dequeue()
Q.dequeue()
```


<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
0 was enqueued at 10914464
0 10914464
------------------------------
1 was enqueued at 10914496
0 10914464
1 10914496
------------------------------
2 was enqueued at 10914528
0 10914464
1 10914496
2 10914528
------------------------------
It was dequeued and present state of queue is as follow
1 10914496
2 10914528
------------------------------
It was dequeued and present state of queue is as follow
2 10914528
------------------------------
It was dequeued and present state of queue is as follow
```
<hr class='division3'>
</details>

<br><br><br>

#### LIFO Queue
```python
import queue

data_queue = queue.LifoQueue()
data_queue.put(0)
data_queue.put(1)
data_queue.put(2)
data_queue.put(3)
data_queue.put(4)
```
<br><br><br>

#### Priority Queue
.put((priority, data))
```python
import queue

data_queue = queue.PriorityQueue()
data_queue.put((7,'7th data'))
data_queue.put((1,'1st data'))
data_queue.put((6,'6th data'))
data_queue.put((2,'2nd data'))
data_queue.put((4,'4th data'))
```
<br><br><br>
<hr class="division2">

## **Recursion**

<br><br><br>

<hr class="division2">

## **Linked List**

<br><br><br>

<hr class="division2">

## **Stack**

<br><br><br>

<hr class="division2">



## **Recursion**

### ***Factorial***

---

### ***Tower of Hanoi***


<br><br><br>

<hr class="division2">

## **Tree**

<br><br><br>

<hr class="division2">

## **Priority queue and heap**

<br><br><br>

<hr class="division2">

## **Sorting**

![fq0A8hx](https://user-images.githubusercontent.com/52376448/64517642-b5489c80-d32b-11e9-964a-dd4000ec1956.gif)

<br><br><br>

### ***Bubble Sort***

![giphy (1)](https://user-images.githubusercontent.com/52376448/64515755-25552380-d328-11e9-8b95-060eb452ffd8.gif)

<br><br><br>

---

### ***Selection Sort***

![AccomplishedCourageousGander-size_restricted](https://user-images.githubusercontent.com/52376448/64515021-c216c180-d326-11e9-9dc8-f56651cb0e3a.gif)

<br><br><br>

---

### ***Insertion Sort***

![CornyThickGordonsetter-size_restricted](https://user-images.githubusercontent.com/52376448/64516459-89c4b280-d329-11e9-9c37-e8a89edb07b9.gif)

<br><br><br>

---

### ***Heap Sort***

![Sorting_heapsort_anim](https://user-images.githubusercontent.com/52376448/64515661-f2ab2b00-d327-11e9-9edf-548887660b1e.gif)

<br><br><br>

---

### ***Merge Sort***

![merge-sort-gif-9](https://user-images.githubusercontent.com/52376448/64515971-93014f80-d328-11e9-940c-caa0168e78e0.gif)

<br><br><br>

---

### ***Quicksort***

![Sorting_quicksort_anim](https://user-images.githubusercontent.com/52376448/64516995-6b12eb80-d32a-11e9-83e7-9f1a4e2609e5.gif)

<br><br><br>

---

### ***Cocktail shaker sort***

![Sorting_shaker_sort_anim](https://user-images.githubusercontent.com/52376448/64516827-1a9b8e00-d32a-11e9-9851-1fe82ed50af6.gif)

<br><br><br>

<hr class="division2">

## **Search**

### ***Linear search***

<br><br><br>

---

### ***Binary search***

<br><br><br>

---

### ***Ternary search***

<br><br><br>

<hr class="division2">

## **Table and Hash**

<br><br><br>

<hr class="division2">

## **Graph**

### ***Breadth-First search***

<br><br><br>

---

### ***Depth-First search***

<br><br><br>

---

### ***Bellman-Ford algorithm***

<br><br><br>

---

### ***Dijkstra's algorithm***

<br><br><br>

---

### ***A' algorithm***
## **Algorithm**


<br><br><br>

<hr class="division1">

List of posts followed by this article
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-arithmetic-analysis.html' class='jb-medium'>arithmetic analysis</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-backtracking.html' class='jb-medium'>backtracking</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-blockchain.html' class='jb-medium'>blockchain</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-boolean-algebra.html' class='jb-medium'>boolean algebra</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-ciphers.html' class='jb-medium'>ciphers</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-compression.html' class='jb-medium'>compression</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-conversions.html' class='jb-medium'>conversions</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-digital-image-processing.html' class='jb-medium'>digital image processing</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-divide-and-conquer.html' class='jb-medium'>divide and conquer</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-dynamic-programming.html' class='jb-medium'>dynamic programming</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-file-transfer.html' class='jb-medium'>file transfer</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-linear-algebra.html' class='jb-medium'>linear algebra</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-machine-learning.html' class='jb-medium'>machine learning</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-maths.html' class='jb-medium'>maths</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-matrix.html' class='jb-medium'>matrix</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-networking-flow.html' class='jb-medium'>networking-flow</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-neural-network.html' class='jb-medium'>neural-network</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-other.html' class='jb-medium'>other</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-strings.html' class='jb-medium'>strings</a>
- <a href='https://userdyk-github.github.io/pl03-topic03/PL03-Topic03-traversals.html' class='jb-medium'>traversals</a>


---

Reference

- <a href="https://dojang.io/course/index.php?categoryid=1" target="_blank">dojang</a>
- <a href="https://www.fun-coding.org/daveblog.html" target="_blank">fun coding</a>
- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://visualgo.net/en' target="_blank">visualgo</a>
- <a href='https://github.com/TheAlgorithms/Python' target="_blank">TheAlgorithms</a>
- <a href='http://mai1408.github.io/index.html' target="_blank">Visualizing Algorithms through animation</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

