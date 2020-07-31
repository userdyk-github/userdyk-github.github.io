---
layout : post
title : PL00, Parallel computing
categories: [PL00]
comments : true
tags : [PL00]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜<a href='https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL00/2019-08-13-PL00-Parallel-computing.md' target="_blank">page management</a> <br>
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

## **Introduction to CUDA Programming**

|Hardward|Bus standards|
|:--|:--|
|CPU|pci-e|
|GPU|nv-link|

<br><br><br>


### ***The history of high-performance computing***
![image](https://user-images.githubusercontent.com/52376448/89057382-7f276780-d398-11ea-97e9-a0f3a974b3e0.png)

<br><br><br>

---

### ***Heterogeneous computing***
![image](https://user-images.githubusercontent.com/52376448/89057506-b6961400-d398-11ea-8f53-253e6c0a3d79.png)

<br><br><br>

---

### ***Low latency versus higher throughput***
![image](https://user-images.githubusercontent.com/52376448/89057597-e9d8a300-d398-11ea-9751-00e51dc36e63.png)

<br><br><br>

---

### ***GPU architecture***
![image](https://user-images.githubusercontent.com/52376448/89059253-d24ee980-d39b-11ea-81c8-6a482bac163f.png)

|hardware|software|
|:--|:--|
|CUDA Core/SIMD code|CUDA thread|
|Streaming multiprocessor|CUDA block|
|GPU device|GRID/kernel|

---

<br><br><br>

<hr class="division2">


## **CUDA Memory Management**
![image](https://user-images.githubusercontent.com/52376448/89059845-e5ae8480-d39c-11ea-8892-6b264eb2f919.png)
![image](https://user-images.githubusercontent.com/52376448/89059873-f3fca080-d39c-11ea-9609-d91d803c4547.png)

<br><br><br>

### ***Coalesced versus uncoalesced global memory access***
![image](https://user-images.githubusercontent.com/52376448/89060424-fc091000-d39d-11ea-841a-d0cf6e043e8a.png)
![image](https://user-images.githubusercontent.com/52376448/89060458-088d6880-d39e-11ea-974f-8889c31e8cdf.png)
![image](https://user-images.githubusercontent.com/52376448/89060529-235fdd00-d39e-11ea-89d6-8b4ae5cc40ea.png)
![image](https://user-images.githubusercontent.com/52376448/89060565-31adf900-d39e-11ea-9416-86b442875c4f.png)
![image](https://user-images.githubusercontent.com/52376448/89060621-4b4f4080-d39e-11ea-8c7d-8787a0dfcbf3.png)
![image](https://user-images.githubusercontent.com/52376448/89060657-5904c600-d39e-11ea-969c-08b597ad0cbc.png)

<br><br><br>

---

### ***Bank conflicts and its effect on shared memory***
![image](https://user-images.githubusercontent.com/52376448/89060759-82bded00-d39e-11ea-834c-8197347def41.png)
![image](https://user-images.githubusercontent.com/52376448/89060780-8b162800-d39e-11ea-9d51-d8ec8cc9c503.png)
![image](https://user-images.githubusercontent.com/52376448/89060791-90737280-d39e-11ea-87a5-69da8922e23d.png)
![image](https://user-images.githubusercontent.com/52376448/89060814-9bc69e00-d39e-11ea-8807-f3553ca24460.png)

<br><br><br>

---

### ***Read-only data/cache***
![image](https://user-images.githubusercontent.com/52376448/89060887-be58b700-d39e-11ea-8b4c-cfdf214fa543.png)

<br><br><br>

---

### ***Registers in GPU***
![image](https://user-images.githubusercontent.com/52376448/89060957-dcbeb280-d39e-11ea-930a-588f93142f20.png)

<br><br><br>

---

### ***Pinned memory***

<br><br><br>

---

### ***Unified memory***
![image](https://user-images.githubusercontent.com/52376448/89061088-0d9ee780-d39f-11ea-9426-9baf54d099f4.png)

<br><br><br>

#### Understanding unified memory page allocation and transfer
1. First, we need to allocate new pages on the GPU and CPU (first-touch basis). If the page is not present and mapped to another, a device page table page fault occurs. When *x, which resides in page 2, is accessed in the GPU that is currently mapped to CPU memory, it gets a page fault. Take a look at the following diagram:
![image](https://user-images.githubusercontent.com/52376448/89061228-52c31980-d39f-11ea-90ba-695b5550fa2a.png)

2. In the next step, the old page on the CPU is unmapped, as shown in the following diagram:
![image](https://user-images.githubusercontent.com/52376448/89061244-58b8fa80-d39f-11ea-92ac-90fe4eb75ef4.png)

3. Next, the data is copied from the CPU to the GPU, as shown in the following diagram:
![image](https://user-images.githubusercontent.com/52376448/89061257-5e164500-d39f-11ea-898b-2c3ff1ca0bb3.png)

4. Finally, the new pages are mapped on the GPU, while the old pages are freed on the CPU, as shown in the following diagram:
![image](https://user-images.githubusercontent.com/52376448/89061269-640c2600-d39f-11ea-9e70-7ddd06462847.png)

<br><br><br>

---

<hr class="division2">


## **CUDA Thread Programming**

<br><br><br>
<hr class="division2">

## **Kernel Execution Model and Optimization Strategies**

<br><br><br>
<hr class="division2">

## **CUDA Application Profiling and Debugging**

<br><br><br>
<hr class="division2">

## **Scalable Multi-GPU Programming**

<br><br><br>
<hr class="division2">

## **Parallel Programming Patterns in CUDA**

<br><br><br>
<hr class="division2">

## **Programming with Libraries and Other Languages**
### ***Linear algebra operation using cuBLAS***

|level in cuBLAS|operation|
|:--|:--|
|level 1|vector-vector|
|level 2|matrix-vector|
|level 3|matrix-matrix|

<br><br><br>

---

<br><br><br>
<hr class="division2">

## **GPU Programming Using OpenACC**

<br><br><br>
<hr class="division2">

## **Deep Learning Acceleration with CUDA**

<br><br><br>

<hr class="division1">

List of posts followed by this article
- Jaegeun Han, Bharatkumar Sharma - Learn CUDA Programming_ A beginner's guide to GPU programming and parallel computing with CUDA 10.x and C_C++-Packt Publishing (2019)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='' target="_blank"></a>
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


