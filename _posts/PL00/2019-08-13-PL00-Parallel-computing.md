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

### ***Global memory/device memory***
#### Coalesced versus uncoalesced global memory access
![image](https://user-images.githubusercontent.com/52376448/89060424-fc091000-d39d-11ea-841a-d0cf6e043e8a.png)
![image](https://user-images.githubusercontent.com/52376448/89060458-088d6880-d39e-11ea-974f-8889c31e8cdf.png)
![image](https://user-images.githubusercontent.com/52376448/89060529-235fdd00-d39e-11ea-89d6-8b4ae5cc40ea.png)
![image](https://user-images.githubusercontent.com/52376448/89060565-31adf900-d39e-11ea-9416-86b442875c4f.png)
![image](https://user-images.githubusercontent.com/52376448/89060621-4b4f4080-d39e-11ea-8c7d-8787a0dfcbf3.png)
![image](https://user-images.githubusercontent.com/52376448/89060657-5904c600-d39e-11ea-969c-08b597ad0cbc.png)

<br><br><br>

---

### ***Shared memory***
#### Bank conflicts and its effect on shared memory
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
### ***CUDA threads, blocks, and the GPU***
![image](https://user-images.githubusercontent.com/52376448/89061918-88b4cd80-d3a0-11ea-804f-771d55f69e4f.png)
![image](https://user-images.githubusercontent.com/52376448/89061930-8d798180-d3a0-11ea-8615-1fe85149c7b2.png)
![image](https://user-images.githubusercontent.com/52376448/89061943-92d6cc00-d3a0-11ea-8aeb-b780e5cc1214.png)

<br><br><br>

---

### ***Understanding parallel reduction***
![image](https://user-images.githubusercontent.com/52376448/89062028-b4d04e80-d3a0-11ea-800f-2d688763ad6e.png)

<br><br><br>


#### Naive parallel reduction using global memory
![image](https://user-images.githubusercontent.com/52376448/89062087-c9ace200-d3a0-11ea-848e-14c2eb52e3fe.png)

<br><br><br>

---

#### Reducing kernels using shared memory
![image](https://user-images.githubusercontent.com/52376448/89062130-dc271b80-d3a0-11ea-9357-e76aeedc9769.png)

<br><br><br>

---

### ***Minimizing the CUDA warp divergence effect***
![image](https://user-images.githubusercontent.com/52376448/89062181-f660f980-d3a0-11ea-9481-5aa3c014d70e.png)

<br><br><br>

#### Determining divergence as a performance bottleneck

<span class="frame3">Interleaved addressing</span><br>
![image](https://user-images.githubusercontent.com/52376448/89062223-0bd62380-d3a1-11ea-8e7d-88a8e86055d3.png)


<span class="frame3">Sequential addressing</span><br>
![image](https://user-images.githubusercontent.com/52376448/89062273-1c869980-d3a1-11ea-94c4-608488719dc9.png)

<br><br><br>

---

### ***Performance modeling and balancing the limiter***
#### The Roofline model
![image](https://user-images.githubusercontent.com/52376448/89062327-345e1d80-d3a1-11ea-9fa0-0f79a3025483.png)

<br><br><br>

---

### ***Warp-level primitive programming***
![image](https://user-images.githubusercontent.com/52376448/89062490-869f3e80-d3a1-11ea-96cb-48d580456bfb.png)
![image](https://user-images.githubusercontent.com/52376448/89062536-961e8780-d3a1-11ea-88a7-5cbdec6f528e.png)
![image](https://user-images.githubusercontent.com/52376448/89062557-9f0f5900-d3a1-11ea-97c7-92066e08261d.png)

<br><br><br>

#### Parallel reduction with warp primitives
![image](https://user-images.githubusercontent.com/52376448/89062592-b3ebec80-d3a1-11ea-9e67-4ccd7b5184a0.png)
![image](https://user-images.githubusercontent.com/52376448/89062605-b9493700-d3a1-11ea-8ef1-8c0dc410d0e9.png)

<br><br><br>

---

### ***Cooperative Groups for flexible thread handling***
#### Cooperative Groups in a CUDA thread block
![image](https://user-images.githubusercontent.com/52376448/89062659-d120bb00-d3a1-11ea-889b-158bb6620752.png)

<br><br><br>

#### Benefits of Cooperative Groups
<span class="frame3">Modularity</span><br>
![image](https://user-images.githubusercontent.com/52376448/89062707-e72e7b80-d3a1-11ea-87f1-a63b465c6be3.png)

<br><br><br>

---

### ***Atomic operations***
![image](https://user-images.githubusercontent.com/52376448/89062760-fca3a580-d3a1-11ea-8b7b-b9b9cc44fc11.png)

<br><br><br>

---

### ***Low/mixed precision operations***
#### Dot product operations and accumulation for 8-bit integers and 16-bit data (DP4A and DP2A)
![image](https://user-images.githubusercontent.com/52376448/89062823-147b2980-d3a2-11ea-8107-046584bb2dd5.png)

<br><br><br>

---

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


