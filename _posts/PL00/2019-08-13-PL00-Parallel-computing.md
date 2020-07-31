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
### ***Kernel execution with CUDA streams***
#### The usage of CUDA streams
![image](https://user-images.githubusercontent.com/52376448/89064317-bef44c00-d3a4-11ea-8417-cfe20775fee5.png)
![image](https://user-images.githubusercontent.com/52376448/89064381-ddf2de00-d3a4-11ea-91a3-ed6c1c1d2f0a.png)

<br><br><br>

#### Stream-level synchronization
![image](https://user-images.githubusercontent.com/52376448/89064430-f19e4480-d3a4-11ea-8ff8-61f5fd3612ef.png)

<br><br><br>

#### Working with the default stream
![image](https://user-images.githubusercontent.com/52376448/89064465-04b11480-d3a5-11ea-80f8-8f4c1b2d2f00.png)

<br><br><br>

---

### ***Pipelining the GPU execution***
#### Concept of GPU pipelining
![image](https://user-images.githubusercontent.com/52376448/89064532-1f838900-d3a5-11ea-9531-07e84071b539.png)
![image](https://user-images.githubusercontent.com/52376448/89064545-24483d00-d3a5-11ea-91cd-07efc3ef1208.png)
![image](https://user-images.githubusercontent.com/52376448/89064562-2a3e1e00-d3a5-11ea-937c-2bdd1f0eae98.png)


<br><br><br>

#### Building a pipelining execution
![image](https://user-images.githubusercontent.com/52376448/89064623-48a41980-d3a5-11ea-8784-568a4844dea9.png)
![image](https://user-images.githubusercontent.com/52376448/89064654-58bbf900-d3a5-11ea-9e89-a923a56f3b51.png)

<br><br><br>

---

### ***The CUDA callback function***
![image](https://user-images.githubusercontent.com/52376448/89064703-712c1380-d3a5-11ea-9282-8c4c40d129e6.png)

<br><br><br>

---

### ***CUDA streams with priority***
#### Stream execution with priorities
![image](https://user-images.githubusercontent.com/52376448/89064781-9587f000-d3a5-11ea-9797-dbd0d8219807.png)

<br><br><br>

---

### ***Kernel execution time estimation using CUDA events***
#### Using CUDA events
![image](https://user-images.githubusercontent.com/52376448/89064901-c405cb00-d3a5-11ea-8f44-da7a77a72d12.png)

<br><br><br>

---

### ***CUDA dynamic parallelism***
#### Usage of dynamic parallelism
![image](https://user-images.githubusercontent.com/52376448/89064999-eef01f00-d3a5-11ea-9602-7b9626422dd8.png)

<br><br><br>

---

### ***Grid-level cooperative groups***
#### Understanding grid-level cooperative groups
![image](https://user-images.githubusercontent.com/52376448/89065062-0a5b2a00-d3a6-11ea-8a23-e1ee33f23003.png)

<br><br><br>

---

### ***CUDA kernel calls with OpenMP***
#### CUDA kernel calls with OpenMP
![image](https://user-images.githubusercontent.com/52376448/89065215-3d9db900-d3a6-11ea-987c-6543001e6345.png)

<br><br><br>

---

### ***Multi-Process Service***
![image](https://user-images.githubusercontent.com/52376448/89065273-51491f80-d3a6-11ea-853c-226911c2ff83.png)

<br><br><br>

#### Enabling MPS
![image](https://user-images.githubusercontent.com/52376448/89065319-64f48600-d3a6-11ea-811a-624e89497f0d.png)

<br><br><br>

#### Profiling an MPI application and understanding MPS operation
![image](https://user-images.githubusercontent.com/52376448/89065383-82c1eb00-d3a6-11ea-900e-5434085274c2.png)
![image](https://user-images.githubusercontent.com/52376448/89065394-87869f00-d3a6-11ea-9a50-92664f7fb140.png)
![image](https://user-images.githubusercontent.com/52376448/89065407-8f464380-d3a6-11ea-971f-ac4ecdc08785.png)

<br><br><br>

---

### ***Kernel execution overhead comparison***
#### Comparison of three executions
![image](https://user-images.githubusercontent.com/52376448/89065470-bac92e00-d3a6-11ea-9fe8-035779bfe182.png)

<br><br><br>

<hr class="division2">

## **CUDA Application Profiling and Debugging**

<br><br><br>
<hr class="division2">

## **Scalable Multi-GPU Programming**
### ***Solving a linear equation using Gaussian elimination***
#### Single GPU hotspot analysis of Gaussian elimination
![image](https://user-images.githubusercontent.com/52376448/89065705-23b0a600-d3a7-11ea-84a1-f4555ea18381.png)

<br><br><br>

---

### ***GPUDirect peer to peer***
![image](https://user-images.githubusercontent.com/52376448/89065748-3aef9380-d3a7-11ea-960c-cd59ba0f8407.png)
![image](https://user-images.githubusercontent.com/52376448/89065774-4347ce80-d3a7-11ea-91ee-30429673df96.png)
![image](https://user-images.githubusercontent.com/52376448/89065788-480c8280-d3a7-11ea-990a-4c60fcdedca8.png)
![image](https://user-images.githubusercontent.com/52376448/89065811-4e9afa00-d3a7-11ea-9671-6af0832f6022.png)

<br><br><br>

#### Single node – multi-GPU Gaussian elimination
![image](https://user-images.githubusercontent.com/52376448/89065851-66727e00-d3a7-11ea-9a92-7f87afb780ed.png)

<br><br><br>

---

### ***GPUDirect RDMA***
![image](https://user-images.githubusercontent.com/52376448/89065917-7ee29880-d3a7-11ea-9799-ec43c2e86c05.png)
![image](https://user-images.githubusercontent.com/52376448/89065926-843fe300-d3a7-11ea-840f-095dfe9dc65b.png)

<br><br><br>

#### CUDA-aware MPI
![image](https://user-images.githubusercontent.com/52376448/89065983-9de12a80-d3a7-11ea-9322-99119bd2072f.png)

<br><br><br>

#### Multinode – multi-GPU Gaussian elimination
![image](https://user-images.githubusercontent.com/52376448/89066052-c0734380-d3a7-11ea-9aa5-86b4f5218b5e.png)

<br><br><br>

---

### ***CUDA streams***
#### Application 1 – using multiple streams to overlap data transfers with kernel execution
![image](https://user-images.githubusercontent.com/52376448/89066154-ec8ec480-d3a7-11ea-8efc-1bc2174e84ee.png)
![image](https://user-images.githubusercontent.com/52376448/89066163-f284a580-d3a7-11ea-8525-080a8dd00968.png)

<br><br><br>

#### Application 2 – using multiple streams to run kernels on multiple devices
![image](https://user-images.githubusercontent.com/52376448/89066200-05977580-d3a8-11ea-911a-d90594826c3a.png)

<br><br><br>

---

### ***Additional tricks***
#### Collective communication acceleration using NCCL
![image](https://user-images.githubusercontent.com/52376448/89066291-2cee4280-d3a8-11ea-9c35-f299a8b6ca34.png)
![image](https://user-images.githubusercontent.com/52376448/89066320-3aa3c800-d3a8-11ea-98f7-550f15d9b116.png)

<br><br><br>

<hr class="division2">

## **Parallel Programming Patterns in CUDA**
### ***Matrix multiplication optimization***
![image](https://user-images.githubusercontent.com/52376448/89066427-66bf4900-d3a8-11ea-9651-b17107d5196e.png)
![image](https://user-images.githubusercontent.com/52376448/89066453-72127480-d3a8-11ea-9529-471c500798b7.png)

<br><br><br>
#### Performance analysis of the tiling approach
![image](https://user-images.githubusercontent.com/52376448/89066494-87879e80-d3a8-11ea-8eda-96da95d0a774.png)

<br><br><br>

---

### ***Convolution***
#### Convolution operation in CUDA
![image](https://user-images.githubusercontent.com/52376448/89066549-a0904f80-d3a8-11ea-8fa4-9a22b2292278.png)

<br><br><br>
#### Optimization strategy
![image](https://user-images.githubusercontent.com/52376448/89066601-b867d380-d3a8-11ea-8456-f805b2ff9886.png)
![image](https://user-images.githubusercontent.com/52376448/89066618-bd2c8780-d3a8-11ea-9632-d439adbbe417.png)

<br><br><br>

---

### ***Prefix sum (scan)***
![image](https://user-images.githubusercontent.com/52376448/89066695-dd5c4680-d3a8-11ea-92f6-bf00336c5206.png)
![image](https://user-images.githubusercontent.com/52376448/89066769-f49b3400-d3a8-11ea-8e20-fd6be4ec157a.png)

<br><br><br>

#### Building a global size scan
![image](https://user-images.githubusercontent.com/52376448/89066823-14caf300-d3a9-11ea-86b2-ac282e3ccf72.png)

<br><br><br>

---

### ***Compact and split***
![image](https://user-images.githubusercontent.com/52376448/89066881-31672b00-d3a9-11ea-8555-3f979dfcd443.png)
![image](https://user-images.githubusercontent.com/52376448/89066894-36c47580-d3a9-11ea-81b4-b6dba278dcb9.png)
![image](https://user-images.githubusercontent.com/52376448/89066905-3c21c000-d3a9-11ea-9218-568d4c3bcdbd.png)
![image](https://user-images.githubusercontent.com/52376448/89066917-404ddd80-d3a9-11ea-97ab-f38bb023f2c9.png)

<br><br><br>

---

### ***N-body***
#### Implementing an N-body simulation on GPU
![image](https://user-images.githubusercontent.com/52376448/89066991-64a9ba00-d3a9-11ea-97ca-c17f14449458.png)

<br><br><br>

---

### ***Histogram calculation***
#### Understanding a parallel histogram
![image](https://user-images.githubusercontent.com/52376448/89067082-8d31b400-d3a9-11ea-931e-6966ffe93c6f.png)

<br><br><br>

---

### ***Quicksort and CUDA dynamic parallelism***
#### Quicksort in CUDA using dynamic parallelism
![image](https://user-images.githubusercontent.com/52376448/89067157-b6524480-d3a9-11ea-9100-0ace77434e94.png)
![image](https://user-images.githubusercontent.com/52376448/89067172-bbaf8f00-d3a9-11ea-9b2a-40749d7b8949.png)

<br><br><br>

---

### ***Radix sort***

<br><br><br>

---


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
### ***OpenACC directives***
#### Parallel and loop directives
![image](https://user-images.githubusercontent.com/52376448/89068844-0979c680-d3ad-11ea-89f7-7726c5d5d16b.png)
![image](https://user-images.githubusercontent.com/52376448/89068878-11d20180-d3ad-11ea-8356-676188a41c46.png)

<br><br><br>
#### Data directive
![image](https://user-images.githubusercontent.com/52376448/89068922-244c3b00-d3ad-11ea-8c67-c774550a0507.png)
![image](https://user-images.githubusercontent.com/52376448/89068949-2f9f6680-d3ad-11ea-955a-5d8b8e2e0bcf.png)

<br><br><br>

---

### ***Asynchronous programming in OpenACC***
![image](https://user-images.githubusercontent.com/52376448/89069022-4fcf2580-d3ad-11ea-8bed-c7cd6221e648.png)

<br><br><br>
#### Applying the unstructured data and async directives to merge image code
![image](https://user-images.githubusercontent.com/52376448/89069071-66757c80-d3ad-11ea-86ea-9d972854d0c2.png)

<br><br><br>

---

### ***Additional important directives and clauses***
#### Gang/vector/worker
![image](https://user-images.githubusercontent.com/52376448/89069157-96bd1b00-d3ad-11ea-982c-822e5bee52f6.png)

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


