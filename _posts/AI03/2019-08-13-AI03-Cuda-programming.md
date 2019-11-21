---
layout : post
title : AI03, Cuda programming
categories: [AI03]
comments : true
tags : [AI03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/pl03/PL03-Libraries.html' target="_blank">Python Libraries</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Introduction to CPU Parallel Programming**

### ***1.1 EVOLUTION OF PARALLEL PROGRAMMING***
### ***1.2 MORE CORES, MORE PARALLELISM***
### ***1.3 CORES VERSUS THREADS***
#### 1.3.1 More Threads or More Cores to Parallelize?
#### 1.3.2 Influence of Core Resource Sharing
#### 1.3.3 Influence of Memory Resource Sharing
### ***1.4 OUR FIRST SERIAL PROGRAM***
#### 1.4.1 Understanding Data Transfer Speeds
#### 1.4.2 The main() Function in imflip.c
#### 1.4.3 Flipping Rows Vertically: FlipImageV()
#### 1.4.4 Flipping Columns Horizontally: FlipImageH()
### ***1.5 WRITING, COMPILING, RUNNING OUR PROGRAMS***
#### 1.5.1 Choosing an Editor and a Compiler
#### 1.5.2 Developing in Windows 7, 8, and Windows 10 Platforms
#### 1.5.3 Developing in a Mac Platform
#### 1.5.4 Developing in a Unix Platform
### ***1.6 CRASH COURSE ON UNIX***
#### 1.6.1 Unix Directory-Related Commands
#### 1.6.2 Unix File-Related Commands
### ***1.7 DEBUGGING YOUR PROGRAMS***
#### 1.7.1 gdb
#### 1.7.2 Old School Debugging
#### 1.7.3 valgrind
### ***1.8 PERFORMANCE OF OUR FIRST SERIAL PROGRAM***
#### 1.8.1 Can We Estimate the Execution Time?
#### 1.8.2 What Does the OS Do When Our Code Is Executing?
#### 1.8.3 How Do We Parallelize It?
#### 1.8.4 Thinking About the Resources
<br><br><br>

<hr class="division2">

## **Parallel Programming using CUDA C**

### ***2.1 OUR FIRST PARALLEL PROGRAM***
#### 2.1.1 The main() Function in imflipP.c
#### 2.1.2 Timing the Execution
#### 2.1.3 Split Code Listing for main() in imflipP.c
#### 2.1.4 Thread Initialization
#### 2.1.5 Thread Creation
#### 2.1.6 Thread Launch/Execution
#### 2.1.7 Thread Termination (Join)
#### 2.1.8 Thread Task and Data Splitting
### ***2.2 WORKING WITH BITMAP (BMP) FILES***
#### 2.2.1 BMP is a Non-Lossy/Uncompressed File Format
#### 2.2.2 BMP Image File Format
#### 2.2.3 Header File ImageStuff.h
#### 2.2.4 Image Manipulation Routines in ImageStuff.c
### ***2.3 TASK EXECUTION BY THREADS***
#### 2.3.1 Launching a Thread
#### 2.3.2 Multithreaded Vertical Flip: MTFlipV()
#### 2.3.3 Comparing FlipImageV() and MTFlipV()
#### 2.3.4 Multithreaded Horizontal Flip: MTFlipH()
### ***2.4 TESTING/TIMING THE MULTITHREADED CODE***

<br><br><br>

<hr class="division2">

## **Developing Our First Parallel CPU Program**
### ***3.1 EFFECT OF THE “PROGRAMMER” ON PERFORMANCE***
### ***3.2 EFFECT OF THE “CPU” ON PERFORMANCE***
#### 3.2.1 In-Order versus Out-Of-Order Cores 55
#### 3.2.2 Thin versus Thick Threads 57
### ***3.3 PERFORMANCE OF IMFLIPP***
### ***3.4 EFFECT OF THE “OS” ON PERFORMANCE***
#### 3.4.1 Thread Creation 59
#### 3.4.2 Thread Launch and Execution 59
#### 3.4.3 Thread Status 60
#### 3.4.4 Mapping Software Threads to Hardware Threads 61
#### 3.4.5 Program Performance versus Launched Pthreads 62
### ***3.5 IMPROVING IMFLIPP***
#### 3.5.1 Analyzing Memory Access Patterns in MTFlipH() 64
#### 3.5.2 Multithreaded Memory Access of MTFlipH() 64
#### 3.5.3 DRAM Access Rules of Thumb 66
### ***3.6 IMFLIPPM: OBEYING DRAM RULES OF THUMB***
#### 3.6.1 Chaotic Memory Access Patterns of imflipP 67
#### 3.6.2 Improving Memory Access Patterns of imflipP 68
#### 3.6.3 MTFlipHM(): The Memory Friendly MTFlipH() 69
#### 3.6.4 MTFlipVM(): The Memory Friendly MTFlipV() 71
### ***3.7 PERFORMANCE OF IMFLIPPM.C***
#### 3.7.1 Comparing Performances of imflipP.c and imflipPM.c 72
#### 3.7.2 Speed Improvement: MTFlipV() versus MTFlipVM() 73
#### 3.7.3 Speed Improvement: MTFlipH() versus MTFlipHM() 73
#### 3.7.4 Understanding the Speedup: MTFlipH() versus MTFlipHM() 73
### ***3.8 PROCESS MEMORY MAP***
### ***3.9 INTEL MIC ARCHITECTURE: XEON PHI***
### ***3.10 WHAT ABOUT THE GPU?***
### ***3.11 CHAPTER SUMMARY***

<br><br><br>

<hr class="division2">

## **Improving Our First Parallel CPU Program**

<br><br><br>

<hr class="division2">

## **Understanding the Cores and Memory**

<br><br><br>

<hr class="division2">

## **Thread Management and Synchronization**

<br><br><br>

<hr class="division2">

## **Introduction to GPU Parallelism and CUDA**

<br><br><br>

<hr class="division2">

## **CUDA Host/Device Programming Model**

<br><br><br>

<hr class="division2">

## **Understanding GPU Hardware Architecture**

<br><br><br>

<hr class="division2">

## **Understanding GPU Cores**

<br><br><br>

<hr class="division2">

## **Understanding GPU Memory**

<br><br><br>

<hr class="division2">

## **CUDA Streams**

<br><br><br>

<hr class="division2">

## **CUDA Libraries**

<br><br><br>

<hr class="division2">

## **Introduction to OpenCL**

<br><br><br>

<hr class="division2">

## **Other GPU Programming Languages**

<br><br><br>

<hr class="division2">

## **Deep Learning Using CUDA**

<br><br><br>


<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- [post1](https://userdyk-github.github.io/)
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


