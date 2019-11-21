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

## **Developing Our First Parallel CPU Program**

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

## **Improving Our First Parallel CPU Program**

### ***3.1 EFFECT OF THE “PROGRAMMER” ON PERFORMANCE***
### ***3.2 EFFECT OF THE “CPU” ON PERFORMANCE***
#### 3.2.1 In-Order versus Out-Of-Order Cores
#### 3.2.2 Thin versus Thick Threads
### ***3.3 PERFORMANCE OF IMFLIPP***
### ***3.4 EFFECT OF THE “OS” ON PERFORMANCE***
#### 3.4.1 Thread Creation
#### 3.4.2 Thread Launch and Execution
#### 3.4.3 Thread Status
#### 3.4.4 Mapping Software Threads to Hardware Threads
#### 3.4.5 Program Performance versus Launched Pthreads
### ***3.5 IMPROVING IMFLIPP***
#### 3.5.1 Analyzing Memory Access Patterns in MTFlipH()
#### 3.5.2 Multithreaded Memory Access of MTFlipH()
#### 3.5.3 DRAM Access Rules of Thumb
### ***3.6 IMFLIPPM: OBEYING DRAM RULES OF THUMB***
#### 3.6.1 Chaotic Memory Access Patterns of imflipP
#### 3.6.2 Improving Memory Access Patterns of imflipP
#### 3.6.3 MTFlipHM(): The Memory Friendly MTFlipH()
#### 3.6.4 MTFlipVM(): The Memory Friendly MTFlipV()
### ***3.7 PERFORMANCE OF IMFLIPPM.C***
#### 3.7.1 Comparing Performances of imflipP.c and imflipPM.c
#### 3.7.2 Speed Improvement: MTFlipV() versus MTFlipVM()
#### 3.7.3 Speed Improvement: MTFlipH() versus MTFlipHM()
#### 3.7.4 Understanding the Speedup: MTFlipH() versus MTFlipHM()
### ***3.8 PROCESS MEMORY MAP***
### ***3.9 INTEL MIC ARCHITECTURE: XEON PHI***
### ***3.10 WHAT ABOUT THE GPU?***
### ***3.11 CHAPTER SUMMARY***

<br><br><br>

<hr class="division2">

## **Understanding the Cores and Memory**

### ***4.1 ONCE UPON A TIME ... INTEL ...***
### ***4.2 CPU AND MEMORY MANUFACTURERS***
### ***4.3 DYNAMIC (DRAM) VERSUS STATIC (SRAM) MEMORY***
#### 4.3.1 Static Random Access Memory (SRAM)
#### 4.3.2 Dynamic Random Access Memory (DRAM)
#### 4.3.3 DRAM Interface Standards
#### 4.3.4 Influence of DRAM on our Program Performance
#### 4.3.5 Influence of SRAM (Cache) on our Program Performance
### ***4.4 IMAGE ROTATION PROGRAM: IMROTATE.C***
#### 4.4.1 Description of the imrotate.c
#### 4.4.2 imrotate.c: Parametric Restrictions and Simplifications
#### 4.4.3 imrotate.c: Theory of Operation
### ***4.5 PERFORMANCE OF IMROTATE***
#### 4.5.1 Qualitative Analysis of Threading Efficiency
#### 4.5.2 Quantitative Analysis: Defining Threading Efficiency
### ***4.6 THE ARCHITECTURE OF THE COMPUTER***
#### 4.6.1 The Cores, L1$ and L2$***
#### 4.6.2 Internal Core Resources***
#### 4.6.3 The Shared L3 Cache Memory (L3$)***
#### 4.6.4 The Memory Controller***
#### 4.6.5 The Main Memory***
#### 4.6.6 Queue, Uncore, and I/O***
### ***4.7 IMROTATEMC: MAKING IMROTATE MORE EFFICIENT***
#### 4.7.1 Rotate2(): How Bad is Square Root and FP Division?
#### 4.7.2 Rotate3() and Rotate4(): How Bad Is sin() and cos()?
#### 4.7.3 Rotate5(): How Bad Is Integer Division/Multiplication?
#### 4.7.4 Rotate6(): Consolidating Computations
#### 4.7.5 Rotate7(): Consolidating More Computations
#### 4.7.6 Overall Performance of imrotateMC
### ***4.8 CHAPTER SUMMARY***

<br><br><br>

<hr class="division2">

## **Thread Management and Synchronization**
### ***5.1 EDGE DETECTION PROGRAM: IMEDGE.C***
#### 5.1.1 Description of the imedge.c
#### 5.1.2 imedge.c: Parametric Restrictions and Simplifications
#### 5.1.3 imedge.c: Theory of Operation
### ***5.2 IMEDGE.C : IMPLEMENTATION***
#### 5.2.1 Initialization and Time-Stamping
#### 5.2.2 Initialization Functions for Different Image Representations
#### 5.2.3 Launching and Terminating Threads
#### 5.2.4 Gaussian Filter
#### 5.2.5 Sobel
#### 5.2.6 Threshold
### ***5.3 PERFORMANCE OF IMEDGE***
### ***5.4 IMEDGEMC: MAKING IMEDGE MORE EFFICIENT***
#### 5.4.1 Using Precomputation to Reduce Bandwidth
#### 5.4.2 Storing the Precomputed Pixel Values
#### 5.4.3 Precomputing Pixel Values
#### 5.4.4 Reading the Image and Precomputing Pixel Values
#### 5.4.5 PrGaussianFilter
#### 5.4.6 PrSobel
#### 5.4.7 PrThreshold
### ***5.5 PERFORMANCE OF IMEDGEMC***
### ***5.6 IMEDGEMCT: SYNCHRONIZING THREADS EFFICIENTLY***
#### 5.6.1 Barrier Synchronization
#### 5.6.2 MUTEX Structure for Data Sharing
### ***5.7 IMEDGEMCT: IMPLEMENTATION***
#### 5.7.1 Using a MUTEX: Read Image, Precompute
#### 5.7.2 Precomputing One Row at a Time
### ***5.8 PERFORMANCE OF IMEDGEMCT***
<br><br><br>

<hr class="division2">

## **Introduction to GPU Parallelism and CUDA**

### ***6.1 ONCE UPON A TIME ... NVIDIA ...***
#### 6.1.1 The Birth of the GPU
#### 6.1.2 Early GPU Architectures
#### 6.1.3 The Birth of the GPGPU
#### 6.1.4 Nvidia, ATI Technologies, and Intel
### ***6.2 COMPUTE-UNIFIED DEVICE ARCHITECTURE (CUDA)***
#### 6.2.1 CUDA, OpenCL, and Other GPU Languages
#### 6.2.2 Device Side versus Host Side Code
### ***6.3 UNDERSTANDING GPU PARALLELISM***
#### 6.3.1 How Does the GPU Achieve High Performance?
#### 6.3.2 CPU versus GPU Architectural Differences
### ***6.4 CUDA VERSION OF THE IMAGE FLIPPER: IMFLIPG.CU***
#### 6.4.1 imflipG.cu: Read the Image into a CPU-Side Array
#### 6.4.2 Initialize and Query the GPUs
#### 6.4.3 GPU-Side Time-Stamping
#### 6.4.4 GPU-Side Memory Allocation
#### 6.4.5 GPU Drivers and Nvidia Runtime Engine
#### 6.4.6 CPU→GPU Data Transfer
#### 6.4.7 Error Reporting Using Wrapper Functions
#### 6.4.8 GPU Kernel Execution
#### 6.4.9 Finish Executing the GPU Kernel
#### 6.4.10 Transfer GPU Results Back to the CPU
#### 6.4.11 Complete Time-Stamping
#### 6.4.12 Report the Results and Cleanup
#### 6.4.13 Reading and Writing the BMP File
#### 6.4.14 Vflip(): The GPU Kernel for Vertical Flipping
#### 6.4.15 What Is My Thread ID, Block ID, and Block Dimension?
#### 6.4.16 Hflip(): The GPU Kernel for Horizontal Flipping
#### 6.4.17 Hardware Parameters: threadIDx.x, blockIdx.x, blockDim.x
#### 6.4.18 PixCopy(): The GPU Kernel for Copying an Image
#### 6.4.19 CUDA Keywords
### ***6.5 CUDA PROGRAM DEVELOPMENT IN WINDOWS***
#### 6.5.1 Installing MS Visual Studio 2015 and CUDA Toolkit 8.0
#### 6.5.2 Creating Project imflipG.cu in Visual Studio 2015
#### 6.5.3 Compiling Project imflipG.cu in Visual Studio 2015
#### 6.5.4 Running Our First CUDA Application: imflipG.exe
#### 6.5.5 Ensuring Your Program’s Correctness
### ***6.6 CUDA PROGRAM DEVELOPMENT ON A MAC PLATFORM***
#### 6.6.1 Installing XCode on Your Mac
#### 6.6.2 Installing the CUDA Driver and CUDA Toolkit
#### 6.6.3 Compiling and Running CUDA Applications on a Mac
### ***6.7 CUDA PROGRAM DEVELOPMENT IN A UNIX PLATFORM***
#### 6.7.1 Installing Eclipse and CUDA Toolkit
#### 6.7.2 ssh into a Cluster
#### 6.7.3 Compiling and Executing Your CUDA Code
<br><br><br>

<hr class="division2">

## **CUDA Host/Device Programming Model**

### ***7.1 DESIGNING YOUR PROGRAM’S PARALLELISM***
#### 7.1.1 Conceptually Parallelizing a Task
#### 7.1.2 What Is a Good Block Size for Vflip()?
#### 7.1.3 imflipG.cu: Interpreting the Program Output
#### 7.1.4 imflipG.cu: Performance Impact of Block and Image Size
### ***7.2 KERNEL LAUNCH COMPONENTS***
#### 7.2.1 Grids
#### 7.2.2 Blocks
#### 7.2.3 Threads
#### 7.2.4 Warps and Lanes
### ***7.3 IMFLIPG.CU: UNDERSTANDING THE KERNEL DETAILS***
#### 7.3.1 Launching Kernels in main() and Passing Arguments to Them
#### 7.3.2 Thread Execution Steps
#### 7.3.3 Vflip() Kernel Details
#### 7.3.4 Comparing Vflip() and MTFlipV()
#### 7.3.5 Hflip() Kernel Details
#### 7.3.6 PixCopy() Kernel Details
### ***7.4 DEPENDENCE OF PCI EXPRESS SPEED ON THE CPU***
### ***7.5 PERFORMANCE IMPACT OF PCI EXPRESS BUS***
#### 7.5.1 Data Transfer Time, Speed, Latency, Throughput, and Bandwidth
#### 7.5.2 PCIe Throughput Achieved with imflipG.cu
### ***7.6 PERFORMANCE IMPACT OF GLOBAL MEMORY BUS***
### ***7.7 PERFORMANCE IMPACT OF COMPUTE CAPABILITY***
#### 7.7.1 Fermi, Kepler, Maxwell, Pascal, and Volta Families
#### 7.7.2 Relative Bandwidth Achieved in Different Families
#### 7.7.3 imflipG2.cu: Compute Capability 2.0 Version of imflipG.cu
#### 7.7.4 imflipG2.cu: Changes in main()
#### 7.7.5 The PxCC20() Kernel
#### 7.7.6 The VfCC20() Kernel
### ***7.8 PERFORMANCE OF IMFLIPG2.CU***
### ***7.9 OLD-SCHOOL CUDA DEBUGGING***
#### 7.9.1 Common CUDA Bugs
#### 7.9.2 return Debugging
#### 7.9.3 Comment-Based Debugging
#### 7.9.4 printf() Debugging
### ***7.10 BIOLOGICAL REASONS FOR SOFTWARE BUGS***
#### 7.10.1 How Is Our Brain Involved in Writing/Debugging Code?
#### 7.10.2 Do We Write Buggy Code When We Are Tired?
##### 7.10.2.1 Attention
##### 7.10.2.2 Physical Tiredness
##### 7.10.2.3 Tiredness Due to Heavy Physical Activity
##### 7.10.2.4 Tiredness Due to Needing Sleep
##### 7.10.2.5 Mental Tiredness
<br><br><br>

<hr class="division2">

## **Understanding GPU Hardware Architecture**

### ***8.1 GPU HARDWARE ARCHITECTURE***
### ***8.2 GPU HARDWARE COMPONENTS***
#### 8.2.1 SM: Streaming Multiprocessor
#### 8.2.2 GPU Cores
#### 8.2.3 Giga-Thread Scheduler
#### 8.2.4 Memory Controllers
#### 8.2.5 Shared Cache Memory (L2$)
#### 8.2.6 Host Interface
### ***8.3 NVIDIA GPU ARCHITECTURES***
#### 8.3.1 Fermi Architecture
#### 8.3.2 GT, GTX, and Compute Accelerators
#### 8.3.3 Kepler Architecture
#### 8.3.4 Maxwell Architecture
#### 8.3.5 Pascal Architecture and NVLink
### ***8.4 CUDA EDGE DETECTION: IMEDGEG.CU***
#### 8.4.1 Variables to Store the Image in CPU, GPU Memory
##### 8.4.1.1 TheImage and CopyImage
##### 8.4.1.2 GPUImg
##### 8.4.1.3 GPUBWImg
##### 8.4.1.4 GPUGaussImg
##### 8.4.1.5 GPUGradient and GPUTheta
##### 8.4.1.6 GPUResultImg
#### 8.4.2 Allocating Memory for the GPU Variables
#### 8.4.3 Calling the Kernels and Time-Stamping Their Execution
#### 8.4.4 Computing the Kernel Performance
#### 8.4.5 Computing the Amount of Kernel Data Movement
#### 8.4.6 Reporting the Kernel Performance
### ***8.5 IMEDGEG: KERNELS***
#### 8.5.1 BWKernel()
#### 8.5.2 GaussKernel()
#### 8.5.3 SobelKernel()
#### 8.5.4 ThresholdKernel()
### ***8.6 PERFORMANCE OF IMEDGEG.CU***
#### 8.6.1 imedgeG.cu: PCIe Bus Utilization
#### 8.6.2 imedgeG.cu: Runtime Results
#### 8.6.3 imedgeG.cu: Kernel Performance Comparison
### ***8.7 GPU CODE: COMPILE TIME***
#### 8.7.1 Designing CUDA Code
#### 8.7.2 Compiling CUDA Code
#### 8.7.3 GPU Assembly: PTX, CUBIN
### ***8.8 GPU CODE: LAUNCH***
#### 8.8.1 OS Involvement and CUDA DLL File
#### 8.8.2 GPU Graphics Driver
#### 8.8.3 CPU←→GPU Memory Transfers
### ***8.9 GPU CODE: EXECUTION (RUN TIME)***
#### 8.9.1 Getting the Data
#### 8.9.2 Getting the Code and Parameters
#### 8.9.3 Launching Grids of Blocks
#### 8.9.4 Giga Thread Scheduler (GTS)
#### 8.9.5 Scheduling Blocks
#### 8.9.6 Executing Blocks
#### 8.9.7 Transparent Scalability

<br><br><br>

<hr class="division2">

## **Understanding GPU Cores**
### ***9.1 GPU ARCHITECTURE FAMILIES***
#### 9.1.1 Fermi Architecture
#### 9.1.2 Fermi SM Structure
#### 9.1.3 Kepler Architecture
#### 9.1.4 Kepler SMX Structure
#### 9.1.5 Maxwell Architecture
#### 9.1.6 Maxwell SMM Structure
#### 9.1.7 Pascal GP100 Architecture
#### 9.1.8 Pascal GP100 SM Structure
#### 9.1.9 Family Comparison: Peak GFLOPS and Peak DGFLOPS
#### 9.1.10 GPU Boost
#### 9.1.11 GPU Power Consumption
#### 9.1.12 Computer Power Supply
### ***9.2 STREAMING MULTIPROCESSOR (SM) BUILDING BLOCKS***
#### 9.2.1 GPU Cores
#### 9.2.2 Double Precision Units (DPU)
#### 9.2.3 Special Function Units (SFU)
#### 9.2.4 Register File (RF)
#### 9.2.5 Load/Store Queues (LDST)
#### 9.2.6 L1$ and Texture Cache
#### 9.2.7 Shared Memory
#### 9.2.8 Constant Cache
#### 9.2.9 Instruction Cache
#### 9.2.10 Instruction Buffer
#### 9.2.11 Warp Schedulers
#### 9.2.12 Dispatch Units
### ***9.3 PARALLEL THREAD EXECUTION (PTX) DATA TYPES***
#### 9.3.1 INT8 : 8-bit Integer
#### 9.3.2 INT16 : 16-bit Integer
#### 9.3.3 24-bit Integer
#### 9.3.4 INT32 : 32-bit Integer
#### 9.3.5 Predicate Registers (32-bit)
#### 9.3.6 INT64 : 64-bit Integer
#### 9.3.7 128-bit Integer
#### 9.3.8 FP32: Single Precision Floating Point (float)
#### 9.3.9 FP64: Double Precision Floating Point (double)
#### 9.3.10 FP16: Half Precision Floating Point (half)
#### 9.3.11 What is a FLOP?
#### 9.3.12 Fused Multiply-Accumulate (FMA) versus Multiply-Add(MAD)
#### 9.3.13 Quad and Octo Precision Floating Point
#### 9.3.14 Pascal GP104 Engine SM Structure
### ***9.4 IMFLIPGC.CU: CORE-FRIENDLY IMFLIPG***
#### 9.4.1 Hflip2(): Precomputing Kernel Parameters
#### 9.4.2 Vflip2(): Precomputing Kernel Parameters
#### 9.4.3 Computing Image Coordinates by a Thread
#### 9.4.4 Block ID versus Image Row Mapping
#### 9.4.5 Hflip3(): Using a 2D Launch Grid
#### 9.4.6 Vflip3(): Using a 2D Launch Grid
#### 9.4.7 Hflip4(): Computing Two Consecutive Pixels
#### 9.4.8 Vflip4(): Computing Two Consecutive Pixels
#### 9.4.9 Hflip5(): Computing Four Consecutive Pixels
#### 9.4.10 Vflip5(): Computing Four Consecutive Pixels
#### 9.4.11 PixCopy2(), PixCopy3(): Copying 2,4 Consecutive Pixels at a Time
### ***9.5 IMEDGEGC.CU: CORE-FRIENDLY IMEDGEG 299***
#### 9.5.1 BWKernel2(): Using Precomputed Values and 2D Blocks
#### 9.5.2 GaussKernel2(): Using Precomputed Values and 2D Blocks
<br><br><br>

<hr class="division2">




<br><br><br>

<hr class="division2">

## **Understanding GPU Memory**

### ***10.1 GLOBAL MEMORY***
### ***10.2 L2 CACHE***
### ***10.3 TEXTURE/L1 CACHE***
### ***10.4 SHARED MEMORY***
#### 10.4.1 Split versus Dedicated Shared Memory
#### 10.4.2 Memory Resources Available Per Core
#### 10.4.3 Using Shared Memory as Software Cache
#### 10.4.4 Allocating Shared Memory in an SM
### ***10.5 INSTRUCTION CACHE***
### ***10.6 CONSTANT MEMORY***
### ***10.7 IMFLIPGCM.CU: CORE AND MEMORY FRIENDLY IMFLIPG***
#### 10.7.1 Hflip6(),Vflip6(): Using Shared Memory as Buffer
#### 10.7.2 Hflip7(): Consecutive Swap Operations in Shared Memory
#### 10.7.3 Hflip8(): Using Registers to Swap Four Pixels
#### 10.7.4 Vflip7(): Copying 4 Bytes (int) at a Time
#### 10.7.5 Aligned versus Unaligned Data Access in Memory
#### 10.7.6 Vflip8(): Copying 8 Bytes at a Time
#### 10.7.7 Vflip9(): Using Only Global Memory, 8 Bytes at a Time
#### 10.7.8 PixCopy4(), PixCopy5(): Copying One versus 4 Bytes Using Shared Memory
#### 10.7.9 PixCopy6(), PixCopy7(): Copying One/Two Integers Using Global Memory
### ***10.8 IMEDGEGCM.CU: CORE- & MEMORY-FRIENDLY IMEDGEG***
#### 10.8.1 BWKernel3(): Using Byte Manipulation to Extract RGB
#### 10.8.2 GaussKernel3(): Using Constant Memory
#### 10.8.3 Ways to Handle Constant Values
#### 10.8.4 GaussKernel4(): Buffering Neighbors of 1 Pixel in Shared Memory
#### 10.8.5 GaussKernel5(): Buffering Neighbors of 4 Pixels in Shared Memory
#### 10.8.6 GaussKernel6(): Reading 5 Vertical Pixels into Shared Memory
#### 10.8.7 GaussKernel7(): Eliminating the Need to Account for Edge Pixels
#### 10.8.8 GaussKernel8(): Computing 8 Vertical Pixels
### ***10.9 CUDA OCCUPANCY CALCULATOR***
#### 10.9.1 Choosing the Optimum Threads/Block
#### 10.9.2 SM-Level Resource Limitations
#### 10.9.3 What is “Occupancy”?
#### 10.9.4 CUDA Occupancy Calculator: Resource Computation
#### 10.9.5 Case Study: GaussKernel7()
#### 10.9.6 Case Study: GaussKernel8()

<br><br><br>

<hr class="division2">

## **CUDA Streams**
### ***11.1 WHAT IS PIPELINING?***
#### 11.1.1 Execution Overlapping
#### 11.1.2 Exposed versus Coalesced Runtime
### ***11.2 MEMORY ALLOCATION***
#### 11.2.1 Physical versus Virtual Memory
#### 11.2.2 Physical to Virtual Address Translation
#### 11.2.3 Pinned Memory
#### 11.2.4 Allocating Pinned Memory with cudaMallocHost()
### ***11.3 FAST CPU←→GPU DATA TRANSFERS***
#### 11.3.1 Synchronous Data Transfers
#### 11.3.2 Asynchronous Data Transfers
### ***11.4 CUDA STREAMS***
#### 11.4.1 CPU→GPU Transfer, Kernel Exec, GPU→CPUTransfer
#### 11.4.2 Implementing Streaming in CUDA
#### 11.4.3 Copy Engine
#### 11.4.4 Kernel Execution Engine
#### 11.4.5 Concurrent Upstream and Downstream PCIe Transfers
#### 11.4.6 Creating CUDA Streams
#### 11.4.7 Destroying CUDA Streams
#### 11.4.8 Synchronizing CUDA Streams
### ***11.5 IMGSTR.CU: STREAMING IMAGE PROCESSING***
#### 11.5.1 Reading the Image into Pinned Memory
#### 11.5.2 Synchronous versus Single Stream
#### 11.5.3 Multiple Streams
#### 11.5.4 Data Dependence Across Multiple Streams
##### 11.5.4.1 Horizontal Flip: No Data Dependence
##### 11.5.4.2 Edge Detection: Data Dependence
##### 11.5.4.3 Preprocessing Overlapping Rows Synchronously
##### 11.5.4.4 Asynchronous Processing the Non-Overlapping Rows
### ***11.6 STREAMING HORIZONTAL FLIP KERNEL***
### ***11.7 IMGSTR.CU: STREAMING EDGE DETECTION***
### ***11.8 PERFORMANCE COMPARISON: IMGSTR.CU***
#### 11.8.1 Synchronous versus Asynchronous Results
#### 11.8.2 Randomness in the Results
#### 11.8.3 Optimum Queuing
#### 11.8.4 Best Case Streaming Results
#### 11.8.5 Worst Case Streaming Results
### ***11.9 NVIDIA VISUAL PROFILER: NVVP***
#### 11.9.1 Installing nvvp and nvprof
#### 11.9.2 Using nvvp
#### 11.9.3 Using nvprof
#### 11.9.4 imGStr Synchronous and Single-Stream Results
#### 11.9.5 imGStr 2- and 4-Stream Results

<br><br><br>

<hr class="division2">

## **CUDA Libraries**
### ***12.1 cuBLAS***
#### 12.1.1 BLAS Levels
#### 12.1.2 cuBLAS Datatypes
#### 12.1.3 Installing cuBLAS
#### 12.1.4 Variable Declaration and Initialization
#### 12.1.5 Device Memory Allocation
#### 12.1.6 Creating Context
#### 12.1.7 Transferring Data to the Device
#### 12.1.8 Calling cuBLAS Functions
#### 12.1.9 Transfer Data Back to the Host
#### 12.1.10 Deallocating Memory
#### 12.1.11 Example cuBLAS Program: Matrix Scalar
### ***12.2 CUFFT***
#### 12.2.1 cuFFT Library Characteristics
#### 12.2.2 A Sample Complex-to-Complex Transform
#### 12.2.3 A Sample Real-to-Complex Transform
### ***12.3 NVIDIA PERFORMANCE PRIMITIVES (NPP)***
### ***12.4 THRUST LIBRARY***

<br><br><br>

<hr class="division2">

## **Introduction to OpenCL**
### ***13.1 WHAT IS OpenCL?***
#### 13.1.1 Multiplatform
#### 13.1.2 Queue-Based
### ***13.2 IMAGE FLIP KERNEL IN OPENCL***
### ***13.3 RUNNING OUR KERNEL***
#### 13.3.1 Selecting a Device
#### 13.3.2 Running the Kernel
##### 13.3.2.1 Creating a Compute Context
##### 13.3.2.2 Creating a Command Queue
##### 13.3.2.3 Loading Kernel File
##### 13.3.2.4 Setting Up Kernel Invocation
#### 13.3.3 Runtimes of Our OpenCL Program
### ***13.4 EDGE DETECTION IN OpenCL***

<br><br><br>

<hr class="division2">

## **Other GPU Programming Languages**
### ***14.1 GPU PROGRAMMING WITH PYTHON***
#### 14.1.1 PyOpenCL Version of imflip
#### 14.1.2 PyOpenCL Element-Wise Kernel
### ***14.2 OPENGL***
### ***14.3 OPENGL ES: OPENGL FOR EMBEDDED SYSTEMS***
### ***14.4 VULKAN***
### ***14.5 MICROSOFT’S HIGH-LEVEL SHADING LANGUAGE (HLSL)***
#### 14.5.1 Shading
#### 14.5.2 Microsoft HLSL
### ***14.6 APPLE’S METAL API***
### ***14.7 APPLE’S SWIFT PROGRAMMING LANGUAGE***
### ***14.8 OPENCV***
#### 14.8.1 Installing OpenCV and Face Recognition
#### 14.8.2 Mobile-Cloudlet-Cloud Real-Time Face Recognition
#### 14.8.3 Acceleration as a Service (AXaas)

<br><br><br>

<hr class="division2">

## **Deep Learning Using CUDA**
### ***CUDNN LIBRARY FOR DEEP LEARNING***
<a href="https://developer.nvidia.com/" target="_blank">cuDNN Download</a>

<br><br><br>
#### Creating a Layer
```cuda
struct Conv_Layer
{
int inputs, outputs, kernelSize;
int inputWidth, inputHeight, outputWidth, outputHeight;
std::vector<float> convV;
std::vector<float> biasV;
...
};
struct Maxpool_Layer
{
int size, stride;
...
};
struct Fully_Connected_Layer
{
int inputs, outputs;
std::vector<float> neuronsV;
std::vector<float> biasV;
...
};
```
<br><br><br>

#### Creating a Network
```cuda
struct My_Network
{
cudnnTensorDescriptor_t dataTensorDesc, convTensorDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnActivationDescriptor_t lastLayerActDesc;
cudnnFilterDescriptor_t filterDesc;
cudnnPoolingDescriptor_t poolDesc;
void createHandles()
{
//General tensors and layers used in the network.
//These need to be initialized by a descriptor.
cudnnCreateTensorDescriptor(&dataTensorDesc);
cudnnCreateTensorDescriptor(&convTensorDesc);
cudnnCreateConvolutionDescriptor(&convDesc);
cudnnCreateActivationDescriptor(&lastLayerActDesc);
cudnnCreateFilterDescriptor(&filterDesc);
cudnnCreatePoolingDescriptor(&poolDesc);
}
void destroyHandles()
{
cudnnDestroyTensorDescriptor(&dataTensorDesc);
cudnnDestroyTensorDescriptor(&convTensorDesc);
cudnnDestroyConvolutionDescriptor(&convDesc);
cudnnDestroyActivationDescriptor(&lastLayerActDesc);
cudnnDestroyFilterDescriptor(&filterDesc);
cudnnDestroyPoolingDescriptor(&poolDesc);
}
...
};

```
<br><br><br>

#### Forward Propagation
```cuda
convoluteForward(...)
{
cudnnSetTensor4dDescriptor(dataTensorDesc, ...);
cudnnSetFilter4dDescriptor(filterDesc, ...);
cudnnSetConvolution2dDescriptor(convDesc, ...);
cudnnConvolutionForward(...);
}
```
<br><br><br>

#### Backpropagation
```cuda
cudnnActivationBackward(...)
cudnnPoolingBackward(...)
cudnnConvolutionBackwardBias(...)
cudnnConvolutionBackwardFilter(...)
```
<br><br><br>

#### Using cuBLAS in the Network
```cuda
fullyConenctedForward(...)
{
...
cublasSgemv(...);
...
}
```
<br><br><br>

---

### ***KERAS***
```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from keras import losses

model = Sequential()
model.add(Dense(units=..., input_dim=...))
model.add(Activation(’relu’))
model.add(Conv2D(..., activation=’relu’))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(..., activation=’softmax’))
model.compile(loss=losses.mean_squared_error, optimizer=’sgd’, metrics=[’accuracy’])
model.fit([training data input], [training data output], batch_size=..., epochs=...)
score = model.evaluate([test data input], [test data output])
```
Note that if Keras is running on the TensorFlow or CNTK backends, it automatically runs on the GPU if any GPU is detected. If the backend is Theano, there are multiple methods to use the GPU. One way is manually setting the device of the Theano configuration, as follows:
```python
import theano

theano.config.device = ’gpu’
theano.config.floatX = ’float32’
```

<br><br><br>


<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- Horst Simon, GPU Parallel Program Development Using CUDA
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


