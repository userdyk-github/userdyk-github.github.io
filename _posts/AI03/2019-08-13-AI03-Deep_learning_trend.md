---
layout : post
title : AI03, Deep learning trend
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

## **Researcher**

> LeCun ｜ Hinton ｜ Fei-Fei Li ｜ Krizhevsky

<br><br><br>
<hr class="division2">

## **Framework**

Past
> Caffe2 ｜ MatConvNet

<br>

Present
> Tensorflw ｜ Pytorch

<br><br><br>
<hr class="division2">

## **Cloud Platform**

> AWS ｜ Google Cloud Platform ｜ Microsoft Azure

Enterprise IT(legacy IT) > Infrastructure(as a Service, IaaS) > Platform(as a Service, PaaS) > Software(as a Service, SaaS)


<br><br><br>
<hr class="division2">

## **Hardware**

> NVIDIA(GTX, Titan, TESLA)

<br><br><br>

<hr class="division2">

## **CUDA(Compute Unified Device Architecture) Programming**

### ***Present situation of	CUDA and deep learning framework***

Multiprocessing system and programming model concept for using CUDA/cuDNN, by grasping graphics card and hardware model in relation to Caffe, Caffe2, Tensorflow.

- CUDA DeviceInfo
- 1D/2D Matrix sum, product based on CUDA
- Optimizing parallel reduction
- Code review for Caffe, Caffe2, Tensorflow

<br><br><br>

---

### ***GPU Memory usage***

Deep understanding of the principle of data transfer between CPU and GPUs, and the characteristics of the different kinds of memory used by GPUs, and we learn efficient memory utilization techniques and implement them into CUDA.


메모리 동기/비동기 복사
공유 메모리/고정 메모리 복사
글로벌 메모리/Zero-Copy 메모리 복사
통합 메모리 복사

<br><br><br>

---

### ***GPU Memory and stream usage***	

Stream concept and access technique for maximizing resource utlization of GPU, with implementation.

메모리 정합/정렬 액세스
메모리 뱅크 충돌과 패딩 회피
데이터 전송 스트림과 이벤트 구현
스트림 동기화 구현

<br><br><br>

---

### ***CUDA debugging profiling, cuDNN usage***
CUDA 프로그램을 실질적으로 디버깅하거나 성능 최적화하는 방법을 이해하고 이때 사용하는 도구들을 활용해봅니다. 병렬처리 성능을 극대화시키기 위해cuDNN 을 학습한 다음, 효율적인 Convolution연산을 위한 GEMM(General Matrix Multiplication) 알고리즘을 학습하고 직접 구현해봅니다.

[실습] CUDA 디버깅 도구 활용
[실습] CUDA 시각화/프로파일링 도구 활용
[실습] GEMM 구현
[실습] CUDA/cuDNN 기반 Convolution Layer

<br><br><br>

---

### ***Implement of deep learning with CUDA***	
CUDA및 cuDNN을 활용해서 MaxPooling, Activation, FC 레이어 등을 구현해보고, 이들을 통합해서 Object Detection을 위한 YOLO v2를 구현해봅니다.

[실습] CUDA/cuDNN 기반 MaxPooling Layer
[실습] CUDA/cuDNN 기반 Activation Layer
[실습] CUDA/cuDNN 기반 FullyConnected Layer
[실습] YOLO v2 구현

<br><br><br>

---

### ***Implement of custom layers based on CUDA/cuDNN***

CUDA/cuDNN 구현 테크닉들을 기반으로, Caffe/Caffe2/Tensorflow에서 각각 적용해볼 수 있는 자신만의 사용자 정의 레이어를 구현해봅니다.

[실습] Caffe Custom Layer(CPP, CUDA, cuDNN)구현
[실습] Caffe2 Custom Operator(CPP, CUDA, cuDNN)구현
[실습] Tensorflow Custom Operator(CPP, CUDA, cuDNN)구현

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
