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

### ***1	CUDA와 딥러닝 프레임워크 현***

CUDA/cuDNN을 사용하기 위한 병렬처리 개념과 프로그래밍 모델, 그래픽 카드 하드웨어 모델 등을 이해하고 병렬처리 개념을 소개합니다. 배운 내용들이 Caffe, Caffe2, Tensorflow에서 어떻게 쓰이고 있는지 소개합니다.

[실습] CUDA DeviceInfo
[실습] CUDA 기반 1차원/2차원 행렬 합,곱
[실습] 중첩 Reduce 연산 테크닉
[실습] Caffe, Caffe2, Tensorflow 코드 리뷰

<br><br><br>

---

### ***2	GPU 메모리활용 1***

CPU-GPU 간 데이터 복사가 이뤄지는 원리와, GPU 가 사용하는 여러 메모리 종류들의 특성들을 깊이있게 이해하고, 이를 바탕으로 효율적인 메모리 활용 테크닉을 학습하여 CUDA로 구현해봅니다.

[실습] 메모리 동기/비동기 복사
[실습] 공유 메모리/고정 메모리 복사
[실습] 글로벌 메모리/Zero-Copy 메모리 복사
[실습] 통합 메모리 복사

<br><br><br>

---

### ***3	GPU 메모리 활용 2/ 스트림 활용***	

GPU의 리소스 활용을 최대화하기 위한 액세스 테크닉과 스트림의 개념을 이해하고 CUDA로 구현해봅니다.

[실습] 메모리 정합/정렬 액세스
[실습] 메모리 뱅크 충돌과 패딩 회피
[실습] 데이터 전송 스트림과 이벤트 구현
[실습] 스트림 동기화 구현

<br><br><br>

---

### ***4	CUDA 디버깅과 프로파일링, cuDNN 활용***
CUDA 프로그램을 실질적으로 디버깅하거나 성능 최적화하는 방법을 이해하고 이때 사용하는 도구들을 활용해봅니다. 병렬처리 성능을 극대화시키기 위해cuDNN 을 학습한 다음, 효율적인 Convolution연산을 위한 GEMM(General Matrix Multiplication) 알고리즘을 학습하고 직접 구현해봅니다.

[실습] CUDA 디버깅 도구 활용
[실습] CUDA 시각화/프로파일링 도구 활용
[실습] GEMM 구현
[실습] CUDA/cuDNN 기반 Convolution Layer

<br><br><br>

---

### ***5	CUDA 딥러닝 구현***	
CUDA및 cuDNN을 활용해서 MaxPooling, Activation, FC 레이어 등을 구현해보고, 이들을 통합해서 Object Detection을 위한 YOLO v2를 구현해봅니다.

[실습] CUDA/cuDNN 기반 MaxPooling Layer
[실습] CUDA/cuDNN 기반 Activation Layer
[실습] CUDA/cuDNN 기반 FullyConnected Layer
[실습] YOLO v2 구현

<br><br><br>

---

### ***6	딥러닝 프레임워크의 커스텀레이어 구현***

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
