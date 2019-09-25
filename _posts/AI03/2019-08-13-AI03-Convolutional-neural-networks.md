---
layout : post
title : AI03, Convolutional neural networks
categories: [AI03]
comments : true
tags : [AI03]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
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

## **Convolution Operation**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Analog and Digital Signals**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **2D Convolution**

### ***2D Convolution of an Image to Different LSI System Responses***

```python
## Illustate 2D convolution of images through an example
import scipy.signal 
import numpy as np

# Take a 7x7 image as example
image = np.array([[1, 2, 3, 4, 5, 6, 7],  
                  [8, 9, 10, 11, 12, 13, 14],    
                  [15, 16, 17, 18, 19, 20, 21],   
                  [22, 23, 24, 25, 26, 27, 28],       
                  [29, 30, 31, 32, 33, 34, 35],        
                  [36, 37, 38, 39, 40, 41, 42],      
                  [43, 44, 45, 46, 47, 48, 49]])

# Defined an image-processing kernel 
filter_kernel = np.array([[-1, 1, -1],   
                          [-2, 3, 1],         
                          [2, -4, 0]])

# Convolve the image with the filter kernel through scipy 2D convolution to produce an output image of same dimension as that of the input
I = scipy.signal.convolve2d(image, filter_kernel,mode='same', boundary='fill', fillvalue=0) 
print(I)

# We replicate the logic of a scipy 2D convolution by going through the following steps 
# a) The boundaries need to be extended in both directions for the image and padded with zeroes.
#  For convolving the 7x7 image by 3x3 kernel, the dimensions need to be extended by  (3-1)/2—i.e., 1— 
#on either side for each dimension. So a skeleton image of 9x9 image would be created 
# in which the boundaries of 1 pixel are pre-filled with zero.
# b) The kernel needs to be flipped—i.e., rotated—by 180 degrees 
# c) The flipped kernel needs to placed at each coordinate location for the image and then the sum of
#coordinate-wise product with the image intensities needs to be computed. These sums for each coordinate would give 
#the intensities for the output image.

row,col=7,7

## Rotate the filter kernel twice by 90 degrees to get 180 rotation 
filter_kernel_flipped = np.rot90(filter_kernel,2) 

## Pad the boundaries of the image with zeroes and fill the rest from the original image 
image1 = np.zeros((9,9)) 
for i in range(row):  
    for j in range(col):  
        image1[i+1,j+1] = image[i,j] 
        print(image1)
        
## Define the output image
image_out = np.zeros((row,col))

## Dynamic shifting of the flipped filter at each image coordinate and then computing the convolved sum.
for i in range(1,1+row):  
    for j in range(1,1+col):   
        arr_chunk = np.zeros((3,3))
        for k,k1 in zip(range(i-1,i+2),range(3)):     
            for l,l1 in zip(range(j-1,j+2),range(3)):        
                arr_chunk[k1,l1] = image1[k,l]
        image_out[i-1,j-1] = np.sum(np.multiply(arr_chunk,filter_kernel_flipped))
print(image_out)
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
[[ -2  -8  -7  -6  -5  -4  28]
 [  5  -3  -4  -5  -6  -7  28]
 [ -2 -10 -11 -12 -13 -14  28]
 [ -9 -17 -18 -19 -20 -21  28]
 [-16 -24 -25 -26 -27 -28  28]
 [-23 -31 -32 -33 -34 -35  28]
 [-29  13  13  13  13  13  27]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 5. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 5. 6. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 5. 6. 7. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 5. 6. 7. 0.]
 [0. 8. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 2. 3. 4. 5. 6. 7. 0.]
 [0. 8. 9. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44. 45.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44. 45. 46.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44. 45. 46. 47.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44. 45. 46. 47. 48.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  0.]
 [ 0.  8.  9. 10. 11. 12. 13. 14.  0.]
 [ 0. 15. 16. 17. 18. 19. 20. 21.  0.]
 [ 0. 22. 23. 24. 25. 26. 27. 28.  0.]
 [ 0. 29. 30. 31. 32. 33. 34. 35.  0.]
 [ 0. 36. 37. 38. 39. 40. 41. 42.  0.]
 [ 0. 43. 44. 45. 46. 47. 48. 49.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
[[ -2.  -8.  -7.  -6.  -5.  -4.  28.]
 [  5.  -3.  -4.  -5.  -6.  -7.  28.]
 [ -2. -10. -11. -12. -13. -14.  28.]
 [ -9. -17. -18. -19. -20. -21.  28.]
 [-16. -24. -25. -26. -27. -28.  28.]
 [-23. -31. -32. -33. -34. -35.  28.]
 [-29.  13.  13.  13.  13.  13.  27.]]
```
<hr class='division3'>
</details>

<hr class="division2">

## **Common Image-Processing Filters**

### ***Mean Filter***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Median Filter***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Gaussian Filter***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Gradient-based Filters*** 

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Sobel Edge-Detection Filter*** 

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Convolution Neural Networks**

### ***Convolution Layer***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---


### ***Pooling Layer***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Components of Convolution Neural Networks**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Backpropagation Through the Convolutional Layer**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Backpropagation Through the Pooling Layers**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Weight Sharing Through Convolution and Its Advantages**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Translation Equivariance**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Translation Invariance Due to Pooling**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Dropout Layers and Regularization**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Convolutional Neural Network for Digit Recognition on the MNIST Dataset**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Convolutional Neural Network for Solving Real-World Problems**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Batch Normalization**

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>

<hr class="division2">

## **Different Architectures in Convolutional Neural Networks**

### ***LeNet***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***AlexNet***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***VGG16***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***ResNet***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">

## **Transfer Learning**

### ***Guidelines for Using Transfer Learning***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Transfer Learning with Google’s InceptionV3***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
<br><br><br>

---

### ***Transfer Learning with Pre-trained VGG16***

```python
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>
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
<hr class='division3'>
</details>

