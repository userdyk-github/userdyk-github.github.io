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
# Convolution of an Image with Mean Filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import cv2

img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray,cmap='gray')
mean = 0
var = 100
sigma = var**0.5
row,col = 650,442 
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col) 
gray_noisy = gray + gauss
plt.imshow(gray_noisy,cmap='gray') 

## Mean filter
Hm = np.array([[1,1,1],[1,1,1],[1,1,1]])/float(9)
Gm = convolve2d(gray_noisy,Hm,mode='same')
plt.imshow(Gm,cmap='gray')
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (2)](https://user-images.githubusercontent.com/52376448/65568030-4e65ed00-df93-11e9-90a9-fc5b9e1db35c.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Median Filter***

```python
## Generate random integers from 0 to 20 
## If the value is zero we will replace the image pixel with a low value of 0 that corresponds to a black pixel
## If the value is 20 we will replace the image pixel with a high value of 255 that corresponds to a white pixel
## We have taken 20 integers, out of which we will only tag integers 1 and 20 as salt and pepper noise 
## Hence, approximately 10% of the overall pixels are salt and pepper noise. If we want to reduce it
## to 5% we can take integers from 0 to 40 and then treat 0 as an indicator for a black pixel and 40 as an indicator for a white pixel.
np.random.seed(0)
gray_sp = gray*1 
sp_indices = np.random.randint(0,21,[row,col]) 
for i in range(row):    
    for j in range(col):    
        if sp_indices[i,j] == 0:
            gray_sp[i,j] = 0  
        if sp_indices[i,j] == 20:     
            gray_sp[i,j] = 255 
plt.imshow(gray_sp,cmap='gray')
                
## Now we want to remove the salt and pepper noise through a Median filter.
## Using the opencv Median filter for the same
gray_sp_removed = cv2.medianBlur(gray_sp,3)
plt.imshow(gray_sp_removed,cmap='gray')

##Implementation of the 3x3 Median filter without using opencv
gray_sp_removed_exp = gray*1 
for i in range(row):  
    for j in range(col):   
        local_arr = []      
        for k in range(np.max([0,i-1]),np.min([i+2,row])):    
            for l in range(np.max([0,j-1]),np.min([j+2,col])):  
                local_arr.append(gray_sp[k,l])     
        gray_sp_removed_exp[i,j] = np.median(local_arr) 
plt.imshow(gray_sp_removed_exp,cmap='gray')
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/65568096-7ead8b80-df93-11e9-8f9d-404cf4ce5599.png)
<hr class='division3'>
</details>
<br><br><br>

---

### ***Gaussian Filter***

```python
Hg = np.zeros((20,20)) 
for i in range(20):   
    for j in range(20):   
        Hg[i,j] = np.exp(-((i-10)**2 + (j-10)**2)/10)
plt.imshow(Hg,cmap='gray')
gray_blur = convolve2d(gray,Hg,mode='same') 
plt.imshow(gray_blur,cmap='gray')
gray_high = 0.1
gray_enhanced = gray + 0.025 * gray_high
plt.imshow(gray_enhanced,cmap='gray')
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (4)](https://user-images.githubusercontent.com/52376448/65568118-8ff69800-df93-11e9-8380-69e072681c53.png)
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
# Convolution Using a Sobel Filter
Hx = np.array([[ 1,0, -1],[2,0,-2],[1,0,-1]],dtype=np.float32)
Gx = convolve2d(gray,Hx,mode='same') 
plt.imshow(Gx,cmap='gray')

Hy = np.array([[ -1,-2, -1],[0,0,0],[1,2,1]],dtype=np.float32) 
Gy = convolve2d(gray,Hy,mode='same') 
plt.imshow(Gy,cmap='gray')

G = (Gx*Gx + Gy*Gy)**0.5 
plt.imshow(G,cmap='gray')
```

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![다운로드 (5)](https://user-images.githubusercontent.com/52376448/65568145-a3a1fe80-df93-11e9-9048-c943cb039fc9.png)
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

