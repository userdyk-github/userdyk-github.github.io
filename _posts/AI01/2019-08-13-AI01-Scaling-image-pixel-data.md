---
layout : post
title : AI01, Scaling image pixel data
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Scaling-image-pixel-data.md" target="_blank">page management</a><br>
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
## **numpy**
### ***Sample image***
[boat.png][1]
<img width="640" alt="boat" src="https://user-images.githubusercontent.com/52376448/71426209-b8547680-26e9-11ea-9f17-8088e89db405.png">
<br><br><br>

---

### ***Normalize Pixel Values***

`STEP 1`
```python
# example of pixel normalization
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)

# confirm pixel range is 0-255
print(pixels.shape)
print(pixels.dtype)
print(pixels.min(), pixels.max())
pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT1</summary>
<hr class='division3'>
<p>
    (856, 1280, 4)<br>
    uint8<br>
    0 255
</p>
```
array([[[221, 223, 226, 255],
        [210, 212, 215, 255],
        [191, 192, 195, 255],
        ...,
        [191, 192, 195, 255],
        [210, 212, 215, 255],
        [221, 223, 226, 255]],

       [[213, 215, 217, 255],
        [190, 192, 194, 255],
        [206, 207, 208, 255],
        ...,
        [206, 207, 208, 255],
        [190, 192, 194, 255],
        [213, 215, 217, 255]],

       [[199, 201, 204, 255],
        [196, 198, 199, 255],
        [236, 234, 236, 255],
        ...,
        [236, 234, 236, 255],
        [196, 198, 199, 255],
        [199, 201, 204, 255]],

       ...,

       [[193, 193, 193, 255],
        [180, 180, 180, 255],
        [151, 152, 152, 255],
        ...,
        [154, 154, 155, 255],
        [180, 180, 180, 255],
        [193, 193, 193, 255]],

       [[197, 197, 197, 255],
        [192, 192, 192, 255],
        [179, 179, 179, 255],
        ...,
        [179, 179, 179, 255],
        [192, 192, 192, 255],
        [197, 197, 197, 255]],

       [[198, 198, 198, 255],
        [196, 196, 196, 255],
        [192, 192, 192, 255],
        ...,
        [192, 192, 192, 255],
        [196, 196, 196, 255],
        [198, 198, 198, 255]]], dtype=uint8)
```
<hr class='division3'>
</details>

<br>

`STEP 2`
```python
# convert from integers to floats
pixels = pixels.astype('float32')

# normalize to the range 0-1
pixels /= 255.0

# confirm the normalization
print(pixels.min(), pixels.max())

pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT2</summary>
<hr class='division3'>
<p>
    0.0 0.003921569
</p>
```
array([[[0.00339869, 0.00342945, 0.00347559, 0.00392157],
        [0.00322953, 0.00326028, 0.00330642, 0.00392157],
        [0.00293733, 0.00295271, 0.00299885, 0.00392157],
        ...,
        [0.00293733, 0.00295271, 0.00299885, 0.00392157],
        [0.00322953, 0.00326028, 0.00330642, 0.00392157],
        [0.00339869, 0.00342945, 0.00347559, 0.00392157]],

       [[0.00327566, 0.00330642, 0.00333718, 0.00392157],
        [0.00292195, 0.00295271, 0.00298347, 0.00392157],
        [0.00316801, 0.00318339, 0.00319877, 0.00392157],
        ...,
        [0.00316801, 0.00318339, 0.00319877, 0.00392157],
        [0.00292195, 0.00295271, 0.00298347, 0.00392157],
        [0.00327566, 0.00330642, 0.00333718, 0.00392157]],

       [[0.00306036, 0.00309112, 0.00313725, 0.00392157],
        [0.00301423, 0.00304498, 0.00306036, 0.00392157],
        [0.00362937, 0.00359862, 0.00362937, 0.00392157],
        ...,
        [0.00362937, 0.00359862, 0.00362937, 0.00392157],
        [0.00301423, 0.00304498, 0.00306036, 0.00392157],
        [0.00306036, 0.00309112, 0.00313725, 0.00392157]],

       ...,

       [[0.00296809, 0.00296809, 0.00296809, 0.00392157],
        [0.00276817, 0.00276817, 0.00276817, 0.00392157],
        [0.00232218, 0.00233756, 0.00233756, 0.00392157],
        ...,
        [0.00236832, 0.00236832, 0.0023837 , 0.00392157],
        [0.00276817, 0.00276817, 0.00276817, 0.00392157],
        [0.00296809, 0.00296809, 0.00296809, 0.00392157]],

       [[0.0030296 , 0.0030296 , 0.0030296 , 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.00275279, 0.00275279, 0.00275279, 0.00392157],
        ...,
        [0.00275279, 0.00275279, 0.00275279, 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.0030296 , 0.0030296 , 0.0030296 , 0.00392157]],

       [[0.00304498, 0.00304498, 0.00304498, 0.00392157],
        [0.00301423, 0.00301423, 0.00301423, 0.00392157],
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        ...,
        [0.00295271, 0.00295271, 0.00295271, 0.00392157],
        [0.00301423, 0.00301423, 0.00301423, 0.00392157],
        [0.00304498, 0.00304498, 0.00304498, 0.00392157]]], dtype=float32)
```
<hr class='division3'>
</details>


<br><br><br>

<hr class="division2">


### ***Center Pixel Values***

- <strong>Global Centering</strong>: Calculating and subtracting the mean pixel value <strong>across color channels</strong>. [mean:O, std:X]
- <strong>Local Centering</strong>: Calculating and subtracting the mean pixel value <strong>per color channel</strong>. [mean:O, std:O]

<br><br><br>


#### Global Centering

`STEP1`
```python
# example of global centering (subtract mean)
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)
pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
array([[[221, 223, 226, 255],
        [210, 212, 215, 255],
        [191, 192, 195, 255],
        ...,
        [191, 192, 195, 255],
        [210, 212, 215, 255],
        [221, 223, 226, 255]],

       [[213, 215, 217, 255],
        [190, 192, 194, 255],
        [206, 207, 208, 255],
        ...,
        [206, 207, 208, 255],
        [190, 192, 194, 255],
        [213, 215, 217, 255]],

       [[199, 201, 204, 255],
        [196, 198, 199, 255],
        [236, 234, 236, 255],
        ...,
        [236, 234, 236, 255],
        [196, 198, 199, 255],
        [199, 201, 204, 255]],

       ...,

       [[193, 193, 193, 255],
        [180, 180, 180, 255],
        [151, 152, 152, 255],
        ...,
        [154, 154, 155, 255],
        [180, 180, 180, 255],
        [193, 193, 193, 255]],

       [[197, 197, 197, 255],
        [192, 192, 192, 255],
        [179, 179, 179, 255],
        ...,
        [179, 179, 179, 255],
        [192, 192, 192, 255],
        [197, 197, 197, 255]],

       [[198, 198, 198, 255],
        [196, 196, 196, 255],
        [192, 192, 192, 255],
        ...,
        [192, 192, 192, 255],
        [196, 196, 196, 255],
        [198, 198, 198, 255]]], dtype=uint8)
```
<hr class='division3'>
</details>

<br>

`STEP2`
```python
# convert from integers to floats
pixels = pixels.astype('float32')
print(pixels.shape)

# calculate global mean across color channels
mean = pixels.mean()
print('Mean: %.3f'% mean)
print('Min: %.3f, Max: %.3f'% (pixels.min(), pixels.max()))

# global centering of pixels
pixels -= mean

# confirm it had the desired effect
mean = pixels.mean()
print('Mean: %.3f'% mean)
print('Min: %.3f, Max: %.3f'% (pixels.min(), pixels.max()))
pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
(856, 1280, 4)<br>
Mean: 184.501<br>
Min: 0.000, Max: 255.000<br>
Mean: -0.000<br>
Min: -184.501, Max: 70.499
</p>
```
array([[[ 36.49881,  38.49881,  41.49881,  70.49881],
        [ 25.49881,  27.49881,  30.49881,  70.49881],
        [  6.49881,   7.49881,  10.49881,  70.49881],
        ...,
        [  6.49881,   7.49881,  10.49881,  70.49881],
        [ 25.49881,  27.49881,  30.49881,  70.49881],
        [ 36.49881,  38.49881,  41.49881,  70.49881]],

       [[ 28.49881,  30.49881,  32.49881,  70.49881],
        [  5.49881,   7.49881,   9.49881,  70.49881],
        [ 21.49881,  22.49881,  23.49881,  70.49881],
        ...,
        [ 21.49881,  22.49881,  23.49881,  70.49881],
        [  5.49881,   7.49881,   9.49881,  70.49881],
        [ 28.49881,  30.49881,  32.49881,  70.49881]],

       [[ 14.49881,  16.49881,  19.49881,  70.49881],
        [ 11.49881,  13.49881,  14.49881,  70.49881],
        [ 51.49881,  49.49881,  51.49881,  70.49881],
        ...,
        [ 51.49881,  49.49881,  51.49881,  70.49881],
        [ 11.49881,  13.49881,  14.49881,  70.49881],
        [ 14.49881,  16.49881,  19.49881,  70.49881]],

       ...,

       [[  8.49881,   8.49881,   8.49881,  70.49881],
        [ -4.50119,  -4.50119,  -4.50119,  70.49881],
        [-33.50119, -32.50119, -32.50119,  70.49881],
        ...,
        [-30.50119, -30.50119, -29.50119,  70.49881],
        [ -4.50119,  -4.50119,  -4.50119,  70.49881],
        [  8.49881,   8.49881,   8.49881,  70.49881]],

       [[ 12.49881,  12.49881,  12.49881,  70.49881],
        [  7.49881,   7.49881,   7.49881,  70.49881],
        [ -5.50119,  -5.50119,  -5.50119,  70.49881],
        ...,
        [ -5.50119,  -5.50119,  -5.50119,  70.49881],
        [  7.49881,   7.49881,   7.49881,  70.49881],
        [ 12.49881,  12.49881,  12.49881,  70.49881]],

       [[ 13.49881,  13.49881,  13.49881,  70.49881],
        [ 11.49881,  11.49881,  11.49881,  70.49881],
        [  7.49881,   7.49881,   7.49881,  70.49881],
        ...,
        [  7.49881,   7.49881,   7.49881,  70.49881],
        [ 11.49881,  11.49881,  11.49881,  70.49881],
        [ 13.49881,  13.49881,  13.49881,  70.49881]]], dtype=float32)
```
<hr class='division3'>
</details>

<br><br><br>

---

#### Local Centering

`STEP1`
```python
# example of per-channel centering (subtract mean)
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)
print(pixels.shape)

# convert from integers to floats
pixels = pixels.astype('float32')
pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (856, 1280, 4)
</p>
```
array([[[221., 223., 226., 255.],
        [210., 212., 215., 255.],
        [191., 192., 195., 255.],
        ...,
        [191., 192., 195., 255.],
        [210., 212., 215., 255.],
        [221., 223., 226., 255.]],

       [[213., 215., 217., 255.],
        [190., 192., 194., 255.],
        [206., 207., 208., 255.],
        ...,
        [206., 207., 208., 255.],
        [190., 192., 194., 255.],
        [213., 215., 217., 255.]],

       [[199., 201., 204., 255.],
        [196., 198., 199., 255.],
        [236., 234., 236., 255.],
        ...,
        [236., 234., 236., 255.],
        [196., 198., 199., 255.],
        [199., 201., 204., 255.]],

       ...,

       [[193., 193., 193., 255.],
        [180., 180., 180., 255.],
        [151., 152., 152., 255.],
        ...,
        [154., 154., 155., 255.],
        [180., 180., 180., 255.],
        [193., 193., 193., 255.]],

       [[197., 197., 197., 255.],
        [192., 192., 192., 255.],
        [179., 179., 179., 255.],
        ...,
        [179., 179., 179., 255.],
        [192., 192., 192., 255.],
        [197., 197., 197., 255.]],

       [[198., 198., 198., 255.],
        [196., 196., 196., 255.],
        [192., 192., 192., 255.],
        ...,
        [192., 192., 192., 255.],
        [196., 196., 196., 255.],
        [198., 198., 198., 255.]]], dtype=float32)
```
<hr class='division3'>
</details>

<br>

`STEP2`
```python
# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))

# per-channel centering of pixels
pixels -= means

# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
pixels
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    Means: [158.43480487 159.58662109 164.9829202  255.        ]<br>
    Mins: [  0.   0.   0. 255.], Maxs: [255. 255. 255. 255.]<br>
    Means: [-3.06365524e-07 -1.24562507e-06  4.88580506e-07  0.00000000e+00]<br>
    Mins: [-158.4348  -159.58662 -164.98293    0.     ], Maxs: [96.56519  95.413376 90.01708   0.      ]<br>
</p>
```
array([[[ 62.565197 ,  63.41338  ,  61.01708  ,   0.       ],
        [ 51.565197 ,  52.41338  ,  50.01708  ,   0.       ],
        [ 32.565197 ,  32.41338  ,  30.01708  ,   0.       ],
        ...,
        [ 32.565197 ,  32.41338  ,  30.01708  ,   0.       ],
        [ 51.565197 ,  52.41338  ,  50.01708  ,   0.       ],
        [ 62.565197 ,  63.41338  ,  61.01708  ,   0.       ]],

       [[ 54.565197 ,  55.41338  ,  52.01708  ,   0.       ],
        [ 31.565195 ,  32.41338  ,  29.01708  ,   0.       ],
        [ 47.565197 ,  47.41338  ,  43.01708  ,   0.       ],
        ...,
        [ 47.565197 ,  47.41338  ,  43.01708  ,   0.       ],
        [ 31.565195 ,  32.41338  ,  29.01708  ,   0.       ],
        [ 54.565197 ,  55.41338  ,  52.01708  ,   0.       ]],

       [[ 40.565197 ,  41.41338  ,  39.01708  ,   0.       ],
        [ 37.565197 ,  38.41338  ,  34.01708  ,   0.       ],
        [ 77.56519  ,  74.413376 ,  71.01708  ,   0.       ],
        ...,
        [ 77.56519  ,  74.413376 ,  71.01708  ,   0.       ],
        [ 37.565197 ,  38.41338  ,  34.01708  ,   0.       ],
        [ 40.565197 ,  41.41338  ,  39.01708  ,   0.       ]],

       ...,

       [[ 34.565197 ,  33.41338  ,  28.01708  ,   0.       ],
        [ 21.565195 ,  20.41338  ,  15.017079 ,   0.       ],
        [ -7.434805 ,  -7.5866213, -12.982921 ,   0.       ],
        ...,
        [ -4.434805 ,  -5.5866213,  -9.982921 ,   0.       ],
        [ 21.565195 ,  20.41338  ,  15.017079 ,   0.       ],
        [ 34.565197 ,  33.41338  ,  28.01708  ,   0.       ]],

       [[ 38.565197 ,  37.41338  ,  32.01708  ,   0.       ],
        [ 33.565197 ,  32.41338  ,  27.01708  ,   0.       ],
        [ 20.565195 ,  19.41338  ,  14.017079 ,   0.       ],
        ...,
        [ 20.565195 ,  19.41338  ,  14.017079 ,   0.       ],
        [ 33.565197 ,  32.41338  ,  27.01708  ,   0.       ],
        [ 38.565197 ,  37.41338  ,  32.01708  ,   0.       ]],

       [[ 39.565197 ,  38.41338  ,  33.01708  ,   0.       ],
        [ 37.565197 ,  36.41338  ,  31.01708  ,   0.       ],
        [ 33.565197 ,  32.41338  ,  27.01708  ,   0.       ],
        ...,
        [ 33.565197 ,  32.41338  ,  27.01708  ,   0.       ],
        [ 37.565197 ,  36.41338  ,  31.01708  ,   0.       ],
        [ 39.565197 ,  38.41338  ,  33.01708  ,   0.       ]]],
      dtype=float32)
```
<hr class='division3'>
</details>

<br><br><br>
`SUPPLEMENT1`
```python
import numpy as np
a = np.array([[1, 2], 
              [3, 4]])

print(a.shape)
print(np.mean(a))          # (1+2+3+4)/4 = 2.5
print(np.mean(a, axis=0))  # (1+3)/2 = 2, (2+4)/2 = 3
print(np.mean(a, axis=1))  # (1+2)/2 = 2.5, (3+4)/2 = 3.5
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (2, 2)<br>
    2.5 <br>
    [2. 3.] <br>
    [1.5 3.5] 
</p>
<hr class='division3'>
</details>

<br>
`SUPPLEMENT2`
```python
import numpy as np
a = np.array([[[1, 2], 
               [3, 4]],
              
               [[5,6],
                [7,8]]])

print(a.shape)

print(np.mean(a),'\n\n')                # (1+2+3+4+5+6+7+8)/8



print(np.mean(a, axis=0))               # (1+5)/2 = 3, (2+6)/2 = 4
                                        # (3+7)/2 = 5, (4+8)/2 = 6
print(np.mean(a, axis=(0,1)))           # ([1+5]/2 + [3+7]/2)/2 = 4
                                        # ([2+6]/2 + [4+8]/2)/2 = 5
print(np.mean(a, axis=(0,2)),'\n\n')    # ([1+5]/2 + [2+6]/2)/2 = 3.5
                                        # ([3+7]/2 + [4+8]/2)/2 = 5.5



print(np.mean(a, axis=1))               # (1+3)/2 = 2, (2+4)/2 = 3
                                        # (5+7)/2 = 6, (6+8)/2 = 7
print(np.mean(a, axis=(1,0)))           # ([1+3]/2 + [5+7]/2)/2 = 4
                                        # ([2+4]/2 + [6+8]/2)/2 = 5
print(np.mean(a, axis=(1,2)),'\n\n')    # ([1+3]/2 + [2+4]/2)/2 = 2.5
                                        # ([5+7]/2 + [6+8]/2)/2 = 6.5



print(np.mean(a, axis=2))               # (1+2)/2 = 1.5, (3+4)/2 = 3.5
                                        # (5+6)/2 = 5.5, (7+8)/2 = 7.5
print(np.mean(a, axis=(2,0)))           # ([1+2]/2 + [5+6]/2)/2 = 3.5
                                        # ([3+4]/2 + [7+8]/2)/2 = 5.5
print(np.mean(a, axis=(2,1)),'\n\n')    # ([1+2]/2 + [3+4]/2)/2 = 2.5
                                        # ([5+6]/2 + [7+8]/2)/2 = 6.5
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    (2, 2, 2)<br>
    4.5 <br><br>


    [[3. 4.]<br>
     [5. 6.]]<br>
    [4. 5.]<br>
    [3.5 5.5] <br><br>


    [[2. 3.]<br>
     [6. 7.]]<br>
    [4. 5.]<br>
    [2.5 6.5] <br><br>


    [[1.5 3.5]<br>
     [5.5 7.5]]<br>
    [3.5 5.5]<br>
    [2.5 6.5] <br><br>

</p>
<hr class='division3'>
</details>


<br><br><br>


<hr class="division2">

### ***Standardize Pixel Values***

***For consistency of the input data***, it may make more sense to standardize images per-channel using statistics calculated per minibatch or across the training dataset, if possible.
<br><br><br>


#### Global Standardization

```python
# example of global pixel standardization
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# global standardization of pixels
pixels = (pixels - mean) / std

# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    Mean: 184.501, Standard Deviation: 73.418<br>
    Mean: -0.000, Standard Deviation: 1.000
</p>
<hr class='division3'>
</details>

<br><br><br>

---

#### Positive Global Standardization

```python
# example of global pixel standardization shifted to positive domain
from numpy import asarray
from numpy import clip
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# global standardization of pixels
pixels = (pixels - mean) / std

# clip pixel values to [-1,1]
pixels = clip(pixels, -1.0, 1.0)

# shift from [-1,1] to [0,1] with 0.5 mean
pixels = (pixels + 1.0) / 2.0

# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
```
Mean: 184.501, Standard Deviation: 73.418
[[[ 0.4971365   0.52437776  0.5652396   0.9602377 ]
  [ 0.34730968  0.3745509   0.41541278  0.9602377 ]
  [ 0.08851784  0.10213846  0.14300032  0.9602377 ]
  ...
  [ 0.08851784  0.10213846  0.14300032  0.9602377 ]
  [ 0.34730968  0.3745509   0.41541278  0.9602377 ]
  [ 0.4971365   0.52437776  0.5652396   0.9602377 ]]

 [[ 0.38817152  0.41541278  0.442654    0.9602377 ]
  [ 0.07489721  0.10213846  0.1293797   0.9602377 ]
  [ 0.29282716  0.3064478   0.32006842  0.9602377 ]
  ...
  [ 0.29282716  0.3064478   0.32006842  0.9602377 ]
  [ 0.07489721  0.10213846  0.1293797   0.9602377 ]
  [ 0.38817152  0.41541278  0.442654    0.9602377 ]]

 [[ 0.19748281  0.22472405  0.26558593  0.9602377 ]
  [ 0.15662095  0.1838622   0.19748281  0.9602377 ]
  [ 0.7014459   0.6742046   0.7014459   0.9602377 ]
  ...
  [ 0.7014459   0.6742046   0.7014459   0.9602377 ]
  [ 0.15662095  0.1838622   0.19748281  0.9602377 ]
  [ 0.19748281  0.22472405  0.26558593  0.9602377 ]]

 ...

 [[ 0.11575908  0.11575908  0.11575908  0.9602377 ]
  [-0.06130901 -0.06130901 -0.06130901  0.9602377 ]
  [-0.45630705 -0.44268644 -0.44268644  0.9602377 ]
  ...
  [-0.4154452  -0.4154452  -0.40182456  0.9602377 ]
  [-0.06130901 -0.06130901 -0.06130901  0.9602377 ]
  [ 0.11575908  0.11575908  0.11575908  0.9602377 ]]

 [[ 0.17024156  0.17024156  0.17024156  0.9602377 ]
  [ 0.10213846  0.10213846  0.10213846  0.9602377 ]
  [-0.07492963 -0.07492963 -0.07492963  0.9602377 ]
  ...
  [-0.07492963 -0.07492963 -0.07492963  0.9602377 ]
  [ 0.10213846  0.10213846  0.10213846  0.9602377 ]
  [ 0.17024156  0.17024156  0.17024156  0.9602377 ]]

 [[ 0.1838622   0.1838622   0.1838622   0.9602377 ]
  [ 0.15662095  0.15662095  0.15662095  0.9602377 ]
  [ 0.10213846  0.10213846  0.10213846  0.9602377 ]
  ...
  [ 0.10213846  0.10213846  0.10213846  0.9602377 ]
  [ 0.15662095  0.15662095  0.15662095  0.9602377 ]
  [ 0.1838622   0.1838622   0.1838622   0.9602377 ]]]
Mean: 0.563, Standard Deviation: 0.396
Min: 0.000, Max: 0.980
```
<hr class='division3'>
</details>

<br><br><br>

`SUPPLEMENT`
```python
import numpy as np
clip(np.array([1,2,3,4,5]),2,4)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>array([2, 2, 3, 4, 4])</p>
<hr class='division3'>
</details>

<br><br><br>

---

#### Local Standardization

```python
# example of per-channel pixel standardization
from numpy import asarray
from PIL import Image

# load image
image = Image.open('boat.png')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))

# per-channel standardization of pixels
pixels = (pixels - means) / stds

# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<p>
    Means: [158.43480487 159.58662109 164.9829202  255.        ], Stds: [70.63586854 70.73750037 70.1171148   0.        ]<br>
    Means: [-3.98300453e-13 -1.93157989e-13  3.25967320e-13             nan], Stds: [ 1.  1.  1. nan]
</p>
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **keras**
### ***MNIST Handwritten Image Classiﬁcation Dataset***

```python
# example of loading the MNIST dataset
from keras.datasets import mnist

# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT 1</summary>
<hr class='division3'>
```python
# summarize dataset type and shape
print(type(train_images), train_images.dtype, train_images.shape)
print(type(train_labels), train_labels.dtype, train_labels.shape)
print(type(test_images), test_images.dtype, test_images.shape)
print(type(test_labels), test_labels.dtype, test_labels.shape)

# summarize pixel values
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
<p>
  <class 'numpy.ndarray'> uint8 (60000, 28, 28)<br>
  <class 'numpy.ndarray'> uint8 (60000,)<br>
  <class 'numpy.ndarray'> uint8 (10000, 28, 28)<br>
  <class 'numpy.ndarray'> uint8 (10000,)<br>
  Train 0 255 33.318421449829934 78.56748998339798<br>
  Test 0 255 33.791224489795916 79.17246322228644<br>
</p>
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT 2</summary>
<hr class='division3'>
```python
print(train_images[0].shape)
io.imshow(train_images[0])
```

<p>
  (28, 28)<br>
  <matplotlib.image.AxesImage at 0x23244de4fd0>
</p>
![다운로드 (3)](https://user-images.githubusercontent.com/52376448/63792062-bba44500-c937-11e9-9747-e048df95e1a6.png)
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


### ***ImageDataGenerator Class for Pixel Scaling***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

```python
# create data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split 


"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator()
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)


"""model design"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.fit_generator(train_iterator, validation_data=val_iterator, epochs=10, steps_per_epoch=10, validation_steps=10)


"""evaluation"""
# evaluate model loss on test dataset
result = model.evaluate_generator(test_iterator, steps=10)
for i in range(len(model.metrics_names)):  
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))
    
model.predict_generator(test_iterator)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">(train_images, train_labels), (test_images, test_labels) = mnist.load_data()</summary>
<hr class='division3'>
```python
# summarize dataset shape, pixel values for train
print('Train', train_images.shape, train_labels.shape)
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())

# summarize dataset shape, pixel values for test
print('Test', (test_images.shape, test_labels.shape))
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
```
Train (60000, 28, 28) (60000,)
Train 0 255 33.318421449829934 78.56748998339798
Test ((10000, 28, 28), (10000,))
Test 0 255 33.791224489795916 79.17246322228644
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)</summary>
<hr class='division3'>
```python
# summarize dataset shape, pixel values for train
print('Train', train_images.shape, train_labels.shape)
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())

# summarize dataset shape, pixel values for val
print('Val', valX.shape, valy.shape)
print('Val', valX.min(), valX.max(), valX.mean(), valX.std())

# summarize dataset shape, pixel values for test
print('Test', (test_images.shape, test_labels.shape))
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```
```
Train (48000, 28, 28) (48000,)
Train 0 255 33.29773514562075 78.54482970203107
Val (12000, 28, 28) (12000,)
Val 0 255 33.40116666666667 78.65801142483167
Test ((10000, 28, 28), (10000,))
Test 0 255 33.791224489795916 79.17246322228644
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">train_iterator, val_iterator, test_iterator</summary>
<hr class='division3'>
```python
train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=30.790, std=75.816
val batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=34.835, std=80.186
test batch shape=(32, 28, 28, 1), min=0.000, max=255.000, mean=36.032, std=81.371
```
<hr class='division3'>
</details>
<br><br><br>

<hr class="division2">


### ***How to Normalize Images With ImageDataGenerator***
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0)
```
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(rescale=1.0/255.0)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(rescale=1.0/255.0)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.130, std=0.307
val batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.126, std=0.302
test batch(32) shape=(32, 28, 28, 1), min=0.000, max=1.000, mean=0.123, std=0.302
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=0.000, max=1.000, mean=0.131, std=0.308
val batch(all) shape=(12000, 28, 28, 1), min=0.000, max=1.000, mean=0.131, std=0.308
test batch(all) shape=(10000, 28, 28, 1), min=0.000, max=1.000, mean=0.133, std=0.310
```
<hr class='division3'>
</details><br>


<br><br><br>

---

<hr class="division2">


### ***How to Center Images With ImageDataGenerator***
#### feature-wise centering
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=True)
```
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
[[[33.29781]]] None
[[[33.40119]]] None
[[[33.79124]]] None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-1.533, std=77.921
val batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-2.625, std=76.458
test batch(32) shape=(32, 28, 28, 1), min=-33.791, max=221.209, mean=-3.413, std=75.195
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.494, std=78.545
val batch(all) shape=(12000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.390, std=78.658
test batch(all) shape=(10000, 28, 28, 1), min=-33.791, max=221.209, mean=-0.000, std=79.172
```
<hr class='division3'>
</details>

<br><br><br>

---

#### sample-wise centering
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(samplewise_center=True)
```
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-54.675, max=236.723, mean=0.000, std=79.433
val batch(32) shape=(32, 28, 28, 1), min=-60.829, max=238.736, mean=0.000, std=79.335
test batch(32) shape=(32, 28, 28, 1), min=-62.398, max=242.120, mean=-0.000, std=77.976
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-90.477, max=248.513, mean=-0.000, std=77.764
val batch(all) shape=(12000, 28, 28, 1), min=-101.381, max=247.806, mean=-0.000, std=77.882
test batch(all) shape=(10000, 28, 28, 1), min=-83.435, max=247.832, mean=-0.000, std=78.383
```
<hr class='division3'>
</details>

<br><br><br>

<hr class="division2">


### ***How to Standardize Images With ImageDataGenerator***
#### feature-wise Standardization
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
```
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
[[[33.29781]]] [[[78.54484]]]
[[[33.40119]]] [[[78.65801]]]
[[[33.79124]]] [[[79.172455]]]
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=0.004, std=1.006
val batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=0.029, std=1.034
test batch(32) shape=(32, 28, 28, 1), min=-0.427, max=2.794, mean=-0.026, std=0.968
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.006, std=0.992
val batch(all) shape=(12000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.005, std=0.994
test batch(all) shape=(10000, 28, 28, 1), min=-0.427, max=2.794, mean=-0.000, std=1.000
```
<hr class='division3'>
</details>

<br><br><br>

---

#### sample-wise Standardization
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
```
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
datagen.fit(train_images)
datagen.fit(valX)
datagen.fit(test_images)

# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split 

"""data preprocessing"""
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, valX, train_labels, valy = train_test_split(train_images, train_labels, test_size=0.2,random_state=2018)

# reshape to rank 4
train_images = train_images.reshape(48000,28,28,1)
valX = valX.reshape(12000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)   

print('train shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_images.shape, train_images.min(), train_images.max(), train_images.mean(), train_images.std()))
print('val shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (valX.shape, valX.min(), valX.max(), valX.mean(), valX.std()))
print('test shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_images.shape, test_images.min(), test_images.max(), test_images.mean(), test_images.std()))
print('--------'*10)

# get batch iterator
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
datagen.fit(train_images)
print(datagen.mean, datagen.std)
datagen.fit(valX)
print(datagen.mean, datagen.std)
datagen.fit(test_images)
print(datagen.mean, datagen.std)
print('--------'*10)



# batch : 32
train_iterator = datagen.flow(train_images, train_labels, batch_size=32)
val_iterator = datagen.flow(valX, valy, batch_size=32)
test_iterator = datagen.flow(test_images, test_labels, batch_size=32)

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(32) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
print('--------'*10)



# batch : all
train_iterator = datagen.flow(train_images, train_labels, batch_size=len(train_images))
val_iterator = datagen.flow(valX, valy, batch_size=len(valX))
test_iterator = datagen.flow(test_images, test_labels, batch_size=len(test_images))

train_batchX, train_batchy = train_iterator.next()
val_batchX, val_batchy = val_iterator.next()
test_batchX, test_batchy = test_iterator.next()

print('train batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (train_batchX.shape, train_batchX.min(), train_batchX.max(), train_batchX.mean(), train_batchX.std()))
print('val batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (val_batchX.shape, val_batchX.min(), val_batchX.max(), val_batchX.mean(), val_batchX.std()))
print('test batch(all) shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (test_batchX.shape, test_batchX.min(), test_batchX.max(), test_batchX.mean(), test_batchX.std()))
```
```
train shape=(48000, 28, 28, 1), min=0.000, max=255.000, mean=33.298, std=78.545
val shape=(12000, 28, 28, 1), min=0.000, max=255.000, mean=33.401, std=78.658
test shape=(10000, 28, 28, 1), min=0.000, max=255.000, mean=33.791, std=79.172
--------------------------------------------------------------------------------
None None
None None
None None
--------------------------------------------------------------------------------
train batch(32) shape=(32, 28, 28, 1), min=-0.600, max=4.298, mean=0.000, std=1.000
val batch(32) shape=(32, 28, 28, 1), min=-0.554, max=4.273, mean=-0.000, std=1.000
test batch(32) shape=(32, 28, 28, 1), min=-0.585, max=4.394, mean=0.000, std=1.000
--------------------------------------------------------------------------------
train batch(all) shape=(48000, 28, 28, 1), min=-0.777, max=7.770, mean=-0.000, std=1.000
val batch(all) shape=(12000, 28, 28, 1), min=-0.851, max=7.249, mean=0.000, std=1.000
test batch(all) shape=(10000, 28, 28, 1), min=-0.732, max=7.578, mean=0.000, std=1.000
```
<hr class='division3'>
</details>

<br><br><br>
<hr class="division2">

## **pytorch**

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

[1]:{{ site.url }}/download/AI01/boat.png
