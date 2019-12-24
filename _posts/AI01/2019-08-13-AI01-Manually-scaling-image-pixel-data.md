---
layout : post
title : AI01, Manually scaling image pixel data
categories: [AI01]
comments : true
tags : [AI01]
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
