---
layout : post
title : AI01, Practical data handling
categories: [AI01]
comments : true
tags : [AI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AI01/2019-08-13-AI01-Practical-data-handling.md" target="_blank">page management</a><br>
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
## **File I/O**
### ***Image***
```python
import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('input_image.jpg')   # load image
plt.imshow(img)
plt.figsave('output_image.jpg')         # save image
```
<br><br><br>

---

### ***Table***
```python
import pandas as pd

df = pd.read_csv('input_table.csv')    # load table
df.to_excel('output_table.xlsx')       # save table
```
<br><br><br>

---

### ***Text***
```python
with open('input_text.txt','r') as f:  # load text
    text = f.read()
with open('output.txt','w') as f:      # save text
    f.write(text)
```
<br><br><br>
<hr class="division2">

## **From WEB**
<ins>Developer tools</ins><br>
<p>F12 : Elements(Inspector, Ctrl + Shift + c), Networks</p>
<p>/robots.txt</p>

![image](https://user-images.githubusercontent.com/52376448/71744017-5ba34980-2ea9-11ea-90fc-40deb5d05e50.png)

<br><br><br>
### ***Scraping***
#### from urllib
<span class="frame3">installation</span><br>
```dos

```
<br><br><br>
<span class="frame3">urlretrieve(from urllib.request) : download file</span><br>

```python
import urllib.request as req

# from : file url
img_url = 'https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png'
html_url = 'https://www.google.com/'

# to : path
img_save_path = r'S:\workspace\2020-01-19\winscp.jpg'
html_save_path = r' S:\workspace\2020-01-19\index.html'

# download file
img_file, img_header = req.urlretrieve(img_url,img_save_path); print(img_header)
html_file, html_header = req.urlretrieve(html_url, html_save_path); print(html_header)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">handling error</summary>
<hr class='division3'>
<hr class='division3'>
</details>

<br><br><br>

<span class="frame3">urlopen(from urllib.request) : save file as an object on python</span><br>
```python
import urllib.request as req

# from : file url
# to : path
file_url = "https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png"
save_path = r"S:\workspace\2020-01-22\winscp.jpg"


# save file as an object on python
response = req.urlopen(file_url)
header_info = response.info()
http_status_code = response.getcode()

# download file
contents = response.read()
with open(save_path, 'wb') as c:
    c.write(contents)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">handling error</summary>
<hr class='division3'>
```python
import urllib.request as req
from urllib.error import URLError, HTTPError

# from : file url
target_url = ["https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png",
              "https://google.com"]

# to : path
path_list = [r"S:\workspace\2020-01-22\winscp.jpg",
             r"S:\workspace\2020-01-22\index.html"]

# download file
for i, url in enumerate(target_url):
    try:
        response = req.urlopen(url)
        contents = response.read()
        print('---------------------------------------------------')
        print('Header Info-{} : {}'.format(i, response.info()))
        print('HTTP Status Code : {}'.format(response.getcode()))
        print('---------------------------------------------------')
        
        with open(path_list[i], 'wb') as c:
            c.write(contents)

    except HTTPError as e:
        print("Download failed.")
        print('HTTPError Code : ', e.code)

    except URLError as e:
        print("Download failed.")
        print('URL Error Reason : ', e.reason)

    else:
        print()
        print("Download Succeed.")
```

<hr class='division3'>
</details>
<br><br><br>
#### from requests
<span class="frame3">installation</span><br>
```dos
pip install requests
pip install lxml
pip install cssselect
```
<br><br><br>

<span class="frame3">get(from requests)</span><br>
```python
import requests
import lxml.html

response = requests.get('https://www.naver.com/')
root = lxml.html.fromstring(response.content)

urls = []
for i in root.cssselect('.api_list .api_item a.api_link'):
    url = i.get('href')
    urls.append(url)

for i in urls:
    print(i)
```
![image](https://user-images.githubusercontent.com/52376448/73355664-f4c45500-42db-11ea-887e-29fde730c5dc.png)

<br><br><br>
#### from Beautiful Soup
<span class="frame3">installation</span><br>
```dos

```
<br><br><br>

<br><br><br>

#### from Selenium
<span class="frame3">installation</span><br>
```dos

```
<br><br><br>

<br><br><br>


---

### ***Example for scraping***
#### EX1, encar
<a href="http://www.encar.com/index.do" target="_blank">encar</a><br>

<br><br><br>
#### EX2, mois
<a href="https://www.mois.go.kr/frt/sub/a08/rss/screen.do" target="_blank">mois</a><br>

<br><br><br>
#### EX3, daum finance
<a href="https://finance.daum.net/" target="_blank">daum finance</a><br>

<br><br><br>

---

### ***Scrapy***
<span class="frame3">installation</span><br>
```dos
pip install scrapy
pip install pypiwin32
```
<br><br><br>

<br><br><br>
<hr class="division2">

## **From DB**
<br><br><br>
<hr class="division2">

## **h5**
```python
import h5py
import numpy as np

f = h5py.File('input_big_data.h5','r')    # load big_data
for i in f.keys():                        
    info = f.get(i)                       # show information about big_data
    print(info)                           
    
    data = np.array(info)                 # show big_data
    print(data)
```
<br><br><br>

<hr class="division2">


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
<summary class='jb-small' style="color:blue">handling error</summary>
<hr class='division3'>
<hr class='division3'>
</details>
