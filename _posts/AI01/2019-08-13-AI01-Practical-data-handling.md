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
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/73355888-73b98d80-42dc-11ea-854e-6298cbb3278d.png)
![image](https://user-images.githubusercontent.com/52376448/73355985-a499c280-42dc-11ea-8e96-77cda40d9bf1.png)
![image](https://user-images.githubusercontent.com/52376448/73356313-694bc380-42dd-11ea-939a-f4843ac5e67c.png)
![image](https://user-images.githubusercontent.com/52376448/73356056-ce52e980-42dc-11ea-8650-727d767c4553.png)
<hr class='division3'>
</details>

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
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/73355888-73b98d80-42dc-11ea-854e-6298cbb3278d.png)
![image](https://user-images.githubusercontent.com/52376448/73355985-a499c280-42dc-11ea-8e96-77cda40d9bf1.png)
<hr class='division3'>
</details>

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
<a href="https://www.w3schools.com/cssref/css_selectors.asp" target="_blank">css_selectors</a><br>
![image](https://user-images.githubusercontent.com/52376448/73358148-5804b600-42e1-11ea-9425-7153061e3a32.png)

<span class="frame3_1">with cssselect</span><br>

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
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/73355748-2b01d480-42dc-11ea-9e50-c97222a8a7cd.png)
![image](https://user-images.githubusercontent.com/52376448/73355664-f4c45500-42db-11ea-887e-29fde730c5dc.png)
<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
response = requests.get('https://www.naver.com/')
print(response)
print(response.content)
```
```
<Response [200]>
```
```
b'<!doctype html>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="Referrer" content="origin">\n<meta http-equiv="Content-Script-Type" content="text/javascript">\n<meta http-equiv="Content-Style-Type" content="text/css">\n<meta http-equiv="X-UA-Compatible" content="IE=edge">\n<meta name="viewport" content="width=1100">\n<meta name="apple-mobile-web-app-title" content="NAVER" />\n<meta name="robots" content="index,nofollow"/>\n<meta name="description" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\xa9\x94\xec\x9d\xb8\xec\x97\x90\xec\x84\x9c \xeb\x8b\xa4\xec\x96\x91\xed\x95\x9c \xec\xa0\x95\xeb\xb3\xb4\xec\x99\x80 \xec\x9c\xa0\xec\x9a\xa9\xed\x95\x9c \xec\xbb\xa8\xed\x85\x90\xec\xb8\xa0\xeb\xa5\xbc \xeb\xa7\x8c\xeb\x82\x98 \xeb\xb3\xb4\xec\x84\xb8\xec\x9a\x94"/>\n<meta property="og:title" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84">\n<meta property="og:url" content="https://www.naver.com/">\n<meta property="og:image" content="https://s.pstatic.net/static/www/
...
...
...
\n\t\t} else if (window.attachEvent) { \n\t\t\twindow.attachEvent("onload", loadJS);\n\t\t} else {\n\t\t\twindow.onload = loadJS;\n\t\t}\n\t\t\n\t</script>\n</body>\n</html>\n'
```
<hr class='division3'>
</details>
<br>
<span class="frame3_1">with session, xpath</span><br>
```python
import requests
import lxml.html

session = requests.Session()
response = session.get('https://www.naver.com/')
root = lxml.html.fromstring(response.content)
root.make_links_absolute(response.url)

urls = {}
for i in root.xpath('//ul[@class="api_list"]/li[@class="api_item"]/a[@class="api_link"]'):
    url = i.get('href')
    name = i.xpath('./img')[0].get('alt'); urls[name] = url

for name, url in urls.items():
    print(name, url)
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/73355748-2b01d480-42dc-11ea-9e50-c97222a8a7cd.png)
![image](https://user-images.githubusercontent.com/52376448/73367823-678cfa80-42f3-11ea-99fd-bcd6b210c52a.png)
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
response = requests.get('https://www.naver.com/')
print(response)
print(response.content)
print(response.url)
```
```
<Response [200]>
```
```
b'<!doctype html>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="Referrer" content="origin">\n<meta http-equiv="Content-Script-Type" content="text/javascript">\n<meta http-equiv="Content-Style-Type" content="text/css">\n<meta http-equiv="X-UA-Compatible" content="IE=edge">\n<meta name="viewport" content="width=1100">\n<meta name="apple-mobile-web-app-title" content="NAVER" />\n<meta name="robots" content="index,nofollow"/>\n<meta name="description" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84 \xeb\xa9\x94\xec\x9d\xb8\xec\x97\x90\xec\x84\x9c \xeb\x8b\xa4\xec\x96\x91\xed\x95\x9c \xec\xa0\x95\xeb\xb3\xb4\xec\x99\x80 \xec\x9c\xa0\xec\x9a\xa9\xed\x95\x9c \xec\xbb\xa8\xed\x85\x90\xec\xb8\xa0\xeb\xa5\xbc \xeb\xa7\x8c\xeb\x82\x98 \xeb\xb3\xb4\xec\x84\xb8\xec\x9a\x94"/>\n<meta property="og:title" content="\xeb\x84\xa4\xec\x9d\xb4\xeb\xb2\x84">\n<meta property="og:url" content="https://www.naver.com/">\n<meta property="og:image" content="https://s.pstatic.net/static/www/
...
...
...
\n\t\t} else if (window.attachEvent) { \n\t\t\twindow.attachEvent("onload", loadJS);\n\t\t} else {\n\t\t\twindow.onload = loadJS;\n\t\t}\n\t\t\n\t</script>\n</body>\n</html>\n'
```
```
https://www.naver.com/
```
<hr class='division3'>
</details>


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
```python
from urllib.request import urlopen
from urllib.parse import urlparse

# with urlopen
response_1 = urlopen("http://www.encar.com/")
print('type : {}'.format(type(response_1)))
print("geturl : {}".format(response_1.geturl()))
print("status : {}".format(response_1.status))
print("headers : {}".format(response_1.getheaders()))
print("getcode : {}".format(response_1.getcode()))
print("read : {}".format(response_1.read(1).decode('utf-8')))

# with urlparse
response_2 = urlparse('http://www.encar.co.kr?test=test')
print('total parse : {}'.format(response_2))
print('partial parse : {}'.format(response_2.query))
```
![image](https://user-images.githubusercontent.com/52376448/73404256-86f84780-4334-11ea-9812-ad6217c8b2b9.png)


<br><br><br>
#### EX2, ipify
<a href="https://www.ipify.org/" target="_blank">ipify</a><br>
```python
import urllib.request
from urllib.parse import urlparse

# request
API = "https://api.ipify.org"            # some request url
values = {'format': 'json'}              # It is also possible to use text, jsonp instead of json 
params = urllib.parse.urlencode(values)  # get parameter by encoding
url = API + "?" + params                 # request url

# response
data = urllib.request.urlopen(url).read() # read response data
text = data.decode("utf-8")               # decode read data
print('response : {}'.format(text))
```

<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
<hr class='division3'>
</details>

![image](https://user-images.githubusercontent.com/52376448/73407480-97acbb80-433c-11ea-8d09-2530154613e7.png)

<br><br><br>
#### EX3, mois
<a href="https://www.mois.go.kr/frt/sub/a08/rss/screen.do" target="_blank">mois</a><br>
```python
import urllib.request
import urllib.parse

API = "http://www.mois.go.kr/gpms/view/jsp/rss/rss.jsp"

params = []
for num in [1001, 1012, 1013, 1014]:
    params.append(dict(ctxCd=num))

for i in params:
    param = urllib.parse.urlencode(i)
    url = API + "?" + param
    res_data = urllib.request.urlopen(url).read()
    contents = res_data.decode("utf-8")
    print(contents)
```
![image](https://user-images.githubusercontent.com/52376448/73420146-dfdfd400-4364-11ea-8127-342872cb0387.png)

<br><br><br>
#### EX4, daum finance
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
