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
`jpg`
```bash
$ pip install matplotlib
```
```python
import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('input_image.jpg')   # load image
plt.imshow(img); plt.show()
plt.figsave('output_image.jpg')         # save image
```
`png`
```bash
$ pip install matplotlib
$ pip install pillow
```
```python
import matplotlib.pyplot as plt
from matplotlib import image

img = image.imread('input_image.png', 0)   # load image
plt.imshow(img); plt.show()
plt.figsave('output_image.png')            # save image
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
### ***Scraping : basic***
#### urllib
<a href="https://docs.python.org/3/library/urllib.html" target="_blank">API</a><br>
<span class="frame3">installation</span><br>
```bash
$ pip install urllib3
$ pip install fake-useragent
```
<br><br><br>
<span class="frame3">urlretrieve(from urllib.request)</span><br>
<span class="frame3_1">download file</span>
```python
from urllib.request import urlretrieve

# from : file url
img_url = 'https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png'
html_url = 'https://www.google.com/'

# to : path
img_save_path = r'S:\workspace\2020-01-19\winscp.jpg'
html_save_path = r'S:\workspace\2020-01-19\index.html'

# download file
img_file, img_header = urlretrieve(img_url,img_save_path); print(img_header)
html_file, html_header = urlretrieve(html_url, html_save_path); print(html_header)
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


<span class="frame3">urlopen(from urllib.request)</span><br>
<span class="frame3_1">response</span>
```python
from urllib.request import urlopen

file_url = "https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png"
response = urlopen(file_url)

print('header_info : {}'.format(response.info()))
print('http_status_code : {}'.format(response.getcode()))
print('geturl : {}'.format(response.geturl()))
print('status : {}'.format(response.status))
print('headers : {}'.format(response.getheaders()))
print('contents : {}'.format(response.read(10)))                          # response binary data, response.content in module 'requests'
print('contents decode: {}'.format(response.read(10).decode('utf-8')))    # response data, response.text in module 'requests'
```
![image](https://user-images.githubusercontent.com/52376448/73856204-70904580-4878-11ea-92d2-647fd336b7c4.png)
<br><br><br>

<span class="frame3_1">save file as an object on python</span>
```python
from urllib.request import urlopen

# from : file url
# to : path
file_url = "https://user-images.githubusercontent.com/52376448/69004181-481c3d80-0952-11ea-98b4-823969ceb0c3.png"
save_path = r"S:\workspace\2020-01-22\winscp.jpg"

# save file as an object on python
response = urlopen(file_url)
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
#### requests
<a href="https://requests.readthedocs.io/en/master/api/" target="_blank">API</a><br>

<span class="frame3">installation</span><br>
```bash
$ pip install requests
$ pip install lxml
$ pip install cssselect
```
<br><br><br>

<span class="frame3">Request methods : GET</span><br>
```python
import requests

response = requests.get("https://www.naver.com")

print(response.text)          # response data, response.read().decode('utf-8') in module 'urlopen'            
print(response.content)       # response binary data, response.read() in module 'urlopen'
print(response.headers)       # header
print(response.status_code)   # status code
print(response.url)           # url
print(response.ok)            # ok
print(response.encoding)      # encoding
```
![image](https://user-images.githubusercontent.com/52376448/73748172-3d788400-479c-11ea-97eb-ffed90c859d0.png)
![image](https://user-images.githubusercontent.com/52376448/73752362-305f9300-47a4-11ea-8712-e5f5bc0da9c8.png)
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT, response.text</summary>
<hr class='division3'>
It can be used response.iter_lines instead of method 'response.text'.
```python
import requests

response = requests.get("https://www.naver.com")

#if response.encoding is None: response.encoding = 'UTF-8'
for line in response.iter_lines(decode_unicode=True):
    print(line)    
```
![image](https://user-images.githubusercontent.com/52376448/73893743-d7841d80-48bd-11ea-931f-b5bc667e4248.png)

<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">with session</summary>
<hr class='division3'>
```python
import requests

session = requests.Session()
response = session.get("https://www.naver.com")

print(response.text)
print(response.content)
print(response.status_code)
print(response.url)
print(response.ok)
print(response.encoding)

session.close()
```

or

```python
import requests

with requests.Session() as session:
    response = session.get("https://www.naver.com")
    
    print(response.text)
    print(response.content)
    print(response.status_code)
    print(response.url)
    print(response.ok)
    print(response.encoding)
```
<hr class='division3'>
</details>


<details markdown="1">
<summary class='jb-small' style="color:blue">with cookies, headers</summary>
<hr class='division3'>
```python
import requests

response1 = requests.get("https://httpbin.org/cookies", cookies={'name':'kim'})
response2 = requests.get("https://httpbin.org", headers={'user-agent':'nice-man_1.0.0_win10_ram16_home_chrome'})

print(response1, response1.text)
print(response2, response2.text)
```
![image](https://user-images.githubusercontent.com/52376448/73901842-47eb6880-48d7-11ea-9722-b6a0ea858e72.png)

<details markdown="1">
<summary class='jb-small' style="color:red">another way carring cookies</summary>
<hr class='division3_1'>
```python
import requests

response = requests.get('https://httpbin.org/cookies')
print(response.text)

jar = requests.cookies.RequestsCookieJar()
jar.set('name', 'niceman', domain='httpbin.org', path='/cookies')
response = requests.get('http://httpbin.org/cookies', cookies=jar)
print(response.text)
```
```
{
  "cookies": {}
}

{
  "cookies": {
    "name": "niceman"
  }
}
```

<hr class='division3_1'>
</details>

<hr class='division3'>
</details>

<details markdown="1">
<summary class='jb-small' style="color:blue">with timeout</summary>
<hr class='division3'>
```python
import requests

response = requests.get('https://github.com', timeout=10)
print(response.text)
```
![image](https://user-images.githubusercontent.com/52376448/73903536-b979e580-48dc-11ea-87e6-e8fd4da0ef16.png)

<hr class='division3'>
</details>


<br><br><br>





<span class="frame3_1">with json</span><br>
```python
import requests

response = requests.get('https://jsonplaceholder.typicode.com/posts/1')

print('.headers : \n',response.headers)
print('.text : \n',response.text)
print('.json() : \n', response.json())
print('.json().keys() : \n', response.json().keys())
print('.json().values() : \n',response.json().values())
```
![image](https://user-images.githubusercontent.com/52376448/73895656-a575ba00-48c3-11ea-962f-8d0c83b30096.png)
<br><br><br>

```python
import requests
import json

response = requests.get('http://httpbin.org/stream/100', stream=True)

#if response.encoding is None: response.encoding = 'UTF-8'
for line in response.iter_lines(decode_unicode=True):
    b = json.loads(line); print(b)    # type(line) = str, type(b) = dict

    for k, v in b.items():
        print("Key: {}, Values: {}".format(k, v))
```
![image](https://user-images.githubusercontent.com/52376448/73894220-1ff00b00-48bf-11ea-91c6-37933bde6741.png)

<br><br><br>

<span class="frame3_1">with lxml</span><br>
<span class="frame3_2">with cssselect</span><br>

<a href="https://www.w3schools.com/cssref/css_selectors.asp" target="_blank">css_selectors</a><br>
![image](https://user-images.githubusercontent.com/52376448/73358148-5804b600-42e1-11ea-9425-7153061e3a32.png)


```python
import requests
import lxml.html

response = requests.get('https://www.naver.com/')
root = lxml.html.fromstring(response.content)

for i in root.cssselect('.api_list .api_item a.api_link'):
    url = i.get('href')
    name = i.cssselect('.api_logo')[0].get('alt');
    
    print(name, url)
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
<span class="frame3_2">with xpath</span><br>
```python
import requests
import lxml.html

response = requests.get('https://www.naver.com/')
root = lxml.html.fromstring(response.content)
root.make_links_absolute(response.url)

for i in root.xpath('//ul[@class="api_list"]/li[@class="api_item"]/a[@class="api_link"]'):
    url = i.get('href')
    name = i.xpath('./img')[0].get('alt')

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

<span class="frame3">Another request methods : POST, DELETE, PUT:UPDATE, REPLACE (FETCH : UPDATE, MODIFY)</span><br>
![image](https://user-images.githubusercontent.com/52376448/73895901-86c3f300-48c4-11ea-8708-52e2e1e41554.png)

```python
import requests

response = requests.post('http://httpbin.org/post', data={'kim':'stellar'})
print(response.text)
print(response.headers)
```
![image](https://user-images.githubusercontent.com/52376448/73903791-4f157500-48dd-11ea-970d-4d06be581235.png)
```python
import requests

payload1 = {'name': 'kim', 'pay': 'true'}
payload2 = (('name', 'park'), ('pay', 'false'))

response1 = requests.post('http://httpbin.org/post', data=payload1)
response2 = requests.post('http://httpbin.org/post', data=payload2)

print(response1.text)
print(response2.text)
```
![image](https://user-images.githubusercontent.com/52376448/73903936-c4814580-48dd-11ea-9e0e-ca14de004283.png)
<br><br><br>

```python
import requests

response = requests.put('http://httpbin.org/put', data={'data': '{"name": "Kim", "grade": "A"}'})
print(response.text)
```
![image](https://user-images.githubusercontent.com/52376448/73904080-42455100-48de-11ea-9a05-df1d5995cc7f.png)

<br><br><br>
```python
import requests

response = requests.delete('http://httpbin.org/delete')
print(response.text)
```
![image](https://user-images.githubusercontent.com/52376448/73904159-89cbdd00-48de-11ea-8ee5-5b5b0a0012c9.png)
```python
import requests

response = requests.delete('https://jsonplaceholder.typicode.com/posts/1')
print(response.text)
```
![image](https://user-images.githubusercontent.com/52376448/73904233-b54ec780-48de-11ea-9bd0-b0602e5a9f33.png)

<br><br><br>




#### BeautifulSoup
<a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" target="_blank">API</a><br>

<span class="frame3">installation</span><br>
```bash
$ pip install beautifulsoup4
```
<br><br><br>
<span class="frame3">Basic</span><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')
print(soup.prettify())
```
```html
<html>
 <head>
  <title>
   The Dormouse's story
  </title>
 </head>
 <body>
  <h1>
   this is h1 area
  </h1>
  <h2>
   this is h2 area
  </h2>
  <p class="title">
   <b>
    The Dormouse's story
   </b>
  </p>
  <p class="story">
   Once upon a time there were three little sisters
   <a class="sister" href="http://example.com/elsie" id="link1">
    Elsie
   </a>
   <a class="sister" href="http://example.com/lacie" id="link2">
    Lacie
   </a>
   <a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">
    Tillie
   </a>
  </p>
  <p class="story">
   story...
  </p>
 </body>
</html>
```

<br><br><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

h1 = soup.html.body.h1; print(h1, h1.string)    # h1 tag
p = soup.html.body.p; print(p, p.string)         # p tag
```
```html
<h1>this is h1 area</h1>, this is h1 area
<p class="title"><b>The Dormouse's story</b></p>, The Dormouse's story
```
<br><br><br>

```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

p = soup.html.body.p; print('p', p)
p2 = p.next_sibling.next_sibling; print('p2', p2)
p3 = p.next_sibling.next_sibling.next_sibling.next_sibling; print('p3', p3)
p4 = p.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling; print('p4', p4)
```
```html
p <p class="title"><b>The Dormouse's story</b></p>
p2 <p class="story">Once upon a time there were three little sisters
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
</p>
p3 <p class="story">story...</p>
p4 None
```
<br><br><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

p = soup.html.body.p
p2 = p.next_sibling.next_sibling

print(list(p2.next_elements))
for i in p2.next_elements:
    print(i)
```
```html
['Once upon a time there were three little sisters\n', <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, 'Elsie', '\n', <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, 'Lacie', '\n', <a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>, 'Tillie', '\n', '\n', <p class="story">story...</p>, 'story...', '\n', '\n', '\n']
Once upon a time there were three little sisters

<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
Elsie


<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
Lacie


<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
Tillie




<p class="story">story...</p>
story...







```
<br><br><br>

<span class="frame3">FIND</span><br>
<span class="frame3_1">find_all</span><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

tag_a = soup.find_all("a", class_='sister')
print(tag_a)

for i in tag_a:
    print(i.text, i.string)
```
```html
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>]
Elsie Elsie
Lacie Lacie
Tillie Tillie
```
<br><br><br>

```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

tag_a = soup.find_all("a", string=["Elsie","Tillie"], id="link1")
print(tag_a)

for i in tag_a:
    print(i.text, i.string)
```
```html
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]
Elsie Elsie
```
<br><br><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

tag_a = soup.find_all("a", limit=2)
print(tag_a)

for i in tag_a:
    print(i.text, i.string)
```
```html
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
Elsie Elsie
Lacie Lacie
```
<br><br><br>

<span class="frame3_1">find</span><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

tag_a = soup.find("a")   # the first tag that was found
print(tag_a)
print(tag_a.text, tag_a.string)
```
```html
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
Elsie Elsie
```
<br><br><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

tag_a = soup.find("a", {"class": "sister", "data-io": "link3"})    # multiple condition
print(tag_a)
print(tag_a.text, tag_a.string)
```
```html
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
Tillie Tillie
```
<br><br><br>



<span class="frame3">SELECT</span><br>
<span class="frame3_1">select_one</span><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

select_b = soup.select_one("p.title > b")
select_idlink1 = soup.select_one("a#link1")
select_valuelink3 = soup.select_one("a[data-io='link3']")

print(select_b, select_b.string)
print(select_idlink1, select_idlink1.string)
print(select_valuelink3, select_valuelink3.string)
```
```html
<b>The Dormouse's story</b> The Dormouse's story
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a> Elsie
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a> Tillie
```
<br><br><br>

<span class="frame3_1">select</span><br>
```python
from bs4 import BeautifulSoup

HTML = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<h1>this is h1 area</h1>
<h2>this is h2 area</h2>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a data-io="link3" href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
</body>
</html>
"""

soup = BeautifulSoup(HTML, 'html.parser')

select_a = soup.select("p.story > a")
select_a2 = soup.select("p.story > a:nth-of-type(2)")
select_classstory = soup.select("p.story")

print(select_a)
print(select_a[0])
print(select_a[1])
print(select_a[2])
print(select_a[0]['href'])
print(select_a[1]['href'])
print(select_a[2]['href'])
print()

print(select_a2)
print(select_a2[0])
print(select_a2[0]['id'])
print()

print(select_classstory)
print(select_classstory[0])
print(select_classstory[1])
print(select_classstory[0]['class'])
```
```html
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>]
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
http://example.com/elsie
http://example.com/lacie
http://example.com/tillie

[<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
link2

[<p class="story">Once upon a time there were three little sisters
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
</p>, <p class="story">story...</p>]
<p class="story">Once upon a time there were three little sisters
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
<a class="sister" data-io="link3" href="http://example.com/tillie" id="link3">Tillie</a>
</p>
<p class="story">story...</p>
['story']
```
<br><br><br>





#### Selenium
<a href="" target="_blank">API</a><br>

<span class="frame3">Installation</span><br>
```bash
$ pip install selenium
```

<br><br><br>

<span class="frame3">Access web(with web driver)</span><br>
web drivers([chrome][1], [firefox][2], [phantomjs][3]) official download links

- <a href="https://sites.google.com/a/chromium.org/chromedriver/downloads" target="_blank">chrome driver</a><br>
- <a href="https://github.com/mozilla/geckodriver/releases">firefox driver</a>
- <a href="https://phantomjs.org/download.html">phantomjs driver</a>

![image](https://user-images.githubusercontent.com/52376448/74072863-9d488680-4a4b-11ea-816a-42d0dea0a549.png)
`without option`
```python
from selenium import webdriver

browser = webdriver.Chrome('webdriver/chromedriver.exe')  # Driver path is important
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)            # maximize_window(), minimize_window()
browser.get('https://www.naver.com')
browser.quit()
```
`with option(headless)`
```python
from selenium import webdriver

chrome_options = Options()
chrome_options.add_argument("--headless")

browser = webdriver.Chrome('webdriver/chromedriver.exe', options=chrome_options)  # Driver path is important
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)            # maximize_window(), minimize_window()
browser.get('https://www.naver.com')
browser.quit()
```
result without option

![image](https://user-images.githubusercontent.com/52376448/74072957-f7e1e280-4a4b-11ea-906e-23e3f5488764.png)

<details markdown="1">
<summary class='jb-small' style="color:blue">web driver method</summary>
<hr class='division3'>
```python
print(browser.page_source)
print(browser.session_id)
print(browser.title)
print(browser.current_url)
print(browser.get_cookies())
```
```
<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml" lang="ko"><head>
<meta charset="utf-8" />
<meta name="Referrer" content="origin" />
<meta http-equiv="Content-Script-Type" content="text/javascript" />
<meta http-equiv="Content-Style-Type" content="text/css" />
<meta http-equiv="X-UA-Compatible" content="IE=edge" />
<meta name="viewport" content="width=1100" />
<meta name="apple-mobile-web-app-title" content="NAVER" />
<meta name="robots" content="index,nofollow" />
<meta name="description" content="네이버 메인에서 다양한 정보와 유용한 컨텐츠를 만나 보세요" />
...
...
...

                if (window.addEventListener) {
                        window.addEventListener("load", function() { loadJS(); }, true);
                } else if (window.attachEvent) {
                        window.attachEvent("onload", loadJS);
                } else {
                        window.onload = loadJS;
                }

        </script>


</body></html>
aa015a79b7acddf96892e138b1e75e31
NAVER
https://www.naver.com/
[{'domain': '.naver.com', 'expiry': 1612652898, 'httpOnly': False, 'name': 'NRTK', 'path': '/', 'secure': False, 'value': 'ag#all_gr#1_ma#-2_si#0_en#0_sp#0'}, {'domain': '.naver.com', 'expiry': 2524640401.665735, 'httpOnly': False, 'name': 'NNB', 'path': '/', 'secure': True, 'value': 'EFLEUDPB5U6V4'}, {'domain': 'www.naver.com', 'expiry': 1581203297.710821, 'httpOnly': True, 'name': 'PM_CK_loc', 'path': '/', 'secure': False, 'value': 'd2d101bc3885853d3f553a325db5c09b55091e808f79a0dfdb0fb274ee3cfd30'}]
```

<hr class='division3'>
</details>

<br><br><br>
<span class="frame3">Search keyword & screen shot</span><br>
```python
from selenium import webdriver

browser = webdriver.Chrome('webdriver/chromedriver.exe')   # Driver path is important
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)            # maximize_window(), minimize_window()
browser.get('https://www.daum.net')

# search keyword
element = browser.find_element_by_css_selector('div.inner_search > input.tf_keyword')
element.send_keys('lion')          # search word(input keyword)
element.submit()                   # form submit

# screen shot
browser.save_screenshot("website_ch1.png")             # saving way 1
browser.get_screenshot_as_file("website_ch2.png")      # saving way 2
browser.quit()
```
![image](https://user-images.githubusercontent.com/52376448/74073622-5e680000-4a4e-11ea-8831-14fa25249fab.png)

<br><br><br>
<span class="frame3">Click(with Explicitly wait)</span><br>
```python
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")

browser = webdriver.Chrome('./webdriver/chromedriver.exe', options=chrome_options)
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)  # maximize_window(), minimize_window()
browser.get('http://prod.danawa.com/list/?cate=112758&15main_11_02')

# Explicitly wait
WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'))).click()
WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.XPATH, '//*[@id="selectMaker_simple_priceCompare_A"]/li[14]/label'))).click()
time.sleep(3)

# bs4 initializer
soup = BeautifulSoup(browser.page_source, "html.parser")
pro_list = soup.select('div.main_prodlist.main_prodlist_list > ul > li')
for v in pro_list:
    if not v.find('div', class_='ad_header'):
        # product name, image, price
        print(v.select('p.prod_name > a')[0].text.strip())
        print(v.select('a.thumb_link > img')[0]['src'])
        print(v.select('p.price_sect > a')[0].text.strip())

browser.quit()
```

<details markdown="1">
<summary class='jb-small' style="color:blue">Click(with Implicitly wait)</summary>
<hr class='division3'>
```python
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")

browser = webdriver.Chrome('./webdriver/chromedriver.exe', options=chrome_options)
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)  # maximize_window(), minimize_window()
browser.get('http://prod.danawa.com/list/?cate=112758&15main_11_02')

# Implicitly wait
time.sleep(3); browser.find_element_by_xpath('//*[@id="dlMaker_simple"]/dd/div[2]/button[1]').click()
time.sleep(2); browser.find_element_by_xpath('//*[@id="selectMaker_simple_priceCompare_A"]/li[14]/label').click()
time.sleep(3)

# bs4 initializer
soup = BeautifulSoup(browser.page_source, "html.parser")
pro_list = soup.select('div.main_prodlist.main_prodlist_list > ul > li')
for v in pro_list:
    if not v.find('div', class_='ad_header'):
        # product name, image, price
        print(v.select('p.prod_name > a')[0].text.strip())
        print(v.select('a.thumb_link > img')[0]['src'])
        print(v.select('p.price_sect > a')[0].text.strip())

browser.quit()
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Application(click page-number)</summary>
<hr class='division3'>
```python
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")

browser = webdriver.Chrome('./webdriver/chromedriver.exe', options=chrome_options)
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)  # maximize_window(), minimize_window()
browser.get('http://prod.danawa.com/list/?cate=112758&15main_11_02')

WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'))).click()
WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.XPATH, '//*[@id="selectMaker_simple_priceCompare_A"]/li[14]/label'))).click()
time.sleep(3)


cur_page_num = 1; target_crawl_num = 5
while cur_page_num <= target_crawl_num:
    # bs4 initializer
    soup = BeautifulSoup(browser.page_source, "html.parser")
    pro_list = soup.select('div.main_prodlist.main_prodlist_list > ul > li')
    
    print('Current Page : {}'.format(cur_page_num))
    for v in pro_list:
        if not v.find('div', class_='ad_header'):
            # product name, image, price
            print(v.select('p.prod_name > a')[0].text.strip())
            print(v.select('a.thumb_link > img')[0]['src'])
            print(v.select('p.price_sect > a')[0].text.strip())
    cur_page_num += 1   # next page
    if cur_page_num > target_crawl_num:
        print('Crawling Succeed.')
        break
    
    WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.number_wrap > a:nth-child({})'.format(cur_page_num)))).click()
    time.sleep(4)

browser.quit()
```
<hr class='division3'>
</details>

<br><br><br>
<span class="frame3">xlsxwriter</span><br>
```bash
$ pip install xlsxwriter
```
```python
import urllib.request as req
from io import BytesIO
import xlsxwriter

import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

workbook = xlsxwriter.Workbook("crawling_result.xlsx")
worksheet = workbook.add_worksheet()

chrome_options = Options()
chrome_options.add_argument("--headless")

browser = webdriver.Chrome('./webdriver/chromedriver.exe', options=chrome_options)
browser.implicitly_wait(5)
browser.set_window_size(1920, 1280)  # maximize_window(), minimize_window()
browser.get('http://prod.danawa.com/list/?cate=112758&15main_11_02')

WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'))).click()
WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.XPATH, '//*[@id="selectMaker_simple_priceCompare_A"]/li[14]/label'))).click()
time.sleep(3)

cur_page_num = 1; target_crawl_num = 5
ins_cnt = 1  # excel row number

while cur_page_num <= target_crawl_num:
    # bs4 initializer
    soup = BeautifulSoup(browser.page_source, "html.parser")
    pro_list = soup.select('div.main_prodlist.main_prodlist_list > ul > li')
    
    print('Current Page : {}'.format(cur_page_num))
    for v in pro_list:
        if not v.find('div', class_='ad_header'):
            # product name, price
            prod_name = v.select('p.prod_name > a')[0].text.strip()
            prod_price = v.select('p.price_sect > a')[0].text.strip()
            # save excel(text)
            worksheet.write('A%s' % ins_cnt, prod_name)
            worksheet.write('B%s' % ins_cnt, prod_price)

            """
            # product image
            img_data = BytesIO(req.urlopen(v.select('a.thumb_link > img')[0]['data-original']).read())
            
            # save excel(image)
            worksheet.insert_image('C%s' % ins_cnt, prod_name, {'image_data': img_data})
            """

            ins_cnt += 1   # next row
    cur_page_num += 1      # next page
    if cur_page_num > target_crawl_num:
        print('Crawling Succeed.')
        break

    WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.number_wrap > a:nth-child({})'.format(cur_page_num)))).click()
    time.sleep(4)

browser.quit()
workbook.close()
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Making excel file</summary>
<hr class='division3'>
```python
import urllib.request as req
from io import BytesIO
import xlsxwriter

workbook = xlsxwriter.Workbook("excel.xlsx")
worksheet = workbook.add_worksheet()

img_data = BytesIO(req.urlopen('http://img.danawa.com/prod_img/500000/866/250/img/5250866_1.jpg?shrink=160:160&_v=20180713162212').read())

worksheet.write('A1', 'prod_name')
worksheet.write('B1', 'prod_price')
worksheet.insert_image('C1', 'prod_name', {'image_data': img_data})

workbook.close()
```
<hr class='division3'>
</details>




<br><br><br>

---


### ***Scraping : advanced(Scrapy)***
<span class="frame3">installation</span><br>
```dos
pip install scrapy
pip install pypiwin32
```
<br><br><br>

#### Shell

<br><br><br>

#### Spider

<br><br><br>

#### Selectors

<br><br><br>

#### Items

<br><br><br>

#### Exports

<br><br><br>

#### Settings

<br><br><br>

#### Pipeline

<br><br><br>

#### Scrapy project

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
```python
values = {'format': 'json'}              # It is also possible to use text, jsonp instead of json 
params = urllib.parse.urlencode(values)  # get parameter by encoding
print(params)
```
```
format=json
```
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
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
for i in params:
    print(i)
    param = urllib.parse.urlencode(i)
    print(param)
```
```
{'ctxCd': 1001}
ctxCd=1001
{'ctxCd': 1012}
ctxCd=1012
{'ctxCd': 1013}
ctxCd=1013
{'ctxCd': 1014}
ctxCd=1014
```
<hr class='division3'>
</details>

![image](https://user-images.githubusercontent.com/52376448/73420146-dfdfd400-4364-11ea-8127-342872cb0387.png)

<br><br><br>
#### EX4, daum finance
<a href="https://finance.daum.net/" target="_blank">daum finance</a><br>
```python
import json
import urllib.request as req
from fake_useragent import UserAgent

ua = UserAgent()
headers = {'User-Agent' : ua.ie,
           'referer' : 'https://finance.daum.net/'}
url = "https://finance.daum.net/api/search/ranks?limit=10"

res = req.urlopen(req.Request(url, headers=headers)).read().decode('utf-8')
rank_json = json.loads(res)['data']   # str -> json

for elm in rank_json:
    print('rank : {}, trade price : {}, name : {}'.format(elm['rank'], elm['tradePrice'], elm['name']), )
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>
```python
from fake_useragent import UserAgent

ua = UserAgent()
print(ua.ie)
print(ua.msie)
print(ua.chrome)
print(ua.safari)
print(ua.random)
```
```
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; InfoPath.3; .NET4.0C; .NET4.0E; .NET CLR 3.5.30729; .NET CLR 3.0.30729; MS-RTC LM 8)
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; InfoPath.3; .NET4.0C; .NET4.0E; .NET CLR 3.5.30729; .NET CLR 3.0.30729; MS-RTC LM 8)
Mozilla/5.0 (X11; CrOS i686 3912.101.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36
Mozilla/5.0 (Windows; U; Windows NT 6.1; zh-HK) AppleWebKit/533.18.1 (KHTML, like Gecko) Version/5.0.2 Safari/533.18.5
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1664.3 Safari/537.36
```
<br><br><br>
```python
res = req.urlopen(req.Request(url, headers=headers)).read().decode('utf-8')
```
```
{"data":[{"rank":1,"rankChange":0,"symbolCode":"A005930","code":"KR7005930003","name":"삼성전자","tradePrice":57100,"change":"FALL","changePrice":2000,"changeRate":0.0338409475,"chartSlideImage":null,"isNew":false},{"rank":2,"rankChange":2,"symbolCode":"A308170","code":"KR7308170000","name":"센트랄모텍","tradePrice":42450,"change":"RISE","changePrice":5750,"changeRate":0.1566757493,"chartSlideImage":null,"isNew":false},{"rank":3,"rankChange":5,"symbolCode":"A068270","code":"KR7068270008","name":"셀트리온","tradePrice":166500,"change":"FALL","changePrice":4500,"changeRate":0.0263157895,"chartSlideImage":null,"isNew":false},{"rank":4,"rankChange":-1,"symbolCode":"A226440","code":"KR7226440006","name":"한송네오텍","tradePrice":1930,"change":"RISE","changePrice":270,"changeRate":0.1626506024,"chartSlideImage":null,"isNew":false},{"rank":5,"rankChange":0,"symbolCode":"A028300","code":"KR7028300002","name":"에이치엘비","tradePrice":96300,"change":"FALL","changePrice":3500,"changeRate":0.0350701403,"chartSlideImage":null,"isNew":false},{"rank":6,"rankChange":-4,"symbolCode":"A215600","code":"KR7215600008","name":"신라젠","tradePrice":13750,"change":"FALL","changePrice":850,"changeRate":0.0582191781,"chartSlideImage":null,"isNew":false},{"rank":7,"rankChange":0,"symbolCode":"A011000","code":"KR7011000007","name":"진원생명과학","tradePrice":5590,"change":"RISE","changePrice":240,"changeRate":0.0448598131,"chartSlideImage":null,"isNew":true},{"rank":8,"rankChange":-1,"symbolCode":"A091990","code":"KR7091990002","name":"셀트리온헬스케어","tradePrice":55600,"change":"FALL","changePrice":500,"changeRate":0.008912656,"chartSlideImage":null,"isNew":false},{"rank":9,"rankChange":0,"symbolCode":"A045060","code":"KR7045060001","name":"오공","tradePrice":7920,"change":"RISE","changePrice":230,"changeRate":0.0299089727,"chartSlideImage":null,"isNew":false},{"rank":10,"rankChange":-4,"symbolCode":"A036540","code":"KR7036540003","name":"SFA반도체","tradePrice":6190,"change":"RISE","changePrice":340,"changeRate":0.0581196581,"chartSlideImage":null,"isNew":false}]}
```
<hr class='division3'>
</details>

![image](https://user-images.githubusercontent.com/52376448/73422799-6f898080-436d-11ea-8602-380b23cf5fe5.png)

<br><br><br>

#### EX5, naver image search
<a href="https://search.naver.com/search.naver?where=image&sm=tab_jum&query=%EC%9D%B4%EB%AF%B8%EC%A7%80" target="_blank">naver image search</a><br>
```python
import os
import urllib.parse as rep
import urllib.request as req
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

opener = req.build_opener()                            # Header info initializer
opener.addheaders = [('User-agent', UserAgent().ie)]   # User-Agent info
req.install_opener(opener)                             # Insert header info

base = "https://search.naver.com/search.naver?where=image&sm=tab_jum&query="   # naver image search URL
quote = rep.quote_plus("신소율")                                               # search word
url = base + quote                                                             # combine basic URL + search word

response = req.urlopen(url) # request

# error handling(being reated to making folder)
save_path = "image"  # image save folder at current path
try:
    if not (os.path.isdir(save_path)):         # check if existing folder is
        os.makedirs(os.path.join(save_path))   # if not, make folder
except OSError as e:
        print("folder creation failed!")
        print("folder name : {}".format(e.filename))
        raise RuntimeError('System Exit!')
else:
    print('folder is created!')


# bs4 initializer
soup = BeautifulSoup(response, "html.parser")
img_list = soup.select("div.img_area > a.thumb._thumb > img")

# download, numbering images
for i, img_list in enumerate(img_list, 1):
    file_name = os.path.join(save_path, save_path + str(i) + '.png'); print('image name : {}'.format(file_name))
    req.urlretrieve(img_list['data-source'], file_name)

print("download succeeded!")
```
```
folder is created!
image name : image\image1.png
image name : image\image2.png
image name : image\image3.png
image name : image\image4.png
image name : image\image5.png
image name : image\image6.png
image name : image\image7.png
image name : image\image8.png
image name : image\image9.png
image name : image\image10.png
image name : image\image11.png
image name : image\image12.png
image name : image\image13.png
image name : image\image14.png
image name : image\image15.png
image name : image\image16.png
image name : image\image17.png
image name : image\image18.png
image name : image\image19.png
image name : image\image20.png
image name : image\image21.png
image name : image\image22.png
image name : image\image23.png
image name : image\image24.png
image name : image\image25.png
image name : image\image26.png
image name : image\image27.png
image name : image\image28.png
image name : image\image29.png
image name : image\image30.png
image name : image\image31.png
image name : image\image32.png
image name : image\image33.png
image name : image\image34.png
image name : image\image35.png
image name : image\image36.png
image name : image\image37.png
image name : image\image38.png
image name : image\image39.png
image name : image\image40.png
image name : image\image41.png
image name : image\image42.png
image name : image\image43.png
image name : image\image44.png
image name : image\image45.png
image name : image\image46.png
image name : image\image47.png
image name : image\image48.png
image name : image\image49.png
image name : image\image50.png
download succeeded!
```

<br><br><br>
#### EX6, danawa log-in
![image](https://user-images.githubusercontent.com/52376448/74026891-786ff700-49ea-11ea-873e-3770696329b1.png)
![image](https://user-images.githubusercontent.com/52376448/74027362-52972200-49eb-11ea-8f70-7d7809e09487.png)
![image](https://user-images.githubusercontent.com/52376448/74027597-d94bff00-49eb-11ea-8774-f4fa68f5bc89.png)

```python
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

# /login/form-data on developer tools
login_info = {
    'redirectUrl': 'http://www.danawa.com/',
    'loginMemberType': 'general',
    'id': '',
    'password': ''
}

# request headers on developer tools
request_headers = {
    'User-Agent': UserAgent().chrome,
    'Referer': 'https://auth.danawa.com/login?url=http%3A%2F%2Fcws.danawa.com%2Fpoint%2Findex.php'
}


with requests.session() as s:
    # Request(try log-in)
    response = s.post('https://auth.danawa.com/login', login_info, headers=request_headers)
    
    # if log-in fail
    if response.status_code != 200:
        raise Exception('Login failed.')

    # move page with session info after log-in
    response = s.get('http://www.danawa.com/member/myPage.php', headers=request_headers)

    # EUC-KR (if korean is not work)
    # response.encoding = 'euc-kr'

    # bs4 initializer
    soup = BeautifulSoup(response.text, "html.parser")

    # check whether log-in is sucessful
    check_name = soup.find('p', class_="p_id")
    
    if check_name is None:
        raise Exception('Login failed. Wrong Password.')
    else:
        print('log-in is successful')
```
```
log-in is successful
```
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
- <a href='https://www.ipify.org/' target="_blank">ipify</a>
- <a href='https://httpbin.org/' target="_blank">httpbin</a>

---

[1]:{{ site.url }}/download/AI01/chromedriver.exe
[2]:{{ site.url }}/download/AI01/geckodriver.exe
[3]:{{ site.url }}/download/AI01/phantomjs.exe

<details markdown="1">
<summary class='jb-small' style="color:blue">handling error</summary>
<hr class='division3'>
<hr class='division3'>
</details>
