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
<a href="https://docs.python.org/3/library/urllib.html" target="_blank">API</a><br>
<span class="frame3">installation</span><br>
```dos
pip install urllib3
pip install fake-useragent
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
#### from requests
<a href="https://requests.readthedocs.io/en/master/api/" target="_blank">API</a><br>

<span class="frame3">installation</span><br>
```dos
pip install requests
pip install lxml
pip install cssselect
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
<summary class='jb-small' style="color:blue">with cookies, headers</summary>
<hr class='division3'>
```python
import requests

response1 = requests.get("https://httpbin.org/cookies", cookies={'name':'kim'})
response2 = requests.get("https://httpbin.org", headers={'user-agent':'nice-man_1.0.0_win10_ram16_home_chrome'})

print(response1, response1.text)
print(response2, response2.text)
```

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
<a href="" target="_blank">API</a><br>

<span class="frame3">installation</span><br>
```dos

```
<br><br><br>

<br><br><br>

#### from Selenium
<a href="" target="_blank">API</a><br>

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
- <a href='https://www.ipify.org/' target="_blank">ipify</a>
- <a href='https://httpbin.org/' target="_blank">httpbin</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">handling error</summary>
<hr class='division3'>
<hr class='division3'>
</details>
