---
layout : post
title : PL03-Topic02, requests
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/PL03/PL03-Topic02/2019-08-13-PL03-Topic02-requests.md" target="_blank">page management</a><br>
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


## **Installation**
### ***For linux***
```bash
$ pip3 install requests
```
<br><br><br>

### ***For windows***
```dos
pip install requests
```
<br><br><br>

### ***Version Control***
```python

```
<br><br><br>


<hr class="division2">

## **get**
```python
import requests

url = 'https://song-eunho.tistory.com/'

resp = requests.get(url)
print(resp.text)
```

<br><br><br>
<hr class="division2">

## **post**
```python
import requests

url = 'https://www.kangcom.com/member/member_check.asp'
data = {'id':'userdyk','pwd':'pwd1'}

resp = requests.post(url, data=data)
print(resp.text)
```

<br><br><br>
<hr class="division2">

## title3

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference

- <a href='https://repl.it/languages/python' target="_blank">Implementation with python2 on web</a>
- <a href='https://repl.it/languages/python3' target="_blank">Implementation with python3 on web</a>
- <a href='https://realpython.com/python-requests/'>Python’s Requests Library (Guide)</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
    <details markdown="1">
    <summary class='jb-small' style="color:red">OUTPUT</summary>
    <hr class='division3_1'>
    <hr class='division3_1'>
    </details>
<hr class='division3'>
</details>




