---
layout : post
title : PL03-Topic02, gspread
categories: [PL03-Topic02]
comments : true
tags : [PL03-Topic02]
---
[Back to the previous page](https://userdyk-github.github.io/pl03/PL03-Libraries.html) <br>
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
$ pip install gspread
```
<br><br><br>

### ***For windows***
```dos
pip install gspread
pip install --upgrade oauth2client
```
<a href="https://cloud.google.com/" target="_blank">Google Cloud Platform(GCP), API & Services, google driver API and google sheets API</a>

<br><br><br>

### ***Version Control***
```python

```
<br><br><br>

<hr class="division2">

## **Google sheets**

```python
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = [
'https://spreadsheets.google.com/feeds',
'https://www.googleapis.com/auth/drive',
]

json_file_name = r'S:\Github\userdyk\yerin\emerald-cumulus-243810-93dc8ef8e5dd.json'
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
gc = gspread.authorize(credentials)

spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1AQ3GXxWGHlx55KtUTTRYqtw8AA_bfjxJ2YVvS_PDSmk/edit#gid=1267764562'

# 스프레스시트 문서 가져오기 
doc = gc.open_by_url(spreadsheet_url)
# 시트 선택하기
worksheet = doc.worksheet('sheet1')
```
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
- <a href='https://console.developers.google.com/' target="_blank">google developers</a>
- <a href='http://hleecaster.com/python-google-drive-spreadsheet-api/'>hleecaster</a>

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




