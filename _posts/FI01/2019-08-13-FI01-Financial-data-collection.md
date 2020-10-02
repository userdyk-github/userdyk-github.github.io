---
layout : post
title : FI01, Financial data collection
categories: [FI01]
comments : true
tags : [FI01]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) ｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/FI01/2019-08-13-FI01-Financial-data-collection.md" target="_blank">page management</a><br>
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

## **Stock Information**
- <a href="https://www.google.com/finance" target="_blank">google finance</a>
  - <a href="https://developers.google.com/gdata/" target="_blank">gdata</a>
    - <a href="https://developers.google.com/gdata/docs/directory" target="_blank">gdata api directory1</a>
    - <a href="https://developers.google.com/gdata/jsdoc/2.2/google/gdata/" target="_blank">gdata api directory2</a>
      - <a href="https://developers.google.com/gdata/jsdoc/2.2/google/gdata/finance/" target="_blank">finance</a>
        - <a href="https://developers.google.com/gdata/jsdoc/2.2/google/gdata/finance/FinanceService" target="_blank">finance service</a>
    - <a href="https://github.com/google/gdata-python-client" target="_blank">github</a>
  - <a href="https://towardsdatascience.com/best-5-free-stock-market-apis-in-2019-ad91dddec984" target="_blank">manual</a>
- <a href="https://github.com/ranaroussi/yfinance">yfinance</a>
  - <a href="https://towardsdatascience.com/best-5-free-stock-market-apis-in-2019-ad91dddec984" target="_blank">manual</a>

`pandas-datareader` : <a href="https://pandas-datareader.readthedocs.io/en/latest/" target="_blank">API URL</a>
```bash
$ pip install pandas-datareader
```
```python
import pandas_datareader.data as web

df = web.DataReader('005380', 'naver', start='2020-01-01', end='2020-06-30')
df.tail()
```
`finance-datareader` : <a href="https://github.com/financedata/financedatareader" target="_blank">API URL Github</a>, <a href="https://github.com/FinanceData/FinanceDataReader/wiki/Users-Guide" target="_blank">github wiki userguide</a>
```bash
$ pip install finance-datareader
```
```python

```
<hr class="division2">

## title2

<hr class="division2">

## title3

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


