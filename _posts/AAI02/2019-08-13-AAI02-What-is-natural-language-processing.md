---
layout : post
title : AAI02, What is natural language processing
categories: [AAI02]
comments : true
tags : [AAI02]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html)｜<a href="https://github.com/userdyk-github/userdyk-github.github.io/blob/master/_posts/AAI02/2019-08-13-AAI02-What-is-natural-language-processing.md" target="_blank">page management</a><br>
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

## **konlpy installation**

<a href="http://konlpy.org/ko/latest/install/" target="_blank">URL</a>
```bash
$ sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl      # Install Java 1.8 or up
$ python3 -m pip install --upgrade pip
$ python3 -m pip install konlpy                                            # Python 3.x
```
<hr class="division2">

## **morpheme analysis**
```python
from konlpy.tag import Hannanum

hannanum = Hannanum()
analyze = hannanum.analyze((u'대한민국은 아름다운 나라이다.'))
morphs = hannanum.morphs((u'대한민국은 아름다운 나라이다.'))
nouns = hannanum.nouns((u'대한민국은 아름다운 나라이다.'))
pos = hannanum.pos((u'대한민국은 아름다운 나라이다.'))

print("analyze :\n", analyze)
print("morphs :\n", morphs)
print("nouns :\n", nouns)
print("pos :\n", pos)
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
- <a href='https://wikidocs.net/book/2155' target="_blank">딥 러닝을 이용한 자연어 처리 입문</a>
- Taweh Beysolow II, Applied Natural Language Processing with Python, 2018
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>



