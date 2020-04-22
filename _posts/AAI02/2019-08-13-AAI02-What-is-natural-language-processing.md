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


<br><br><br>
<hr class="division2">

## **korean morpheme analysis**
### ***Hannanum : KAIST***
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
<br><br><br>

---

### ***Kkma : SNU***
```python
from konlpy.tag import Kkma

kkma = Kkma()
morphs = kkma.morphs((u'대한민국은 아름다운 나라이다.'))
nouns = kkma.nouns((u'대한민국은 아름다운 나라이다.'))
pos = kkma.pos((u'대한민국은 아름다운 나라이다.'))
sentences = kkma.sentences((u'대한민국은 아름다운 나라이다.'))

print("morphs :\n", morphs)
print("nouns :\n", nouns)
print("pos :\n", pos)
print("sentences :\n", sentences)
```
<br><br><br>

### ***Komoran : Shineware***
```python

```
<br><br><br>

### ***Mecab : Eunjeon project***
```bash
$ sudo apt-get install curl git
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
```python
from konlpy.tag import Mecab

mecab = Mecab()
morphs = mecab.morphs((u'대한민국은 아름다운 나라이다.'))
nouns = mecab.nouns((u'대한민국은 아름다운 나라이다.'))
pos = mecab.pos((u'대한민국은 아름다운 나라이다.'))

print("morphs :\n", morphs)
print("nouns :\n", nouns)
print("pos :\n", pos)
```
<br><br><br>

### ***Okt : Twitter***
```python
from konlpy.tag import Okt

okt = Okt()
morphs = okt.morphs((u'대한민국은 아름다운 나라이다.'))
nouns = okt.nouns((u'대한민국은 아름다운 나라이다.'))
pos = okt.pos((u'대한민국은 아름다운 나라이다.'))
phrases = okt.phrases((u'대한민국은 아름다운 나라이다.'))

print("morphs :\n", morphs)
print("nouns :\n", nouns)
print("pos :\n", pos)
print("phrases :\n", phrases)
```
<br><br><br>

<hr class="division2">

## **Text preprocessing**
### ***Tokenization***
#### Word Tokenization
`word_tokenize`
```python
from nltk.tokenize import word_tokenize  

print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))  
```
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']  
<br>
`WordPunctTokenizer`
```python
from nltk.tokenize import WordPunctTokenizer  

print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']  

<br>
`text_to_word_sequence`
```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

<br><br><br>

<span class="frame3">Consideration</span>

- Don't simply exclude <b>punctuation marks</b> or <b>special characters</b>.
  - ex] Ph.D, AT&T, 123,456,789
- In case of <b>abbreviations</b> and <b>spacing within words</b>
  - ex] rock 'n' roll(abbreviation), New York(spacing within words)
- Standard : Penn Treebank Tokenization

`TreebankWordTokenizer`
```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))
```
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.'] 

<br><br><br>

#### Sentence Tokenization
`sent_tokenize`
```python
from nltk.tokenize import sent_tokenize

text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))
```
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']
<br>

```python
from nltk.tokenize import sent_tokenize

text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
```
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
<br><br><br>

#### Part-of-speech tagging
<span class="frame3">English</span>
```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
x=word_tokenize(text)

print(word_tokenize(text))
pos_tag(x)
```
['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']<br>
[('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]

<span class="frame3">Korean</span>
```python

```


<hr class="division2">

## **Language model**
```python

```
<br><br><br>
<hr class="division2">



## **Count based word Representation**
<br><br><br>
<hr class="division2">



## **Document Similarity**
<br><br><br>
<hr class="division2">



## **Topic Modeling**
<br><br><br>
<hr class="division2">


## **Machine Learning**
<br><br><br>
<hr class="division2">



## **Deep Learning**
<br><br><br>
<hr class="division2">



## **Recurrent Neural Network**
<br><br><br>
<hr class="division2">



## **Word Embedding**
<br><br><br>
<hr class="division2">



## **Text Classification**
<br><br><br>
<hr class="division2">



## **Tagging Task**
<br><br><br>
<hr class="division2">



## **Neural Machine Translation**
<br><br><br>
<hr class="division2">



## **Attention Mechanism**
<br><br><br>
<hr class="division2">


## **Transformer**
<br><br><br>
<hr class="division2">


## **Convolution Neural Network**
<br><br><br>
<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href='https://wikidocs.net/book/2155' target="_blank">딥 러닝을 이용한 자연어 처리 입문</a>
- <a href="https://www.edwith.org/deepnlp/joinLectures/17363" target="_blank">NLP lecture</a>
- Taweh Beysolow II, Applied Natural Language Processing with Python, 2018
- <a href='https://userdyk-github.github.io/'>post3</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>



