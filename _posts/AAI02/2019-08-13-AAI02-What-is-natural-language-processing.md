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

print(x)
pos_tag(x)
```
['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']<br>
[('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]

<a href="https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html" target="_blank">Reference</a>

|PRP|personal pronouns|
|VBP|verb|
|RB|adverb|
|VBG|present participle|
|IN|preposition|
|NNP|proper noun|
|NNS|aggregate noun|
|CC|conjunction|
|DT|article|

<br><br><br>
<span class="frame3">Korean</span>
```python
from konlpy.tag import Kkma  

kkma=Kkma()  
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
```
['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']  <br>
[('열심히','MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]  <br>
['코딩', '당신', '연휴', '여행']  <br>

<br><br><br>

### ***cleaning and normalization***

- morphology
  - stem
  - affix

<br><br><br>
#### Lemmatization : conservation of pos

`WordNetLemmatizer`
```python
from nltk.stem import WordNetLemmatizer

n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([n.lemmatize(w) for w in words])
```
['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting'] <br><br>

<p class="jb-small">The above results present inappropriate words that do not have any meaning, such as <b>dy or ha</b>. This is because the lemmatizer must know the information about part or speech of the original word for accurate results.</p>
```python
from nltk.stem import WordNetLemmatizer

n=WordNetLemmatizer()
print(n.lemmatize('dies', 'v'))
print(n.lemmatize('watched', 'v'))
print(n.lemmatize('has', 'v'))
```
'die'<br>
'watch'<br>
'have'

<br><br><br>

#### Stemming : non-conservation of pos
<span class="frame3">stemming through porter algorithm</span>
`PorterStemmer`
```python
from nltk.stem import PorterStemmer

s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])
```
['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
<br>
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)

print(words)
print([s.stem(w) for w in words])
```
['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']<br>
['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']<br><br>

<p class="jb-small">
  The results of the above algorithm include words that are not in the dictionary.
</p>

<br><br><br>
<span class="frame3">stemming through Lancaster Stemmer algorithm</span>
`LancasterStemmer`
```python
from nltk.stem import LancasterStemmer

l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])
```


<br><br><br>
#### Removing Unnecessary Words(noise data)

<span class="frame3">Stopword</span>
<span class="frame3_1">List of Stopword about Eng</span>
`stopwords`
```python
from nltk.corpus import stopwords  
stopwords.words('english')[:10]
```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']  <br>
<span class="frame3_1">Removing Stopword about Eng</span>
```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 

print(word_tokens) 
print(result) 
```
['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']<br>
['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
<br>

<span class="frame3_1">Removing Stopword about Kor</span>
```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)
```
['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']<br>
['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']<br>
<br><br><br>

<span class="frame3">Rare words</span>

<br><br><br>

<span class="frame3">words with very a short length</span>

```python
import re

text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')

print(shortword.sub('', text))
```
was wondering anyone out there could enlighten this car.

<br><br><br>

#### Regular Expression
```python
```
<br><br><br>
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



