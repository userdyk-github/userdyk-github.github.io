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

## **Introduction**

### Upstream-task

- Tokenize
- Embedding
  - Factorization based(Matrix decomposition)
    - GloVe, Swivel
  - Prediction based
    - Word2Vec, FastText, BERT, ELMo, GPT
  - Topic based
    - LDA
  
<br><br><br>

---

### Downstream-task

- Part of Speech tagging
- Named Entity Recognition
- Semantic Rule Labeling

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

---

### ***Komoran : Shineware***
```python

```
<br><br><br>

---

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

---

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

<span class="frame3">Consideration</span><br>

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
<span class="frame3">English</span><br>
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
<span class="frame3">Korean</span><br>
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
#### Tokenization with regular expression
```python
import nltk
from nltk.tokenize import RegexpTokenizer

tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```
['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop'] <br>

```python
import nltk
from nltk.tokenize import RegexpTokenizer

tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop'] <br>

<br><br><br>

---

### ***Cleaning and normalization***

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
<span class="frame3">stemming through porter algorithm</span><br>
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
<span class="frame3">stemming through Lancaster Stemmer algorithm</span><br>
`LancasterStemmer`
```python
from nltk.stem import LancasterStemmer

l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])
```


<br><br><br>
#### Removing Unnecessary Words(noise data)

<span class="frame3">Stopword</span><br>
<span class="frame3_1">List of Stopword about Eng</span><br>
`stopwords`
```python
from nltk.corpus import stopwords  
stopwords.words('english')[:10]
```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']  <br>
<span class="frame3_1">Removing Stopword about Eng</span><br>
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
<span class="frame3_1">List of Stopword about Kor</span><br>

- [https://www.ranks.nl/stopwords/korean](https://www.ranks.nl/stopwords/korean)
- [https://bab2min.tistory.com/544](https://bab2min.tistory.com/544)

<br>
<span class="frame3_1">Removing Stopword about Kor</span><br>
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

<span class="frame3">Rare words</span><br>

<br><br><br>

<span class="frame3">words with very a short length</span><br>

```python
import re

text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')

print(shortword.sub('', text))
```
was wondering anyone out there could enlighten this car.

<br><br><br>

---

### ***Encoding***

#### Integer encoding

<br><br><br>

#### One-hot encoding

<br><br><br>

#### Byte Pair encoding

<br><br><br>
<hr class="division2">


## **Language model**

- Assign probabilities for word sequences
  - Machine Translation
  - Spell Correction
  - Speech Recognition

<br><br><br>

### ***Statistical Language Model, SLM***

- Prediction of the next word from a given previous word
  - P(W) = P(w_1,w_2,w_3,...,w_n)
  - P(w_n｜w_1,...,w_n-1) = P(w_1,w_2,...,w_n)/P(w_1,w_2,w_3,...,w_n-1)

![image](https://user-images.githubusercontent.com/52376448/79956207-cc06c280-84ba-11ea-9e93-31406b685431.png)

Cause sparsity problem.
<br><br><br>

#### N-gram Language Model

![image](https://user-images.githubusercontent.com/52376448/79959958-c2338e00-84bf-11ea-85a0-69a6c7aa255f.png)


- Sparsity problem
- Trade off (<b>sparsity</b> vs <b>accuracy</b>)
  - The higher the value of n, the stronger the sparsity problem and the accuracy.
  - The lower the value of n, the weaker the sparsity problem and the accuracy.

for general,
![image](https://user-images.githubusercontent.com/52376448/79956207-cc06c280-84ba-11ea-9e93-31406b685431.png)
for n-gram,
![image](https://user-images.githubusercontent.com/52376448/79959150-bd220f00-84be-11ea-95fc-aade43cfc0fc.png)

<br><br><br>

#### Evaluation: Perplexity(Branching factor)

- Evaluation
  - extrinsic
  - intrinsic : Perplexity

![image](https://user-images.githubusercontent.com/52376448/79959035-9237bb00-84be-11ea-9d87-2b38e92d6deb.png)

<br><br><br>

---

### ***Neural Network Based Language Model***

#### Feed Forward Neural Network Language Model, FFNNLM : Neural Probabilistic Language Model

- Improvement : Solving sparsity problem
- Limitation : Fixed-length input

<br><br><br>

#### Recurrent Neural Network Language Model, RNNLM
```python

```
<br><br><br>

<hr class="division2">


## **Quantification : Word representation**
![image](https://user-images.githubusercontent.com/52376448/79962375-eba1e900-84c2-11ea-9b94-6ff5e3d45bf1.png)

### ***Local Representation***

- WDM, Word-Document Matrix
- TF-IDF, Term Frequency-Inverse Document Frequency
- WCM, Word-Context Matrix
- PMIM, Point-wise Mutual Information Matrix

<br><br><br>

#### Count based word Representation(1) : BoW 
<span class="frame3">Bag of Words</span><br>
```python
from konlpy.tag import Okt
import re  
okt=Okt()  

token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token=okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index={}  
bow=[]  
for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca]=len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index)-1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index=word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index]=bow[index]+1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)
print(bow)
```
('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)  <br>
[1, 2, 1, 1, 2, 1, 1, 1, 1, 1]  <br>
<br><br><br>

<span class="frame3">Create Bag of Words with CountVectorizer class</span><br>
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```
[[1 1 2 1 2 1]]<br>
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}<br>
<br><br><br>

<span class="frame3">Remove stopwords in Bag of Words</span><br>
<span class="frame3_1">Custom stopwords</span><br>
```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])

print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```
[[1 1 1 1 1]]<br>
{'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}<br>
<span class="frame3_1">CountVectorizer stopwords</span><br>
```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")

print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```
[[1 1 1]]<br>
{'family': 0, 'important': 1, 'thing': 2}<br>
<span class="frame3_1">NLTK stopwords</span><br>
```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```
[[1 1 1 1]]<br>
{'family': 1, 'important': 2, 'thing': 3, 'everything': 0}<br>
<br><br><br>

#### Count based word Representation(2) : DTM
<span class="frame3">Document-Term Matrix, DTM</span><br>
![image](https://user-images.githubusercontent.com/52376448/79975907-ae475680-84d6-11ea-8f6e-6bf83477376d.png)

- Limitations : Sparse representation

<br><br><br>

#### Count based word Representation(3) : TD-IDF
<span class="frame3">Term Frequency-Inverse Document Frequency</span><br>
<span class="frame3_1">Implementation : pandas</span><br>
```python
import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()



N = len(docs) # 총 문서의 수

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)
    


result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]        
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
print(tf_)
```
![image](https://user-images.githubusercontent.com/52376448/79976981-7ccf8a80-84d8-11ea-947d-feaeeca628bd.png)
```python
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
print(idf_)
```
![image](https://user-images.githubusercontent.com/52376448/79977054-983a9580-84d8-11ea-8489-014bebdac7fe.png)
```python
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]

        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
print(tfidf_)
```
![image](https://user-images.githubusercontent.com/52376448/79977104-a983a200-84d8-11ea-8c86-b4a31f9a0c94.png)

<span class="frame3_1">Implementation : scikit-learn</span><br>
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```
[[0 1 0 1 0 1 0 1 1]<br>
 [0 0 1 0 0 0 0 1 0]<br>
 [1 0 0 0 1 0 1 0 0]]<br>
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}<br>
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)

print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
```
[[0.         0.46735098 0.         0.46735098 0.         0.46735098 0.         0.35543247 0.46735098]<br>
 [0.         0.         0.79596054 0.         0.         0.         0.         0.60534851 0.        ]<br>
 [0.57735027 0.         0.         0.         0.57735027 0.         0.57735027 0.         0.        ]]<br>
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}<br>

<span class="frame3_1">Implementation : keras</span><br>
```python

```
<br><br><br>


---


### ***Continuous Representation***
#### Topic modeling(1) : LSA
<span class="frame3">Singular Value Decomposition, SVD</span><br>
```python
import numpy as np

A = np.array([[0,0,0,1,0,1,1,0,0],
              [0,0,0,1,1,0,1,0,0],
              [0,1,1,0,2,0,0,0,0],
              [1,0,0,0,0,0,0,1,1]])
              
# Full SVD              
U, s, VT = np.linalg.svd(A, full_matrices = True)  
S = np.zeros(np.shape(A))         # 대각 행렬의 크기인 4 x 9의 임의의 행렬 생성
S[:4, :4] = np.diag(s)            # 특이값을 대각행렬에 삽입

# Truncated SVD
S=S[:2,:2]
U=U[:,:2]
VT=VT[:2,:]
```
<br><br><br>

<span class="frame3">Latent Semantic Analysis, LSA</span><br>
```python

```

<br><br><br>
#### Topic modeling(2) : LDA
<span class="frame3">Latent Dirichlet Allocation, LDA</span><br>


<br><br><br>


#### Word Embedding

<br><br><br>

---

### ***Document Similarity***

<br><br><br>
<hr class="division2">



## **Recurrent Neural Network**
### ***Word-level***
```python

```
<br><br><br>

### ***Character-level***
```python

```
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
<a href="https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225" target="_blank">attention explanation URL</a>
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
- <a href="https://torchtext.readthedocs.io/en/latest/index.html" target="_blank">torchtext</a>
- <a href='https://wikidocs.net/book/2155' target="_blank">딥 러닝을 이용한 자연어 처리 입문</a>
- <a href="https://wikidocs.net/book/2159" target="_blank"> 딥 러닝을 이용한 자연어 처리 심화</a>
- <a href="https://www.edwith.org/deepnlp/joinLectures/17363" target="_blank">NLP lecture(1)</a>
- <a href="https://www.youtube.com/playlist?list=PLVNY1HnUlO26qqZznHVWAqjS1fWw0zqnT" target="_blank">NLP lecture(2)</a>
- Taweh Beysolow II, Applied Natural Language Processing with Python, 2018
- <a href='https://wikidocs.net/book/2788'> PyTorch로 시작하는 딥 러닝 입문 </a>
- <a href="http://jalammar.github.io/" target="_blank">jalammar</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
<hr class='division3'>
</details>



