## Data preprocessing(English)
# 1. Cleansing
# 2. Tokenization
# 3. Stopwords
# 4. Stemming
# 5. Lemmatization


## 1. Cleansing

import re


def data_processing(text):
    text_re = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', text)

    # Single character removal
    text_re = re.sub(r"\s+[a-zA-Z]\s+", ' ', text_re)

    # Removing multiple spaces
    text_re = re.sub(r'\s+', ' ', text_re)

    return text_re


text = "DF+#$%^ $^&@$%}a"
data_processing(text)

"""
'DF } '
"""

## 2. Tokenization

from nltk import sent_tokenize
text_sample = 'The Beluga XL is the successor to the Beluga, or Airbus A300-600ST, which has been in operation since 1995. \
Its design was adapted from an A330 airliner, with Airbus engineers lowering the flight deck and grafting a huge cargo bay onto the fuselage to create its distinctive shape.\
Through an upward-opening forward hatch on the "bubble," completed aircraft wings, fuselage sections and other components easily slide in and out.'
sentences = sent_tokenize(text = text_sample)
print(type(sentences),len(sentences))
print(sentences)

"""
<class 'list'> 2
['The Beluga XL is the successor to the Beluga, or Airbus A300-600ST, which has been in operation since 1995.', 'Its design was adapted from an A330 airliner, with Airbus engineers lowering the flight deck and grafting a huge cargo bay onto the fuselage to create its distinctive shape.Through an upward-opening forward hatch on the "bubble," completed aircraft wings, fuselage sections and other components easily slide in and out.']

"""

## 3. Stopwords

from nltk import word_tokenize
import nltk
nltk.download('stopwords')
print("how many stopword ", len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[0:30])

"""
영어 stopword 개수 179
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself'
, 'it', "it's", 'its', 'itself']

"""


## 4. Stemming
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()
word_list = ['loved', 'loving', 'lovely', 'going', 'went']
for word in word_list:
    print(stemmer.stem(word))

"""
lov
lov
lov
"""


## 5. Lemmatization
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing','v'))
print(lemma.lemmatize('beautiful','a'))
print(lemma.lemmatize('fanciest','a'))

"""
amuse
beautiful
fancy

"""





