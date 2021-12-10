import re
from num2words import num2words
import contractions
import string
import unicodedata
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt') 
#nltk.download('wordnet')

'''
Text normalization created by: AraujoVE
'''


# convert text to ascii
def toAscii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

# convert numbers to words
def processNumbers(text):
    return re.sub(r'(^|\s)(\d+)($|\s)', lambda n: n.group(1)+num2words(int(n.group(2))).replace(" ","-")+n.group(3), text)

# replace multiple spaces with single space
def adjustSpaces(text):
    return re.sub(r'\s+', ' ', text)

# lowercase text
def lowerCase(text):
    return text.lower()

# remove leading and trailing whitespaces
def trimSpaces(text):
    return text.strip()

# replace contractions with their expanded forms
def replaceContractions(text):
    return contractions.fix(text)

# remove punctuation
def removePunctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# remove stop words
def removeStopWords(text): 
    text = text.split()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = [w for w in text if not w in stopwords]
    text = ' '.join(text)
    return text

# pos tag conversion
def posTagConv(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('R') or tag == 'WRB':
        return 'r'
    else:
        return 'n'

# lemmatize text
def lemmatizeText(text):
    #Completar com todas as possibilidades
    chosenPosTags = ['NNS','FW']
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tag = nltk.pos_tag(tokens)
    text = " ".join([lemmatizer.lemmatize(wordPair[0]) for wordPair in pos_tag if wordPair[1] in chosenPosTags])
    return text


# normalize text
def normalizeText(text):
    text = lowerCase(text)
    text = trimSpaces(text)
    text = replaceContractions(text)
    text = adjustSpaces(text)
    text = lemmatizeText(text)
    text = removeStopWords(text)
    text = removePunctuation(text)
    text = processNumbers(text)
    text = adjustSpaces(text)
    return trimSpaces(text)

def dividedNormalizedText(text,delimiter=","):
    return normalizeText(text).replace(" ",delimiter)