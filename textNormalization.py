import re
from typing import List
from num2words import num2words
import contractions
import string
import unicodedata

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from nltk_resources import download_all_nltk_resources_if_needed
download_all_nltk_resources_if_needed()  # TODO: download only what is needed

'''
Text normalization created by: AraujoVE & marcuscastelo with license CC BY-SA 4.0
'''


# convert text to ascii
def toAscii(text: str):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


# convert numbers to words
def processNumbers(text: str):
    return re.sub(r'(?:^|\s)(\d+)(?:$|\s)', lambda n: num2words(int(n.group(2))).replace(" ", "-"), text)


# replace multiple spaces with single space
def adjustSpaces(text: str):
    return re.sub(r'\s+', ' ', text)


# lowercase text
def lowerCase(text: str):
    return text.lower()


# remove leading and trailing whitespaces
def trimSpaces(text: str):
    return text.strip()


# replace contractions with their expanded forms
def replaceContractions(text: str):
    return contractions.fix(text)


# remove punctuation
def removePunctuation(text: str):
    # Remove punctuations and replace '' with '' (weird syntax)
    return text.translate(str.maketrans('', '', string.punctuation))


# remove stop words
def removeStopWords(text: str):
    text = text.split()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = [w for w in text if not w in stopwords]
    text = ' '.join(text)
    return text


# pos tag conversion
def posTagConv(tag: str):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('R') or tag == 'WRB':
        return 'r'
    else:
        return 'n'


# lemmatize text
def lemmatizeText(text: str):
    # Completar com todas as possibilidades https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
    acceptedPosTags = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'FW', 'CD']
    acceptedPosTags = [posTagConv(tag) for tag in acceptedPosTags] #TODO: check if correct

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tag = nltk.pos_tag(tokens)

    def filterTokensByPos(acceptedPosTags: List[str]):
        return [wordPair for wordPair in pos_tag if wordPair[1] in acceptedPosTags]

    text = " ".join([lemmatizer.lemmatize(wordPair[0]) for wordPair in filterTokensByPos(acceptedPosTags)])
    return text


# normalize text
def normalizeText(text: str):
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


def normalizeWithDelimiter(text: str, delimiter: str = ","):
    return normalizeText(text).replace(" ", delimiter)
