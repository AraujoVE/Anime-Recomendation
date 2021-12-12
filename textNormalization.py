import re
from sys import prefix
from typing import Any, List
from num2words import num2words # type: ignore 
import contractions # type: ignore
import string
import unicodedata

import nltk # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from nltk.tokenize import word_tokenize
from numpy import number # type: ignore

from nltk_resources import download_all_nltk_resources_if_needed
download_all_nltk_resources_if_needed()  # TODO: download only what is needed

'''
Text normalization created by: AraujoVE & marcuscastelo
'''

# Convert text to ASCII
def convertAscii(text: str):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


# Convert numbers to words (e.g. "123" -> "hundred twenty three")
def convertNumbers(text: str) -> str:
    ISOLATED_NUMBER_REGEX = r'(^|\s)([\d.]+)($|\s)'

    def remove_commas(text: str) -> str:
        return text.replace(",", "")

    # Replace numbers with words (e.g. "123" -> "hundred twenty three")
    def number_to_words(regexMatch):
        prefix = regexMatch.group(1)
        numberStr = regexMatch.group(2)
        suffix = regexMatch.group(3)
        wordsStr = num2words(int(numberStr), lang='en').replace("-", " ")
        return prefix + wordsStr + suffix
        

    text = remove_commas(text)
    return re.sub(ISOLATED_NUMBER_REGEX, number_to_words, text)


# Convert multiple spaces to single spaces
def removeDoubleSpaces(text: str):
    return re.sub(r'\s+', ' ', text)


# Convert text to lower case
def lowerCase(text: str):
    return text.lower()


# Remove leading and trailing whitespaces
def trimSpaces(text: str):
    return text.strip()


# Replace contractions with their expanded forms
def replaceContractions(text: str):
    return contractions.fix(text)


# Remove punctuation (such as commas, periods, exclamation points, etc.)
def removePunctuation(text: str):
    # Remove punctuations and replace '' with '' (weird syntax)
    return text.translate(str.maketrans('', '', string.punctuation))


# Remove stop words (such as "the", "a", "an")
def removeStopWords(tokens: List[str]):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = [token for token in tokens if not token in stopwords]
    return text

# Split text into tokens
def tokenize_text(text: str) -> List[str]:
    return word_tokenize(text)

def join_tokens(tokens: List[str], delimiter: str = " "):
    return delimiter.join(tokens)

# Filter out tokens with POS (Part of Speech) tag not in acceptedPosTags
def filter_tokens_by_pos(tokens: List[str], acceptedPosTags: List[str]) -> List[str]:
    token_pos_pairs = nltk.pos_tag(tokens)
    return [ token for (token, pos) in token_pos_pairs if pos in acceptedPosTags ]

# Lemmatize tokens (i.e. convert verbs, nouns, adjectives to their base form)
def lemmatizeTokens(tokens: List[str]):
    lemmatizer = WordNetLemmatizer()
    return [ lemmatizer.lemmatize(token) for token in tokens ]

def lemmatizeText(text: str):
    return lemmatizeTokens(tokenize_text(text))

# Apply all text normalization functions
def normalizeTextToKeywords(text: str, delimiter: str = ",") -> str:
    #TODO: move to GUI config 
    # Completar com todas as possibilidades https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
    acceptedPosTags = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'FW', 'CD']

    # Text normalization
    text = lowerCase(text)
    text = trimSpaces(text)
    text = replaceContractions(text)
    text = removeDoubleSpaces(text)

    # Token normalization
    tokens = tokenize_text(text)
    tokens = filter_tokens_by_pos(tokens, acceptedPosTags)
    tokens = lemmatizeTokens(tokens)
    tokens = removeStopWords(tokens)

    # Final text normalization
    text = join_tokens(tokens)
    text = removePunctuation(text)
    text = convertNumbers(text)
    text = removeDoubleSpaces(text)
    text = trimSpaces(text)

    text.replace(" ", delimiter)

    return text

def test_all():
    print("Testing lowerCase()")
    print(lowerCase("This is a test"))

    print("Testing trimSpaces()")
    print(trimSpaces("  This is a test  "))

    print("Testing replaceContractions()")
    print(replaceContractions("I'm testing this"))

    print("Testing removePunctuation()")
    print(removePunctuation("This, is a test!"))

    print("Testing removeStopWords()")
    print(removeStopWords("This is a test"))

    print("Testing lemmatizeText()")
    print(lemmatizeTokens("This is a test"))

    print("Testing normalizeText()")
    print(normalizeTextToKeywords("This is a test"))

    print("Testing normalizeWithDelimiter()")
    print(normalizeWithDelimiter("This is a test"))

    print("Testing processNumbers()")
    print(convertNumbers("This is a test"))
    print(convertNumbers("This is a test 123"))
    print(convertNumbers("This is a test 123,456"))
    print(convertNumbers("This is a test 123,456.789"))
    print(convertNumbers("This is a test 123,456.789,000"))
    print(convertNumbers("This is a test 123,456.789,000.000"))
    print(convertNumbers("This is a test 123,456.789,000.000.000"))


def test_lemma():
    print("Testing lemmatizeText()")
    print(lemmatizeTokens("This is a test"))
    print(lemmatizeTokens("Tanjiro is a good person that walks with his friends"))

def test_numbers():
    print("Testing convertNumbers()")
    print(convertNumbers("This is a test"))
    print(convertNumbers("This is a test 123"))
    print(convertNumbers("This is a test 123,456"))
    print(convertNumbers("This is a test 123,456.789"))
    print(convertNumbers("This is a test 123,456.789,000"))
    print(convertNumbers("This is a test 123,456.789,000.000"))
    print(convertNumbers("This is a test 123,456.789,000.000.000"))
    print(convertNumbers("A2.B is a friend of B2, but R2D2 is another robot"))
    print(convertNumbers("I have 10 friends, one of them is 2-years-old and the other is called 15 (or ichigo in japanese)"))
    print(convertNumbers("We are on 2021, the year of the robot"))

if __name__ == '__main__':
    # test_all()
    #test_lemma()
    test_numbers()