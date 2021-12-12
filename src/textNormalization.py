import re
from sys import prefix
from typing import Any, List
from num2words import num2words  # type: ignore
import contractions  # type: ignore
import string
import unicodedata

import nltk  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from numpy import number  # type: ignore

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
        wordsStr = num2words(int(numberStr), lang='en').replace(" ", "-").replace(",", "")
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

# Join tokens into a single string (e.g. ["hello", "world"] -> "hello world")
def join_tokens(tokens: List[str], delimiter: str = " "):
    assert isinstance(delimiter, str), f"delimiter must be a string, not {type(delimiter)}"
    assert isinstance(tokens, list), f"tokens must be a list, not {type(tokens)}"

    return delimiter.join(tokens) if tokens else ""

# Filter out tokens with POS (Part of Speech) tag not in acceptedPosTags
def filter_tokens_by_pos(tokens: List[str], acceptedPosTags: List[str]) -> List[str]:
    token_pos_pairs = nltk.pos_tag(tokens)
    return [token for (token, pos) in token_pos_pairs if pos in acceptedPosTags]

# Lemmatize tokens (i.e. convert verbs, nouns, adjectives to their base form)
def lemmatizeTokens(tokens: List[str]):
    def convert_treebank_wordnet(treebankPos: str):
        if treebankPos.startswith('J'):
            return wordnet.ADJ
        elif treebankPos.startswith('V'):
            return wordnet.VERB
        elif treebankPos.startswith('N'):
            return wordnet.NOUN
        elif treebankPos.startswith('R') or treebankPos == 'WRB':
            return wordnet.ADV
        else:
            return None


    lemmatizer = WordNetLemmatizer()
    def lemmatize_token(token: str, pos: str) -> str:
        assert token is not None, f"token must not be None"
        assert pos is not None, f"pos must not be None"

        wordnet_pos = convert_treebank_wordnet(pos)
        if wordnet_pos is None:
            return token # Wordnet does not know this POS tag, so it will not lemmatize it
        else:
            return lemmatizer.lemmatize(token, wordnet_pos)
            

    # Get token POS (Part of Speech) tag
    token_pos_pairs = nltk.pos_tag(tokens)

    # Lemmatize tokens
    return [ lemmatize_token(token, pos) for (token, pos) in token_pos_pairs]

# Lemmatize text (tokens separated by space)
def lemmatizeText(text: str) -> List[str]:
    return lemmatizeTokens(tokenize_text(text))

# Apply all text normalization functions
def normalizeTextToKeywords(text: str, delimiter: str = ",") -> str:
    # TODO: move acceptedPOSTags to GUI config
    # Completar com todas as possibilidades https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b

    acceptedAdjectives = ['JJ', 'JJR', 'JJS']
    acceptedNouns = ['NN', 'NNS', 'NNP', 'NNPS']
    acceptedVerbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # acceptedAdverbs = ['RB', 'RBR', 'RBS']
    # acceptedPrepositions = ['IN']
    # acceptedConjunctions = ['CC']
    # acceptedPronouns = ['PRP', 'PRP$', 'WP', 'WP$']
    # acceptedDeterminers = ['DT']
    acceptedNumbers = ['CD']
    # acceptedParticles = ['RP']
    # acceptedOther = ['UH', 'MD', 'EX', 'POS', 'FW']

    acceptedPosTags = acceptedAdjectives + acceptedVerbs + acceptedNumbers + acceptedNouns + ['FW'] #foreing words

    # Text normalization
    text = lowerCase(text)
    text = trimSpaces(text)
    text = replaceContractions(text)
    text = removeDoubleSpaces(text)

    # Token normalization
    tokens = tokenize_text(text)
    assert None not in tokens, f"tokens contains None: {tokens}"
    tokens = filter_tokens_by_pos(tokens, acceptedPosTags)
    assert None not in tokens, f"tokens contains None: {tokens}"
    tokens = lemmatizeTokens(tokens)
    assert None not in tokens, f"tokens contains None: {tokens}"
    tokens = removeStopWords(tokens)
    assert None not in tokens, f"tokens contains None: {tokens}"


    # Final text normalization
    text = join_tokens(tokens)
    text = removePunctuation(text)
    text = convertNumbers(text)
    text = removeDoubleSpaces(text)
    text = trimSpaces(text)

    text.replace(" ", delimiter)

    return text


texts = [
    "This is a test",
    "I'm testing this",
    "This, is a test!",
    "Tanjiro is a good person that walks with his friends",
    "I was walking with my dog while I was eating",
    "This is a test with numbers 123",
    "This is a test with numbers 123,456",
    "This is a test with numbers 123,456,789",
    "This is a test with numbers 123,456,789,000",
    "A2.B and A.2B are not the same!",
    "A2.B is a friend of B2, but R2D2 is another robot",
    "I have 10 friends, one of them is 2-years-old and the other is called 15 (or ichigo in japanese)",
    "We are on 2021, the year of the robot",
]

def test_lowerCase():
    for text in texts:
        print(f"lowerCase() with: {text} = \n{lowerCase(text)}")

def test_trimSpaces():
    for text in texts:
        print(f"trimSpaces() with: {text} = \n{trimSpaces(text)}")
    
def test_removeDoubleSpaces():
    for text in texts:
        print(f"removeDoubleSpaces() with: {text} = \n{removeDoubleSpaces(text)}")

def test_replaceContractions():
    for text in texts:
        print(f"replaceContractions() with: {text} = \n{replaceContractions(text)}")

def test_removePunctuation():
    for text in texts:
        print(f"removePunctuation() with: {text} = \n{removePunctuation(text)}")

def test_removeStopWords():
    for text in texts:
        print(f"removeStopWords() with: {text} = \n{removeStopWords(tokenize_text(text))}")

def test_tokenize_text():
    for text in texts:
        print(f"tokenize_text() with: {text} = \n{tokenize_text(text)}")

def test_filter_tokens_by_pos():
    for text in texts:
        print(f"filter_tokens_by_pos() with: {text} = \n{filter_tokens_by_pos(tokenize_text(text), ['NN'])}")

def test_lemmatizeTokens():
    for text in texts:
        print(f"lemmatizeTokens() with: {text} = \n{lemmatizeTokens(tokenize_text(text))}")

def test_lemmatizeText():
    for text in texts:
        print(f"lemmatizeText() with: {text} = \n{lemmatizeText(text)}")


def test_join_tokens():
    for text in texts:
        print(f"join_tokens() with: {tokenize_text(text)} = \n{join_tokens(tokenize_text(text))}")

def test_normalizeTextToKeywords():
    for text in texts:
        print(f"normalizeTextToKeywords() with: {text} = \n{normalizeTextToKeywords(text)}")


def test_all():
    print("Testing lowerCase()")
    test_lowerCase()

    print("Testing trimSpaces()")
    test_trimSpaces()

    print("Testing replaceContractions()")
    test_replaceContractions()

    print("Testing removePunctuation()")
    test_removePunctuation()

    print("Testing removeStopWords()")
    test_removeStopWords()

    print("Testing tokenize_text()")
    test_tokenize_text()

    print("Testing filter_tokens_by_pos()")
    test_filter_tokens_by_pos()

    print("Testing lemmatizeTokens()")
    test_lemmatizeTokens()

    print("Testing lemmatizeText()")
    test_lemmatizeText()

    print("Testing join_tokens()")
    test_join_tokens()

    print("Testing normalizeTextToKeywords()")
    test_normalizeTextToKeywords()

if __name__ == '__main__':
    # test_all()
    test_normalizeTextToKeywords()
