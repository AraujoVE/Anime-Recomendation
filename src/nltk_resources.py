import nltk # type: ignore
from dataclasses import dataclass

@dataclass
class NLTK_Resource:
    path: str
    name: str



PUNKT_RES = NLTK_Resource('tokenizers/punkt', 'punkt')
AVG_PERCEPTRON_TAGGER_RES = NLTK_Resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
WORDNET_RES = NLTK_Resource('corpora/wordnet', 'wordnet')
STOPWORDS_RES = NLTK_Resource('corpora/stopwords', 'stopwords')

all_resources = [
    PUNKT_RES,
    AVG_PERCEPTRON_TAGGER_RES,
    WORDNET_RES,
    STOPWORDS_RES
]

def download_nltk_resource_if_needed(resource: NLTK_Resource):
    try:
        nltk.data.find(resource.path)
    except LookupError:
        nltk.download(resource.name)

def download_all_nltk_resources_if_needed():
    for resource in all_resources:
        assert type(resource) == NLTK_Resource, f'resource {resource} must be of type NLTK_Resource, got {type(resource)}'
        download_nltk_resource_if_needed(resource)
