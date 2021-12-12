import nltk # type: ignore
from dataclasses import dataclass

@dataclass
class NLTK_Resource:
    path: str
    name: str

needed_resources = [
    NLTK_Resource('tokenizers/punkt', 'punkt'),
    NLTK_Resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    NLTK_Resource('corpora/wordnet', 'wordnet'),
    NLTK_Resource('corpora/stopwords', 'stopwords')
]

def download_nltk_resource_if_needed(resource: NLTK_Resource):
    try:
        nltk.data.find(resource.path)
    except LookupError:
        nltk.download(resource.name)

def download_all_nltk_resources_if_needed():
    for resource in needed_resources:
        download_nltk_resource_if_needed(resource)
