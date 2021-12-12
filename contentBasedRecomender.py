from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from getKeywords import *
from src.anime_dataset import read_anime_df
from src.utils import *

def create_bow_df(anime_df: pd.DataFrame):
    print("Creating BOW dataframe")
    # # Create bag of words for with many datas
    # bagOfWordsCols = ['Synopsis_Keywords', 'Genres', 'Studios',
    #                 'Producers', 'Type', 'Episodes', 'Source', 'Duration']
    bagOfWordsCols = ['Synopsis_Keywords', 'Genres', 'Studios', 'Producers', 'Type', 'Episodes', 'Source', 'Duration']

    bow_df = pd.DataFrame()
    bow_df['title'] = anime_df['Name']

    # Setting keywords
    bow_df['bag_of_words'] = anime_df.apply(createPrefixedKeywords, axis='columns', args=(bagOfWordsCols,))

    return bow_df


@dataclass
class ExecutionParams:
    animeName: str
    selectionRange: int

def execute(params: ExecutionParams, bow_df: pd.DataFrame):
    print(f"Executing (1) recomender for {params.animeName=} with selection range {params.selectionRange=}")

    print("Applying TfidfVectorizer")
    # Defining a TF-IDF Vectorizer Object.
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

    # Constructing the required TF-IDF matrix by fitting and transforming the data. TF-IDF represents how important is a word in the phrase to a document.
    tfidf_matrix = tfidf.fit_transform(bow_df['bag_of_words'])

    featureNames: List[str] = tfidf.get_feature_names()
    # TODO: remove '' feature from tfidf

    # Creating list of indices for later matching
    indices = pd.Series(bow_df.index, index=bow_df['title']).drop_duplicates().reset_index()

    colsNames = bow_df['bag_of_words'].values.tolist()

    print("Calculating cosine similarity")
    # Computing the cosine similarity matrix: Option 2
    cosine_sim = calcAnimeSimilarity(tfidf_matrix, featureNames)
    print(cosine_sim)

    print(f"Executing (2) recomender for {params.animeName=} with selection range {params.selectionRange=}")
    # Get most similar to chosen anime
    mostSimilar = getContentBasedRecommendation(
        bow_df, cosine_sim, indices, params.animeName, params.selectionRange)

    return mostSimilar


def main():
    anime_df = read_anime_df()
    bow_df = create_bow_df(anime_df)

    params = ExecutionParams(
        'Kiss x Sis (TV)', selectionRange=7
    )

    result = execute(params, bow_df)
    print(result)


def test():
    anime_df = read_anime_df()
    bow_df = create_bow_df(anime_df)
    bow_df.to_csv('bowdf.csv')
    

if __name__ == "__main__":
    main()
    # test()