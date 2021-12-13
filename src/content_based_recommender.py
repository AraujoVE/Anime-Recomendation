from dataclasses import dataclass
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rake_keywords import getKeywords
from anime_dataset import ANIME_DATASET, AnimeDataset
from utils import *
import cols

class ContentBasedRecommender:
    def __init__(self, anime_dataset: AnimeDataset):
        self._set_default_prefs()

        self.anime_dataset = anime_dataset
        self.bow_df = self._create_bow_df()

    def _set_default_prefs(self) -> None:
        ''' Sets the default preferences for the recommender '''
        self.colsToUse = [ cols.SYNOPSIS_KEYWORDS, cols.GENRES, cols.STUDIOS, cols.PRODUCERS, cols.TYPE, cols.EPISODES, cols.SOURCE, cols.DURATION]

    def _create_bow_df(self) -> pd.DataFrame:
        ''' Creates a dataframe with the bow (Bag of Words) representation of the anime titles '''
        colsToApplyBOW = self.colsToUse

        bow_df = pd.DataFrame()
        bow_df[cols.MAL_ID] = self.anime_dataset.anime_df[cols.MAL_ID]
        bow_df[cols.NAME] = self.anime_dataset.anime_df[cols.NAME]

        bow_df[cols.BAG_OF_WORDS] = self.anime_dataset.anime_df.apply(createPrefixedKeywords, axis='columns', args=(colsToApplyBOW,))

        return bow_df

    def _create_tfidf_matrix(self) -> pd.DataFrame:
        ''' Creates a dataframe with the tfidf representation of the anime titles '''
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.bow_df[cols.BAG_OF_WORDS])

        # Convert tifidf matrix to a dataframe
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        return tfidf_df

    def get_recommendations(self, mal_id: int, n: int = 10) -> pd.DataFrame:
        ''' Returns a dataframe with the top n recommendations for the given anime_id '''

        pass

    def execute(self, animeName: str, selectionRange: int):
        bow_df = self.bow_df
        print(f"[ContentBasedRecomender] Executing for {animeName}")

        print("[ContentBasedRecomender] Creating TF-IDF matrix")
        # Defining a TF-IDF Vectorizer Object.
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

        # Constructing the required TF-IDF matrix by fitting and transforming the data. 
        # TF-IDF represents how important is a word in the phrase to a document.
        tfidf_matrix = tfidf.fit_transform(bow_df[cols.BAG_OF_WORDS])
        featureNames = tfidf.get_feature_names_out()
        assert (len(featureNames) == tfidf_matrix.shape[1]), "Feature names and matrix dimensions do not match"

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

        # Creating list of indices for later matching
        indices = pd.Series(bow_df.index, index=bow_df[cols.NAME])
        animeNames = indices.index.values

        # # Computing the cosine similarity matrix: Option 2
        # if cosine folder already exists, do not compute cosine similarity again
        if not os.path.exists('cosine/merged.csv'):
            print("[ContentBasedRecomender] Computing cosine similarity matrix...")
            calcAnimeSimilarityMatrix(animeNames, tfidf_matrix, featureNames)
        else:
            print("[ContentBasedRecomender] Cosine similarity matrix already exists")
        # Get most similar to chosen anime
        print("[ContentBasedRecomender] Looking for similar anime...")
        mostSimilar = getContentBasedRecommendation(
            bow_df, 'cosine/merged.csv', indices, animeName, selectionRange)

        return mostSimilar



def prompt_anime(suggest_similar: bool = True) -> str:
    ''' Prompts the user for a valid anime name '''
    while True:
        anime_name = input(">>> Enter anime name: ")

        exact_anime = ANIME_DATASET.get_by_name(anime_name)
        if exact_anime is not None:
            return anime_name

        if suggest_similar:
            print("[ContentBasedRecomender] Anime not found, did you mean one of these?")
            print_similar_anime(anime_name)
        else:
            print("[ContentBasedRecomender] Anime not found")
            

def print_similar_anime(anime_name: str):
    search_results = ANIME_DATASET.search_by_name(anime_name)
    if not search_results.empty:
        print(search_results[[cols.MAL_ID, cols.NAME]])
    else:
        print("[ContentBasedRecomender] No similar anime found")

def main():
    # Initialize recommender and dataset
    print("[ContentBasedRecomender] Initializing...")
    anime_dataset = ANIME_DATASET
    recomender = ContentBasedRecommender(anime_dataset)

    # Prompt user for anime name
    anime_name = prompt_anime()

    print(f"[ContentBasedRecomender] Selected anime: {anime_name}")

    result = recomender.execute(anime_name, selectionRange=7)

    print("Similar animes:")
    print(result)

if __name__ == "__main__":
    main()