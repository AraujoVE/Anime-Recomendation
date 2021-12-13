from dataclasses import dataclass
from pathy import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rake_keywords import getKeywords
from anime_dataset import ANIME_DATASET, AnimeDataset
from utils import *
import cols

class ContentBasedRecomender:
    def __init__(self, anime_dataset: AnimeDataset):
        self._set_default_prefs()

        self.anime_dataset = anime_dataset
        self.bow_df = self._create_bow_df()

    def _set_default_prefs(self) -> None:
        ''' Sets the default preferences for the recommender '''
        self.colsToUse = [cols.GENRES, cols.STUDIOS, cols.PRODUCERS, cols.TYPE, cols.EPISODES, cols.SOURCE, cols.DURATION]

    def _create_bow_df(self) -> pd.DataFrame:
        ''' Creates a dataframe with the bow (Bag of Words) representation of the anime titles '''
        colsToApplyBOW = [ cols.GENRES, cols.STUDIOS, cols.PRODUCERS, cols.TYPE, cols.EPISODES, cols.SOURCE, cols.DURATION]

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
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())

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

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())

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


def main():
    
    anime_dataset = ANIME_DATASET
    recomender = ContentBasedRecomender(anime_dataset)

    params = (
        # 'Kiss x Sis (TV)', 7
        # 'Kimetsu no Yaiba', 7
        # 'Cowboy Bebop', 7
        # 'Mai-Otome', 7
        # 'Shin Shirayuki-hime Densetsu Prétear', 7
        'Sen to Chihiro no Kamikakushi', 7
        # 'Air', 7
        # 'Blood+', 7
        # 'Futakoi', 7
        # 'Tokyo Underground', 7
        # 'GetBackers', 7
        # 'Green Green', 7
    )

    print('Anime info:')
    print(anime_dataset.get_by_name(params[0]))
    print('iloc = ', anime_dataset.convert_id_to_index(anime_dataset.get_by_name(params[0])[cols.MAL_ID]))
    input('Press enter to continue...')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 160)
    result = recomender.execute(*params)
    print (f'Final result: \n{result}')


def test():
    recomender = ContentBasedRecomender(ANIME_DATASET)
    bow_df = recomender.bow_df

    unified_bow_series = bow_df[cols.BAG_OF_WORDS]

    all_keywords = []
    for bow in unified_bow_series:
        all_keywords.extend(bow.split(' '))

    print(all_keywords[:100])

if __name__ == "__main__":
    main()
    # test()
    # print(ANIME_DATASET.convert_index_to_id(100))