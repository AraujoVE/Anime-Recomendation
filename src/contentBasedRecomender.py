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

# def execute(params: ExecutionParams, bow_df: pd.DataFrame):
#     print(f"[ContentBasedRecomender] Executing for {params.animeName}")

#     print("[ContentBasedRecomender] Creating TF-IDF matrix")
#     # Defining a TF-IDF Vectorizer Object.
#     tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

#     # Constructing the required TF-IDF matrix by fitting and transforming the data. 
#     # TF-IDF represents how important is a word in the phrase to a document.
#     tfidf_matrix = tfidf.fit_transform(bow_df['bag_of_words'])[:,1:]

#     tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names()[1:])
#     print(tfidf_df.head())
#     exit()
    
#     indices = pd.Series(bow_df.index, index=bow_df['title'])

#     anime_df = ANIME_DATASET.anime_df

#     featureNames = tfidf.get_feature_names_out()

#     kimetsu_index = bow_df.index[bow_df['title'] == 'Kimetsu no Yaiba'].tolist()[0]
#     print(f"Kimetsu index: {kimetsu_index}")
#     kimetsu_row = tfidf_matrix.toarray()[kimetsu_index:kimetsu_index+1]
#     kimetsu_row_list = kimetsu_row[0]
#     # Lists all non-zero feature names for kimetsu
#     kimetsu_features_idx = [i for i in range(len(kimetsu_row_list)) if kimetsu_row_list[i] != 0]
#     kimetsu_features = [featureNames[i] for i in range(len(kimetsu_row_list)) if kimetsu_row_list[i] != 0]
#     print(f"[ContentBasedRecomender] Kimetsu features idx: {kimetsu_features_idx}")
#     print(f"[ContentBasedRecomender] Kimetsu features: {kimetsu_features}")


#     print(tfidf_df[kimetsu_features])

#     exit()
    


   

#     anime_df = ANIME_DATASET.anime_df

#     kimestu_malid = datasetIndexToMALID(kimetsu_index)
#     print(f"[ContentBasedRecomender] Kimetsu MAL ID: {kimestu_malid}")
#     print(f"[ContentBasedRecomender] Kimetsu title: {anime_df[anime_df['MAL_ID'] == kimestu_malid]['Name'].values[0]}")

#     print(bow_df.iloc[kimetsu_index]['bag_of_words'])

#     # Save list of feature names
#     with open('featureNames.txt', 'w') as f:
#         for featureName in featureNames:
#             f.write(f'{featureName}\n')

#     # Creating list of indices for later matching
#     animeNames = indices.index.values

#     # # Computing the cosine similarity matrix: Option 2
#     # if cosine folder already exists, do not compute cosine similarity again
#     if not os.path.exists('cosine'):
#         print("[ContentBasedRecomender] Computing cosine similarity matrix...")
#         calcAnimeSimilarityMatrix(animeNames, tfidf_matrix, featureNames)
#     else:
#         print("[ContentBasedRecomender] Cosine similarity matrix already exists")
#     # Get most similar to chosen anime
#     print("[ContentBasedRecomender] Looking for similar anime...")
#     mostSimilar = getContentBasedRecommendation(
#         bow_df, 'cosine/merged.csv', indices, params.animeName, params.selectionRange)

#     return mostSimilar


# def main():
#     anime_dataset = AnimeDataset()
#     anime_df = anime_dataset.anime_df
#     bow_df = create_bow_df(anime_df)

#     params = ExecutionParams(
#         # 'Kiss x Sis (TV)', selectionRange=7
#         # 'Kimetsu no Yaiba', selectionRange=7
#         'Cowboy Bebop', selectionRange=7
#     )

#     result = execute(params, bow_df)
#     print(result)


def test():
    recomender = ContentBasedRecomender(ANIME_DATASET)
    bow_df = recomender.bow_df

    unified_bow_series = bow_df[cols.BAG_OF_WORDS]

    all_keywords = []
    for bow in unified_bow_series:
        all_keywords.extend(bow.split(' '))

    print(all_keywords[:100])

if __name__ == "__main__":
    # main()
    test()