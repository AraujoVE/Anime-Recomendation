import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from anime_dataset import ANIME_DATASET, AnimeDataset
from utils import *
import cols
import tfidf
import cosine

SIMILARITY_MATRIX_PATH = 'cosine/merged.csv'


class ContentBasedRecommender:
    def __init__(self, anime_dataset: AnimeDataset):
        self._set_default_prefs()

        self.anime_dataset = anime_dataset
        self.bow_df = self._create_bow_df()
        self._initialize_tfidf()

    def _set_default_prefs(self) -> None:
        ''' Sets the default preferences for the recommender '''
        self.colsToUse = [cols.SYNOPSIS_KEYWORDS, cols.GENRES, cols.STUDIOS,
                          cols.PRODUCERS, cols.TYPE, cols.EPISODES, cols.SOURCE, cols.DURATION]

        self.colWeights = {
            'synopsis_keywords': 100,
            'genres': 50,
            'type': 5,
            'episodes': 2,
            'studios': 2,
            'producers': 1,
            'source': 1,
            'duration': 1,
        }

    def _create_bow_df(self) -> pd.DataFrame:
        ''' Creates a dataframe with the bow (Bag of Words) representation of the anime titles '''
        colsToApplyBOW = self.colsToUse

        bow_df = pd.DataFrame()
        bow_df[cols.MAL_ID] = self.anime_dataset.anime_df[cols.MAL_ID]
        bow_df[cols.NAME] = self.anime_dataset.anime_df[cols.NAME]

        bow_df[cols.BAG_OF_WORDS] = self.anime_dataset.anime_df.apply(
            createPrefixedKeywords, axis='columns', args=(colsToApplyBOW,))

        return bow_df

    def _initialize_tfidf(self) -> None:
        ''' Creates a dataframe with the tfidf representation of the anime titles '''
        # Defining a TF-IDF Vectorizer Object.
        # TF-IDF represents how important is a word in the phrase to a document.
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(' '))

        # Constructing the required TF-IDF matrix by fitting and transforming the data.
        print("[ContentBasedRecomender] Creating TF-IDF matrix")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.bow_df[cols.BAG_OF_WORDS])

        self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        assert '' not in self.tfidf_feature_names
        assert (len(self.tfidf_feature_names) ==
                self.tfidf_matrix.shape[1]), "Feature names and matrix dimensions do not match"

    def _use_disk_similarity_matrix(self) -> None:
        ''' Creates a similarity matrix in disk for the anime titles if it does not exist '''
        print('[ContentBasedRecomender] - Looking for similarity matrix in disk')

        if os.path.exists(SIMILARITY_MATRIX_PATH):
            print(
                f"[CalcAnimeSimilarity] - Similarity matrix already exists at {SIMILARITY_MATRIX_PATH}")
            return

        print(
            f"[CalcAnimeSimilarity] - Similarity matrix does not exist at {SIMILARITY_MATRIX_PATH}")
        print("[CalcAnimeSimilarity] - Computing similarity matrix...")

        print('[CalcAnimeSimilarity] - Calculating feature weights...')

        feature_weights = tfidf.calcFeatureWeights(
            self.tfidf_feature_names, self.colWeights)

        # Applying weights to u and v instead of in the cosine function (it was too slow)
        # https://www.tutorialguruji.com/python/how-does-the-parameter-weights-work-in-scipy-spatial-distance-cosine/
        # https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity/448904#448904
        weighted_matrix = self.tfidf_matrix * np.sqrt(feature_weights) 
        #TODO: check if this is correct (csr_matrix, it was np.array before)

        params = PartialMatrixCreationParams(
            partials_folder='cosine/partials',
            merged_filename='cosine/merged.csv',
            step_size=100,
            keep_partials=False,
        )

        print('[CalcAnimeSimilarity] - Creating partial cosine similarity matrices...')
        cosine.create_cosine_similarity_matrix(
            headers='PLACEHOLDER_HEADER', #TODO: remove this placeholder
            matrix=weighted_matrix,
            params=params,
        )

    def _find_similar_anime(self, anime_index: int, count: int) -> pd.DataFrame:
        HEADER_PRESENT = True # TODO: remove this placeholder
        if HEADER_PRESENT:
            anime_index += 1

        # Reads the similarity matrix from disk in the specific line (anime_index)
        anime_simililarity_row = pd.Series(pd.read_csv(SIMILARITY_MATRIX_PATH, skiprows=anime_index, nrows=1, header=None).iloc[0])

        # Sorts the similarity matrix in descending order
        anime_simililarity_row.sort_values(ascending=False, inplace=True)

        # Excludes itself from the similarity matrix
        anime_simililarity_row.drop(anime_simililarity_row.index[anime_simililarity_row.index == anime_index], inplace=True)

        # Returns the top n most similar anime titles

        similar_anime_df = pd.DataFrame()
        similar_anime_df[cols.NAME] = self.anime_dataset.anime_df[cols.NAME].iloc[anime_simililarity_row.index[:count]]
        similar_anime_df['Similarity'] = anime_simililarity_row.iloc[:count]
        similar_anime_df[cols.GENRES] = self.anime_dataset.anime_df[cols.GENRES].iloc[anime_simililarity_row.index[:count]]

        return similar_anime_df

    def execute(self, animeName: str, selectionRange: int) -> pd.DateOffset:
        print(f"[ContentBasedRecomender] Executing for {animeName}")

        anime = ANIME_DATASET.get_by_name(animeName)

        self._use_disk_similarity_matrix()

        # Get most similar to chosen anime
        print("[ContentBasedRecomender] Looking for similar anime...")
        anime_index = ANIME_DATASET.convert_id_to_index(anime[cols.MAL_ID])
        similarity_df = self._find_similar_anime(anime_index, selectionRange)

        return similarity_df


def prompt_anime(suggest_similar: bool = True) -> str:
    ''' Prompts the user for a valid anime name '''
    while True:
        anime_name = input(">>> Enter anime name: ")

        exact_anime = ANIME_DATASET.get_by_name(anime_name)
        if exact_anime is not None:
            return anime_name

        if suggest_similar:
            print(
                "[ContentBasedRecomender] Anime not found, did you mean one of these?")
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

    # Get recommendations
    recommended_df = recomender.execute(anime_name, selectionRange=7)

    # Print results
    print("Similar animes:")
    print(recommended_df)


if __name__ == "__main__":
    main()
