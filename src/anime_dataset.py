from typing import Optional
import pandas as pd
import cols
from utils import lowerCaseCols, removeInvalidRows, splitCols, standardizeDuration

DATASET_FOLDER = 'dataset'

class AnimeDataset:
    def __init__(self):
        from cols import MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS, SYNOPSIS_KEYWORDS
        anime_csv_cols = [ MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS, SYNOPSIS_KEYWORDS ]

        print('[Anime Dataset] Loading anime dataset...')
        self.anime_df = pd.read_csv(f'{DATASET_FOLDER}/anime_merged.csv', usecols=anime_csv_cols)
        print('[Anime Dataset] Preprocessing dataset...')
        self._preprocess_df()
        print('[Anime Dataset] Ready!')

    def _preprocess_df(self):
        self.anime_df = removeInvalidRows(self.anime_df, cols.GENRES, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.STUDIOS, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.PRODUCERS, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.SOURCE, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.TYPE, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.EPISODES, ['Unknown', 'unknown'])
        # self.anime_df = removeInvalidRows(self.anime_df, cols.DURATION, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.SYNOPSIS, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.SYNOPSIS_KEYWORDS, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.SCORE, ['Unknown', 'unknown'])

        self.anime_df.dropna(inplace=True) # drop rows with missing values
        self.anime_df = removeInvalidRows(self.anime_df, cols.TYPE,  [ 'Music', 'Unknown' ]) #TODO: string constants and GUI choice
        self.anime_df.drop_duplicates(subset=['Name'], inplace=True)

        lowerCasedCols = ['Synopsis_Keywords', 'Genres', 'Type',
                            'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
        self.anime_df = lowerCaseCols(self.anime_df, lowerCasedCols)
        self.anime_df[cols.DURATION] = self.anime_df[cols.DURATION].apply(standardizeDuration)

        pluralCols = [ cols.SYNOPSIS_KEYWORDS, cols.GENRES, cols.STUDIOS, cols.PRODUCERS ]
        self.anime_df = splitCols(self.anime_df, pluralCols)

        self.anime_df.reset_index(inplace=True, drop=True)

    def get_by_name(self, name: str) -> Optional[pd.Series]:
        results = self.anime_df[self.anime_df[cols.NAME] == name]
        if len(results) == 0:
            return None
        return results.iloc[0]

    def get_by_id(self, mal_id: int) -> Optional[pd.Series]:
        retults = self.anime_df[self.anime_df[cols.MAL_ID] == mal_id]
        if len(retults) == 0:
            return None
        return retults.iloc[0]

    def search_by_name(self, name: str) -> pd.DataFrame:
        results = self.anime_df[self.anime_df[cols.NAME].str.contains(name, case=False)]
        return results

    def convert_index_to_id(self, index: int) -> int:
        return self.anime_df.iloc[index][cols.MAL_ID]

    def convert_id_to_index(self, mal_id: int) -> int:
        return self.anime_df[self.anime_df[cols.MAL_ID] == mal_id].index[0]

ANIME_DATASET = AnimeDataset()


def test():
    kimetsu = 'Kimetsu no Yaiba'
    search_results = ANIME_DATASET.get_by_name(kimetsu)
    pd.set_option('display.max_colwidth', -1)
    print(search_results)

    

if __name__ == '__main__':
    test()