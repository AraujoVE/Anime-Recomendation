from typing import Optional
from unittest import case
import pandas as pd
import cols
from utils import lowerCaseCols, removeInvalidRows, splitCols, standardizeDuration

DATASET_FOLDER = 'dataset'

class AnimeDataset:
    def __init__(self):
        from cols import MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS, SYNOPSIS_KEYWORDS
        anime_csv_cols = [ MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS, SYNOPSIS_KEYWORDS ]

        self.anime_df = pd.read_csv(f'{DATASET_FOLDER}/anime_merged.csv', usecols=anime_csv_cols)
        self._preprocess_df()

    def _preprocess_df(self):
        # Drop if genre is unknown

        self.anime_df = removeInvalidRows(self.anime_df, cols.GENRES, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.STUDIOS, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.PRODUCERS, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.SOURCE, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.TYPE, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.EPISODES, ['Unknown', 'unknown'])
        self.anime_df = removeInvalidRows(self.anime_df, cols.DURATION, ['Unknown', 'unknown'])
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

        self.anime_df.to_csv(f'anime_merged_preprocessed.csv', index=False)

    def search_by_name(self, name) -> pd.DataFrame:
        return self.anime_df[self.anime_df['Name'].str.contains(name, case=False)].sort_values(by=['Score'], ascending=False)

    def searh_by_id(self, mal_id) -> Optional[pd.Series]:
        retults = self.anime_df[self.anime_df[cols.MAL_ID] == mal_id]
        if len(retults) == 0:
            return None
        return retults.iloc[0]

ANIME_DATASET = AnimeDataset()

def datasetIndexToMALID(index: int) -> int:
    return ANIME_DATASET.anime_df.iloc[index][cols.MAL_ID]

def getAnimeByMALID(mal_id: int) -> Optional[pd.Series]:
    return ANIME_DATASET.searh_by_id(mal_id)

if __name__ == '__main__':
    iloc = 597
    print(ANIME_DATASET.anime_df.iloc[iloc]['Name'])
    print(ANIME_DATASET.anime_df.iloc[iloc]['MAL_ID'])