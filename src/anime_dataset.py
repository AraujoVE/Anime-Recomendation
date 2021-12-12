from unittest import case
import pandas as pd
import cols
from utils import lowerCaseCols, removeInvalidRows, splitCols, standardizeDuration

DATASET_FOLDER = 'dataset'

class AnimeDataset:
    def __init__(self):
        from cols import MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS_KEYWORDS
        anime_csv_cols = [ MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS_KEYWORDS ]

        self.anime_df = pd.read_csv(f'{DATASET_FOLDER}/anime_merged.csv', usecols=anime_csv_cols)
        self._preprocess_df()

    def _preprocess_df(self):
        self.anime_df.dropna(inplace=True) # drop rows with missing values
        self.anime_df = removeInvalidRows(self.anime_df, cols.TYPE,  [ 'Music', 'Unknown' ]) #TODO: string constants and GUI choice
        self.anime_df.drop_duplicates(subset=['Name'], inplace=True)

        lowerCasedCols = ['Synopsis_Keywords', 'Genres', 'Type',
                            'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
        lowerCaseCols(self.anime_df, lowerCasedCols)
        self.anime_df[cols.DURATION] = self.anime_df[cols.DURATION].apply(standardizeDuration)

        pluralCols = [ cols.SYNOPSIS_KEYWORDS, cols.GENRES, cols.STUDIOS, cols.PRODUCERS ]
        splitCols(self.anime_df, pluralCols)

    def search_by_name(self, name) -> pd.DataFrame:
        return self.anime_df[self.anime_df['Name'].str.contains(name, case=False)].sort_values(by=['Score'], ascending=False)

ANIME_DATASET = AnimeDataset()