from dataclasses import dataclass
from enum import Enum
import pandas as pd

DATASET_FOLDER = 'dataset'

class AnimeDatasetCol(Enum):
    MAL_ID = 0
    NAME = 1
    SCORE = 2
    GENRES = 3
    TYPE = 4
    EPISODES = 5
    STUDIOS = 6
    PRODUCERS = 7
    SOURCE = 8
    DURATION = 9


class AnimeDataset:
    def __init__(self):
        self.anime_df = pd.read_csv(f'{DATASET_FOLDER}/anime.csv')

    pass

def read_anime_df():
    print("Reading anime data")
    cols_to_use = ['MAL_ID', 'Synopsis_Keywords']
    sypnopsis = pd.read_csv(
        'dataset/anime_with_synopsis_keywords.csv', usecols=cols_to_use)

    # Get anime data
    cols_to_use = ['MAL_ID', 'Name', 'Score', 'Genres', 'Type',
                'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
    anime_data = pd.read_csv('dataset/anime.csv', usecols=cols_to_use)

    # Merge synopsis and anime data
    anime_df = anime_data.merge(sypnopsis, how='left', on='MAL_ID')
    anime_df.dropna(inplace=True)

    # Remove rows with 'Music' and 'Unknown' type
    remove_label = ['Music', 'Unknown']
    anime_df = removeInvalidRows(anime_df, 'Type', remove_label)

    # Remove duplicates
    anime_df.drop_duplicates(subset='Name', inplace=True)

    # Set data to lower case and replace spaces with underscores
    lowerCasedCols = ['Synopsis_Keywords', 'Genres', 'Type',
                    'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
    lowerCaseCols(anime_df, lowerCasedCols)

    # Standardize durations
    anime_df["Duration"] = anime_df["Duration"].apply(standardizeDuration)

    # Split given cols into lists
    toSplitCols = ['Synopsis_Keywords', 'Genres', 'Studios', 'Producers']
    splitCols(anime_df, toSplitCols)

    return anime_df