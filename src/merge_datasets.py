from matplotlib import use
import pandas as pd
from pathy import os

from synopsis_keywords import ANIME_SYNOPSIS_KEYWORDS_FILE, generateSynopsisKeywords

DATASET_FOLDER = "dataset"
ANIME_CSV = f'{DATASET_FOLDER}/anime.csv'
ANIME_WITH_SYNOPSIS_KEYWORDS_CSV = f'{DATASET_FOLDER}/anime_with_synopsis_keywords.csv'
ANIME_MERGED_CSV = f'{DATASET_FOLDER}/anime_merged.csv'

def merge_datasets():
    print('[MergeDatasets] Start')

    print('[MergeDatasets] Check if merged dataset exists')
    if os.path.exists(ANIME_MERGED_CSV):
        print('[MergeDatasets] Merged dataset exists. Skip merging')
        return

    print('[MergeDatasets] Merged dataset does not exist')

    print(f'[MergeDatasets] Checking if {ANIME_SYNOPSIS_KEYWORDS_FILE} exists')
    # Convert anime_with_synopsis.csv to anime_with_synopsis_keywords.csv
    if os.path.exists(ANIME_WITH_SYNOPSIS_KEYWORDS_CSV):
        print(f'[MergeDatasets] {ANIME_WITH_SYNOPSIS_KEYWORDS_CSV} exists')
    else:
        print(f'[MergeDatasets] {ANIME_WITH_SYNOPSIS_KEYWORDS_CSV} does not exist')
        generateSynopsisKeywords()

    # Merge everything together
    from cols import MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION, SYNOPSIS_KEYWORDS

    # Read anime.csv
    print(f'[MergeDatasets] Reading {ANIME_CSV}...')
    anime_cols = [ MAL_ID, NAME, SCORE, GENRES, TYPE, EPISODES, STUDIOS, PRODUCERS, SOURCE, DURATION ]
    anime = pd.read_csv(ANIME_CSV, usecols=anime_cols)

    # Read anime_with_synopsis.csv
    print(f'[MergeDatasets] Reading {ANIME_WITH_SYNOPSIS_KEYWORDS_CSV}...')
    anime_with_synopsis_cols = [ MAL_ID, SYNOPSIS_KEYWORDS ]
    anime_with_synopsis = pd.read_csv(ANIME_WITH_SYNOPSIS_KEYWORDS_CSV, usecols=anime_with_synopsis_cols)

    # Merge datasets
    print(f'[MergeDatasets] Merging {ANIME_CSV} and {ANIME_WITH_SYNOPSIS_KEYWORDS_CSV}...')
    anime_merged = pd.merge(anime, anime_with_synopsis, how='left', on=MAL_ID)

    print(f'[MergeDatasets] Writing {ANIME_MERGED_CSV}...')
    anime_merged.to_csv(ANIME_MERGED_CSV, index=False)

    print('[MergeDatasets] Done.')

if __name__ == "__main__":
    merge_datasets()