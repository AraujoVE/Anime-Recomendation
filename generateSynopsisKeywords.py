import pandas as pd
import numpy as np
from getKeywords import *
from textNormalization import normalizeWithDelimiter

from nltk_resources import download_all_nltk_resources_if_needed
download_all_nltk_resources_if_needed() # TODO: download only needed resources

ANIME_SYNOPSIS_FILE = 'anime_synopsis.csv'

cols_to_use = ['MAL_ID', 'sypnopsis']
anime_df = pd.read_csv(ANIME_SYNOPSIS_FILE, usecols = cols_to_use)

anime_df.dropna(inplace = True)
anime_df['sypnopsis'] = anime_df['sypnopsis'].astype(str)
anime_df = anime_df[(anime_df['sypnopsis'].isnull() == False) & (~anime_df['sypnopsis'].str.contains('No synopsis'))]

anime_df['Synopsis_Keywords'] = anime_df['sypnopsis'].apply(normalizeWithDelimiter)

#write csv file
anime_df.to_csv(ANIME_SYNOPSIS_FILE, index = False)
