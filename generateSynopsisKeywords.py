import pandas as pd
import numpy as np
from getKeywords import *
from textNormalization import dividedNormalizedText

cols_to_use =['MAL_ID', 'sypnopsis']
anime_df = pd.read_csv('dataset/anime_with_synopsis.csv', usecols = cols_to_use)

anime_df.dropna(inplace = True)
anime_df['sypnopsis'] = anime_df['sypnopsis'].astype(str)
anime_df = anime_df[(anime_df['sypnopsis'].isnull() == False) & (~anime_df['sypnopsis'].str.contains('No synopsis'))]

anime_df['Synopsis_Keywords'] = anime_df['sypnopsis'].apply(dividedNormalizedText)

#write csv file
anime_df.to_csv('dataset/anime_with_synopsis_keywords.csv', index = False)
