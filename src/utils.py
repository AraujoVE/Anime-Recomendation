import re
from time import sleep
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cols

from cosine import PartialMatrixCreationParams, create_weighted_cosine_similarity_matrix

animeDurations = [
    {'time': 120, 'type': 'curtíssimo'},
    {'time': 600, 'type': 'curto'},
    {'time': 1200, 'type': 'medio'},
    {'time': 2400, 'type': 'grande'},
    {'time': 4800, 'type': 'grandíssimo'}
]

biggestDurationType = 'maior'

def splitCols(df, cols, list_delimiter=','): 
    for col in cols:
        df[col] = df[col].apply(lambda cell: [item.strip('_') for item in cell.lower().split(list_delimiter)])
    return df

def standardizeDuration(duration):
    #print(duration)
    if duration == 'unknown':
        #print('Duration is unknown')
        return duration

    hours = re.match(r'(\d+)_hr\.',duration)
    if hours:
        hours = 60 * 60 * int(hours.group(1))
    else:
        hours = 0

    minutes = re.match(r'(\d+)_min\.',duration)
    if minutes:
        minutes = 60 * int(minutes.group(1))
    else:
        minutes = 0

    seconds = re.match(r'(\d+)_sec\.',duration)
    if seconds:
        seconds = int(seconds.group(1))
    else:
        seconds = 0
    
    time = str(hours + minutes + seconds)
    '''
    timeName = biggestDurationType
    for durationType in animeDurations:
        if time <= durationType['time']:
            timeName = durationType['type']
            break

    return timeName
    '''
    return time

def removeInvalidRows(df: pd.DataFrame, col: str, remove_rows_labels: List[str]):
    for label in remove_rows_labels:
        matches = df[col] == label
        df.drop(df[matches].index, inplace=True)
    return df

def lowerCaseCols(df,cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: re.sub(r'\s+','_',x.lower().strip()))
    return df

def createPrefixedKeywords(line: pd.Series, chosenCols: List[str]) -> str:
    # Cells from a column 'col' become 'col___value'
    # They are all append to a string, separated by spaces

    def filter_unknown (keyword: str): 
        return not keyword.strip().lower().startswith('unknown')

    prefixedKeywordsList = []
    for colName in chosenCols:

        prefix = colName + '___'
        cell = line[colName]

        colKeywords = []
        if isinstance(cell, list):
            colKeywords = cell
        elif isinstance(cell, str):
            colKeywords = [cell]
        else:
            print("Unexpected type: ", type(cell))
            raise Exception("Unexpected type: ", type(cell))

        colKeywords = list(filter(filter_unknown, colKeywords))

        if colKeywords:
            prefixedColKeywords = [prefix + keyword for keyword in colKeywords]
            prefixedKeywordsList.extend(prefixedColKeywords)

        else:
            # print(f'!!! Empty col: {colName} !!!')
            # print(f'Cell: {cell}')
            pass


    result = ' '.join(prefixedKeywordsList)
    return result

def getContentBasedRecommendation(df_bow: pd.DataFrame, merged_cos_sim_filename: str, indices: pd.Series, title: str, selectionRange: int):
    if not title in indices.index.tolist():
        raise Exception(f'Title {title} not found in indices')

    # Get the index of the movie that matches the title
    title_index = indices[title] + 1 # +1 account for the header

    with open(merged_cos_sim_filename, 'rb') as cos_sim_file:
        cos_sim_matrix_sliced = pd.read_csv(cos_sim_file, skiprows=title_index, nrows=1, header=None)
        cos_sim_series = pd.Series(cos_sim_matrix_sliced.iloc[0])

    # Reorder it by similarity, excluding itself
    cos_sim_series.sort_values(ascending=False, inplace=True)
    cos_sim_series_most_similar = cos_sim_series[1:selectionRange+1]

    # Get first selectionRange titles
    top_similar_animes_indices = cos_sim_series_most_similar.index.tolist()

    # Get the actual titles of the selectionRange most similar movies
    similiar_animes = df_bow.loc[top_similar_animes_indices]

    # Remove BOW column
    similiar_animes.drop(cols.BAG_OF_WORDS, axis=1, inplace=True)

    # Add cosine similarity to the dataframe
    similiar_animes['Similarity'] = cos_sim_series_most_similar.copy()

    return pd.DataFrame(similiar_animes) # Return the titles of the top 'selectionRange' most similar anime