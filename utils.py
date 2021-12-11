import re
from time import sleep
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from cosine import PartialMatrixCreationParams, create_weighted_cosine_similarity_matrix

animeDurations = [
    {'time': 120, 'type': 'curtíssimo'},
    {'time': 600, 'type': 'curto'},
    {'time': 1200, 'type': 'medio'},
    {'time': 2400, 'type': 'grande'},
    {'time': 4800, 'type': 'grandíssimo'}
]

biggestDurationType = 'maior'

def splitCols(df, cols, delimiter=','): 
    for col in cols:
        df[col] = df[col].apply(lambda x: [elem.strip('_') for elem in x.lower().split(delimiter)])
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

def removeInvalidRows(df, col, remove_rows_labels):
    for label in remove_rows_labels:
        df.drop(df[df[col] == label].index, inplace = True)
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

        prefix = ' ' + colName + '___'
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
            prefixedKeywordsStr = f'{prefix}{prefix.join(colKeywords)}' # Extra prefix in the beginning, because join doens't place it in the beginning

            # print(f'{colName} -> {prefixedKeywordsStr}')
            prefixedKeywordsList.append(prefixedKeywordsStr)

        else:
            # print(f'!!! Empty col: {colName} !!!')
            # print(f'Cell: {cell}')
            pass

    result = ''.join(prefixedKeywordsList)
    # print(f'Result: {result}')
    # sleep(1)
    return result

def getContentBasedRecommendation(df_bow: pd.DataFrame, cosine_sim, indices, title: str, selectionRange: int):
    print("getContentBasedRecommendation()")
    if not title in indices['title'].tolist():
        return "Ops, title not in our database..."

    # Get the index of the movie that matches the title
    title_index = indices[indices['title'] == title].index[0]

    # Cosine similarity scores of anime titles in descending order (most similar is on top)
    scores = pd.Series(cosine_sim[title_index]).sort_values(ascending = False)

    # Top 'selectionRange' most similar anime indexes
    top_animes_rec = list(scores.iloc[1:selectionRange].index) # from 1 to 'selectionRange', because 0 is the searched title always
  
    return pd.DataFrame(df_bow['title'].iloc[top_animes_rec]) # Return the titles of the top 'selectionRange' most similar anime

def getColWeights(_, prefixedKeywords: List[str]):

    print("getCossineWeights() - creating weights...")

    # TODO: assert the following dict is on the same order of the original columns
    colWeights = {
        'synopsis_keywords' : 100,
        'genres' : 10,
        'type' : 1,
        'episodes' : 1,
        'studios' : 1,
        'producers' : 1,
        'source' : 1,
        'duration' : 1,
        'none': 0
    }

    # Normalize the weights (sum of all weights = 1)
    totalWeights = sum(colWeights.values())
    for colName in colWeights.keys():
        colWeights[colName] /= totalWeights


    colFrequency = {}

    # Count the frequency of each column on the prefixedKeywords list
    for prefixedKeyword in prefixedKeywords:
        if prefixedKeyword == '':
            print("WARNING: Empty prefixedKeyword, this shouldn't happen!!") #TODO: remove this
            colName = 'none'
        else:
            colName, keyword = prefixedKeyword.split('___')

        if colName in colFrequency:
            colFrequency[colName] += 1
        else:
            colFrequency[colName] = 1

    # Reduce the weights of columns that are too frequent
    for colName, colFreq in colFrequency.items():
        if colFreq > 0:
            colWeights[colName] /= colFreq

    colNames = list(colWeights.keys()) # TODO: assert the following dict is on the same order of the original columns

    # Convert weights to a list
    weights = np.array([ colWeights[colName] for colName in colNames ])

    return weights

def calcAnimeSimilarity(tf_IdfMatrix, featureNames: List[str]):
    print("calcAnimeSimilarity()")

    print("calcAnimeSimilarity() - creating weights for each col...")
    cossineWeights = getColWeights(tf_IdfMatrix, featureNames)

    params = PartialMatrixCreationParams(
        partials_folder='cosine/partials',
        merged_filename='cosine/merged.csv',
    )

    np_tdidf: np.ndarray = tf_IdfMatrix.toarray()
    create_weighted_cosine_similarity_matrix(np_tdidf, cossineWeights, params)
