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
            prefixedKeywordsList.append(prefixedKeywordsStr)

        else:
            # print(f'!!! Empty col: {colName} !!!')
            # print(f'Cell: {cell}')
            pass

    result = ' '.join(prefixedKeywordsList)
    return result

def getContentBasedRecommendation(df_bow: pd.DataFrame, merged_cos_sim_filename: str, indices: pd.Series, title: str, selectionRange: int):
    print("getContentBasedRecommendation()")
    if not title in indices.index.tolist():
        return "Ops, title not in our database..."

    # Get the index of the movie that matches the title
    title_index = indices[title] + 1 # +1 account for the header

    with open(merged_cos_sim_filename, 'rb') as cos_sim_file:
        cos_sim_matrix_sliced = pd.read_csv(cos_sim_file, skiprows=title_index, nrows=1, header=None)
        cos_sim_series = pd.Series(cos_sim_matrix_sliced.iloc[0])


    # Reorder it by similarity, excluding itself
    cos_sim_series.sort_values(ascending=False, inplace=True)
    cos_sim_series_most_similar = cos_sim_series[0:selectionRange+1]

    # Get first selectionRange titles
    print("top_similar_animes cos: ", cos_sim_series_most_similar)
    top_similar_animes_indices = cos_sim_series_most_similar.index.tolist()


    our_anime = df_bow.iloc[title_index]
    print("our_anime: \n", our_anime)
    print("bag of words: \n", our_anime['bag_of_words'])
    print()

    # Get the actual titles of the selectionRange most similar movies
    similiar_animes = df_bow.iloc[top_similar_animes_indices]
    for sim_anime in similiar_animes.itertuples():
        print(sim_anime[1])
        print("Bag of words: \n", sim_anime[2])
        print()


    return pd.DataFrame(similiar_animes) # Return the titles of the top 'selectionRange' most similar anime

def getColWeights(_, features: List[str]):
    print('[ColWeights] Starting calculations...')
    # TODO: assert the following dict is on the same order of the original columns
    colWeights = {
        'synopsis_keywords' : 0,
        'genres' : 10,
        'type' : 0,
        'episodes' : 0,
        'studios' : 0,
        'producers' : 0,
        'source' : 0,
        'duration' : 0,
        'none': 0
    }

    def featureColName(feature: str):
        if feature == '': return 'none'
        return feature.split('___')[0]

    # Normalize the weights (sum of all weights = 1)
    print('[ColWeights] Normalizing input weights...')
    
    activeCols = { featureColName(feature) for feature in features if featureColName(feature) in colWeights }
    inactiveCols = { colName for colName in colWeights if colName not in activeCols }


    print('[ColWeights] Active cols: ', activeCols)
    totalActiveColsWeight = sum([colWeights[col] for col in activeCols])
    print('[ColWeights] Total active cols weight: ', totalActiveColsWeight)
    for colName in activeCols:
        colWeights[colName] /= totalActiveColsWeight

    for colName in inactiveCols:
        colWeights[colName] = 0

    print('[ColWeights] Normalized col weights: ', colWeights)
    assert (abs(sum(colWeights.values()) - 1) < 0.001), "Sum of weights is not 1"

    # Calculate the weights of each column
    print('[ColWeights] Analyzing col frequencies...')
    colFrequency = { featureColName(feature): 0 for feature in features }

    for feature in features:
        colFrequency[featureColName(feature)] += 1

    # Reduce the weights of columns that are too frequent
    # Count the frequency of each column on the feature list
    print('[ColWeights] Reducing weights...')
    for colName, colFreq in colFrequency.items():
        if colFreq > 0:
            colWeights[colName] /= colFreq
 

    print(colWeights)

    # Convert weights to a list
    print('[ColWeights] Expand weights to match features...')
    featureWeights = np.array([ colWeights[featureColName(feature)] for feature in features ])

    assert (abs(sum(featureWeights) - 1) < 0.001), "Sum of weights is not 1"

    return featureWeights

def calcAnimeSimilarityMatrix(animeNames: List[str], tf_IdfMatrix, featureNames: List[str]):
    print('[CalcAnimeSimilarity] - Starting...')

    print('[CalcAnimeSimilarity] - Calculating cosine similarity...')
    colWeights = getColWeights(tf_IdfMatrix, featureNames)

    params = PartialMatrixCreationParams(
        partials_folder='cosine/partials',
        merged_filename='cosine/merged.csv',
        step_size=100,
    )

    np_tfIdf: np.ndarray = tf_IdfMatrix.toarray()

    print('[CalcAnimeSimilarity] - Creating partial cosine similarity matrices...')
    create_weighted_cosine_similarity_matrix(
        headers=animeNames,
        matrix=np_tfIdf,
        params=params,
        colWeights=colWeights
    )