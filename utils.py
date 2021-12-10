import re
from time import sleep
from typing import List
import pandas as pd
import numpy as np
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
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

def getCossineWeights(_, featureNames: List[str]):
    print("getCossineWeights() - creating weights...")
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

    # Normalize the weights
    totalWeights = sum(colWeights.values())
    for colName in colWeights.keys():
        colWeights[colName] /= totalWeights

    foundColNames = []
    
    for col in featureNames:
        if col == '':
            print ("Empty col still exists!")
            foundColNames.append('none')
        else:
            foundColNames.append(col.split('___')[0])

    for colName in colWeights.keys():
        count = foundColNames.count(colName)
        if count > 0:
            colWeights[colName] /= count

    weights = np.zeros(len(foundColNames))
    for i in range(len(foundColNames)):
        weights[i] = colWeights[foundColNames[i]]        

    print("getCossineWeights() - weights created!")
    return weights

def calcWeightedCosSim(tf_IdfMatrix, featureNames: List[str]):
    print("calcWeightedCosSim() - Calculating cosine similarity...")

    linesNo = tf_IdfMatrix.shape[0]
    # colsNo = tf_IdfMatrix.shape[1]
    # cosine_sim = np.zeros((linesNo, linesNo))
    cossineWeights = getCossineWeights(tf_IdfMatrix, featureNames)
    # for i in range(linesNo):
    #     # print(f'{i=}')
    #     for j in range(i+1,linesNo):
    #         # elem1 = tf_IdfMatrix[i].toarray().flatten()
    #         # elem2 = tf_IdfMatrix[j].toarray().flatten()
    #         # cosine_sim[i,j] = cosine(elem1, elem2,w=cossineWeights)
    #     #     cosine_sim[j,i] = cosine_sim[i,j]
    #     # cosine_sim[i,i] = 1
    #         pass

    # pass

    np_tdidf: np.ndarray = tf_IdfMatrix.toarray()

    # Applying weights to u and v instead of in the cosine function (it was too slow)
    # https://www.tutorialguruji.com/python/how-does-the-parameter-weights-work-in-scipy-spatial-distance-cosine/
    # https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity/448904#448904

    squareRootWeights = np.sqrt(cossineWeights)
    np_tdidf_weighted = np_tdidf * squareRootWeights

    print(f"** CALCULATING COSINE SIMILARITY **")
    print(f"This part takes a while...")
    # in cosine_similarity(matrixA, matrixB = None) if matrixB is None, 
    # it calculates the cosine similarity between matrixA and matrixA
    cosine_sim = cosine_similarity(np_tdidf) # Cosine similarity matrix between lines of the same matrix

    print("calcWeightedCosSim() - Done!")
    return cosine_sim
