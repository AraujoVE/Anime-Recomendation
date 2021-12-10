import re
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

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

def createPrefixedKeywords(line: pd.Series, chosenCols: List[str]):
    # Cells from a column 'col' become 'col___value'
    # They are all append to a string, separated by spaces

    prefixedKeywords = []
    for colName in chosenCols:
        prefix = ' ' + colName + '___'
        cell = line[colName]

        localKeywords = []
        if isinstance(cell, list):
            localKeywords = cell
        elif isinstance(cell, str):
            localKeywords = [cell]        


        prefixedKeywords.append(prefix.join(localKeywords))

    filter_unknown = lambda keyword: not keyword.startswith('unknown')
    filteredKeywords = filter(filter_unknown, prefixedKeywords)

    return list(filteredKeywords)

def getContentBasedRecommendation(df_bow: pd.DataFrame, cosine_sim, indices, title: str, selectionRange: int):
    if not title in indices['title'].tolist():
        return "Ops, title not in our database..."

    # Get the index of the movie that matches the title
    title_index = indices[indices['title'] == title].index[0]

    # Cosine similarity scores of anime titles in descending order (most similar is on top)
    scores = pd.Series(cosine_sim[title_index]).sort_values(ascending = False)

    # Top 'selectionRange' most similar anime indexes
    top_animes_rec = list(scores.iloc[1:selectionRange].index) # from 1 to 'selectionRange', because 0 is the searched title always
  
    return pd.DataFrame(df_bow['title'].iloc[top_animes_rec]) # Return the titles of the top 'selectionRange' most similar anime

def getCossineWeights(df, colsIndices: List[str]):
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
    print(f'{colsIndices=}')
    for col in colsIndices:
        # print(f'{col=}')
        if col == '':
            print ("Empty col still exists!")
            foundColNames.append('none')
        else:
            foundColNames.append(col.split('___')[0])
    
    # for colName in colWeights.keys():
    #     colWeights[colName] /= foundColNames.count(colName)
    
    for i in range(len(foundColNames)):
        foundColNames[i] = colWeights[foundColNames[i]]        

    return np.array(foundColNames)

def calcWeightedCosSim(tf_IdfMatrix, featureNames: List[str]):
    linesNo = tf_IdfMatrix.shape[0]
    colsNo = tf_IdfMatrix.shape[1]
    cosine_sim = np.zeros((linesNo, linesNo))
    print("Começou :( ")
    cossineWeights = getCossineWeights(tf_IdfMatrix, featureNames)
    print("Acabou :D ")
    for i in range(linesNo):
        for j in range(i+1,linesNo):
            elem1 = tf_IdfMatrix[i].toarray().flatten()
            elem2 = tf_IdfMatrix[j].toarray().flatten()
            cosine_sim[i,j] = cosine(elem1, elem2,w=cossineWeights)
            cosine_sim[j,i] = cosine_sim[i,j]
        cosine_sim[i,i] = 1
    return cosine_sim