import re
from numpy.core.numeric import indices
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

def setKeywords(line, chossenCols):
    returnText = ''
    for col in chossenCols:
        tag = ' ' + col + '___'
        returnText += tag
        if isinstance(line[col], list):
            joinedTexts = tag.join(line[col])
            returnText += joinedTexts
            #if 'taiwan' in joinedTexts:
            #    print(returnText)
        else:
            returnText += str(line[col])
    returnText = re.sub(r'.*___unknown(?=$|\s)','',returnText).strip()
    return returnText

def getContentBasedRecommendation(df_final, cosine_sim, indices, title, selectionRange):
    movies = []

    if not title in indices['title'].tolist():
        return "Ops, title not in our database..."

    # Get the index of the movie that matches the title
    title_index = indices[indices['title'] == title].index[0]

    # Cosine similarity scores of anime titles in descending order (most similar is on top)
    scores = pd.Series(cosine_sim[title_index]).sort_values(ascending = False)

    # Top 'selectionRange' most similar anime indexes
    top_animes_rec = list(scores.iloc[1:selectionRange].index) # from 1 to 'selectionRange', because 0 is the searched title always
  
    return pd.DataFrame(df_final['title'].iloc[top_animes_rec]) # Return the titles of the top 'selectionRange' most similar anime

def getCossineWeights(df,colsIndices):
    weightsPerType = {
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
    totalWeights = sum(weightsPerType.values())
    for weight in weightsPerType.keys():
        weightsPerType[weight] /= totalWeights

    weights = []

    for col in colsIndices:
        if col == '':
            weights.append('none')
        else:
            weights.append(col.split('___')[0])
    
    for weight in weightsPerType.keys():
        weightsPerType[weight] /= weights.count(weight)
    
    for i in range(len(weights)):
        weights[i] = weightsPerType[weights[i]]        

    return np.array(weights)

def calcWeightedCosSim(tf_IdfMatrix,indices):
    linesNo = tf_IdfMatrix.shape[0]
    colsNo = tf_IdfMatrix.shape[1]
    cosine_sim = np.zeros((linesNo, linesNo))
    print("Começou :( ")
    cossineWeights = getCossineWeights(tf_IdfMatrix,indices)
    print("Acabou :D ")
    for i in range(linesNo):
        for j in range(i+1,linesNo):
            elem1 = tf_IdfMatrix[i] #TODO:Está em dimensão (1,colsNo). Converter para (colsNo,)
            elem2 = tf_IdfMatrix[j] #TODO:Está em dimensão (1,colsNo). Converter para (colsNo,)
            print("elem1 = ",elem1.shape)
            print("coss",cossineWeights.shape)
            cosine_sim[i,j] = cosine(elem1, elem2,w=cossineWeights)
            cosine_sim[j,i] = cosine_sim[i,j]
        cosine_sim[i,i] = 1
    return cosine_sim
