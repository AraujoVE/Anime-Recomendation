from typing import List
import numpy as np

def calcFeatureWeights(features: List[str], colWeights: dict) -> np.ndarray:
    print('[ColWeights] Starting calculations...')
    

    def featureColName(feature: str):
        if feature == '': raise Exception('Empty feature')
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
    assert len(featureWeights) == len(features), "Weights and features are not the same length"

    assert (abs(sum(featureWeights) - 1) < 0.001), "Sum of weights is not 1"


    return featureWeights