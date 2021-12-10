from re import split
import pandas as pd
import numpy as np
from getKeywords import *
from utils import *

# Get synopsis keywords data
cols_to_use =['MAL_ID','Synopsis_Keywords']
sypnopsis = pd.read_csv('dataset/anime_with_synopsis_keywords.csv', usecols = cols_to_use)

# Get anime data
cols_to_use =['MAL_ID', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
anime_data = pd.read_csv('dataset/anime.csv', usecols = cols_to_use)

# Merge synopsis and anime data
anime_df = anime_data.merge(sypnopsis, how = 'left', on = 'MAL_ID')
anime_df.dropna(inplace = True)

# Remove rows with 'Music' and 'Unknown' type
remove_label = [ 'Music', 'Unknown' ]
anime_df = removeInvalidRows(anime_df, 'Type', remove_label)

# Remove duplicates
anime_df.drop_duplicates(subset = 'Name', inplace = True)

# Set data to lower case and replace spaces with underscores
lowerCasedCols = ['Synopsis_Keywords', 'Genres', 'Type', 'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
lowerCaseCols(anime_df,lowerCasedCols)

# Standardize durations
anime_df["Duration"] = anime_df["Duration"].apply(standardizeDuration)

# Split given cols into lists
toSplitCols = ['Synopsis_Keywords', 'Genres', 'Studios', 'Producers']
splitCols(anime_df, toSplitCols)

# Create bag of words for with many datas
#bagOfWordsCols = ['Synopsis_Keywords', 'Genres', 'Type', 'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
bagOfWordsCols = ['Synopsis_Keywords','Genres','Studios','Producers','Type', 'Episodes', 'Source', 'Duration']


df_final = pd.DataFrame()
df_final['title'] = anime_df['Name']

# Setting keywords
df_final['bag_of_words'] = anime_df.apply(setKeywords, axis = 'columns', args = (bagOfWordsCols,))

from sklearn.feature_extraction.text import TfidfVectorizer


# Defining a TF-IDF Vectorizer Object.
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

# Constructing the required TF-IDF matrix by fitting and transforming the data. TF-IDF represents how important is a word in the phrase to a document.
tfidf_matrix = tfidf.fit_transform(df_final['bag_of_words'])

featureNames = tfidf.get_feature_names()
#remove '' feature from tfidf

#print("bla")
#print(featureNames)
#print("blu")

# apply get_feature_names_out



#print(tfidf_matrix)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity matrix: Option 1
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Creating list of indicess for later matching
indicess = pd.Series(df_final.index, index = df_final['title']).drop_duplicates().reset_index()

colsNames = df_final['bag_of_words'].values.tolist()


# Computing the cosine similarity matrix: Option 2
cosine_sim = calcWeightedCosSim(tfidf_matrix,featureNames)



# Set parameters
chosenAnime = "Kiss x Sis (TV)"
selectionRange = 7
print(cosine_sim)
# Get most similar to chosen anime
mostSimilar = getContentBasedRecommendation(df_final,cosine_sim,indicess,chosenAnime, selectionRange)

# Print results
print(mostSimilar)