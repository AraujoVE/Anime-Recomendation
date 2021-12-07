import pandas as pd
import numpy as np

cols_to_use =['MAL_ID', 'sypnopsis']
sypnopsis = pd.read_csv('anime_with_synopsis.csv', usecols = cols_to_use)

cols_to_use =['MAL_ID', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Studios', 'Producers', 'Source', 'Duration']
anime_data = pd.read_csv('anime.csv', usecols = cols_to_use)
anime_data.head()

display(sypnopsis.shape)
display(anime_data.shape)

# Merge the two datasets into 'anime_df'
anime_df = anime_data.merge(sypnopsis, how = 'left', on = 'MAL_ID')

display(anime_df.shape)
anime_df.head()

# Checking the types of the anime 
display(anime_df['Type'].value_counts())

def removeInvalidRows(df, col, remove_rows_labels):
  for label in remove_rows_labels:
    df.drop(df[df[col] == label].index, inplace = True)

  return df

def replaceInvalidRows(df, col, replace_rows_labels):
  return

# Removing NaNs
anime_df.dropna(inplace = True)

# Filtering the 'Music' and 'Unknown' types. We just want "watchable" content
remove_label = [ 'Music', 'Unknown' ]
anime_df = removeInvalidRows(anime_df, 'Type', remove_label)

print("\nNew types in our dataframe (after filtering):")
display(anime_df['Type'].value_counts())

# In "Score" -> must replace "Unknown" with 0
# In "Producers" and "Studios" -> we could remove "Unknown" but I'm not sure if it's a good idea
# Too many "Unknown" values in "Premiered", we can't use that column...
# anime_df.drop("Premiered", axis = 'columns', inplace = True)

anime_df.describe(include = 'all')

# Removing duplicate animes 
anime_df.drop_duplicates(subset = 'Name', inplace = True)
display(anime_df.shape)

# Defining content with no synopsis 
no_synopsis = anime_df['sypnopsis'].mode()[0] # most commom synopsis text must be the text when there's no synopsis at all
display("No synopsis text:", no_synopsis)

# Removing content with no synopsis
anime_df = anime_df[(anime_df['sypnopsis'].isnull() == False) & (anime_df['sypnopsis'] != no_synopsis)]
display(anime_df.shape)

no_synopsis = anime_df['sypnopsis'].mode()[0]
display("No synopsis text (series):", no_synopsis)

anime_df = anime_df[(anime_df['sypnopsis'] != no_synopsis)]

anime_df.describe(include = 'all')

# Converting synopsis to string type
anime_df['sypnopsis'] = anime_df['sypnopsis'].astype(str)
print(anime_df['sypnopsis'].head())

from rake_nltk import Rake # https://pypi.org/project/rake-nltk/
import nltk

nltk.download('stopwords')
nltk.download('punkt')

anime_df['Keywords'] = ''

# function to get keywords from a text
def getKeywordsFromText(text):
  # Initialize Rake
  # Uses stopwords for english from NLTK, and all puntuation characters by default
  rake = Rake()
  
  # Extracting keywords from text
  rake.extract_keywords_from_text(text)
  
  # Get dictionary with keywords and scores
  scores = rake.get_word_degrees()
  
  # Return new keywords as list, ignoring scores
  # Obs.: we could change this, selecting only the words with higher score
  return(list(scores.keys()))

# Apply function to generate keywords
anime_df['Keywords'] = anime_df['sypnopsis'].apply(getKeywordsFromText)
anime_df.drop('sypnopsis', axis='columns', inplace = True)
anime_df.head()

# Spliting features into list 
# Removing upper case and splitting words separated by ','
def tokenize(x): 
  if isinstance(x, list): # checking if entry is a list
    return [i.lower().split(", ") for i in x]
  else:
    if isinstance(x, str): # checking if entry is a string
      return x.lower().split(", ")
    else:
      return ''

anime_df['Genres'] = anime_df['Genres'].apply(tokenize)
anime_df['Studios'] = anime_df['Studios'].apply(tokenize)
anime_df['Producers'] = anime_df['Producers'].apply(tokenize)
anime_df.head()

# Preparing our final dataframe

df_final = pd.DataFrame()

df_final['title'] = anime_df['Name']
df_final['bag_of_words'] = ''

# We must later include Producers, Episodes, Studios, Source and Duration to the bag of words!!

def bag_words(x):
 return(' '.join(x['Genres']) + ' ' + ' '.join(x['Keywords']) + ' ')
  
df_final['bag_of_words'] = anime_df.apply(bag_words, axis = 'columns')

df_final.head()

from sklearn.feature_extraction.text import TfidfVectorizer

# Defining a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a' etc.
tfidf = TfidfVectorizer(stop_words='english')

# Replacing NaN with an empty string
df_final['bag_of_words'].fillna('', inplace = True)

# Constructing the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df_final['bag_of_words'])

display(tfidf_matrix.shape)

# Array mapping from feature integer indices to feature name.
display(tfidf.get_feature_names()[1000:1010])

print("Ammount of feature names:", len(tfidf.get_feature_names()))

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

display(cosine_sim.shape)

print("Array of cosine similarity for 0:")
cosine_sim[0]

# Creating list of indices for later matching
indices = pd.Series(df_final.index, index = df_final['title']).drop_duplicates().reset_index()

print("Anime indices (title):")
display(indices)

def getContentBasedRecommendation(title, selection_range = 10, cosine_sim = cosine_sim):
  movies = []

  if title not in indices['title']:
    print("Ops, title not in our database...")
    return
  
  title_index = indices['title' == title].index

  # Cosine similarity scores of anime titles in descending order (most similar is on top)
  scores = pd.Series(cosine_sim[title_index]).sort_values(ascending = False)

  # Top 'selection_range' most similar anime indexes
  top_animes_rec = list(scores.iloc[1:selection_range].index) # from 1 to 'selection_range', because 0 is the searched title always
      
  return pd.DataFrame(df_final['title'].iloc[top_animes_rec])

getContentBasedRecommendation("Naruto", 7)

getContentBasedRecommendation("Kimetsu no Yaiba", 7)
