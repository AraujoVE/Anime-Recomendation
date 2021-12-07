import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# import anime_name dataset
cols_to_use = ['MAL_ID', 'Name']
anime_name_df = pd.read_csv('anime.csv', usecols = cols_to_use)

display(anime_name_df.shape)
anime_name_df.head()

# Importing rating dataset

rating_df = pd.read_csv('rating_complete.csv')

display(rating_df.shape)
rating_df.head()

# Counting number of unique users (check user_id)
users_count = rating_df.groupby('user_id').size().reset_index()
users_count.columns = ["user_id", "anime_count"]
users_count.set_index('user_id', inplace = True)
print('Numbers of unique users : ', users_count.shape[0])

MIN_ANIME_REV = 500

# Filtering the users
# Only getting the users that reviewed at least MIN_ANIME_REV animes
filtered_users = users_count[users_count.anime_count >= MIN_ANIME_REV]
users_ids = list(filtered_users.index)
print('Numbers of unique users with {} or more anime ratings: {}'.format(MIN_ANIME_REV, len(users_ids)))

# Getting only users that are in our users_ids list (users with >= MIN_ANIME_REV anime reviews)
rating_df = rating_df[rating_df['user_id'].isin(users_ids)]

print("Rating shape:", rating_df.shape)
print(rating_df.info())

# Vectorization
unique_users = {int(x): i for i, x in enumerate(rating_df.user_id.unique())} # getting unique user_id from rating_df
unique_animes = {int(x): i for i, x in enumerate(anime_name_df.MAL_ID.unique())} # getting unique animes from anime_name_df

print(len(unique_animes), len(unique_users))

# Creating and filling collaborative-filter matrix (maybe it's sparse, we should check and optimize memory used...)
anime_collaborative_filter = np.zeros((len(unique_animes), len(unique_users)))

# For each combination of user, anime and rating given by that user for that anime (rating)...
for user_id, MAL_ID, rating in rating_df.values:
  anime_collaborative_filter[unique_animes[MAL_ID]][unique_users[user_id]] = rating # ... we insert the rating in our cf matrix
display(anime_collaborative_filter) # each line represents an anime and each column represents all ratings given by a user to all animes (0 = no rating or actually 0)

def getCFRecommendation(title, n_neighbors = 10):
  # Creating KNN to classify our animes based on the ratings
  model_knn = NearestNeighbors(metric='cosine', n_neighbors = n_neighbors)
  model_knn.fit(csr_matrix(anime_collaborative_filter))
  
  query_index = anime_name_df[anime_name_df['Name']==title].index[0]

  distances, indices = model_knn.kneighbors(anime_collaborative_filter[query_index,:].reshape(1, -1), n_neighbors = n_neighbors)
  
  result = []
  for i in range(0, len(distances.flatten())):
    index = indices.flatten()[i]
    if index == query_index:
      continue
    result.append(anime_name_df.iloc[index])
        
  return pd.DataFrame(result)

getCFRecommendation('Kimetsu no Yaiba')

getCFRecommendation('No Game No Life', 7)

getCFRecommendation('Kiss x Sis (TV)', 7)
