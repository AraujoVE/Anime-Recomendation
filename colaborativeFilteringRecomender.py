import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sklearn.neighbors as ng

MIN_ANIME_REV = 500


def getCFRecommendation(anime_collaborative_filter: np.ndarray, anime_name_df: DataFrame, title: str, n_neighbors: int = 10):
    # Creating KNN to classify our animes based on the ratings
    # kd_tree = efficient for large datasets and small dimensions -> O(D N log N) where D = num. of dimensions and N = num. of points
    model_knn = NearestNeighbors(
        metric='minkowski', n_neighbors=n_neighbors, algorithm='kd_tree')
    # Another option for metric + algo. is bruteforce + cosine distance
    model_knn.fit(csr_matrix(anime_collaborative_filter))

    query_index = anime_name_df[anime_name_df['Name'].str.lower() == title.lower()].index[0]

    distances, indices = model_knn.kneighbors(anime_collaborative_filter[query_index, :].reshape(1, -1), n_neighbors=n_neighbors)

    result = []
    for i in range(0, len(distances.flatten())):
        index = indices.flatten()[i]
        if index == query_index:
            continue
        result.append(anime_name_df.iloc[index])

    return pd.DataFrame(result)


def createCFMatrix(unique_animes: int, unique_users: int, rating_df: DataFrame):
    # Creating and filling collaborative-filter matrix (maybe it's sparse, we should check and optimize if possible...)
    anime_collaborative_filter = np.zeros(
        (len(unique_animes), len(unique_users)))

    # For each combination of user, anime and rating given by that user for that anime (rating)...
    for user_id, MAL_ID, rating in rating_df.values:
        # ... we insert the rating in our cf matrix
        anime_collaborative_filter[unique_animes[MAL_ID]][unique_users[user_id]] = rating

    print("-> CF matrix shape:", anime_collaborative_filter.shape)
    # each line represents an anime and each column represents all ratings given by a user to all animes (0 = no rating or actually 0)
    print(anime_collaborative_filter)

    return anime_collaborative_filter


def getUniqueIdsCount(anime_name_df: DataFrame, rating_df: DataFrame):
    # Vectorization
    # getting unique user_id from rating_df
    unique_users = {int(x): i for i, x in enumerate(
        rating_df.user_id.unique())}
    # getting unique anime names from anime_name_df
    unique_animes = {int(x): i for i, x in enumerate(
        anime_name_df.MAL_ID.unique())}

    print("-> Qty. unique users:", len(unique_users))
    print("-> Qty. uique animes:", len(unique_animes))

    return unique_users, unique_animes


def readAndJoinAnimeCSVs():
    # import anime_name dataset
    cols_to_use = ['MAL_ID', 'Name', 'Completed']
    anime_name_df = pd.read_csv('dataset/anime.csv', usecols=cols_to_use)

    # Showing some info about the dataset
    print("-> Anime name dataset loaded")
    print(anime_name_df.shape)
    print(anime_name_df.head())

    # Importing rating dataset
    rating_df = pd.read_csv('dataset/rating_complete.csv',
                            skiprows=lambda x: x > 0 and x < 1500000, nrows=1500000*2)

    # Showing some info about the dataset
    print("-> Rating dataset loaded")
    print(rating_df.shape)
    print(rating_df.head())

    # Counting number of unique users (check user_id)
    users_count = rating_df.groupby('user_id').size().reset_index()

    # Defining user_id as dataframe index
    users_count.columns = ["user_id", "anime_count"]
    users_count.set_index('user_id', inplace=True)
    print('-> Numbers of unique users: ', users_count.shape[0])

    # Filtering the users
    # Only getting the users that reviewed at least MIN_ANIME_REV animes
    # Avoiding (very) sparse matrix
    filtered_users = users_count[users_count.anime_count >= MIN_ANIME_REV]
    users_ids = list(filtered_users.index)
    print('-> Numbers of unique users with {} or more anime ratings: {}'.format(MIN_ANIME_REV, len(users_ids)))

    # Getting only users that are in our users_ids list (users with >= MIN_ANIME_REV anime reviews)
    rating_df = rating_df[rating_df['user_id'].isin(users_ids)]

    print("-> New rating shape (with filtered users):", rating_df.shape)
    print(rating_df.info())

    return anime_name_df, rating_df


def main():
    anime_name_df, rating_df = readAndJoinAnimeCSVs()
    unique_users, unique_animes = getUniqueIdsCount(anime_name_df, rating_df)
    anime_collaborative_filter = createCFMatrix(
        unique_animes, unique_users, rating_df)

    want_to_search = True
    while (want_to_search):
        search_anime = input("# Entre com o nome de um anime:\n>> ")
        if (search_anime == ""):
            continue

        search_anime = search_anime.strip()

        if not search_anime.lower() in anime_name_df['Name'].str.lower().tolist():
            print("-> Título não encontrado na nossa base de dados...")
            continue

        # if anime is not complete, we can't find recommendations
        MIN_COMPLETED = 500
        if anime_name_df[anime_name_df['Name'].str.lower() == search_anime.lower()]['Completed'].values[0] < MIN_COMPLETED:
            print("-> Avaliações insuficientes, não podemos encontrar boa recomendações...")
            continue

        result = getCFRecommendation(
            anime_collaborative_filter, anime_name_df, search_anime, n_neighbors=10)
        print("# Recomendações para \"{}\":\n".format(search_anime), result)

        user_choise_sn = input("# Deseja buscar por outro anime? (s/n)\n>> ")

        if (user_choise_sn.strip() == "n"):
            want_to_search = False


if __name__ == "__main__":
    main()
