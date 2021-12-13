import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

MIN_ANIME_REV = 500
ANIME_MIN_COMPLETED = 500


def getCFRecommendation(rating_matrix_CF: np.ndarray, anime_name_df: DataFrame, title: str, n_neighbors: int = 10):
    # Creating KNN to classify our animes based on the ratings
    print("\t> Criando modelo de kNN para recomendar animes")

    # Metric + algorithm used : bruteforce + cosine distance
    model_knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors, algorithm='brute')
    
    # Calculating sparsity of rating_matrix_CF
    sparsity = 1.0 - np.count_nonzero(rating_matrix_CF) / rating_matrix_CF.size
    print("\t> 'Sparsity' da matriz de avaliações para o CF: {:.3f}".format(sparsity))

    # CSR matrix = Compressed Sparse Row matrix
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    model_knn.fit(csr_matrix(rating_matrix_CF))

    # [ALTERNATIVE] Creating a KNN model using ball_tree -> good for big datasets and a lot of dimensions (O[D N log N])
    # model_knn = NearestNeighbors(metric='minkowski', n_neighbors=n_neighbors, algorithm='ball_tree')
    # model_knn.fit(rating_matrix_CF)

    query_index = anime_name_df[anime_name_df['Name'].str.lower() == title.lower()].index[0]

    # Finding the K-neighbors of a point.
    # Returns indices of and distances to the neighbors of each point.
    # rating_matrix_CF[query_index, :].reshape(1, -1) -> gets the row of the query anime and returns it as a 1xN array (N = number of users) -> gets all ratings given by all users to the query anime
    print("\t> Procurando animes parecidos com '{}'".format(title))
    distances, indices = model_knn.kneighbors(rating_matrix_CF[query_index, :].reshape(1, -1), n_neighbors=n_neighbors)

    '''
    Obs.: uma forma de visualizar a busca por vizinhos é imaginar que cada anime está em um espaço de dimensão = qtd. de usuários. A posição do vetor do anime em cada eixo é o rating dado por aquele usuário ao anime.
    '''

    result = []
    for i in range(0, len(distances.flatten())):
        index = indices.flatten()[i]
        if index == query_index:
            continue
        result.append(anime_name_df.iloc[index])
        result[-1].loc["Distance"] = distances.flatten()[i]

    return pd.DataFrame(result)


def createCFMatrix(unique_animes: int, unique_users: int, rating_df: DataFrame):
    # Creating and filling collaborative-filter matrix (it's sparse, we should check and optimize if possible...)
    print("\t> Inicializando matriz de filtro de colaboração (collaborative-filter)")
    rating_matrix_CF = np.zeros((len(unique_animes), len(unique_users)))

    # For each combination of user, anime and rating given by that user for that anime (rating)...
    for user_id, MAL_ID, rating in rating_df.values:
        # ... we insert the rating in our cf matrix
        # X = Anime ID
        # Y = User ID
        # matrix[x][y] = rating given by user Y to anime X
         # each line represents an anime and each column represents all ratings given by a user to all animes (0 = no rating or actually 0)
        rating_matrix_CF[unique_animes[MAL_ID]][unique_users[user_id]] = rating

    print("\t> Feito! Shape da matriz:", rating_matrix_CF.shape)
   

    return rating_matrix_CF


def getUniqueIDsCount(anime_name_df: DataFrame, rating_df: DataFrame):
    # Vectorization
    # getting unique user_id from rating_df
    unique_users = {int(x): i for i, x in enumerate(rating_df.user_id.unique())}
    # getting unique anime names from anime_name_df
    unique_animes = {int(x): i for i, x in enumerate(anime_name_df.MAL_ID.unique())}

    print("\t> Qtd. usuários únicos:", len(unique_users))
    print("\t> Qtd. animes únicos:", len(unique_animes))

    return unique_users, unique_animes


def readAndJoinAnimeCSVs():
    # import anime_name dataset
    cols_to_use = ['MAL_ID', 'Name', 'Completed']
    anime_name_df = pd.read_csv('dataset/anime.csv', usecols=cols_to_use)

    # Showing some info about the dataset
    print("\t> Dataset de animes carregado")
    # print(anime_name_df.shape)
    # print(anime_name_df.head())

    # Importing rating dataset
    rating_df = pd.read_csv('dataset/rating_complete.csv', nrows=1500000)

    # Showing some info about the dataset
    print("\t> Dataset de avaliações carregado")
    # print(rating_df.shape)
    # print(rating_df.head())

    # Counting number of unique users (check user_id)
    users_count = rating_df.groupby('user_id').size().reset_index()

    # Defining user_id as dataframe index
    users_count.columns = ["user_id", "anime_count"]
    users_count.set_index('user_id', inplace=True)
    print('\t> Qtd. de usuários únicos: ', users_count.shape[0])

    # Filtering the users
    # Only getting the users that reviewed at least MIN_ANIME_REV animes
    # Avoiding (very) sparse matrix
    filtered_users = users_count[users_count.anime_count >= MIN_ANIME_REV]
    users_ids = list(filtered_users.index)
    print('\t> Qtd. de usuários únicos com {} os mais avaliações realizadas: {}'.format(MIN_ANIME_REV, len(users_ids)))

    # Getting only users that are in our users_ids list (users with >= MIN_ANIME_REV anime reviews)
    rating_df = rating_df[rating_df['user_id'].isin(users_ids)]

    print("\t> Shape da matriz de avaliações (após filtro de usuários):", rating_df.shape)
    # print(rating_df.info())

    return anime_name_df, rating_df


def main():
    print("[*] Lendo CSVs de entrada")
    anime_name_df, rating_df = readAndJoinAnimeCSVs()
    print("[**] Buscando usuários e animes")
    unique_users, unique_animes = getUniqueIDsCount(anime_name_df, rating_df)
    print("[***] Criando matriz de filtro de colaboração")
    rating_matrix_CF = createCFMatrix(unique_animes, unique_users, rating_df)
    print("[!] TUDO OK!")

    want_to_search = True
    while (want_to_search):
        search_anime = input("[?] Entre com o nome de um anime:\n>>> ")
        if (search_anime == ""):
            continue

        search_anime = search_anime.strip()

        if not search_anime.lower() in anime_name_df['Name'].str.lower().tolist():
            print("[@] Título não encontrado na nossa base de dados...")

            similar_animes = anime_name_df[anime_name_df['Name'].str.contains(search_anime, case=False)]
            if not similar_animes.empty:
                print("[!] Animes Similares:")
                print(similar_animes)
        # if anime is not complete, we can't find recommendations
        elif anime_name_df[anime_name_df['Name'].str.lower() == search_anime.lower()]['Completed'].values[0] < ANIME_MIN_COMPLETED:
            print("[@] Avaliações insuficientes, não podemos encontrar boas recomendações...")
        else:
            print("[!] Buscando recomendações...")
            result = getCFRecommendation(rating_matrix_CF, anime_name_df, search_anime, n_neighbors=10)
            # Set index of result to MAL_ID
            result.set_index('MAL_ID', inplace=True)
            # Removing column that's not needed
            result.drop(columns=['Completed'], inplace=True)
            print("# Recomendações para \"{}\":\n".format(search_anime), result)

        user_choise_sn = input("[?] Deseja continuar a busca por animes? (s/n)\n>>> ")

        if (user_choise_sn.strip().lower()[0] == "n"):
            want_to_search = False


if __name__ == "__main__":
    main()
