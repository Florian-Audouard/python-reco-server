import numpy as np
import umap

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor



scaler = StandardScaler()
mlb = MultiLabelBinarizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
reducer = umap.UMAP(n_components=100, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)

def get_statistic_of_movie(ratings, movie_id):
    """
    For eaxh movies, returns the mean rating and the number of person who have seen it.

    Args:
        ratings (pd.DataFrame): DataFrame with columns "movieId" and "rating".
        movie_id (int): The movie ID.

    Returns:
        list: [mean_rating or -1.0 if no ratings, number_of_watch]
    """
    mean = ratings[ratings["movieId"] == movie_id]["rating"].mean()
    number_of_watch = ratings[ratings["movieId"] == movie_id]["rating"].sum()
    return [-1.0 if number_of_watch == 0 else mean,
            number_of_watch]

def get_description_of_movie(synopsis):
    """
    Encodes the movie synopsis into a vector using a SentenceTransformer model.

    Args:
        synopsis (str): The movie synopsis.

    Returns:
        np.ndarray: Embedding vector for the synopsis.
    """
    return model.encode(synopsis)

def get_movie_release(date):
    """
    Extracts the year, month, and day from a release date string.

    Args:
        date (str): Release date in the format "YYYY-MM-DD ...".

    Returns:
        list: [year, month, day] as integers.
    """
    release_date = date.split(" ")[0]
    year, month, day = release_date.split("-")
    return [int(year), int(month), int(day)]

def get_movie_and_binary_vector(movies):
    """
    Converts movie genres into binary vectors for each movie.

    Args:
        movies (pd.DataFrame): DataFrame with columns "movieId" and "genre".

    Returns:
        dict: {movieId: binary genre vector}
    """
    movies_and_binary_vector = dict()
    genre_by_movies = [row["genre"].split("|") for _, row in movies.iterrows()]
    binary_vector = mlb.fit_transform(genre_by_movies)
    for i, movie_id in enumerate(movies["movieId"]):
        movies_and_binary_vector[movie_id] = binary_vector[i]
    return movies_and_binary_vector

def extract_data_and_movies_id(ratings, movies):
    """
    Extracts and concatenates features for each movie (statistics, description, genres, release date),
    and returns the scaled feature matrix and corresponding movie IDs.

    Args:
        ratings (pd.DataFrame): DataFrame of ratings.
        movies (pd.DataFrame): DataFrame of movies.

    Returns:
        tuple: (np.ndarray of features, list of [movieId, mean, number_of_watch])
    """
    extract_data = []
    movie_ids = []
    movies_and_binary_vector = get_movie_and_binary_vector(movies)
    descriptions = model.encode(movies['plot'].tolist(), batch_size=64, convert_to_numpy=True)

    def extract_data_from_row(index_row):
        index, row = index_row
        movie_id = row["movieId"]
        date = row["released"]

        statistics = get_statistic_of_movie(ratings, movie_id)
        data = [statistics,
                descriptions[index],
                movies_and_binary_vector[movie_id],
                get_movie_release(date)]
        return np.concatenate(data), [movie_id, statistics[0], statistics[1]]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(
            extract_data_from_row,
            list(movies.iterrows())
            )

    for data, movie_id in results:  
        extract_data.append(data)
        movie_ids.append(movie_id)
    return scaler.fit_transform(np.array(extract_data)), movie_ids

def dbscan_clustering(ratings, movies):
    """
    Performs DBSCAN clustering on movies based on extracted features.

    Args:
        ratings (pd.DataFrame): DataFrame of ratings.
        movies (pd.DataFrame): DataFrame of movies.

    Returns:
        dict: {cluster_label: list of [movieId, mean, number_of_watch] in that cluster}
    """
    data_from_movies, movies_ids = extract_data_and_movies_id(ratings, movies)
    
    reduced_data = reducer.fit_transform(data_from_movies)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(reduced_data)
    cluster_labels = clustering.labels_
    
    label_and_movies_data = dict()
    for index, label in enumerate(cluster_labels):
        if label not in label_and_movies_data:
            label_and_movies_data[label] = [movies_ids[index]]
        else:
            label_and_movies_data[label].append(movies_ids[index])
    return label_and_movies_data