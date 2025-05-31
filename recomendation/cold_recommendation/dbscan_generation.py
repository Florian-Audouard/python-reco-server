import numpy as np
import umap

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict



scaler = StandardScaler()
mlb = MultiLabelBinarizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
reducer = umap.UMAP(n_components=150, n_neighbors=30, min_dist=0.1, metric='cosine')

def get_statistic_of_movie(ratings, movie_id):
    """
    For eaxh movies, returns the mean rating and the number of person who have seen it.

    Args:
        ratings (pd.DataFrame): DataFrame with columns "movieId" and "rating".
        movie_id (int): The movie ID.

    Returns:
        list: [mean_rating or 0.0 if no ratings, number_of_watch]
    """
    mean = ratings[ratings["movieId"] == movie_id]["rating"].mean()
    number_of_watch = ratings[ratings["movieId"] == movie_id]["rating"].count()
    return [0.0 if number_of_watch == 0 else mean,
            number_of_watch]


def get_movie_and_binary_vector(movies):
    """
    Converts movie genres into binary vectors for each movie.

    Args:
        movies (pd.DataFrame): DataFrame with columns "movieId" and "genre".

    Returns:
        dict: {movieId: binary genre vector}
    """
    genre_by_movies = movies['genre'].str.split("|").tolist()
    genre_binary_vector = mlb.fit_transform(genre_by_movies)
    return dict(zip(movies["movieId"], genre_binary_vector))

def extract_data_and_movies_id(ratings, movies, recommendable_movies):
    """
    Extracts and concatenates features for each movie (statistics, description, genres, release date),
    and returns the scaled feature matrix and corresponding movie IDs.
    """
    genre_vector = get_movie_and_binary_vector(movies)
    descriptions = model.encode(movies['plot'].tolist(), batch_size=64, convert_to_numpy=True)

    features = []
    movie_ids = []
    for idx, row in enumerate(movies.itertuples(index=False)):
        movie_id = row.movieId
        if movie_id not in recommendable_movies:
            continue
        statistics = get_statistic_of_movie(ratings, movie_id)
        data = np.concatenate([
            descriptions[idx],
            genre_vector[movie_id],
            np.array([row.year, row.month, row.day], dtype=np.float32)
        ])
        features.append(data)
        movie_ids.append([movie_id, statistics[0], statistics[1]])

    if not features:
        return np.array([]), []

    return scaler.fit_transform(np.stack(features)), movie_ids

def dbscan_clustering(ratings, movies, recommendable_movies):
    """
    Performs DBSCAN clustering on movies based on extracted features.

    Args:
        ratings (pd.DataFrame): DataFrame of ratings.
        movies (pd.DataFrame): DataFrame of movies.

    Returns:
        dict: {cluster_label: list of [movieId, mean, number_of_watch] in that cluster}
    """
    data_from_movies, movies_ids = extract_data_and_movies_id(ratings, movies, recommendable_movies)
    if data_from_movies.shape[0] == 0:
        return {}

    reduced_data = reducer.fit_transform(data_from_movies)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(reduced_data)
    cluster_labels = clustering.labels_

    label_and_movies_data = defaultdict(list)
    for label, movie_info in zip(cluster_labels, movies_ids):
        label_and_movies_data[label].append(movie_info)
    return dict(label_and_movies_data)