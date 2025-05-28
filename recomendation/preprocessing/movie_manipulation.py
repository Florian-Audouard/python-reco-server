import sys
import os
import pandas as pd
import time
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.time_util import get_time_func

HOST = "quarkus-app"


def change_host(host):
    """
    Change the host for the URLs.

    Args:
        host (str): The new host to set.
    """
    global HOST
    HOST = host


def get_url_rating():
    """
    Get the URL for the rating service.

    Returns:
        str: The URL for the rating service.
    """
    return f"http://{HOST}:8080/rating/file"


def get_url_rating_noise():
    """
    Get the URL for the noisy rating service.

    Returns:
        str: The URL for the noisy rating service.
    """
    return f"http://{HOST}:8080/rating/fileNoise"


def get_url_movie():
    """
    Get the URL for the movie service.

    Returns:
        str: The URL for the movie service.
    """
    return f"http://{HOST}:8080/movie/file"


FOLDER = "data"


def delete_unknow_movie_in_ratings(ratings, movies):
    """
    Delete unknown movies in ratings DataFrame.

    Args:
        ratings (pd.DataFrame): DataFrame containing ratings data.
        movies (pd.DataFrame): DataFrame containing movies data.

    Returns:
        pd.DataFrame: Updated ratings DataFrame with unknown movies removed.
    """
    # Remove unknown movies from ratings
    unknown_movies = set(ratings["movieId"]) - set(movies["movieId"])
    return ratings[~ratings["movieId"].isin(unknown_movies)]


def load_data_from_file(folder):
    """
    Load movies, ratings, and tags data from CSV files.

    Args:
        folder (str): Folder containing the CSV files.

    Returns:
        tuple: Tuple containing three DataFrames (movies, ratings, tags).
    """
    # Create data directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, FOLDER)
    os.makedirs(data_dir, exist_ok=True)

    full_path = os.path.join(data_dir, folder)
    ratings = pd.read_csv(os.path.join(full_path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(full_path, "movies.csv"))
    ratings = delete_unknow_movie_in_ratings(ratings, movies)
    return ratings, movies


def fetch_csv_from_url(url):
    response = requests.get(url, timeout=20)
    return pd.read_csv(StringIO(response.text))


def load_data_from_url():
    """
    Load data from URLs in parallel threads.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ratings = executor.submit(fetch_csv_from_url, get_url_rating())
        future_movies = executor.submit(fetch_csv_from_url, get_url_movie())

        ratings = future_ratings.result()
        movies = future_movies.result()
    ratings = delete_unknow_movie_in_ratings(ratings, movies)
    return ratings, movies


def load_data_from_url_noise():
    """
    Load noisy data from URLs in parallel threads.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ratings = executor.submit(fetch_csv_from_url, get_url_rating_noise())
        future_movies = executor.submit(fetch_csv_from_url, get_url_movie())

        ratings = future_ratings.result()
        movies = future_movies.result()
    ratings = delete_unknow_movie_in_ratings(ratings, movies)
    return ratings, movies


def _testing_main():
    """
    Fonction de test pour le module de manipulation de films.
    """
    ratings, movies = load_data_from_file("ml-0.1m")
    print(ratings.head())
    print(movies.head())
    ratings, movies = load_data_from_url()
    print(ratings.head())
    print(movies.head())


# Group by user to avoid cold-start users in val set
def __user_based_split(group, test_size):
    if len(group) < 5:
        return group, pd.DataFrame()  # skip splitting small groups
    return train_test_split(group, test_size=test_size, shuffle=False)


def smart_split(df, test_size=0.2):
    """
    Split a DataFrame into two parts based on a given ratio.

    Args:
        df (pd.DataFrame): DataFrame to split.
        ratio (float): Ratio for splitting the DataFrame.

    Returns:
        tuple: Tuple containing two DataFrames (train_df, test_df).
    """
    train_list = []
    val_list = []

    for _, user_group in df.groupby("userId"):
        train, val = __user_based_split(user_group, test_size=test_size)
        train_list.append(train)
        val_list.append(val)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    return train_df, val_df


if __name__ == "__main__":
    # Exemple d'utilisation
    _testing_main()
