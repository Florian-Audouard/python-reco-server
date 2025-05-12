import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.time_util import get_time_func


@get_time_func
def group_data(movies, tags):
    """
    Combine les genres et les tags pour chaque film dans une seule description.

    Args:
        movies (DataFrame): Films avec leurs genres.
        tags (DataFrame): Tags associés aux films.

    Returns:
        DataFrame: Films enrichis d'une colonne 'full_desc' combinant genres et tags.
    """
    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )
    movies_merged = movies.merge(movie_tags, on="movieId", how="left")
    movies_merged["full_desc"] = (
        movies_merged["genres"] + " " + movies_merged["tag"].fillna("")
    )
    return movies_merged


FOLDER = "data"


def load_data(folder):
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
    movies = pd.read_csv(os.path.join(full_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(full_path, "ratings.csv"))
    tags = pd.read_csv(os.path.join(full_path, "tags.csv"))

    return movies, ratings, tags


def _testing_main():
    """
    Fonction de test pour le module de manipulation de films.
    """
    movies, _, tags = load_data("ml-0.1m")
    movies_merged = group_data(movies, tags)
    print(movies_merged.head())


if __name__ == "__main__":
    # Exemple d'utilisation
    _testing_main()
