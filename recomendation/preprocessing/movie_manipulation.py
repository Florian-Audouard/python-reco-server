import sys
import os
import pandas as pd
import requests
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.time_util import get_time_func


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
    ratings = pd.read_csv(os.path.join(full_path, "ratings.csv"))

    return ratings


def load_data_from_url(url):
    """
    Load data from a URL.

    """
    response = requests.get(url, timeout=20)
    data = pd.read_csv(StringIO(response.text))
    return data


def _testing_main():
    """
    Fonction de test pour le module de manipulation de films.
    """
    ratings = load_data("ml-0.1m")
    print(ratings.head())


if __name__ == "__main__":
    # Exemple d'utilisation
    _testing_main()
