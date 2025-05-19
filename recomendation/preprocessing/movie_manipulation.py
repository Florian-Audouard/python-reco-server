import sys
import os
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split

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
