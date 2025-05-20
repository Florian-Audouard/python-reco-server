import pandas as pd

from recomendation.svd.svd_recommender import SVDRecommender
from recomendation.preprocessing.movie_manipulation import load_data_from_file
from io import StringIO
import requests
from utils.logger import log

folder = "0.1"
FORCE_TRAINING = False
PRODUCTION = False

ALGORITHM = SVDRecommender


def init_algo(data):
    """
    Initialize the algorithm
    """

    algo = ALGORITHM(FORCE_TRAINING, folder)
    algo.testing_main(data)
    return algo


def init_algo_from_file():
    """
    Initialize the algorithm from a file
    """
    log.info("Loading data from file\n")
    data = load_data_from_file(f"ml-{folder}m")
    return init_algo(data)


def init_algo_from_url():
    """
    Initialize the algorithm from a URL
    """
    log.info("Loading data from URL\n")
    url = "http://localhost:8080/rating/file"
    response = requests.get(url, timeout=20)
    data = pd.read_csv(StringIO(response.text))
    return init_algo(data)


algo_bruit = init_algo_from_url()
print()
algo_claire = init_algo_from_file()
