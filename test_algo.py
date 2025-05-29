from concurrent.futures import ThreadPoolExecutor
from recomendation.random.random_recommender import RandomRecommender
from recomendation.svd.svd_recommender import SVDRecommender
from recomendation.deep.deep_recommender_v5 import (
    DeepLearningRecommender as DeepLearningRecommenderV5,
)
from recomendation.preprocessing.movie_manipulation import (
    load_data_from_url,
    load_data_from_url_noise,
    change_host,
)

from utils.logger import log

folder = "0.1"
FORCE_TRAINING = True
PRODUCTION = False

ALGORITHM = SVDRecommender

change_host("localhost")


def init_algo(ratings, movies, noise=False, real_data=None):
    """
    Initialize the algorithm
    """

    algo = ALGORITHM(PRODUCTION, FORCE_TRAINING, noise=noise, real_data=real_data)
    algo.testing_main(ratings, movies)
    return algo


ratings_clean, movies_clean = (None, None)
ratings_noise, movies_noise = (None, None)
with ThreadPoolExecutor(max_workers=2) as executor:
    future_clean = executor.submit(load_data_from_url)
    future_noise = executor.submit(load_data_from_url_noise)

    ratings_clean, movies_clean = future_clean.result()
    ratings_noise, movies_noise = future_noise.result()

algo_claire = init_algo(ratings_clean, movies_clean, noise=False)
print()
algo_bruit = init_algo(ratings_noise, movies_noise, noise=True, real_data=ratings_clean)
print()
