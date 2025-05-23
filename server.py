from fastapi import FastAPI, Query, File, UploadFile


from recomendation.cold_recommendation.dbscan_recommender import DBSCANRecommender
from recomendation.svd.svd_recommender import SVDRecommender
from recomendation.preprocessing.movie_manipulation import load_data_from_url

app = FastAPI()


FORCE_TRAINING = False
PRODUCTION = True

ratings, movies = load_data_from_url()

algo = SVDRecommender(PRODUCTION, FORCE_TRAINING)
algo.init_data(ratings, movies)
algo.load()

cold_algo = DBSCANRecommender(PRODUCTION, FORCE_TRAINING)
cold_algo.init_data(ratings, movies)
cold_algo.load()


@app.get("/")
def read_root():
    return "Welcome to the Movie Recommender API!"


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_n: int = Query(..., gt=0)):
    """
    Get recommendations for a given user
    Args:
        user_id (int): ID of the target user
        top_n (int): Number of recommendations to return
    Returns:
        List of recommended items
    """
    print(f"Getting recommendations for user {user_id} with top_n={top_n}")
    return algo.get_recommendations(user_id=user_id, top_n=top_n)


@app.get("/cold_recommendations")
def get_cold_recommendations(top_n: int = Query(..., gt=0)):
    """
    Get cold recommendations for a given user
    Args:
        top_n (int): Number of recommendations to return
    Returns:
        List of recommended items
    """
    return cold_algo.get_recommendations(user_id=None, top_n=top_n)
