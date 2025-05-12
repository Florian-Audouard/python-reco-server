from fastapi import FastAPI, Query


from recomendation.fitting.svd import SVDRecommender
from recomendation.preprocessing.movie_manipulation import load_data

app = FastAPI()

folder = "0.1"
FORCE_TRAINING = False

data = load_data(f"ml-{folder}m")

algo = SVDRecommender(FORCE_TRAINING)
algo.init_data(data)
algo.load()


@app.get("/")
def read_root():
    return algo.predict(user_id=1, top_n=5)


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id, top_n: int = Query(..., gt=0)):
    """
    Get recommendations for a given user
    Args:
        user_id (int): ID of the target user
        top_n (int): Number of recommendations to return
    Returns:
        List of recommended items
    """
    return algo.predict(user_id=user_id, top_n=top_n)
