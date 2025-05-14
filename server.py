from fastapi import FastAPI, Query, File, UploadFile


from recomendation.svd.svd_recommender import SVDRecommender
from recomendation.preprocessing.movie_manipulation import load_data

app = FastAPI()

folder = "0.1"
FORCE_TRAINING = True
PRODUCTION = True

ALGORITHM = SVDRecommender
algo = SVDRecommender(FORCE_TRAINING, folder)
algo.init_data(load_data(f"ml-{folder}m"))
algo.load()


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
    tmp = algo.get_recommendations(user_id=user_id, top_n=top_n)
    print(tmp)
    return tmp


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    algo = ALGORITHM(FORCE_TRAINING, folder)
    algo.init_data(contents)
    algo.load()
