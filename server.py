from fastapi import FastAPI, Query, File, UploadFile, Request
import pandas as pd

from recomendation.svd.svd_recommender import SVDRecommender
from recomendation.preprocessing.movie_manipulation import load_data
from io import StringIO
from fastapi.responses import JSONResponse

app = FastAPI()

folder = "0.1"
FORCE_TRAINING = False
PRODUCTION = True

ALGORITHM = SVDRecommender
algo = None
algo = ALGORITHM(FORCE_TRAINING, folder)
algo.init_data(load_data(f"ml-{folder}m"))
algo.load()


class AlgorithmNotInitialized(Exception):
    def __init__(self):
        self.name = "Algorithm not initialized"


@app.exception_handler(AlgorithmNotInitialized)
async def my_custom_exception_handler(request: Request, exc: AlgorithmNotInitialized):
    return JSONResponse(
        status_code=418,
        content={"message": f"{exc.name}"},
    )


def verify_algo():
    """
    Verify if the algorithm is initialized
    """
    if algo is None:
        raise AlgorithmNotInitialized()


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
    verify_algo()
    print(f"Getting recommendations for user {user_id} with top_n={top_n}")
    tmp = algo.get_recommendations(user_id=user_id, top_n=top_n)
    print(tmp)
    return tmp


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df_ratings = None
    try:
        df_ratings = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        return {"error": f"Could not parse uploaded file: {e}"}
    global algo
    algo = ALGORITHM(FORCE_TRAINING, folder)
    algo.init_data(df_ratings)
    algo.load()
