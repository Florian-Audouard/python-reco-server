import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from recomendation.ia import model
from recomendation.preprocessing import movie_manipulation
from utils.time_util import get_time_func

os.chdir(os.path.dirname(__file__))

FOLDER_PATH_IA = "recomendation/ia/" + model.FOLDER_PATH


@get_time_func
def read_data(folder):
    """
    Lit les fichiers CSV de films, évaluations et tags depuis le dossier donné.

    Args:
        folder (str): Sous-dossier dans le dossier 'data'.

    Returns:
        tuple: DataFrames pour les films, évaluations, et tags.
    """
    movies = pd.read_csv(f"{FOLDER_PATH_IA}/ml-{folder}m/movies.csv")
    ratings = pd.read_csv(f"{FOLDER_PATH_IA}/ml-{folder}m/ratings.csv")
    tags = pd.read_csv(f"{FOLDER_PATH_IA}/ml-{folder}m/tags.csv")
    return movies, ratings, tags


def main():
    """
    Fonction principale pour exécuter le script.
    """
    movies, ratings, tags = read_data("0.1")
    movies_merged = movie_manipulation.group_data(movies, tags)
    while True:
        user_input = int(input("Entrez un user ID : "))
        reco = model.hybrid_recommendations(
            user_input, movies, ratings, tags, movies_merged, cosine_sim, algo
        )
        print("\n🎥 Recommandations pour l'utilisateur", user_input)
        for title, score in reco:
            print(f" - {title} (score: {score:.2f})")


if __name__ == "__main__":
    main()
