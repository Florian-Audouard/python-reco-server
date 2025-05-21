import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file


FORCE_TRAINING = True
FOLDER_SET = "0.1"


class RandomRecommender(Model):
    """
    A simple recommendation model that returns random movies.
    """

    def __init__(self, production=False, force_training=False):
        super().__init__(production, force_training)
        self.filename = "random_recommender"

    def save(self):
        # Rien à sauvegarder pour un modèle aléatoire
        pass

    def load_impl(self):
        # Rien à charger pour un modèle aléatoire
        pass

    def training_impl(self):
        # Aucun entraînement nécessaire pour un modèle aléatoire
        pass

    def init_data_impl(self):
        # Pas de traitement spécial
        pass

    def predict(self, user_id, list_movie_id):
        """
        Predict ratings for the given user and list of movie IDs by returning random movie IDs.

        Returns:
            list: List of movie IDs in random order
        """
        movie_ids = list(list_movie_id)
        random.shuffle(movie_ids)
        return movie_ids

    def get_recommendations_impl(self, user_id, top_n):
        # Rien à faire ici car la logique est dans predict
        pass


if __name__ == "__main__":
    # Exemple d'utilisation
    recommender = RandomRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
