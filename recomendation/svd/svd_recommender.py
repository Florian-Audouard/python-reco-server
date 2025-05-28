"""
Module de recommandation de films basé sur une approche hybride (filtrage collaboratif + filtrage de contenu).
"""

import os
import sys
from surprise import SVD, Dataset, Reader
import joblib
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file


FORCE_TRAINING = True
FOLDER_SET = "0.1"


class SVDRecommender(Model):
    """
    SVDRecommender is a recommendation system model based on Singular Value Decomposition (SVD).
    """

    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "svd_model.joblib"
        self.surprise_trainset = None
        self.threshold = 3.5

    def init_data_impl(self):
        reader = Reader(rating_scale=(self.min, self.max))
        self.surprise_trainset = Dataset.load_from_df(
            self.trainset[["userId", "movieId", "rating"]], reader
        ).build_full_trainset()

    def training_impl(self):
        """
        Train the SVD model using the initialized data
        """
        self.model = SVD()
        self.model.fit(self.surprise_trainset)

    def save(self):
        """
        Save the trained SVD model to a file
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        joblib.dump(self.model, self.get_full_path())

    def load_impl(self):
        """
        Load a trained SVD model from a file
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = joblib.load(self.get_full_path())

    def predict(self, user_id, candidates):
        """
        Predict the rating for a given user and movie
        Args:
            user_id (int): ID of the target user
            candidates (list): list of candidate movies
        Returns:
            float: Predicted rating
        """
        results = []
        user_id = str(user_id)

        for movie_id in candidates:
            pred_rating = self.model.predict(int(user_id), int(movie_id))
            results.append((int(movie_id), float(pred_rating.est)))
        results = list(filter(lambda x: x[1] >= self.threshold, results))

        results.sort(key=lambda x: x[1], reverse=True)
        results = [x[0] for x in results]
        return results

    def get_recommendations_impl(self, user_id, top_n):
        """
        Generate hybrid recommendations for a given user
        Args:
            user_id (int): ID of the target user
            top_n (int): Number of recommendations to return
        Returns:
            list: List of (title, score) tuples, sorted by descending score
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")


if __name__ == "__main__":
    # Exemple d'utilisation
    recommender = SVDRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
