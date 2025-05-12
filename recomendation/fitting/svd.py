"""
Module de recommandation de films basé sur une approche hybride (filtrage collaboratif + filtrage de contenu).
"""

import os
import sys
from surprise import SVD
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model


FORCE_TRAINING = False


class SVDRecommender(Model):
    """
    SVDRecommender is a recommendation system model based on Singular Value Decomposition (SVD).
    """

    def __init__(self, force_training):
        super().__init__(force_training)
        self.model = None
        self.filename = "svd_model.joblib"

    def training(self):
        """
        Train the SVD model using the initialized data
        """
        self.model = SVD()
        self.model.fit(self.trainset)

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

    def load(self):
        """
        Load a trained SVD model from a file
        Args:
            filepath (str): Path to the saved model file
        """
        super().load()
        self.model = joblib.load(self.get_full_path())

    def predict(self, user_id, top_n):
        """
        Generate hybrid recommendations for a given user
        Args:
            user_id (int): ID of the target user
            top_n (int): Number of recommendations to return
            alpha (float): Weight between collaborative (alpha) and content (1-alpha)
        Returns:
            list: List of (title, score) tuples, sorted by descending score
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")

        watched = self.ratings[self.ratings["userId"] == user_id]["movieId"].values
        candidates = self.movies_merged[~self.movies_merged["movieId"].isin(watched)]

        results = []
        for _, row in candidates.iterrows():
            movie_id = row["movieId"]

            pred_rating = self.model.predict(user_id, movie_id).est

            results.append((row["title"], pred_rating))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def get_prediction_set(self):
        """
        Generate a set of predictions for the validation set
        Returns:
            list: List of (user_id, movie_id, actual_rating) tuples
        """
        return self.model.test(self.validation_set)


if __name__ == "__main__":
    # Exemple d'utilisation
    recommender = SVDRecommender(FORCE_TRAINING)
    recommender.testing_main("0.1", FORCE_TRAINING)
