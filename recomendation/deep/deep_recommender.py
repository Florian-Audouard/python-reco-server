"""
Deep Learning-based Recommender System using Neural Collaborative Filtering (NCF).
This model follows the structure of the SVD recommender and uses the Model base class.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model

FORCE_TRAINING = False


class DeepRecommender(Model):
    def __init__(self, production, force_training):
        super().__init__(production, force_training)

    def training_impl(self, epochs=5, batch_size=256, lr=0.001):
        pass

    def save(self):
        pass

    def load_impl(self):
        pass

    def predict(self, user_id, candidates):
        pass

    def get_recommendations(self, user_id, top_n):
        pass

    def get_prediction_set(self):
        pass


if __name__ == "__main__":
    recommender = DeepRecommender(False, FORCE_TRAINING)
    recommender.testing_main("0.1")
