import os
import sys
import numpy as np
import scipy.sparse

from markov_generation import build_movieid_mapping, markov_matrix, get_user_vector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model
from preprocessing.movie_manipulation import load_data

FORCE_TRAINING = True
FOLDER_SET = "0.1"

class MarkovRecommender(Model):
    """
    MarkovRecommender is a recommendation system model based on Markov chains.
    It builds a transition matrix between movies based on user co-occurrences
    and uses it to recommend movies to users.
    """

    def __init__(self, production=False, force_training=False):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "markov_model.npz"
        self.movieId_to_idx = None
        self.idx_to_movieId = None

    def init_data(self, ratings):
        """
        Initialize data and build movie ID mappings.
        """
        super().init_data(ratings)
        # build mapping after ratings is set
        self.movieId_to_idx, self.idx_to_movieId = build_movieid_mapping(self.ratings)

    def training_impl(self):
        """
        Trains the Markov model by building the Markov matrix from the training set.
        """
        self.model = markov_matrix(self.trainset, self.movieId_to_idx)

    def save(self):
        """
        Saves the trained Markov model (as a sparse matrix) to disk.
        Raises:
            RuntimeError: If the model is not trained.
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        os.makedirs(self.data_dir, exist_ok=True)
        scipy.sparse.save_npz(self.get_full_path(), self.model)

    def load_impl(self):
        """
        Loads the Markov model from disk.
        Raises:
            RuntimeError: If the model is not initialized.
        """
        # no need to check self.model
        os.makedirs(self.data_dir, exist_ok=True)
        self.model = scipy.sparse.load_npz(self.get_full_path())

    def predict(self, user_id, list_movie_id):
        """
        Predicts scores for a given user and a list of movies using the Markov model.
        Returns a list of (movie_id, predicted_score) tuples.
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        user_vector = get_user_vector(self.ratings, user_id, self.movieId_to_idx)
        scores = self.model.dot(user_vector)/user_vector.sum()  

        # sort and select top candidates
        sorted_idxs = np.argsort(scores)[::-1]
        recommended_idxs = [index for index in sorted_idxs if index not in list_movie_id]
        # map internal indices back to movie IDs
        return [(self.idx_to_movieId[idx], scores[idx]) for idx in recommended_idxs]

    def get_recommendations(self, user_id, top_n):
        """
        Generates the top-N recommendations for a given user.
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)
        return super().get_recommendations(user_id, top_n)

    def get_prediction_set(self):
        """
        Generates a prediction set for validation.
        Returns dict: user_id -> (predictions, masked_index)
        """
        real_and_prediction_data_for_user = {}
        for (user_id, _, _) in self.validation_set:
            if user_id in real_and_prediction_data_for_user:
                continue
            user_vector = get_user_vector(self.ratings, user_id, self.movieId_to_idx)
            seen_idxs = np.nonzero(user_vector)[0]
            if seen_idxs.size == 0:
                continue
            hide = np.random.choice(seen_idxs)
            user_vector[hide] = 0
            preds = self.predict(user_id, list(seen_idxs[seen_idxs != hide]))
            real_and_prediction_data_for_user[user_id] = (preds, hide)
        return real_and_prediction_data_for_user

    def accuracy(self, k=100):
        """
        Computes precision, recall, and F1-score for the model.
        Uses leave-one-out: checks if hidden movie is in top-k.
        """
        data = self.get_prediction_set()
        relevant_count = 0
        for user_id,(preds, hide) in data.items():
            top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
            recommended_ids = {movie_id for (movie_id, _) in top_k}
            if hide in recommended_ids:
                relevant_count += 1

        total = len(data)
        precision = relevant_count / total if total else 0
        return {"precision": precision}

if __name__ == "__main__":
    recommender = MarkovRecommender(False, True)
    data = load_data(f"ml-{FOLDER_SET}m")
    recommender.testing_main(data)
