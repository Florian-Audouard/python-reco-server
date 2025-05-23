import os
import sys
import numpy as np
import scipy.sparse

from markov_generation import build_movieid_mapping, markov_matrix, get_user_vector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


class MarkovRecommender(Model):
    """
    MarkovRecommender is a recommendation system model based on Markov chains.
    It builds a transition matrix between movies based on user co-occurrences
    and uses it to recommend movies to users.
    """

    def __init__(self, production=False, force_training=False):
        """
        Initializes the MarkovRecommender model.

        Args:
            production (bool): Production mode flag.
            force_training (bool): If True, forces retraining of the model.
        """
        super().__init__(production, force_training)
        self.model = None
        self.filename = "markov_model.npz"
        self.movieId_to_idx = None
        self.idx_to_movieId = None

    def init_data(self, ratings):
        """
        Initializes data and builds movie ID mappings.

        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings.
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

    def predict(self, user_vector, list_movie_id):
        """
        Predicts scores for a given user and a list of movies using the Markov model.

        Args:
            user_vector (np.ndarray): Binary vector indicating movies seen by the user.
            list_movie_id (list): List of movieIds to exclude (already seen).

        Returns:
            list: List of (movieId, predicted_score) tuples, sorted by score descending.
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        scores = self.model.dot(user_vector)

        # Convert movieIds to internal indices
        excluded_idxs = {
            self.movieId_to_idx[mid]
            for mid in list_movie_id
            if mid in self.movieId_to_idx
        }

        # Sort and filter out seen movies
        sorted_idxs = np.argsort(scores)[::-1]
        recommended_idxs = [
            index for index in sorted_idxs if index not in excluded_idxs
        ]

        # Map internal indices back to movie IDs
        return [(self.idx_to_movieId[idx], scores[idx]) for idx in recommended_idxs]

    def get_recommendations(self, user_id, top_n):
        """
        Generates the top-N recommendations for a given user.

        Args:
            user_id (int): The user ID.
            top_n (int): Number of recommendations to return.

        Returns:
            list: List of (movieId, score) tuples, sorted by score descending.
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)
        return super().get_recommendations(user_id, top_n)

    def get_prediction_set(self):
        """
        Generates a prediction set for validation (leave-one-out).
        For each user, hides one seen movie and predicts on the others.

        Returns:
            dict: user_id -> (list of predictions, masked index)
        """
        real_and_prediction_data_for_user = {}
        for user_id, _, _ in self.validation_set:
            if user_id in real_and_prediction_data_for_user:
                continue
            user_vector = get_user_vector(self.ratings, user_id, self.movieId_to_idx)
            seen_idxs = np.nonzero(user_vector)[0]
            hide = np.random.choice(seen_idxs)
            user_vector[hide] = 0

            seen_movie_ids = [
                self.idx_to_movieId[idx] for idx in seen_idxs if idx != hide
            ]
            preds = self.predict(user_vector, seen_movie_ids)

            real_and_prediction_data_for_user[user_id] = (preds, hide)
        return real_and_prediction_data_for_user

    def extract_top_k(self, sorted_prediction, k):
        """
        Returns the top-k items, including all ties with the k-th score.

        Args:
            sorted_prediction (list): List of (movieId, score) tuples, sorted by score descending.
            k (int): Minimum number of items to return.

        Returns:
            list: List of (movieId, score) with length >= k (if ties).
        """
        if not sorted_prediction:
            return []
        top_k = sorted_prediction[:k]
        if len(sorted_prediction) <= k:
            return top_k

        kth_score = sorted_prediction[k - 1][1]
        for item in sorted_prediction[k:]:
            if item[1] == kth_score:
                top_k.append(item)
            else:
                break
        return top_k

    def accuracy(self, k=10):
        """
        Computes precision for the model using leave-one-out:
        checks if the hidden movie is in the top-k recommendations.

        Args:
            k (int): Number of recommendations to consider.

        Returns:
            dict: {"precision": value}
        """
        data = self.get_prediction_set()
        relevant_count = 0
        for user_id, (preds, hide) in data.items():
            sorted_prediction = sorted(preds, key=lambda x: x[1], reverse=True)
            top_k = self.extract_top_k(sorted_prediction, k)
            recommended_ids = {ids for (ids, _) in top_k}
            if self.idx_to_movieId[hide] in recommended_ids:
                relevant_count += 1
        total = len(data)
        precision = relevant_count / total if total else 0
        return {"precision": precision}


if __name__ == "__main__":
    recommender = MarkovRecommender(False, True)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
