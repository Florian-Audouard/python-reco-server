import os
import sys
from abc import ABC, abstractmethod
from preprocessing.movie_manipulation import load_data
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.time_util import get_time_func
from utils.logger import log


class Model(ABC):
    """
    Abstract base class for recomendation models.
    """

    def __init__(self, production=False, force_training=False):
        """
        Initialize the model with default values.
        """
        self.ratings = None
        self.trainset = None
        self.validation_set = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, "data")
        self.filename = None
        self.min = None
        self.max = None
        self.force_training = force_training
        self.production = production

    @get_time_func
    def init_data(self, ratings):
        """
        Initialize the model with data
        Args:
            data (tuple): Tuple containing (movies, ratings, tags) DataFrames
        """
        self.ratings = ratings
        # Convert ratings DataFrame to Surprise dataset
        self.min = self.ratings["rating"].min()
        self.max = self.ratings["rating"].max()
        reader = Reader(rating_scale=(self.min, self.max))
        data = Dataset.load_from_df(
            self.ratings[["userId", "movieId", "rating"]], reader
        )

        # Split the data into trainset and validation set
        if self.production:
            self.trainset = data.build_full_trainset()
            return
        self.trainset, self.validation_set = train_test_split(data, test_size=0.2)

    def get_full_path(self):
        if self.filename is None:
            raise RuntimeError("Filename not set")
        return os.path.join(self.data_dir, self.filename)

    @abstractmethod
    @get_time_func
    def save(self):
        """
        Save the model to a file
        """

    @abstractmethod
    def load_impl(self):
        """
        function called when loading the model
        """

    @abstractmethod
    def training_impl(self):
        """
        Train the model on the given data
        """

    @abstractmethod
    def predict(self, user_id, list_movie_id):
        """
        Predict the rating for a given user and a list of movie
        Args:
            user_id: User ID
            list_movie_id: Movie ID list
        Returns:
            Predicted rating for each movie
        """

    @get_time_func
    def load(self):
        """
        Load the model from a file
        """
        full_path = self.get_full_path()
        train = False
        if not os.path.exists(full_path):
            log.error("File %s does not exist", full_path)
            train = True
        if self.force_training:
            log.info("FORCE TRAINING MODE")
            train = True
        if train:
            self.training_save()
            return

        log.info("LOADING FROM %s", full_path)
        self.load_impl()

    @get_time_func
    def training(self):
        """
        Train the model on the given data
        """
        if self.trainset is None:
            raise RuntimeError("Model not initialized or trained")
        self.training_impl()

    def training_save(self):
        """
        Train the model on the given data
        and save the model to a file
        """
        if self.trainset is None:
            raise RuntimeError("Model not initialized or trained")
        self.training()
        self.save()

    @get_time_func
    def get_recommendations(self, user_id, top_n):
        """
        Make predictions using the trained model
        Args:
            user_id: Input features to make predictions on
            top_n: Number of recommendations to return
        Returns:
            Predictions made by the model
        """
        if self.trainset is None:
            raise RuntimeError("Data not initialized")

        candidates = self.ratings[self.ratings["userId"] != user_id]["movieId"].unique()

        results = self.predict(user_id, candidates)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    @abstractmethod
    def get_prediction_set(self):
        """
        Get the prediction set
        Returns:
            The prediction set
        """

    def testing_main(self, folder):
        # Exemple d'utilisation
        data = load_data(f"ml-{folder}m")
        self.init_data(data)
        self.load()
        self.get_recommendations(user_id=2, top_n=5)
        accuracy = self.accuracy()
        for key, value in accuracy.items():
            log.info("%s: %s", key, value)

    def accuracy(self, k=10):
        """
        Calculate the accuracy of the model's predictions

        Returns:
            float: Accuracy score
        """
        threshold = (self.max - self.min) / 2
        predictions = self.get_prediction_set()
        user_est_true = defaultdict(list)
        for pred in predictions:
            user_est_true[pred.uid].append((pred.iid, pred.est, pred.r_ui))

        precisions = dict()
        recalls = dict()

        for uid, user_ratings in user_est_true.items():
            user_est_true[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)

            top_k = user_ratings[:k]

            relevant = sum((true_r >= threshold) for (_, _, true_r) in user_ratings)
            recommended = sum((est >= threshold) for (_, est, _) in top_k)
            relevant_and_recommended = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (_, est, true_r) in top_k
            )

            precisions[uid] = (
                relevant_and_recommended / recommended if recommended else 0
            )
            recalls[uid] = relevant_and_recommended / relevant if relevant else 0

        # Moyennes sur tous les utilisateurs
        avg_precision = sum(precisions.values()) / len(precisions)
        avg_recall = sum(recalls.values()) / len(recalls)
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

        return {
            "rmse": rmse(predictions, verbose=False),
            "mae": mae(predictions, verbose=False),
            "precision": avg_precision,
            "recall": avg_recall,
            "F1": f1,
        }
