import os
import sys
from abc import ABC, abstractmethod
from preprocessing.movie_manipulation import group_data, load_data
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
        self.movies = None
        self.ratings = None
        self.tags = None
        self.movies_merged = None
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
    def init_data(self, data):
        """
        Initialize the model with data
        Args:
            data (tuple): Tuple containing (movies, ratings, tags) DataFrames
        """
        self.movies, self.ratings, self.tags = data
        self.movies_merged = group_data(self.movies, self.tags)
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
    @get_time_func
    def load(self):
        """
        Load the model from a file
        """
        full_path = self.get_full_path()
        train = False
        if not os.path.exists(full_path):
            log.error(f"File {full_path} does not exist")
            train = True
        if self.force_training:
            log.info("Force training mode")
            train = True
        if train:
            self.training_save()

    @abstractmethod
    @get_time_func
    def training(self):
        """
        Train the model on the given data
        """
        if self.filename is None:
            raise RuntimeError("Filename not set")

    @get_time_func
    def training_save(self):
        """
        Train the model on the given data
        and save the model to a file
        """
        if self.trainset is None:
            raise RuntimeError("Model not initialized or trained")
        self.training()
        self.save()

    @abstractmethod
    @get_time_func
    def predict(self, user_id, top_n):
        """
        Make predictions using the trained model
        Args:
            user_id: Input features to make predictions on
            top_n: Number of recommendations to return
        Returns:
            Predictions made by the model
        """
        if self.movies_merged is None:
            raise RuntimeError("Model not initialized or trained")

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
        recommendations = self.predict(user_id=1, top_n=5)
        recommendations2 = self.predict(user_id=10, top_n=5)
        print(recommendations)
        print(recommendations2)
        accuracy = self.accuracy()
        print(accuracy)

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
