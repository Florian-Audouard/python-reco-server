import os
import sys
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.time_util import get_time_func
from utils.logger import log
from preprocessing.movie_manipulation import smart_split


class Model(ABC):
    """
    Abstract base class for recomendation models.
    """

    ERROR_MESSAGE = "Model not initialized or trained"

    def __init__(self, production=False, force_training=False):
        """
        Initialize the model with default values.
        """
        self.ratings = None
        self.movies = None
        self.trainset = None
        self.validation_set = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, "data")
        self.filename = None
        self.min = None
        self.max = None
        self.force_training = force_training
        self.production = production
        self.threshold = None

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

    @abstractmethod
    def init_data_impl(self):
        """
        Initialize the model with data
        Args:
            data (tuple): Tuple containing (movies, ratings, tags) DataFrames
        """

    def get_recommendations_impl(self, user_id, top_n):
        """
        Get recommendations for a given user
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
        Returns:
            Recommendations for the user
        """
        pass

    def accuracy_impl(self):
        """
        Calculate the accuracy of the model
        Args:
            top_n: Number of recommendations to consider
            note: Threshold rating to consider a movie relevant
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        return {}

    @get_time_func
    def init_data(self, ratings, movies):
        """
        Initialize the model with data
        Args:
            data (tuple): Tuple containing (movies, ratings, tags) DataFrames
        """
        self.ratings = ratings
        self.movies = movies
        self.movies["released"] = pd.to_datetime(self.movies["released"])
        self.movies["year"] = self.movies["released"].dt.year
        self.movies["month"] = self.movies["released"].dt.month
        self.movies["day"] = self.movies["released"].dt.day
        # Convert ratings DataFrame to Surprise dataset
        self.min = self.ratings["rating"].min()
        self.max = self.ratings["rating"].max()

        # Split the data into trainset and validation set
        if self.production:
            self.trainset = ratings
            self.init_data_impl()
            return
        self.trainset, self.validation_set = train_test_split(
            self.ratings,
            test_size=0.2,
            random_state=3,
            stratify=self.ratings.rating.values,
        )
        self.init_data_impl()

    @get_time_func
    def load(self):
        """
        Load the model from a file
        """
        full_path = self.get_full_path()
        train = False
        if not os.path.exists(full_path):
            log.warning("File %s does not exist", full_path)
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
            raise RuntimeError(self.ERROR_MESSAGE)
        self.training_impl()

    def training_save(self):
        """
        Train the model on the given data
        and save the model to a file
        """
        if self.trainset is None:
            raise RuntimeError(self.ERROR_MESSAGE)
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

        candidates = self.ratings[self.ratings["userId"] != user_id]["movieId"].unique()

        return self.get_recommendations_with_candidates(
            user_id=user_id, candidates=candidates, top_n=top_n
        )

    def testing_main(self, ratings, movies):
        # Exemple d'utilisation
        self.init_data(ratings, movies)
        self.load()
        reco = self.get_recommendations(user_id=2, top_n=10)
        log.info("Recommendations for user 2: %s", reco)
        accuracy = self.accuracy()
        for key, value in accuracy.items():
            log.info("%s: %s", key, value)

    def get_recommendations_with_candidates(self, user_id, candidates, top_n):
        """
        Make predictions using the trained model
        Args:
            user_id: Input features to make predictions on
            top_n: Number of recommendations to return
        Returns:
            Predictions made by the model
        """
        self.get_recommendations_impl(user_id, top_n)
        if self.trainset is None:
            raise RuntimeError("Data not initialized")

        results = self.predict(user_id, candidates)
        if top_n == -1:
            return results
        return results[:top_n]

    def get_pres_recall(self, user, note, top_n):
        candidates = self.validation_set[self.validation_set["userId"] == user]
        relevant_items = set(
            candidates[candidates["rating"] >= note]["movieId"].unique()
        )
        candidates = candidates["movieId"].unique()
        recommendations = set(
            self.get_recommendations_with_candidates(user, candidates, -1)
        )
        hits = recommendations.intersection(relevant_items)
        precision = len(hits) / len(recommendations) if recommendations else 0
        recall = len(hits) / len(relevant_items) if relevant_items else 0
        return precision, recall

    @get_time_func
    def accuracy(self, top_n=5, note=3.5):
        if self.validation_set is None:
            raise RuntimeError("Validation set not initialized")
        if self.threshold is not None:
            self.threshold = note
        precisions = []
        recalls = []
        for user in tqdm(
            self.validation_set["userId"].unique(), desc="Calculating accuracy"
        ):

            precision, recall = self.get_pres_recall(user, note, top_n)
            precisions.append(precision)
            recalls.append(recall)

        # Moyenne des métriques
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": (
                (2 * avg_precision * avg_recall / (avg_precision + avg_recall))
                if avg_precision + avg_recall > 0
                else 0
            ),
        } | self.accuracy_impl()
