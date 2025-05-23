"""
Deep Learning-based Recommender System using Neural Collaborative Filtering (NCF).
This model follows the structure of the SVD recommender and uses the Model base class.
"""

import os

import sys
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from MovieModel import MovieModel


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


def get_tf_dataset(data_df):

    res = tf.data.Dataset.from_tensor_slices(
        {
            "userId": data_df.userId.astype(str).values,
            "movieId": data_df.movieId.astype(str).values,
            "rating": data_df.rating.values.astype(np.float32),
        }
    )
    length = len(data_df.index)
    return res.shuffle(length, reshuffle_each_iteration=False)


class DeepRecommender(Model):
    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.threshold = 3.5
        self.training_data = None
        self.validation_data = None

        self.unique_userIds = None
        self.unique_movieIds = None

        self.filename = "deep_recommender"

    def init_data_impl(self):
        """
        Initialize the data for the recommender system.
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings.
            movies (pd.DataFrame): DataFrame containing movie information.
        """
        self.unique_movieIds = self.ratings.movieId.unique()
        self.unique_userIds = self.ratings.userId.unique()
        self.unique_movieIds = [str(x) for x in self.unique_movieIds]
        self.unique_userIds = [str(x) for x in self.unique_userIds]
        merged_training = self.trainset.merge(self.movies, on="movieId")
        self.training_data = get_tf_dataset(merged_training)
        if not self.production:
            merged_validation = self.validation_set.merge(self.movies, on="movieId")

            self.validation_data = get_tf_dataset(merged_validation)

    def training_impl(self):
        self.model = MovieModel(self.unique_userIds, self.unique_movieIds)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.15))
        cached_train = self.training_data.batch(8192).cache()
        self.model.fit(cached_train, epochs=60)

    def save(self):
        pass

    def load_impl(self):
        pass

    def predict(self, user_id, candidates):
        user_id = str(user_id)
        res = []
        for movie in candidates:
            movie_id = str(movie)
            user_tensor = tf.constant([user_id], dtype=tf.string)
            movie_tensor = tf.constant([movie_id], dtype=tf.string)
            prediction = self.model.ranking_model(user_tensor, movie_tensor)
            res.append((movie_id, prediction.numpy()[0][0]))
        if user_id == "3":
            for value in res:
                print(f"Movie: {value[0]}, Prediction: {value[1]}")
        res = list(filter(lambda x: x[1] >= self.threshold, res))
        res.sort(key=lambda x: x[1], reverse=True)
        res = [int(x[0]) for x in res]
        return res


def main():
    recommender = DeepRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")

    recommender.testing_main(ratings, movies)


if __name__ == "__main__":
    main()
