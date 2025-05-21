"""
Deep Learning-based Recommender System using Neural Collaborative Filtering (NCF).
This model follows the structure of the SVD recommender and uses the Model base class.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from CombinedModel import CombinedModel


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


def get_tf_dataset(data_df, movies):
    res = tf.data.Dataset.from_tensor_slices(
        {
            "userId": data_df["userId"].astype(str).values,
            "movieId": data_df["movieId"].astype(str).values,
            "rating": data_df["rating"].astype(np.float32).values,
        }
    )
    return res


class DeepRecommender(Model):
    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.user_data = None
        self.movie_data = None
        self.training_data = None
        self.validation_data = None

        self.movies_tf = None

        self.user_vocab = None
        self.movie_vocab = None
        self.embedding_dim = 32
        self.user_model = None
        self.movie_model = None
        self.ranking_model = None
        self.ranking_task = None
        self.retrieval_task = None

        self.index = None

        self.filename = "deep_recommender"

    def init_data_impl(self):
        """
        Initialize the data for the recommender system.
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings.
            movies (pd.DataFrame): DataFrame containing movie information.
        """
        self.training_data = get_tf_dataset(self.trainset, self.movies)
        if not self.production:
            self.validation_data = get_tf_dataset(self.validation_set, self.movies)
        self.movies_tf = tf.data.Dataset.from_tensor_slices(
            self.movies["movieId"].astype(str).values
        )

        # 2. Vocabulary layers
        self.user_vocab = tf.keras.layers.StringLookup(mask_token=None)
        self.user_vocab.adapt(self.training_data.map(lambda x: x["userId"]))

        self.movie_vocab = tf.keras.layers.StringLookup(mask_token=None)
        self.movie_vocab.adapt(self.movies_tf)
        self.user_model = tf.keras.Sequential(
            [
                self.user_vocab,
                tf.keras.layers.Embedding(
                    self.user_vocab.vocab_size(), self.embedding_dim
                ),
            ]
        )

        self.movie_model = tf.keras.Sequential(
            [
                self.movie_vocab,
                tf.keras.layers.Embedding(
                    self.movie_vocab.vocab_size(), self.embedding_dim
                ),
            ]
        )
        self.ranking_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        self.ranking_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.movies_tf.batch(128).map(self.movie_model)
            )
        )

    def training_impl(self):

        self.model = CombinedModel(
            self.user_model,
            self.movie_model,
            self.ranking_model,
            self.retrieval_task,
            self.ranking_task,
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

        cached_train = self.training_data.batch(2048).cache()

        self.model.fit(cached_train, epochs=5, verbose=1)

    def save(self):
        pass

    def load_impl(self):
        pass

    def predict(self, user_id, candidates):
        # Convertir movieIds en chaînes
        candidates_str = [str(c) for c in candidates]
        candidates_tf = tf.data.Dataset.from_tensor_slices(candidates_str)

        # Appliquer batch avant l'encodage
        batched_candidates = candidates_tf.batch(128)

        # Obtenir les embeddings
        candidate_embeddings = batched_candidates.map(self.model.movie_model)

        # Reconstruire les IDs en dataset (car ils ont été batchés)
        candidate_ids = tf.data.Dataset.from_tensor_slices(candidates_str).batch(128)

        # Associer IDs et embeddings
        indexed_ds = tf.data.Dataset.zip((candidate_ids, candidate_embeddings))

        # Créer un nouvel index temporaire
        temp_index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)
        temp_index.index_from_dataset(indexed_ds)

        # Prédire pour l'utilisateur
        k = len(candidates)
        scores, recommended_movies = temp_index(tf.constant([str(user_id)]), k=k)
        recommended_movies = recommended_movies[0, :].numpy()

        return [int(m.decode("utf-8")) for m in recommended_movies]


def main():
    recommender = DeepRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")

    recommender.testing_main(ratings, movies)


if __name__ == "__main__":
    main()
