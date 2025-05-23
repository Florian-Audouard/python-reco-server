import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


class DeepLearningRecommender(Model):
    """
    DeepLearningRecommender is a recommendation system model based on deep learning.
    """

    def __init__(self, production, force_training, embedding_size=50):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "deep_recommender_model.h5"
        self.embedding_size = embedding_size
        self.user_id_mapping = None
        self.movie_id_mapping = None
        self.batch_size = 64
        self.epochs = 50
        self.threshold = 3.5

    def init_data_impl(self):
        """
        Initialize the data for training and validation.
        """
        user_ids = self.trainset["userId"].unique()
        movie_ids = self.movies["movieId"].unique()

        self.user_id_mapping = {id: idx for idx, id in enumerate(user_ids)}
        self.movie_id_mapping = {id: idx for idx, id in enumerate(movie_ids)}

        # Map userId and movieId
        self.trainset["userId"] = (
            self.trainset["userId"].map(self.user_id_mapping).fillna(-1).astype(int)
        )
        self.trainset["movieId"] = (
            self.trainset["movieId"].map(self.movie_id_mapping).fillna(-1).astype(int)
        )

    def build_model(self, num_users, num_movies):
        """
        Build the deep learning model.
        """
        user_input = Input(shape=(1,), name="user_input")
        movie_input = Input(shape=(1,), name="movie_input")

        user_embedding = Embedding(input_dim=num_users, output_dim=self.embedding_size)(
            user_input
        )
        movie_embedding = Embedding(
            input_dim=num_movies, output_dim=self.embedding_size
        )(movie_input)

        user_vec = Flatten()(user_embedding)
        movie_vec = Flatten()(movie_embedding)

        concat = Concatenate()([user_vec, movie_vec])

        x = Dense(128, activation="relu")(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output = Dense(1, activation="linear")(x)

        model = KerasModel(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

        return model

    def training_impl(self):
        """
        Train the deep learning model using the initialized data.
        """
        self.model = self.build_model(
            max(self.user_id_mapping.values()) + 1,
            max(self.movie_id_mapping.values()) + 1,
        )

        # Use self.trainset for training
        X = self.trainset[["userId", "movieId"]].values
        y = self.trainset["rating"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(
            [X_train[:, 0], X_train[:, 1]],
            y_train,
            validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
        )

    def save(self):
        """
        Save the trained deep learning model to a file.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")

        # self.model.save(self.get_full_path())

    def load_impl(self):
        """
        Load a trained deep learning model from a file.
        """
        # self.model = KerasModel.load_model(self.get_full_path())

    def predict(self, user_id, candidates):
        """
        Predict the rating for a given user and multiple movies (candidates).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")

        user_idx = self.user_id_mapping[user_id]
        # Map all candidate movie IDs to indices
        movie_indices = np.array(
            [self.movie_id_mapping[np.int64(m)] for m in candidates]
     
    def get_recommendations_impl(self, user_id, top_n):
        """
        Generate recommendations for a given user.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")

   user_array = np.full(shape=len(candidates), fill_value=user_idx, dtype=np.int64)

        # Predict all at once
        preds = self.model.predict([user_array, movie_indices], verbose=0).flatten()
        # Filter based on threshold
        filtered = [
            (movie, pred)
            for movie, pred in zip(candidates, preds)
            if pred >= self.threshold
        ]

        # Sort by predicted rating descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Return just the movie IDs sorted by prediction score
        return [movie for movie, _ in filtered]

    def get_recommendations_impl(self, user_id, top_n):
        """
        Generate recommendations for a given user.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")


if __name__ == "__main__":
    # Example usage
    recommender = DeepLearningRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
