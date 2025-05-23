"""
Deep Learning-based Recommender System using Neural Collaborative Filtering (NCF).
This model follows the structure of the SVD recommender and uses the Model base class.
"""

import os

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from RatingDataset import RatingDataset
from DeepNCF import DeepNCF
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"
VERBOSE = True


def normalize_field(df, field):
    """
    Normalize the field in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        field (str): The field to normalize.
    Returns:
        pd.DataFrame: DataFrame with the normalized field.
    """
    min_val = df[field].min()
    max_val = df[field].max()
    df[field] = (df[field] - min_val) / (max_val - min_val)
    return df


class DeepRecommender(Model):
    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.threshold = 3.5
        self.training_data = None
        self.train_loader = None
        self.validation_data = None
        self.validation_loader = None
        self.num_users = None
        self.num_movies = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.epochs = 10
        self.filename = "deep_recommender"

    def init_data_impl(self):
        """
        Initialize the data for the recommender system.
        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings.
            movies (pd.DataFrame): DataFrame containing movie information.
        """
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.ratings["userId"] = self.user_encoder.fit_transform(self.ratings["userId"])
        self.ratings["movieId"] = self.movie_encoder.fit_transform(
            self.ratings["movieId"]
        )

        self.num_users = self.ratings["userId"].nunique()
        self.num_movies = self.ratings["movieId"].nunique()
        if self.production:
            self.trainset = self.ratings
        else:
            self.trainset, self.validation_set = train_test_split(
                self.ratings, test_size=0.2, random_state=42
            )
            self.validation_data = RatingDataset(self.validation_set)
            self.test_loader = torch.utils.data.DataLoader(
                self.validation_data, batch_size=64
            )

        self.training_data = RatingDataset(self.trainset)
        self.train_loader = torch.utils.data.DataLoader(
            self.training_data, batch_size=64, shuffle=True
        )
        normalize_field(self.ratings, "rating")
        normalize_field(self.movies, "runtime")
        normalize_field(self.movies, "year")
        self.movies["features"] = self.movies.apply(
            lambda row: row["genre_vec"]
            + [row["year"], row["runtime"]]
            + row["country_vec"],
            axis=1,
        )

    def encode_trainset(self, df):
        df["user"] = self.user_encoder.fit_transform(df["userId"])
        df["movie"] = self.movie_encoder.fit_transform(df["movieId"])

    def training_impl(self):
        self.model = DeepNCF(self.num_users, self.num_movies, embedding_dim=50)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        if VERBOSE:
            interator_var = range(self.epochs)
        else:
            interator_var = tqdm(range(self.epochs), desc="Training")

        for epoch in interator_var:
            self.model.train()
            total_loss = 0
            for user, movie, rating in self.train_loader:
                optimizer.zero_grad()
                features = self.movies["features"].iloc[movie].values
                prediction = self.model(user, movie, features)
                loss = loss_fn(prediction, rating)
                loss.backward()
                optimizer.step()
            if VERBOSE:
                print(
                    f"Epoch {epoch+1}, Training Loss: {total_loss/len(self.train_loader):.4f}"
                )
        self.model.eval()

    def save(self):
        pass

    def load_impl(self):
        pass

    def predict(self, user_id, candidates):
        res = []
        for movie_id in candidates:
            prediction = self.model(
                torch.tensor([user_id], dtype=torch.long),
                torch.tensor([movie_id], dtype=torch.long),
            ).item()
            res.append((movie_id, prediction))
        if user_id == 3:
            for result in res:
                print(f"Movie ID: {result[0]}, Prediction: {result[1]}")
        res = list(filter(lambda x: x[1] >= self.threshold, res))
        res = sorted(res, key=lambda x: x[1], reverse=True)
        return [int(x[0]) for x in res]


def main():
    recommender = DeepRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")

    recommender.testing_main(ratings, movies)


if __name__ == "__main__":
    main()
