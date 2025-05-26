import os
import sys
from sympy import plot
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer


from torch.utils.data import DataLoader, Dataset
import pandas as pd

from MovieLensDataset import MovieLensDataset
from RecommenderModel import RecommenderModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


def round_to_half_or_zero(x):
    if x < 0.25:
        return 0
    return round(x * 2) / 2


class DeepLearningRecommender(Model):
    """
    DeepLearningRecommender is a recommendation system model based on deep learning.
    """

    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "deep_recommender_model.h5"
        self.torch_trainset = None
        self.train_loader = None
        self.torch_validation_set = None
        self.threshold = 3.5
        self.epochs = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_data_impl(self):
        """
        Initialize the data for training and validation.
        """
        self.transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.movies["plot_embedding"] = [
            emb.cpu()
            for emb in self.transformer.encode(
                self.movies["plot"].tolist(), convert_to_tensor=True
            )
        ]
        self.torch_trainset = MovieLensDataset(self.trainset, self.movies)
        self.train_loader = DataLoader(
            self.torch_trainset,
            batch_size=32,
            shuffle=True,
        )

    def training_impl(self):
        """
        Train the deep learning model using the initialized data.
        """
        num_users = len(self.torch_trainset.user2idx)
        num_movies = len(self.torch_trainset.movie2idx)
        self.model = RecommenderModel(num_users, num_movies).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                user_ids = batch["users"].to(self.device)
                movie_ids = batch["movies"].to(self.device)
                ratings = batch["ratings"].to(self.device)
                plot_embeddings = batch["plots"].to(self.device)

                optimizer.zero_grad()

                predictions = self.model(user_ids, movie_ids, plot_embeddings)
                loss = criterion(predictions, ratings)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader)}")

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
        self.model.eval()
        user_idx = self.torch_trainset.user2idx.get(user_id)
        if user_idx is None:
            raise ValueError(f"Unknown user ID: {user_id}")

        movie_indices = []
        plot_embeddings = []
        for movie_id in candidates:
            movie_idx = self.torch_trainset.movie2idx.get(movie_id)
            if movie_idx is None:
                raise ValueError(f"Unknown movie ID: {movie_id}")
            movie_indices.append(movie_idx)

            emb = self.torch_trainset.movies_data.iloc[movie_idx]["plot_embedding"]
            if isinstance(emb, torch.Tensor):
                emb = emb.to(self.device)
            else:
                emb = torch.tensor(emb, device=self.device)
            plot_embeddings.append(emb)

        user_tensor = torch.tensor(
            [user_idx] * len(movie_indices), dtype=torch.long
        ).to(self.device)
        movie_tensor = torch.tensor(movie_indices, dtype=torch.long).to(self.device)
        plot_embeddings_tensor = torch.stack(plot_embeddings)

        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor, plot_embeddings_tensor)

        res = []
        for movie_id, prediction in zip(candidates, predictions):
            res.append((int(movie_id), round_to_half_or_zero(prediction.item())))

        if user_id == 3 or user_id == "3":
            print(f"Predictions for user {user_id}:")
            for movie, rating in res:
                print(f"Movie ID: {movie}, Predicted Rating: {rating}")

        res = list(filter(lambda x: x[1] >= self.threshold, res))
        res.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in res]

    def get_recommendations_impl(self, user_id, top_n):
        """
        Generate recommendations for a given user.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")


def main():
    """
    Main function to run the recommender system.
    """
    recommender = DeepLearningRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)


if __name__ == "__main__":
    main()
