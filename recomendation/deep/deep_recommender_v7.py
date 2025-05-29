import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader, Dataset
import pandas as pd


from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from RatingClassifier import RatingClassifier
from RatingLabelDataset import RatingLabelDataset

from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


class DeepLearningRecommender(Model):
    """
    DeepLearningRecommender is a recommendation system model based on deep learning.
    """

    def __init__(self, production, force_training, noise=False, real_data=None):
        super().__init__(production, force_training, noise=noise, real_data=real_data)
        self.model = None
        self.filename = "deep_recommender_model.h5"
        self.torch_trainset = None
        self.train_loader = None
        self.threshold = 3.5
        self.epochs = 5
        self.device = torch.device("cpu")

    def init_data_impl(self):
        """
        Initialize the data for training and validation.
        """
        self.torch_trainset = RatingLabelDataset(self.trainset, self.movies)
        self.train_loader = DataLoader(
            self.torch_trainset,
            batch_size=512,
            shuffle=True,
        )

    def training_impl(self):
        """
        Train the deep learning model using the initialized data.
        """
        num_users = len(self.torch_trainset.user2idx)
        num_movies = len(self.torch_trainset.movie2idx)
        self.model = RatingClassifier(num_users, num_movies)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            for batch in self.train_loader:
                user_ids = batch["users"].to(self.device)
                movie_ids = batch["movies"].to(self.device)
                ratings = batch["ratings"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(user_ids, movie_ids)
                loss = loss_fn(outputs, ratings)
                loss.backward()
                optimizer.step()

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

    def __get_score(self, prediction):
        """
        Convert the model's output to a rating score.
        """
        score = self.torch_trainset.class_to_rating[torch.argmax(prediction).item()]
        return score

    def predict(self, user_id, candidates):
        """_summary_

        Args:
            user_id (_type_): _description_
            candidates (_type_): _description_
        """
        predictions = []
        user_idx = self.torch_trainset.user2idx.get(user_id)
        movie_indices = []
        for movie_id in candidates:
            movie_idx = self.torch_trainset.movie2idx.get(movie_id)
            movie_indices.append(movie_idx)
        with torch.no_grad():
            user_tensor = torch.tensor(
                [user_idx] * len(candidates), dtype=torch.long
            ).to(self.device)
            movie_tensor = torch.tensor(movie_indices, dtype=torch.long).to(self.device)
            self.model.eval()
            predictions = self.model(user_tensor, movie_tensor)
        res = []
        for movie_id, prediction in zip(candidates, predictions):
            res.append((int(movie_id), self.__get_score(prediction)))
        res = list(filter(lambda x: x[1] >= self.threshold, res))
        res = sorted(res, key=lambda x: x[1], reverse=True)
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
