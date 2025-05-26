import os
import sys
import random
import numpy as np
import torch
from torch import nn
import time
from torch import optim

import copy

from sklearn.model_selection import train_test_split

import math

from DatasetBatchIterator import DatasetBatchIterator
from NeuralColabFilteringNet import NeuralColabFilteringNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Model
from preprocessing.movie_manipulation import load_data_from_file


FORCE_TRAINING = True
FOLDER_SET = "0.1"


class RandomRecommender(Model):
    """
    A simple recommendation model that returns random movies.
    """

    def __init__(self, production=False, force_training=False):
        super().__init__(production, force_training)
        self.filename = "random_recommender"
        self.datasets = None
        self.model = None
        self.threshold = 3.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_data_impl(self):
        test_size = 0.2
        random_state = 7
        X = self.trainset[["userId", "movieId"]]
        Y = self.trainset["rating"].astype(np.float32)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        self.datasets = {"train": (X_train, Y_train), "test": (X_test, Y_test)}
        user_count = self.trainset["userId"].nunique()
        movie_count = self.trainset["movieId"].nunique()
        self.model = NeuralColabFilteringNet(user_count, movie_count)

    def save(self):
        # Rien à sauvegarder pour un modèle aléatoire
        pass

    def load_impl(self):
        # Rien à charger pour un modèle aléatoire
        pass

    def training_impl(self):
        self.model.to(self.device)
        self.model.train()
        # Hyper parameters
        lr = 1e-3
        wd = 1e-4
        batch_size = 2046
        max_epochs = 50
        early_stop_epoch_threshold = 3
        no_loss_reduction_epoch_counter = 0
        min_loss = np.inf
        min_loss_model_weights = None
        history = []
        iterations_per_epoch = int(math.ceil(len(self.datasets["train"]) // batch_size))
        min_epoch_number = 1
        epoch_start_time = 0
        loss_criterion = nn.MSELoss(reduction="sum")
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        print(f"Starting training loop...")
        training_start_time = time.perf_counter()
        for epoch in range(max_epochs):
            stats = {"epoch": epoch + 1, "total": max_epochs}
            epoch_start_time = time.perf_counter()
            # Every epoch runs training on train set, followed by eval on test set
            for phase in ("train", "test"):
                is_training = phase == "train"
                self.model.train(is_training)
                running_loss = 0.0
                n_batches = 0

                # Iterate on train/test datasets in batches
                for x_batch, y_batch in DatasetBatchIterator(
                    self.datasets[phase][0],
                    self.datasets[phase][1],
                    batch_size=batch_size,
                    shuffle=is_training,
                ):
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                    # We zero out the loss gradient, since PyTorch by default accumulates gradients
                    optimizer.zero_grad()

                    # We need to compute gradients only during training
                    with torch.set_grad_enabled(is_training):
                        outputs = self.model(x_batch[:, 0], x_batch[:, 1])
                        loss = loss_criterion(outputs, y_batch)

                        if is_training:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()

                # Compute overall epoch loss and update history tracker
                epoch_loss = running_loss / len(self.datasets[phase][0])
                stats[phase] = epoch_loss
                history.append(stats)

                # Handle early stopping
                if phase == "test":
                    stats["time"] = time.perf_counter() - epoch_start_time
                    print(
                        "Epoch [{epoch:03d}/{total:03d}][Time:{time:.2f} sec] Train Loss: {train:.4f} / Validation Loss: {test:.4f}".format(
                            **stats
                        )
                    )

                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        min_loss_model_weights = copy.deepcopy(self.model.state_dict())
                        no_loss_reduction_epoch_counter = 0
                        min_epoch_number = epoch + 1
                    else:
                        no_loss_reduction_epoch_counter += 1
            if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
                print(f"Early stopping applied. Minimal epoch: {min_epoch_number}")
                break
        print(
            f"Training completion duration: {(time.perf_counter() - training_start_time):.2f} sec. Validation Loss: {min_loss}"
        )
        self.model.load_state_dict(min_loss_model_weights)

    def predict(self, user_id, candidates):
        """
        Predict ratings for the given user and list of movie IDs by returning random movie IDs.

        Returns:
            list: List of movie IDs in random order
        """
        self.model.eval()
        movie_tensor = torch.tensor(candidates, dtype=torch.long).to(self.device)
        user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(
            self.device
        )
        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor)
        res = []
        for movie_id, prediction in zip(candidates, predictions):
            res.append((int(movie_id), prediction.item()))

        if user_id == 3 or user_id == "3":
            print(f"Predictions for user {user_id}:")
            for movie, rating in res:
                print(f"Movie ID: {movie}, Predicted Rating: {rating}")

        res = list(filter(lambda x: x[1] >= self.threshold, res))
        res.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in res]

    def get_recommendations_impl(self, user_id, top_n):
        # Rien à faire ici car la logique est dans predict
        pass


if __name__ == "__main__":
    # Exemple d'utilisation
    recommender = RandomRecommender(False, FORCE_TRAINING)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
