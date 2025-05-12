"""
Deep Learning-based Recommender System using Neural Collaborative Filtering (NCF).
This model follows the structure of the SVD recommender and uses the Model base class.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from surprise import Prediction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model

FORCE_TRAINING = False


class NCFDataset(Dataset):
    def __init__(self, ratings):
        self.users = torch.tensor(ratings["userId"].values, dtype=torch.long)
        self.items = torch.tensor(ratings["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=-1)
        out = self.fc_layers(x)
        return out.squeeze()


class DeepRecommender(Model):
    def __init__(self, force_training):
        super().__init__(force_training)
        self.model = None
        self.filename = "deep_recommender.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = None
        self.num_items = None

    def training(self, epochs=5, batch_size=256, lr=0.001):
        self.num_users = int(self.ratings["userId"].max())
        self.num_items = int(self.ratings["movieId"].max())
        self.model = NCF(self.num_users, self.num_items).to(self.device)
        dataset = NCFDataset(self.ratings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for users, items, ratings in loader:
                users, items, ratings = (
                    users.to(self.device),
                    items.to(self.device),
                    ratings.to(self.device),
                )
                optimizer.zero_grad()
                outputs = self.model(users, items)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(ratings)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")

    def save(self):
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")
        os.makedirs(self.data_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.get_full_path())

    def load(self):
        self.num_users = int(self.ratings["userId"].max())
        self.num_items = int(self.ratings["movieId"].max())
        self.model = NCF(self.num_users, self.num_items).to(self.device)
        self.model.load_state_dict(
            torch.load(self.get_full_path(), map_location=self.device)
        )
        self.model.eval()

    def predict(self, user_id, top_n):
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")
        watched = self.ratings[self.ratings["userId"] == user_id]["movieId"].values
        candidates = self.movies_merged[~self.movies_merged["movieId"].isin(watched)]
        user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(
            self.device
        )
        item_tensor = torch.tensor(candidates["movieId"].values, dtype=torch.long).to(
            self.device
        )
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()
        candidates = candidates.copy()
        candidates["score"] = scores
        top_movies = candidates.sort_values("score", ascending=False).head(top_n)
        return list(zip(top_movies["title"], top_movies["score"]))

    def get_prediction_set(self):
        if self.model is None:
            raise RuntimeError("Model not initialized or trained")
        val = self.validation_set
        user_ids = np.array([int(x[0]) for x in val])
        item_ids = np.array([int(x[1]) for x in val])
        ratings = np.array([x[2] for x in val])
        user_tensor = torch.tensor(user_ids, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(item_ids, dtype=torch.long).to(self.device)
        with torch.no_grad():
            preds = self.model(user_tensor, item_tensor).cpu().numpy()
        return [
            Prediction(uid, iid, r_ui, est, None)
            for uid, iid, r_ui, est in zip(user_ids, item_ids, ratings, preds)
        ]


if __name__ == "__main__":
    recommender = DeepRecommender(FORCE_TRAINING)
    recommender.testing_main("0.1")
