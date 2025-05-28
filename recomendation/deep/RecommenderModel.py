import torch
import torch.nn as nn


class RecommenderModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=64,
        hidden_dim=128,
        dropout_rate=0.2,
    ):
        super(RecommenderModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, users, movies):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        output = self.linear_relu_stack(combined)

        return output
