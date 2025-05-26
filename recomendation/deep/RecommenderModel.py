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

        # Hidden layers
        # 384 is for all-MiniLM-L6-v2
        self.fc1 = nn.Linear(2 * self.embedding_size + 384, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, users, movies, plot_embeddings):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        movie_combined = torch.cat([movie_embedded, plot_embeddings], dim=1)
        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_combined], dim=1)
        # Pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output
