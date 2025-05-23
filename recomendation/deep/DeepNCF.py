import torch.nn as nn


class DeepNCF(nn.Module):
    def __init__(self, n_users, n_movies, movie_feature_dim, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.movie_feature_layer = nn.Linear(movie_feature_dim, embedding_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, user_ids, movie_ids, movie_features):
        user_vecs = self.user_embedding(user_ids)
        movie_vecs = self.movie_embedding(movie_ids)
        movie_feature_vecs = self.movie_feature_layer(movie_features)

        combined = torch.cat([user_vecs, movie_vecs, movie_feature_vecs], dim=1)
        return self.output_layer(combined).squeeze()
