import torch


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, df, movie_features=None):
        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)
        self.movie_features = movie_features  # a tensor matrix [movie_id -> features]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        features = (
            self.movie_features[movie]
            if self.movie_features is not None
            else torch.tensor([])
        )
        return user, movie, features, rating
