import torch
from torch.utils.data import Dataset



class MovieLensDataset(Dataset):
    """
    The Movie Lens Dataset class. This class prepares the dataset for training and validation.
    """

    def __init__(self, df_ratings, df_movies):
        """
        Initializes the dataset object with user, movie, and rating data.
        """
        self.ratings_data = df_ratings
        self.movies_data = df_movies

        self.user2idx = {
            uid: i for i, uid in enumerate(self.ratings_data["userId"].unique())
        }

        self.movie2idx = {mid: i for i, mid in enumerate(df_movies["movieId"].unique())}

        self.len_data = len(self.ratings_data)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.len_data

    def __getitem__(self, idx):
        row = self.ratings_data.iloc[idx]
        movie_id = row["movieId"]
        user = torch.tensor(self.user2idx[row["userId"]])
        movie = torch.tensor(self.movie2idx[row["movieId"]])
        rating = torch.tensor(row["rating"], dtype=torch.float32)

        plot_embedding = self.movies_data.loc[
            self.movies_data["movieId"] == movie_id, "plot_embedding"
        ].values[0]

        return {
            "users": user,
            "movies": movie,
            "plots": plot_embedding,
            "ratings": rating,
        }
