import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RatingLabelDataset(Dataset):
    def __init__(self, ratings_data, movies_data):
        self.data = ratings_data
        self.user2idx = {uid: i for i, uid in enumerate(self.data["userId"].unique())}
        self.movie2idx = {
            mid: i for i, mid in enumerate(movies_data["movieId"].unique())
        }
        self.rating_to_class = {
            0.0: 0,
            0.5: 1,
            1.0: 2,
            1.5: 3,
            2.0: 4,
            2.5: 5,
            3.0: 6,
            3.5: 7,
            4.0: 8,
            4.5: 9,
            5.0: 10,
        }
        self.class_to_rating = {v: k for k, v in self.rating_to_class.items()}
        print(f"{self.rating_to_class}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = torch.tensor(self.user2idx[row["userId"]], dtype=torch.long)
        movie = torch.tensor(self.movie2idx[row["movieId"]], dtype=torch.long)
        rating = torch.tensor(self.rating_to_class[row["rating"]], dtype=torch.long)
        return {
            "users": user,
            "movies": movie,
            "ratings": rating,
        }
