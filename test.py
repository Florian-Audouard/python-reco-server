import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("ratings.csv")

# Encode userId and movieId to indices
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
df["user"] = user_encoder.fit_transform(df["userId"])
df["movie"] = movie_encoder.fit_transform(df["movieId"])

num_users = df["user"].nunique()
num_movies = df["movie"].nunique()

# Prepare train/test
train, test = train_test_split(df, test_size=0.2, random_state=42)


# Torch datasets
class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


train_data = RatingDataset(train)
test_data = RatingDataset(test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


# Model
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

    def forward(self, user_ids, movie_ids):
        user_vecs = self.user_embedding(user_ids)
        movie_vecs = self.movie_embedding(movie_ids)
        return (user_vecs * movie_vecs).sum(1)


model = MatrixFactorization(num_users, num_movies)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(10):
    model.train()
    total_loss = 0
    for user, movie, rating in train_loader:
        optimizer.zero_grad()
        prediction = model(user, movie)
        loss = loss_fn(prediction, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    mse = 0
    for user, movie, rating in test_loader:
        prediction = model(user, movie)
        mse += loss_fn(prediction, rating).item()
    print(f"Test MSE: {mse/len(test_loader):.4f}")
