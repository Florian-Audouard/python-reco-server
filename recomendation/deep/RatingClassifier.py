import torch
import torch.nn as nn


class RatingClassifier(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 11)  # 11 classes for the 11 possible ratings

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # logits (no softmax here)
