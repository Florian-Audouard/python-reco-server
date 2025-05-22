import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import tensorflow as tf
import tensorflow_recommenders as tfrs
import seaborn as sns


def printmd(string):
    print(string)


from pathlib import Path

# Load data, specifying column names and types
electronics_data = pd.read_csv(
    "ratings.csv",
    dtype={"rating": "float32"},
    names=["userId", "movieId", "rating", "timestamp"],
    index_col=None,
    header=0,
)

electronics_data.describe()["rating"].reset_index()


recent_prod = electronics_data.drop(columns=["timestamp"])

# Unique user and movie IDs as strings
userIds = recent_prod.userId.unique()
movieIds = recent_prod.movieId.unique()
unique_userIds = [str(x) for x in userIds]
unique_movieIds = [str(x) for x in movieIds]

total_ratings = len(recent_prod.index)

# Convert IDs to strings before creating tf.data.Dataset
ratings = tf.data.Dataset.from_tensor_slices(
    {
        "userId": recent_prod.userId.astype(str).values,
        "movieId": recent_prod.movieId.astype(str).values,
        "rating": recent_prod.rating.values.astype(np.float32),
    }
)

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(int(total_ratings * 0.8))
test = shuffled.skip(int(total_ratings * 0.8)).take(int(total_ratings * 0.2))


class RankingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_userIds, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_userIds) + 1, embedding_dimension),
            ]
        )

        self.product_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movieIds, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_movieIds) + 1, embedding_dimension
                ),
            ]
        )

        self.ratings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, userId, movieId):
        user_embeddings = self.user_embeddings(userId)
        product_embeddings = self.product_embeddings(movieId)
        return self.ratings(tf.concat([user_embeddings, product_embeddings], axis=1))


class amazonModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userId"], features["movieId"])
        return self.task(labels=features["rating"], predictions=rating_predictions)


model = amazonModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=10)
model.evaluate(cached_test, return_dict=True)

# Example inference: convert IDs to strings before passing to the model
user_rand = str(userIds[123])  # Convert to string to match model input type
test_rating = {}

print(f"Top 5 recommended products for User {user_rand}:")

for m in test.take(5):
    movie_id_bytes = m["movieId"].numpy()  # this is bytes
    movie_id_str = movie_id_bytes.decode("utf-8")  # decode directly from bytes to str

    user_tensor = tf.constant([user_rand], dtype=tf.string)
    movie_tensor = tf.constant([movie_id_str], dtype=tf.string)

    prediction = model.ranking_model(user_tensor, movie_tensor)
    test_rating[movie_id_str] = prediction.numpy()[0][0]


# Sort and print top 5 movies by predicted rating
for movie_id in sorted(test_rating, key=test_rating.get, reverse=True):
    print(movie_id)
