import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam


def load_and_preprocess_data(ratings_path="ratings.csv", movies_path="movies.csv"):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_id_mapping = {id: idx for idx, id in enumerate(user_ids)}
    movie_id_mapping = {id: idx for idx, id in enumerate(movie_ids)}

    ratings["userId"] = ratings["userId"].map(user_id_mapping)
    ratings["movieId"] = ratings["movieId"].map(movie_id_mapping)

    return ratings, movies, user_id_mapping, movie_id_mapping


def build_model(num_users, num_movies, embedding_size=50):
    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(
        user_input
    )
    movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(
        movie_input
    )

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    concat = Concatenate()([user_vec, movie_vec])

    x = Dense(128, activation="relu")(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model


def train_model(model, ratings, batch_size=64, epochs=10):
    X = ratings[["userId", "movieId"]].values
    y = ratings["rating"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    history = model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )
    return history


def predict_rating(model, user_id, movie_id, user_id_mapping, movie_id_mapping):
    if user_id not in user_id_mapping or movie_id not in movie_id_mapping:
        print("User  ID or Movie ID not found in training data.")
        return None

    user_idx = user_id_mapping[user_id]
    print(type(movie_id), movie_id)
    movie_idx = movie_id_mapping[movie_id]
    pred = model.predict([np.array([user_idx]), np.array([movie_idx])], verbose=0)
    return pred[0][0]


def recommend_movies(
    model, user_id, movies, user_id_mapping, movie_id_mapping, top_n=10
):
    if user_id not in user_id_mapping:
        print("User  ID not found in training data.")
        return pd.DataFrame()

    user_idx = user_id_mapping[user_id]
    all_movie_indices = np.array(list(movie_id_mapping.values()))

    user_array = np.array([user_idx] * len(all_movie_indices))
    movie_array = all_movie_indices
    preds = model.predict([user_array, movie_array], verbose=0).flatten()

    top_indices = preds.argsort()[-top_n:][::-1]
    top_movie_ids = all_movie_indices[top_indices]

    recommendations = movies[movies["movieId"].isin(top_movie_ids)].copy()

    # Ensure we only assign predicted ratings to the movies in recommendations
    recommendations["predicted_rating"] = preds[top_indices][: len(recommendations)]

    recommendations = recommendations.sort_values(
        by="predicted_rating", ascending=False
    )

    return recommendations[["movieId", "title", "predicted_rating"]]


def main():
    print("Loading and preprocessing data...")
    ratings, movies, user_id_mapping, movie_id_mapping = load_and_preprocess_data()

    print(f"Number of users: {len(user_id_mapping)}")
    print(f"Number of movies: {len(movie_id_mapping)}")

    print("Building model...")
    model = build_model(len(user_id_mapping), len(movie_id_mapping))

    print("Training model...")
    train_model(model, ratings, epochs=10)

    # Example usages
    test_user_id = next(iter(user_id_mapping.keys()))
    test_movie_id = next(iter(movie_id_mapping.keys()))

    print(f"\nPredicting rating for user {test_user_id} and movie {test_movie_id}...")
    pred = predict_rating(
        model, test_user_id, test_movie_id, user_id_mapping, movie_id_mapping
    )
    print(f"Predicted rating: {pred:.2f}")

    print(f"\nTop 10 movie recommendations for user {test_user_id}:")
    recommendations = recommend_movies(
        model, test_user_id, movies, user_id_mapping, movie_id_mapping, top_n=10
    )
    print(recommendations)


if __name__ == "__main__":
    main()
