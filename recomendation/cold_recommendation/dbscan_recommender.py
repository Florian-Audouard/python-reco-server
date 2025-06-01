import os
import sys
import pickle
import random as rd
import math
import numpy as np
import time

try:
    from .dbscan_generation import dbscan_clustering
except ImportError:
    from dbscan_generation import dbscan_clustering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model
from preprocessing.movie_manipulation import load_data_from_file

FORCE_TRAINING = True
FOLDER_SET = "0.1"


class DBSCANRecommender(Model):

    def __init__(self, production, force_training, noise=False, real_data=None):
        super().__init__(production, force_training, noise=noise, real_data=real_data)
        self.model = None
        self.filename = "dbscan_model_and_data.pkl"

        self.limit_watch = 0
        self.limit_rate = 3
        self.recommendable_movies = None

    def training_impl(self):
        recommendable_movies = set()
        for movie_id in self.movies["movieId"]:
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id][
                "rating"
            ].count()
            average_rate = self.ratings[self.ratings["movieId"] == movie_id][
                "rating"
            ].mean()
            if number_of_watch <= self.limit_watch or average_rate >= self.limit_rate:
                recommendable_movies.add(movie_id)

        label_and_movies_data = dbscan_clustering(
            self.ratings, self.movies, recommendable_movies
        )

        self.recommendable_movies = recommendable_movies
        self.model = label_and_movies_data

    def save(self):
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)
        with open(self.get_full_path(), "wb") as f:
            pickle.dump({
                "model": self.model,
                "recommendable_movies": self.recommendable_movies
            }, f)

    def load_impl(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.get_full_path(), "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.recommendable_movies = data["recommendable_movies"]

    def get_select_random_movie(self, model_memory):
        """
        Selects a random movie from a random cluster in the model memory,
        with logic based on the mean rating and number of watches.
        """

        label = rd.choice(list(model_memory.keys()))
        selected_list = model_memory[label]

        if not selected_list:
            return label, None, None, None

        # Utilité basée sur la moyenne
        utilities = np.array([math.exp(-0.01 * movie[1] ** 2) for movie in selected_list])
        probabilities = utilities / utilities.sum()
        idx = np.random.choice(len(selected_list), p=probabilities)
        selected_movie = selected_list[idx][0]
        return label, selected_list, selected_movie, idx

    def get_recommendations(self, user_id, top_n):
        # Copie uniquement les listes de clusters, pas tout le dict
        model_memory = {k: v.copy() for k, v in self.model.items()}
        selected_movies_id = []
        all_movies_id = self.ratings["movieId"].unique()
        if top_n > len(all_movies_id):
            return all_movies_id.tolist()
        while len(selected_movies_id) < top_n and model_memory:
            label, selected_list, selected_movie, index_to_remove = self.get_select_random_movie(model_memory)
            if selected_list is None:
                del model_memory[label]
                continue
            selected_movies_id.append(selected_movie)
            selected_list.pop(index_to_remove)
            if not selected_list:
                del model_memory[label]
        return selected_movies_id

    def init_data_impl(self):
        return
    
    def predict(self, user_id, candidates):
        return

    def get_diversity(self, recommendations):
        genres_list_recommended = set()
        not_watch_movies = set()
        for movie_id in recommendations:
            genres = self.movies[self.movies["movieId"] == movie_id]["genre"].values
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id][
                "rating"
            ].count()
            if number_of_watch <= self.limit_watch:
                not_watch_movies.add(movie_id)
            if len(genres) != 0:
                genres_list_recommended.update(genres[0].split("|"))
        genre_diversity = len(genres_list_recommended)
        unknown_recommended_movies = len(not_watch_movies)
        return genre_diversity, unknown_recommended_movies

    def accuracy(self, k=20):
        if self.validation_set is None:
            raise RuntimeError("Validation set not initialized")

        different_genre_in_db = set()
        different_genre_in_recommendation = set()
        different_unknown_movies = set()

        for movie_id in self.movies["movieId"]:
            genres = self.movies[self.movies["movieId"] == movie_id]["genre"].values
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id][
                "rating"
            ].count()
            if len(genres) != 0:
                different_genre_in_db.update(genres[0].split("|"))
            if number_of_watch <= self.limit_watch:
                different_unknown_movies.add(movie_id)
            if movie_id in self.recommendable_movies and len(genres) != 0:
                different_genre_in_recommendation.update(genres[0].split("|"))

        number_of_genre_in_db = len(different_genre_in_db)
        number_of_genre_in_recommendation = len(different_genre_in_recommendation)
        number_of_unknown_movies = len(different_unknown_movies)
        number_of_movies = len(self.recommendable_movies)

        recommended_diversity = set()
        genre_diversity = []
        unknown_diversity = []

        # Pour la répartition des clusters
        clusters_per_user = []
        cluster_count_global = {}

        # Inverse l'indexation pour retrouver le cluster d'un film
        movie_to_cluster = {}
        for cluster_label, movie_list in self.model.items():
            for movie_info in movie_list:
                movie_id = movie_info[0]
                movie_to_cluster[movie_id] = cluster_label

        films_with_few_ratings_per_user = []

        times = []

        for user_id in self.validation_set["userId"].unique():
            start = time.time()
            recommendations = self.get_recommendations(user_id, k)
            times.append(time.time() - start)
            genre_diversity_recommended, unknown_recommended_movies = self.get_diversity(recommendations)
            recommended_diversity.update(recommendations)
            genre_diversity.append(genre_diversity_recommended)
            unknown_diversity.append(unknown_recommended_movies)

            clusters = set()
            for movie_id in recommendations:
                cluster = movie_to_cluster.get(movie_id)
                if cluster is not None:
                    clusters.add(cluster)
                    cluster_count_global[cluster] = cluster_count_global.get(cluster, 0) + 1
            clusters_per_user.append(len(clusters))
            # Compte les films avec peu de notes dans les recommandations de cet utilisateur
            count_few_ratings = 0
            for movie_id in recommendations:
                number_of_watch = self.ratings[self.ratings["movieId"] == movie_id]["rating"].count()
                if number_of_watch <= self.limit_watch:
                    count_few_ratings += 1
            films_with_few_ratings_per_user.append(count_few_ratings)
        mean_clusters_per_user = np.mean(clusters_per_user) if clusters_per_user else 0
        mean_few_ratings_per_reco = np.mean(films_with_few_ratings_per_user) if films_with_few_ratings_per_user else 0

        cluster_mean_ratings = {}
        for cluster_label, movie_list in self.model.items():
            ratings_list = []
            for movie_info in movie_list:
                movie_id = movie_info[0]
                movie_ratings = self.ratings[self.ratings["movieId"] == movie_id]["rating"]
                if not movie_ratings.empty:
                    ratings_list.extend(movie_ratings.tolist())
            if ratings_list:
                cluster_mean_ratings[cluster_label] = np.mean(ratings_list)
            else:
                cluster_mean_ratings[cluster_label] = None  
        mean_response_time = np.mean(times) if times else 0
        return {
            "number of clusters": len(self.model),
            "mean genre diversity": sum(genre_diversity) / (len(genre_diversity)*number_of_genre_in_db) if number_of_genre_in_db else 0,
            "mean genre inner diversity": sum(genre_diversity) / (len(genre_diversity)*number_of_genre_in_recommendation) if number_of_genre_in_recommendation else 0,
            "mean unknown coverage": sum(unknown_diversity) / (len(unknown_diversity)*number_of_unknown_movies) if number_of_unknown_movies else 0,
            "coverage": len(recommended_diversity) / number_of_movies if number_of_movies else 0,
            "ratio film by user": len(recommended_diversity) / len(self.validation_set["userId"].unique()) if len(self.validation_set["userId"].unique()) else 0,
            "mean clusters per user": mean_clusters_per_user,
            "mean few ratings per reco": mean_few_ratings_per_reco,
            "mean response time": mean_response_time,
        }


if __name__ == "__main__":
    recommender = DBSCANRecommender(False, True)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)
