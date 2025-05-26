import os
import sys
import pickle
import random as rd
import copy
import math

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

    def __init__(self, production=False, force_training=False):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "dbscan_model_and_data.pkl"
        self.movieId_to_idx = None
        self.idx_to_movieId = None

        self.limit_watch = 0
        self.limit_rate = 3
        self.recommendable_movies = None

    def training_impl(self):
        recommendable_movies = set()
        for movie_id in self.movies["movieId"]:
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id]["rating"].count()
            average_rate = self.ratings[self.ratings["movieId"] == movie_id]["rating"].mean()
            if number_of_watch <= self.limit_watch or average_rate >= self.limit_rate:
                recommendable_movies.add(movie_id)
        
        label_and_movies_data = dbscan_clustering(self.ratings, self.movies, recommendable_movies)

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

        Args:
            model_memory (dict): {label: list of (movieId, mean, number_of_watch)}

        Returns:
            tuple: (label, selected_list, selected_movie, index_to_remove)
                - label: the chosen cluster label
                - selected_list: the list of movies in the chosen cluster
                - selected_movie: the selected (movieId, mean, number_of_watch) tuple or None
                - index_to_remove: index of the selected movie in selected_list or None
        """

        label = rd.choice(list(model_memory.keys()))
        selected_list = model_memory[label]

        if len(selected_list) == 0:
            return label, None, None, None
        else : 
            def generate_utility_value(movie):
                average_rate = movie[1]
                return (movie[0], math.exp(-0.2*average_rate**2))
            
            total_value = 0
            probabilities = []

            rd.shuffle(selected_list)
            for index, movie in enumerate(selected_list):
                movie_id, utility_value = generate_utility_value(movie)
                total_value += utility_value
                if index == 0:
                    probabilities.append((movie_id, utility_value))
                else:
                    probabilities.append((movie_id, utility_value + probabilities[index - 1][1]))
            
            random_value = rd.uniform(0, total_value)
            for index in range(len(probabilities)):
                if random_value <= probabilities[index][1]:
                    return label, selected_list, probabilities[index][0], index

    def get_recommendations(self, user_id , top_n):
        model_memory = copy.deepcopy(self.model)
        selected_movies_id = []
        all_movies_id = self.ratings["movieId"].unique()
        if top_n > len(all_movies_id):
            return all_movies_id
        while len(selected_movies_id) < top_n and len(model_memory) > 0:
            if len(model_memory) == 0:
                break 
            label, selected_list, selected_movie, index_to_remove = self.get_select_random_movie(model_memory)
            if selected_list is None:
                del model_memory[label]
                continue
            selected_movies_id.append(selected_movie)
            selected_list.pop(index_to_remove)
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
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id]["rating"].count()
            if number_of_watch <= self.limit_watch:
                not_watch_movies.add(movie_id)
            if len(genres) != 0:
                genres_list_recommended.update(genres[0].split("|"))
        genre_diversity = len(set(genres_list_recommended))
        unknown_recommended_movies = len(not_watch_movies)
        return genre_diversity, unknown_recommended_movies
    

    def accuracy(self, k=30):
        if self.validation_set is None:
            raise RuntimeError("Validation set not initialized")
        
        different_genre_in_db = set()
        different_genre_in_recommendation = set()
        different_unknown_movies = set()

        for movie_id in self.movies["movieId"]:
            genres = self.movies[self.movies["movieId"] == movie_id]["genre"].values
            number_of_watch = self.ratings[self.ratings["movieId"] == movie_id]["rating"].count()
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

        for user_id in self.validation_set["userId"].unique():
            
            recommendations = self.get_recommendations(user_id, k)
            genre_diversity_recommended, unknown_recommended_movies = self.get_diversity(recommendations)

            recommended_diversity.update(recommendations)
            genre_diversity.append(genre_diversity_recommended)
            unknown_diversity.append(unknown_recommended_movies)

        return {
            "mean genre diversity": sum(genre_diversity) / (len(genre_diversity)*number_of_genre_in_db),
            "mean genre inner diversity": sum(genre_diversity) / (len(genre_diversity)*number_of_genre_in_recommendation),
            "mean unknown selection": sum(unknown_diversity) / (len(unknown_diversity)*number_of_unknown_movies),
            "coverage": len(recommended_diversity) / number_of_movies
        }

if __name__ == "__main__":
    recommender = DBSCANRecommender(False, False)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)