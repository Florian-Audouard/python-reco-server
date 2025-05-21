import os
import sys
import pickle
import random as rd
import copy

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
        self.filename = "dbscan_model.pkl"
        self.movieId_to_idx = None
        self.idx_to_movieId = None

    def training_impl(self):
        label_and_movies_data = dbscan_clustering(self.ratings, self.movies)
        for label, movies_data in label_and_movies_data.items():
            label_and_movies_data[label] = sorted(movies_data, key=lambda x: (x[1], -x[2]), reverse=True)
        self.model = label_and_movies_data

    def save(self):
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)
        with open(self.get_full_path(), "wb") as f:
            pickle.dump(self.model, f)

    def load_impl(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.get_full_path(), "rb") as f:
            self.model = pickle.load(f)

    def get_select_random_movie(self, model_memory):
        label = rd.choice(list(model_memory.keys()))
        selected_list = model_memory[label]

        index_to_remove = None

        if selected_list[0][1] > 2.5 and selected_list[-1][2] == 0 :
            if rd.random() < 0.6:
                selected_movie = selected_list[0]
                index_to_remove = 0
            else:
                selected_movie = selected_list[-1]
                index_to_remove = -1
        elif selected_list[0][1] > 2.5:
            selected_movie = selected_list[0]
            index_to_remove = 0
        elif selected_list[-1][2] == 0 :
            selected_movie = selected_list[-1]
            index_to_remove = -1
        else:
            selected_movie = None
            index_to_remove = None
        return label, selected_list, selected_movie, index_to_remove
            

    def get_recommendations(self, user_id , top_n):
        model_memory = copy.deepcopy(self.model)
        selected_movies_id = []
        while len(selected_movies_id) < top_n and len(model_memory) > 0:
            if not model_memory:
                break 
            label, selected_list, selected_movie, index_to_remove = self.get_select_random_movie(model_memory)
            if selected_movie is None:
                del model_memory[label]

            selected_movies_id.append(selected_movie[0])
            del selected_list[index_to_remove]
        return selected_movies_id

    def init_data_impl(self):
        pass

    def get_prediction_set(self):
        pass

    def predict(self, user_id, list_movie_id):
        pass

    def accuracy(self, k=10):
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

if __name__ == "__main__":
    recommender = DBSCANRecommender(False,True)
    ratings, movies = load_data_from_file(f"ml-{FOLDER_SET}m")
    recommender.testing_main(ratings, movies)

