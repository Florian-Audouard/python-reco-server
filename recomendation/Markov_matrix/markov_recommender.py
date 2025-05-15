
import os
import sys
import scipy.sparse 

from markov_generation import markov_matrix, get_user_vector



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Model
from preprocessing.movie_manipulation import load_data

class MarkovRecommender(Model):
    """
    MarkovRecommender is a recommendation system model based on Markov chain.
    """

    def __init__(self, production, force_training):
        super().__init__(production, force_training)
        self.model = None
        self.filename = "markov_model.npz"

    def training_impl(self):
        self.model = markov_matrix(self.trainset)
        print(self.model)

    def save(self):
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        scipy.sparse.save_npz(self.get_full_path(), self.model)


    def load_impl(self):
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)
        
        os.makedirs(self.data_dir, exist_ok=True)
        scipy.sparse.load_npz(self.get_full_path())

    def get_prediction_set(self):
        return super().get_prediction_set()
    
    def predict(self, user_id, list_movie_id):
        """
        Predict the rating for a given user and a list of movies
        Args:
            user_id: User ID
            list_movie_id: Movie ID list
        Returns:
            Predicted rating for each movie
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        # Predict ratings using the model
        matrix = self.model
        user_vector = get_user_vector(self.trainset, user_id)
        stochastic_prediction = matrix.dot(user_vector)

        number_of_recommendations = len(list_movie_id)
        selected_index = stochastic_prediction.argsort()[::-1][:number_of_recommendations]

        return [True if i in selected_index else False for i in list_movie_id]

    def get_recommendations(self, user_id, top_n):
        """
        Generate hybrid recommendations for a given user
        Args:
            user_id (int): ID of the target user
            top_n (int): Number of recommendations to return
        Returns:
            list: List of (title, score) tuples, sorted by descending score
        """
        if self.model is None:
            raise RuntimeError(self.ERROR_MESSAGE)

        return super().get_recommendations(user_id, top_n)
        

    


if __name__ == "__main__":
    # Exemple d'utilisation
    recommender = MarkovRecommender(False, True)
    data = load_data("ml-0.1m")
    recommender.init_data(data)
    recommender.training_impl()
