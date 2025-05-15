import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import itertools as it

def markov_matrix(trainset):
    """
    Create a Markov matrix from the ratings data.

    Args:
        trainset: 
            An object containing user ratings, with the following attributes:
            - n_items (int): total number of movies.
            - ur (dict): dictionary {user_id: [(movie_id, rating), ...]}.

    Returns:
        scipy.sparse.csr_matrix: 
            Normalized Markov matrix (number of co-occurrences between each pair of movies, divided by the total).
            The matrix is of size (number of movies, number of movies).
    """
    number_of_movies = trainset.n_items
    markov_matrix = lil_matrix((number_of_movies, number_of_movies))
    for _, u_ratings in trainset.ur.items():
        movie_watch_by_user = set()
        for movie_id, _ in u_ratings:
            movie_watch_by_user.add(movie_id)
        for i, j in it.combinations(movie_watch_by_user, 2):
            markov_matrix[i, j] += 1
            markov_matrix[j, i] += 1
    markov_matrix = markov_matrix.tocsr()
    row_weight = np.array(markov_matrix.sum(axis=1)).flatten()

    row_index, _ = markov_matrix.nonzero()
    markov_matrix.data /= row_weight[row_index]
    return markov_matrix

def get_user_vector(trainset, user_id):
    """
    Get the user vector for a given user.

    Args:
        trainset: The training set containing user ratings
        user_id: The ID of the user

    Returns:
        list: User vector (binary vector of size n_items, where 1 means the user has seen the movie)
    """
    user_vector = [0] * trainset.n_items
    list_movies_and_rating = trainset.ur[user_id]
    for movie_id, _ in list_movies_and_rating:
        user_vector[movie_id] = 1

    return user_vector
