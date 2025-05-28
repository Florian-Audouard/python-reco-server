import numpy as np
from scipy.sparse import lil_matrix
import itertools as it

def build_movieid_mapping(ratings):
    """
    Builds mappings between movie IDs and internal indices.

    Args:
        ratings (pd.DataFrame): DataFrame containing at least a "movieId" column.

    Returns:
        movie_id_to_idx (dict): Mapping from movieId to internal index (0..N-1).
        idx_to_movie_id (dict): Mapping from internal index to movieId.
    """
    unique_ids = sorted(ratings["movieId"].unique())
    movie_id_to_idx = {}
    idx_to_movie_id = {}
    for i, mid in enumerate(unique_ids):
        movie_id_to_idx[mid] = i
        idx_to_movie_id[i] = mid
    return movie_id_to_idx, idx_to_movie_id

def markov_matrix(trainset, movieid_to_idx):
    """
    Creates a normalized Markov transition matrix (CSR format) of size N×N,
    using movieid_to_idx for compact indexing.

    Args:
        trainset: Training set object with a .ur attribute (user ratings).
        movieid_to_idx (dict): Mapping from movieId to internal index.

    Returns:
        scipy.sparse.csr_matrix: Normalized Markov transition matrix.
    """
    n = len(movieid_to_idx)
    M = lil_matrix((n, n))
    
    for _, u_ratings in trainset.ur.items():
        idxs = { movieid_to_idx[mid] 
                 for (mid, _) in u_ratings 
                 if mid in movieid_to_idx }
        for i, j in it.combinations(idxs, 2):
            M[i, j] += 1
            M[j, i] += 1

    M = M.tocsr()
    row_sum = np.array(M.sum(axis=1)).flatten()
    rows, _ = M.nonzero()
    M.data /= row_sum[rows]
    return M


def get_user_vector(ratings, user_id, movieId_to_idx):
    """
    Returns a binary vector of length N = len(movieId_to_idx),
    where vec[idx] = 1 if the user has seen the movie (via movieId_to_idx), 0 otherwise.

    Args:
        ratings (pd.DataFrame): DataFrame with columns "userId" and "movieId".
        user_id (int): The user ID.
        movieId_to_idx (dict): Mapping from movieId to internal index.

    Returns:
        np.ndarray: Binary vector indicating which movies the user has seen.
    """
    n = len(movieId_to_idx)
    vec = np.zeros(n, dtype=int)

    seen = ratings.loc[ratings["userId"] == user_id, "movieId"]
    for mid in seen:
        idx = movieId_to_idx.get(mid)
        if idx is not None:
            vec[idx] = 1

    return vec