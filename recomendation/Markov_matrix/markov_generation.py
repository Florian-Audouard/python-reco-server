import numpy as np
from scipy.sparse import lil_matrix
import itertools as it

def build_movieid_mapping(ratings):
    """
    Retourne :
      movie_id_to_idx: dict {movieId → idx 0..N-1}
      idx_to_movie_id: dict inverse si besoin
    """
    unique_ids = sorted(ratings["movieId"].unique())
    movie_id_to_idx = {mid: i for i, mid in enumerate(unique_ids)}
    idx_to_movie_id = {i: mid for i, mid in enumerate(unique_ids)}
    return movie_id_to_idx, idx_to_movie_id

def markov_matrix(trainset, movieid2idx):
    """
    Crée la matrice de transition Markov normalisée (CSR) de taille N×N
    en utilisant movieid2idx pour indexer de façon compacte.
    """
    n = len(movieid2idx)
    M = lil_matrix((n, n))
    
    for _, u_ratings in trainset.ur.items():
        # on récupère l'ensemble des indices internes vus par l'utilisateur
        idxs = { movieid2idx[mid] 
                 for (mid, _) in u_ratings 
                 if mid in movieid2idx }
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
    Retourne un vecteur binaire de longueur N=len(movieId_to_idx),
    où vec[idx] = 1 si l'utilisateur a vu le film movieId (via movieId_to_idx).
    """
    n = len(movieId_to_idx)
    vec = np.zeros(n, dtype=int)

    # Pour chaque movieId que user_id a vu, si dans movieid2idx, on positionne 1
    seen = ratings.loc[ratings["userId"] == user_id, "movieId"]
    for mid in seen:
        idx = movieId_to_idx.get(mid)
        if idx is not None:
            vec[idx] = 1

    return vec