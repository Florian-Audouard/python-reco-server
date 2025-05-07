import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@get_time_func
def get_cosine_similarity(movies_merged):
    """
    Calcule la matrice de similarité cosinus basée sur la description complète des films.

    Args:
        movies_merged (DataFrame): Films avec colonnes 'full_desc'.

    Returns:
        ndarray: Matrice de similarité cosinus entre les films.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_merged["full_desc"])
    return cosine_similarity(tfidf_matrix)
