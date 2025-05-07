import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.time_util import get_time_func


@get_time_func
def group_data(movies, tags):
    """
    Combine les genres et les tags pour chaque film dans une seule description.

    Args:
        movies (DataFrame): Films avec leurs genres.
        tags (DataFrame): Tags associés aux films.

    Returns:
        DataFrame: Films enrichis d'une colonne 'full_desc' combinant genres et tags.
    """
    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )
    movies_merged = movies.merge(movie_tags, on="movieId", how="left")
    movies_merged["full_desc"] = (
        movies_merged["genres"] + " " + movies_merged["tag"].fillna("")
    )
    return movies_merged
