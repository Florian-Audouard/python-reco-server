"""
Module de recommandation de films basé sur une approche hybride (filtrage collaboratif + filtrage de contenu).

Ce script charge les données du jeu MovieLens, traite les genres et tags des films pour générer des
descriptions complètes, calcule une matrice de similarité cosinus à partir de ces descriptions,
entraîne un modèle SVD (filtrage collaboratif), puis combine les deux méthodes pour produire
des recommandations personnalisées.

Fonctionnalités :
- Lecture des fichiers de données MovieLens (films, notes, tags).
- Enrichissement des films avec les tags.
- Calcul de similarité de contenu avec TF-IDF + cosinus.
- Entraînement d'un modèle de recommandation basé sur SVD (bibliothèque Surprise).
- Génération de recommandations hybrides pour un utilisateur donné.
- Affichage du temps d'exécution et de l'utilisation mémoire.

Usage :
    Exécuter le script et entrer un user ID lorsqu'invité pour obtenir des recommandations.

Dépendances :
    - pandas
    - scikit-learn
    - surprise
    - pympler
    - log
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.logger import log
from utils.time_util import get_time_func
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


from pympler import asizeof


FOLDER_PATH = "data"


@get_time_func
def get_test_set(ratings):
    """
    Prépare un jeu d'entraînement pour le modèle de recommandation collaboratif.

    Args:
        ratings (DataFrame): Évaluations des utilisateurs.

    Returns:
        Trainset: Jeu d'entraînement Surprise.
    """
    reader = Reader(rating_scale=(0, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    return trainset


@get_time_func
def train_recommender_model(trainset):
    """
    Entraîne un modèle SVD sur le jeu d'entraînement.

    Args:
        trainset (Trainset): Données d'entraînement Surprise.

    Returns:
        AlgoBase: Modèle entraîné.
    """
    algo = SVD()
    algo.fit(trainset)
    return algo


@get_time_func
def hybrid_recommendations(
    user_id, movies, ratings, tags, movies_merged, cosine_sim, algo, top_n=10, alpha=1
):
    """
    Génère des recommandations hybrides (collaboratif + contenu) pour un utilisateur donné.

    Args:
        user_id (int): ID de l'utilisateur cible.
        movies (DataFrame): Liste des films.
        ratings (DataFrame): Notes des utilisateurs.
        tags (DataFrame): Tags de films.
        movies_merged (DataFrame): Films enrichis avec genres + tags.
        cosine_sim (ndarray): Matrice de similarité de contenu.
        algo (AlgoBase): Modèle collaboratif entraîné.
        top_n (int): Nombre de recommandations à retourner.
        alpha (float): Pondération entre collaboratif (alpha) et contenu (1 - alpha).

    Returns:
        list: Liste de tuples (titre, score hybride), triée par score décroissant.
    """
    watched = ratings[ratings["userId"] == user_id]["movieId"].values
    candidates = movies_merged[~movies_merged["movieId"].isin(watched)]

    results = []
    for _, row in candidates.iterrows():
        movie_id = row["movieId"]
        idx = movies_merged.index[movies_merged["movieId"] == movie_id][0]

        pred_rating = algo.predict(user_id, movie_id).est
        content_sim = cosine_sim[idx].mean()

        hybrid_score = alpha * pred_rating + (1 - alpha) * content_sim
        results.append((row["title"], hybrid_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


@get_time_func
def full_process(folder):
    """
    Exécute le pipeline complet : lecture des données, enrichissement, calcul des similarités,
    préparation du modèle et entraînement.

    Args:
        folder (str): Nom du sous-dossier contenant les fichiers CSV.

    Returns:
        tuple: Tous les objets nécessaires pour faire des recommandations.
    """
    movies, ratings, tags = read_data(folder)
    movies_merged = group_data(movies, tags)
    cosine_sim = get_cosine_similarity(movies_merged)
    trainset = get_test_set(ratings)
    algo = train_recommender_model(trainset)
    return movies, ratings, tags, movies_merged, cosine_sim, algo


def display_size(movies, ratings, tags, movies_merged, cosine_sim, algo):
    """
    Affiche la taille mémoire approximative des objets principaux.

    Args:
        movies, ratings, tags, movies_merged, cosine_sim, algo: Objets utilisés dans le pipeline.
    """
    variables = [movies, ratings, tags, movies_merged, cosine_sim, algo]
    names = ["movies", "ratings", "tags", "movies_merged", "cosine_sim", "algo"]
    total = 0
    log.info("")
    for name, var in zip(names, variables):
        size = asizeof.asizeof(var)
        total += size
        log.info("%s: %.2f MB", name, size / 1024 / 1024)
    log.info("")
    log.info("Total memory used: %.2f MB", total / 1024 / 1024)
