import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from functools import wraps
import time
from logger import log


def get_time_func(func):
    """
    Décorateur pour mesurer et afficher le temps d'exécution d'une fonction.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        log.info("%s took %.6fs", func.__name__, end - start)
        return result

    return wrapper
