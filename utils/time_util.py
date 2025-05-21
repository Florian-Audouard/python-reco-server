import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from functools import wraps
import time
from logger import log


from functools import wraps
import time
from utils.logger import log


def get_time_func(func):
    """
    Décorateur pour mesurer et afficher le temps d'exécution d'une fonction.
    Accepte un argument keyword `display_time` (default True) pour activer ou non l'affichage.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        display_time = kwargs.pop("display_time", True)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        if display_time:
            log.info("%s took %.6fs", func.__name__, end - start)
        return result

    return wrapper
