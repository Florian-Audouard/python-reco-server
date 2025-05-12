from logger import log
from pympler import asizeof


def display_size(self):
    """
    Display approximate memory size of main objects
    """
    variables = [
        self.movies,
        self.ratings,
        self.tags,
        self.movies_merged,
        self.algo,
    ]
    names = ["movies", "ratings", "tags", "movies_merged", "algo"]
    total = 0
    log.info("")
    for name, var in zip(names, variables):
        size = asizeof.asizeof(var)
        total += size
        log.info("%s: %.2f MB", name, size / 1024 / 1024)
    log.info("")
    log.info("Total memory used: %.2f MB", total / 1024 / 1024)
