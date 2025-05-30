import numpy as np

from src.utils import non_zero_ratings


def custom_euclidean_distance(Ru, Rv):
    total = 0.0
    for ru, rv in zip(Ru, Rv):
        if non_zero_ratings(ru, rv):  # Increase total only if both are non-zero
            total += (ru - rv) ** 2
    return np.sqrt(total)
