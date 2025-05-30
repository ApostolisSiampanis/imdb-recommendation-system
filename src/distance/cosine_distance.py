import numpy as np

from src.utils import non_zero_ratings


def custom_cosine_distance(Ru, Rv):
    Ru_arr = np.array(Ru)
    Rv_arr = np.array(Rv)

    numerator = 0.0
    sum_u_sq = 0.0
    sum_v_sq = 0.0

    for ru, rv in zip(Ru_arr, Rv_arr):
        if non_zero_ratings(ru, rv):  # Both ratings must be non-zero
            numerator += ru * rv
            sum_u_sq += ru ** 2
            sum_v_sq += rv ** 2

    denominator = np.sqrt(sum_u_sq) * np.sqrt(sum_v_sq)

    if denominator == 0:
        return 1.0  # Maximum cosine distance if no common ratings or zero vector

    return 1 - (numerator / denominator)
