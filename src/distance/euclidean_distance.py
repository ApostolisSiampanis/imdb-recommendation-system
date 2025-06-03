import numpy as np


def euclidean_distance(Ru, Rv):
    common_mask = (Ru != 0) & (Rv != 0)

    # If there are no common items, return the maximum possible distance
    if not np.any(common_mask):
        return np.finfo(float).max

    # Extract the common items
    Ru_common: np.ndarray = Ru[common_mask]
    Rv_common = Rv[common_mask]

    return np.linalg.norm(Ru_common - Rv_common)
