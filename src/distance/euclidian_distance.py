import numpy as np


def euclidean_distance(Ru, Rv, common_mask):
    # If there are no common items, return maximum distance (np.inf)
    if not np.any(common_mask):
        return int(2**32-1)

    # Extract the common items
    Ru_common: np.ndarray = Ru[common_mask]
    Rv_common = Rv[common_mask]

    return np.linalg.norm(Ru_common - Rv_common)
