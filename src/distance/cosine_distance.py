import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def cosine_distance(Ru, Rv, common_mask):
    # If there are no common items, return maximum distance (1.0)
    if not np.any(common_mask):
        return 1.0

    # Extract the common items
    Ru_common = Ru[common_mask].reshape(1, -1)
    Rv_common = Rv[common_mask].reshape(1, -1)

    return cosine_distances(Ru_common, Rv_common)
