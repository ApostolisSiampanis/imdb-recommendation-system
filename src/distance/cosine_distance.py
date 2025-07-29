import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def cosine_distance(Ru, Rv):
    """
    Computes the cosine distance between two user rating vectors,
    considering only the items rated by both users.

    Args:
        Ru (np.ndarray): Rating vector of user u.
        Rv (np.ndarray): Rating vector of user v.

    Returns:
        float: Cosine distance between Ru and Rv in the range [0.0, 1.0],
               where 0.0 indicates an identical direction and 1.0 indicates no common ratings.
    """
    common_mask = (Ru != 0) & (Rv != 0)

    # If there are no common items, return maximum distance (1.0)
    if not np.any(common_mask):
        return 1.0

    # Extract the common items
    Ru_common = Ru[common_mask].reshape(1, -1)
    Rv_common = Rv[common_mask].reshape(1, -1)

    return cosine_distances(Ru_common, Rv_common)
