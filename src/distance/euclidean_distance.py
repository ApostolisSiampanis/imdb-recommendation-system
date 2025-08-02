import numpy as np


def euclidean_distance(Ru, Rv):
    """
    Computes the Euclidean distance between two user rating vectors,
    considering only the items rated by both users.

    Args:
        Ru (np.ndarray): Rating vector of user u.
        Rv (np.ndarray): Rating vector of user v.

    Returns:
        float: Euclidean distance between Ru and Rv. If there are no
               commonly rated items, return the maximum float value.
    """
    common_mask = (Ru != 0) & (Rv != 0)

    # If there are no common items, return the maximum possible distance
    if not np.any(common_mask):
        return np.finfo(float).max

    # Extract the common items
    Ru_common = Ru[common_mask]
    Rv_common = Rv[common_mask]

    return np.linalg.norm(Ru_common - Rv_common)
