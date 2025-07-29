from scipy.spatial.distance import jaccard


def jaccard_distance(Ru, Rv):
    """
    Computes the Jaccard distance between two user rating vectors,
    based on the presence or absence of ratings (binary comparison).

    Args:
        Ru (np.ndarray): Rating vector of user u.
        Rv (np.ndarray): Rating vector of user v.

    Returns:
        float: Jaccard distance in the range [0.0, 1.0], where 0.0 indicates complete overlap
               in rated items and 1.0 indicates no overlap.
    """
    intersection = (Ru != 0) & (Rv != 0)
    union = (Ru != 0) | (Rv != 0)
    return jaccard(intersection, union)
