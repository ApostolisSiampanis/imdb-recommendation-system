from scipy.spatial.distance import jaccard


def jaccard_distance(Ru, Rv):
    intersection = (Ru != 0) & (Rv != 0)
    union = (Ru != 0) | (Rv != 0)
    return jaccard(intersection, union)
