import os
from enum import Enum

import numpy as np
from tqdm import tqdm

from src.constants import DATAFILES_DIR_PATH
from src.distance.cosine_distance import cosine_distance
from src.distance.euclidean_distance import euclidean_distance
from src.distance.jaccard_distance import jaccard_distance


def calculate_distance_matrix(user_item_matrix, metric):
    # If distance matrix is already saved locally, load and return it
    file = os.path.join(DATAFILES_DIR_PATH, f'distance_matrix_{metric.value}.npy')
    if os.path.isfile(file):
        return np.load(os.path.join(DATAFILES_DIR_PATH, f'distance_matrix_{metric.value}.npy'))

    # else calculate the distance matrix
    num_of_users = user_item_matrix.shape[0]
    distance_matrix = np.zeros((num_of_users, num_of_users))

    if metric == Metric.EUCLIDEAN:
        metric_func = euclidean_distance
    elif metric == Metric.COSINE:
        metric_func = cosine_distance
    elif metric == Metric.JACCARD:
        metric_func = jaccard_distance
    else:
        raise ValueError(f'Metric must be one of {Metric._member_names_}')

    with tqdm(total=(num_of_users * (num_of_users - 1)) // 2,
              desc=f'Calculating {metric.value} Distance Matrix') as pbar:
        for i in range(num_of_users):
            Ru = user_item_matrix[i]
            for j in range(i + 1, num_of_users):
                Rv = user_item_matrix[j]
                dist = metric_func(Ru, Rv)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetrical matrix

                pbar.update(1)

    # Save the distance matrix
    np.save(os.path.join(DATAFILES_DIR_PATH, f'distance_matrix_{metric.value}.npy'), distance_matrix)

    return distance_matrix


class Metric(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'
    JACCARD = 'jaccard'
