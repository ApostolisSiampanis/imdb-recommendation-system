import numpy as np
from tqdm import tqdm

from src.distance.cosine_distance import cosine_distance
from src.distance.euclidian_distance import euclidean_distance


def calculate_distance_matrix(user_item_matrix, metric):
    metric_func = euclidean_distance if metric == 'euclidean' else cosine_distance
    num_of_users = user_item_matrix.shape[0]
    distance_matrix = np.zeros((num_of_users, num_of_users))

    with tqdm(total=(num_of_users * (num_of_users - 1)) // 2,
              desc=f'Calculating {metric.capitalize()} Distance Matrix') as pbar:
        for i in range(num_of_users):
            Ru = user_item_matrix[i]
            for j in range(i + 1, num_of_users):
                Rv = user_item_matrix[j]
                common_mask = (Ru != 0) & (Rv != 0)
                dist = metric_func(Ru, Rv, common_mask)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetrical matrix

                pbar.update(1)

    return distance_matrix
