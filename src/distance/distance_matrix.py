import numpy as np
from tqdm import tqdm


def calculate_distance_matrix(user_item_matrix, metric_func, metric):
    num_of_users = user_item_matrix.shape[0]
    distance_matrix = np.zeros((num_of_users, num_of_users))

    with tqdm(total=(num_of_users * (num_of_users - 1)) // 2,
              desc=f'Calculating {metric.capitalize()} Distance Matrix') as pbar:
        for i in range(num_of_users):
            Ru = user_item_matrix[i]
            for j in range(i + 1, num_of_users):
                Rv = user_item_matrix[j]
                dist = metric_func(Ru, Rv)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetrical matrix

                pbar.update(1)

    return distance_matrix
