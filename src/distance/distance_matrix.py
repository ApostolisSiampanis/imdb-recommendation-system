import numpy as np


def calculate_distance_matrix(user_item_matrix, metric_function):
    num_rows = user_item_matrix.shape[0]
    distance_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            dist = metric_function(user_item_matrix.iloc[i].values, user_item_matrix.iloc[j].values)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetrical matrix
    return distance_matrix
