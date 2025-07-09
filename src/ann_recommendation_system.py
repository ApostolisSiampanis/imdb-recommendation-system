from typing import List, Tuple

import numpy as np

from src.constants import N_OF_NEIGHBORS
from src.distance.distance_matrix import Metric
from src.kmedoids_clustering import kmedoids_clustering


def ann_recommendation_system(user_item_matrix, unique_users_indices):
    # TODO add print informational message
    # TODO do we need unique_users_indices
    n_clusters = 4
    clusters_labels, distance_matrix = kmedoids_clustering(user_item_matrix=user_item_matrix,
                                                           unique_users_indices=unique_users_indices,
                                                           metric=Metric.JACCARD,
                                                           n_clusters=n_clusters)

    clusters_of_users = get_clusters_of_users(n_clusters=n_clusters,
                                              clusters_labels=clusters_labels,
                                              user_item_matrix=user_item_matrix,
                                              distance_matrix=distance_matrix)

    # Iterate through clusters and construct training data for each
    for cluster_of_users in clusters_of_users:
        X_train, Y_train = build_training_data_from_cluster(cluster_of_users)
        pass


def get_clusters_of_users(n_clusters, clusters_labels, user_item_matrix, distance_matrix):
    """
    Groups users into clusters and gathers each user's top-N nearest neighbors.

    Returns:
        clusters_of_users: A list of `n_clusters` elements.
            Each element is a list of tuples (neighbors_items, user_index),
            where:
                - neighbors_items is a list of np.ndarrays (item vectors),
                  including the current user (at index 0) and their N nearest neighbors.
                - user_index is the index of the current user in the user_item_matrix.
    """
    clusters_of_users: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(n_clusters)]

    for current_user_index in range(len(clusters_labels)):
        cluster_index = clusters_labels[current_user_index]
        current_user_distances = distance_matrix[current_user_index]

        # Get the current user's top-N neighbors (including self)
        neighbors_indices = np.argsort(current_user_distances)[:N_OF_NEIGHBORS + 1]

        neighbors = []
        for neighbor_index in neighbors_indices:
            neighbor_item_vector = user_item_matrix.to_numpy()[neighbor_index]
            neighbors.append(neighbor_item_vector)

        clusters_of_users[cluster_index].append((neighbors, current_user_index))

    return clusters_of_users


def build_training_data_from_cluster(cluster_of_users):
    """
    Constructs training data (X, Y) for ANN using the i-th user's cluster data.

    Args:
        cluster_of_users (List[Tuple[List[np.ndarray], int]]): List of (neighbors_items, user_index).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X_train: shape (rows, N_OF_NEIGHBORS, number_of_ratings)
            - Y_train: shape (rows, number_of_ratings)
    """
    rows = len(cluster_of_users)
    number_of_ratings = len(cluster_of_users[0][0][0])  # length of a single user's item vector

    X_train = np.empty(shape=(rows, N_OF_NEIGHBORS, number_of_ratings))
    Y_train = np.empty(shape=(rows, number_of_ratings))

    for row_index, (neighbors_items, user_index) in enumerate(cluster_of_users):
        for j, item_vector in enumerate(neighbors_items):
            if j == 0:  # The first item is the current user's own vector
                Y_train[row_index] = item_vector
            else:  # Remaining are neighbors' vectors
                X_train[row_index][j - 1] = item_vector

    return X_train, Y_train
