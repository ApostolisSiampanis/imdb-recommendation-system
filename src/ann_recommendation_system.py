from typing import List, Tuple

import numpy as np

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

    clustered_neighbors = get_clustered_neighbors(n_clusters=n_clusters,
                                                  clusters_labels=clusters_labels,
                                                  user_item_matrix=user_item_matrix,
                                                  distance_matrix=distance_matrix)

    pass


def get_clustered_neighbors(n_clusters, clusters_labels, user_item_matrix, distance_matrix):
    # A list of clusters, where each sublist contains tuples.
    # Each tuple corresponds to one user and contains:
    # (a list of their top-N neighbors' item arrays, current user's index in `user_item_matrix`)
    clustered_neighbors: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(n_clusters)]

    n_neighbors = 5  # Number of nearest neighbors to retrieve (excluding the user themself)

    for current_user_index in range(len(clusters_labels)):
        # Identify which cluster the current user belongs to
        cluster_index = clusters_labels[current_user_index]

        # Extract the row of the distance matrix corresponding to the current user
        current_user_distance_matrix = distance_matrix[current_user_index]

        # Get indices of the 5 nearest neighbors (excluding the current user at index 0)
        neighbors_indices = np.argsort(current_user_distance_matrix)[1:n_neighbors + 1]

        neighbors = []  # Will hold the item vectors of the 5 nearest neighbors

        for neighbor_index in neighbors_indices:
            # Retrieve the item array of the neighbor
            neighbor_item = user_item_matrix.to_numpy()[neighbor_index]
            neighbors.append(neighbor_item)

        # Store the list of neighbor item arrays along with the current user's index
        clustered_neighbors[cluster_index].append((neighbors, current_user_index))

    return clustered_neighbors
