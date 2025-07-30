from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential

from src.constants import N_OF_NEIGHBORS, N_OF_CLUSTERS
from src.distance.distance_matrix import Metric
from src.kmedoids_clustering import kmedoids_clustering


def ann_recommendation_system(user_item_matrix, unique_users_indices):
    """
    Runs an artificial neural network (ANN)-based recommendation system using K-Medoids clustering
    and nearest neighbor information from user-item interactions.

    Args:
        user_item_matrix (pd.DataFrame): Matrix of user-item ratings.
        unique_users_indices (List[int]): Indices of unique users in the user_item_matrix.
    """
    clusters_labels, distance_matrix = kmedoids_clustering(user_item_matrix=user_item_matrix,
                                                           unique_users_indices=unique_users_indices,
                                                           metric=Metric.JACCARD,
                                                           n_clusters=N_OF_CLUSTERS)

    clusters_of_users = get_clusters_of_users(n_clusters=N_OF_CLUSTERS,
                                              clusters_labels=clusters_labels,
                                              user_item_matrix=user_item_matrix,
                                              distance_matrix=distance_matrix)

    # Iterate through clusters and construct training data for each
    for i, cluster_of_users in enumerate(clusters_of_users):
        X, y = build_training_data_from_cluster(cluster_of_users)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        X_train_flat, X_test_flat = X_train.reshape((X_train.shape[0], -1)), X_test.reshape((X_test.shape[0], -1))
        ann_model = build_ann_model(X_train_flat, y_train)

        # Train the model on training data
        print(f'\nANN model training started for cluster {i + 1}...')
        ann_model.fit(
            X_train_flat,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.15,  # Split off 15% for validation
            verbose=1
        )

        # Evaluate model on training data
        y_train_pred = ann_model.predict(X_train_flat)
        y_train_pred_clipped = np.clip(y_train_pred, 1.0, 10.0)

        mask_train = y_train != 0
        train_mae = mean_absolute_error(y_train[mask_train], y_train_pred_clipped[mask_train])

        # Evaluate model on test data
        y_pred = ann_model.predict(X_test_flat)
        y_pred_clipped = np.clip(y_pred, 1.0, 10.0)

        mask_test = y_test != 0
        test_mae = mean_absolute_error(y_test[mask_test], y_pred_clipped[mask_test])

        print('\nResults:')
        print(f'\tTraining MAE: {train_mae:.4f}')
        print(f'\tTest MAE:     {test_mae:.4f}')


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
    Constructs training data (X, y) for ANN using the i-th user's cluster data.

    Args:
        cluster_of_users (List[Tuple[List[np.ndarray], int]]): List of (neighbors_items, user_index).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: shape (rows, N_OF_NEIGHBORS, number_of_ratings)
            - y: shape (rows, number_of_ratings)
    """
    rows = len(cluster_of_users)
    number_of_ratings = len(cluster_of_users[0][0][0])  # length of a single user's item vector

    X = np.empty(shape=(rows, N_OF_NEIGHBORS, number_of_ratings))
    y = np.empty(shape=(rows, number_of_ratings))

    for row_index, (neighbors_items, user_index) in enumerate(cluster_of_users):
        for j, item_vector in enumerate(neighbors_items):
            if j == 0:  # The first item is the current user's own vector
                y[row_index] = item_vector
            else:  # Remaining are neighbors' vectors
                X[row_index][j - 1] = item_vector

    return X, y


def masked_mae(y_true, y_pred):
    """
    Computes mean absolute error (MAE) only over the non-zero (rated) elements.

    Args:
        y_true (tf.Tensor): Ground truth rating matrix.
        y_pred (tf.Tensor): Predicted rating matrix.

    Returns:
        tf.Tensor: Masked MAE loss value.
    """
    mask = tf.not_equal(y_true, 0)  # Mask where rating exists
    mask = tf.cast(mask, tf.float32)  # Convert to float

    error = tf.abs(y_true - y_pred)  # Absolute error
    masked_error = error * mask  # Keep only where mask = 1

    # Prevent division by zero
    loss = tf.reduce_sum(masked_error) / tf.reduce_sum(mask)
    return loss


def build_ann_model(X_train_flat, y_train):
    """
    Builds and compiles an artificial neural network model for rating prediction.

    Args:
        X_train_flat (np.ndarray): Flattened training features of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples, n_outputs).

    Returns:
        tf.keras.Model: Compiled ANN model.
    """
    model = Sequential([
        InputLayer(shape=(X_train_flat.shape[1],)),
        Dense(512, activation='relu'),
        Dropout(0.2),  # prevents overfitting
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss=masked_mae)
    return model
