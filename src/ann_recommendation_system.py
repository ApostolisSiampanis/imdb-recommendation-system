from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
import os
import json
import gc

from src.constants import N_OF_NEIGHBORS, N_OF_CLUSTERS, EPOCHS, BATCH_SIZE
from src.distance.distance_matrix import Metric
from src.kmedoids_clustering import kmedoids_clustering

# Enable dynamic GPU memory growth (safe no-op on CPU-only systems)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:  # pragma: no cover
    print(f"GPU memory growth setup skipped: {e}")


def ann_recommendation_system(user_item_matrix, unique_users_indices, export_dir: str | None = None):
    """
    Train per-cluster ANN models to predict a user's full rating vector from neighbor rating vectors.

    Pipeline:
      1. Cluster users via K-Medoids (Jaccard distance) into N_OF_CLUSTERS clusters.
      2. For each user, collect their top N_OF_NEIGHBORS nearest neighbors (plus self) to build training rows.
      3. For each cluster, train an ANN that maps concatenated neighbor rating vectors -> target user rating vector.
      4. Optionally export artifacts for inference (models, cluster labels, neighbor indices, metadata).

    Args:
        user_item_matrix (pd.DataFrame): User-item ratings (rows=users, cols=items, 0 means no rating).
        unique_users_indices (List[int]): Indices corresponding to rows of user_item_matrix.
        export_dir (str | None): Directory to export artifacts. If None, no export performed.
    """
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        models_dir = os.path.join(export_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

    clusters_labels, distance_matrix = kmedoids_clustering(user_item_matrix=user_item_matrix,
                                                           unique_users_indices=unique_users_indices,
                                                           metric=Metric.JACCARD,
                                                           n_clusters=N_OF_CLUSTERS)

    # Pre-compute cluster user groupings and neighbor indices
    clusters_of_users = get_clusters_of_users(n_clusters=N_OF_CLUSTERS,
                                              clusters_labels=clusters_labels,
                                              user_item_matrix=user_item_matrix,
                                              distance_matrix=distance_matrix)

    if export_dir:
        # Save cluster labels
        np.save(os.path.join(export_dir, 'cluster_labels.npy'), clusters_labels)
        # Save neighbor indices (excluding self) for each user for fast inference reuse
        num_users = distance_matrix.shape[0]
        neighbors_matrix = np.empty((num_users, N_OF_NEIGHBORS), dtype=np.int32)
        for u in range(num_users):
            sorted_idx = np.argsort(distance_matrix[u])
            sorted_idx = sorted_idx[sorted_idx != u][:N_OF_NEIGHBORS]
            # Pad if dataset smaller than N_OF_NEIGHBORS+1
            if len(sorted_idx) < N_OF_NEIGHBORS:
                pad = np.full(N_OF_NEIGHBORS - len(sorted_idx), -1)
                neighbors_matrix[u] = np.concatenate([sorted_idx, pad])
            else:
                neighbors_matrix[u] = sorted_idx
        np.save(os.path.join(export_dir, 'neighbors_indices.npy'), neighbors_matrix)

    metadata = {
        'n_clusters': N_OF_CLUSTERS,
        'n_neighbors': N_OF_NEIGHBORS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'models': []
    } if export_dir else None

    # Iterate through clusters and construct training data for each
    for i, cluster_of_users in enumerate(clusters_of_users):
        cluster_id = i + 1
        if len(cluster_of_users) == 0:
            print(f"Skipping empty cluster {cluster_id}.")
            continue

        X, y = build_training_data_from_cluster(cluster_of_users)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))

        model = build_ann_model(X_train_flat, y_train)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-5)

        print(f"\nTraining ANN for cluster {cluster_id} | samples={X_train_flat.shape[0]} features={X_train_flat.shape[1]}")
        history = model.fit(
            X_train_flat,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )

        # Evaluation
        y_train_pred = model.predict(X_train_flat, batch_size=BATCH_SIZE)
        y_train_pred = np.clip(y_train_pred, 1.0, 10.0)
        mask_train = y_train != 0
        train_mae = mean_absolute_error(y_train[mask_train], y_train_pred[mask_train])

        y_test_pred = model.predict(X_test_flat, batch_size=BATCH_SIZE)
        y_test_pred = np.clip(y_test_pred, 1.0, 10.0)
        mask_test = y_test != 0
        test_mae = mean_absolute_error(y_test[mask_test], y_test_pred[mask_test])

        print("Results:")
        print(f"  Cluster {cluster_id} Train MAE: {train_mae:.4f}")
        print(f"  Cluster {cluster_id} Test  MAE: {test_mae:.4f}")

        if export_dir:
            model_path = os.path.join(models_dir, f'cluster_{cluster_id}.keras')
            model.save(model_path, include_optimizer=False)
            metadata['models'].append({
                'cluster': cluster_id,
                'samples': int(X_train.shape[0] + X_test.shape[0]),
                'input_features': int(X_train_flat.shape[1]),
                'output_dim': int(y_train.shape[1]),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'model_path': os.path.relpath(model_path, export_dir)
            })

        # Cleanup
        del X, y, X_train, X_test, y_train, y_test, X_train_flat, X_test_flat, y_train_pred, y_test_pred, history, model
        tf.keras.backend.clear_session()
        gc.collect()

    if export_dir and metadata:
        with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def get_clusters_of_users(n_clusters, clusters_labels, user_item_matrix, distance_matrix):
    """Group users into clusters and collect each user's top-N neighbors (including self at index 0)."""
    clusters_of_users: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(n_clusters)]
    user_item_np = user_item_matrix.to_numpy()
    num_users = len(clusters_labels)
    for current_user_index in range(num_users):
        cluster_index = clusters_labels[current_user_index]
        current_user_distances = distance_matrix[current_user_index]
        neighbors_indices = np.argsort(current_user_distances)[:N_OF_NEIGHBORS + 1]
        neighbors = [user_item_np[idx] for idx in neighbors_indices]
        clusters_of_users[cluster_index].append((neighbors, current_user_index))
    return clusters_of_users


def build_training_data_from_cluster(cluster_of_users):
    """Create X (neighbors) and y (target user) arrays for one cluster."""
    rows = len(cluster_of_users)
    number_of_ratings = len(cluster_of_users[0][0][0])
    X = np.empty((rows, N_OF_NEIGHBORS, number_of_ratings), dtype=np.float32)
    y = np.empty((rows, number_of_ratings), dtype=np.float32)
    for row_index, (neighbors_items, _user_index) in enumerate(cluster_of_users):
        for j, item_vector in enumerate(neighbors_items):
            if j == 0:
                y[row_index] = item_vector
            else:
                if j - 1 < N_OF_NEIGHBORS:
                    X[row_index, j - 1] = item_vector
    return X, y


def masked_mae(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    error = tf.abs(y_true - y_pred)
    masked_error = error * mask
    return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)


def build_ann_model(X_train_flat, y_train):
    input_dim = X_train_flat.shape[1]
    output_dim = y_train.shape[1]
    model = Sequential([
        InputLayer(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss=masked_mae)
    return model
