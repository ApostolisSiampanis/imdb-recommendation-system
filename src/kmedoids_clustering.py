import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

from src.distance.distance_matrix import calculate_distance_matrix, Metric
from src.plot import plot_clusters


def kmedoids_clustering(user_item_matrix, unique_users_indices, metric, n_clusters):
    """
    Performs K-Medoids clustering on user-item interaction data using a specified distance metric.
    Also applies PCA for 2D visualization and saves a plot of the resulting clusters.

    Args:
        user_item_matrix (pd.DataFrame): Matrix of user-item ratings, shape (n_users, n_items).
        unique_users_indices (List[int]): Indices of the users corresponding to the rows of the matrix.
        metric (Metric): Distance metric to use (EUCLIDEAN, COSINE, or JACCARD).
        n_clusters (int): Number of clusters for the K-Medoids algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - clusters_labels: Cluster labels assigned to each user.
            - distance_matrix: Pairwise distance matrix used for clustering.
    """
    print(f'\nK-Medoids Clustering with {n_clusters} clusters... ({metric.value} distance)')

    # Calculate the custom distance matrix
    distance_matrix = calculate_distance_matrix(user_item_matrix.to_numpy(), metric)

    if metric == Metric.EUCLIDEAN:
        # Normalize the entire distance matrix to the range [0, 1] for consistency and comparability
        distance_matrix = MinMaxScaler().fit_transform(distance_matrix)

    kmedoids_model = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='k-medoids++',
                              random_state=42)
    clusters_labels = kmedoids_model.fit_predict(distance_matrix)  # Pass the precomputed distance matrix

    # Apply PCA for visualization to 2D
    print(f'Applying PCA for visualization (from {user_item_matrix.shape[1]} dimensions to 2D)...')
    reducer = PCA(n_components=2, random_state=42)

    # PCA is applied on original data, not the distance matrix
    user_coordinates = reducer.fit_transform(user_item_matrix)
    print('PCA dimensionality reduction complete.')
    user_coordinates_df = pd.DataFrame(data=user_coordinates, index=unique_users_indices,
                                       columns=['component_1', 'component_2'])

    user_coordinates_df['cluster'] = clusters_labels

    plot_clusters(user_coordinates_df,
                  title=f'User Clusters with {metric.value.capitalize()} Distance (clusters={n_clusters}) - K-Medoids, '
                        f'Visualized with PCA',
                  filename=f'clusters_{metric.value}_n{n_clusters}_kmedoids_pca.png')

    return clusters_labels, distance_matrix
