import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

from src.constants import *
from src.distance.distance_matrix import calculate_distance_matrix, Metric
from src.utils import plot_clusters


def kmedoids_clustering(user_item_matrix, unique_users_indices):
    # Apply PCA for visualization to 2D
    print(f'Applying PCA for visualization (from {user_item_matrix.shape[1]} dimensions to 2D)...')
    reducer = PCA(n_components=2, random_state=42)

    # PCA is applied on original data, not the distance matrix
    user_coordinates = reducer.fit_transform(user_item_matrix)
    print('PCA dimensionality reduction complete.')
    user_coordinates_df = pd.DataFrame(user_coordinates, index=unique_users_indices,
                                       columns=['component_1', 'component_2'])

    # Calculate the custom distance matrices
    for metric in [Metric.EUCLIDEAN, Metric.COSINE]:
        distance_matrix = calculate_distance_matrix(user_item_matrix.to_numpy(), metric)

        if metric == Metric.EUCLIDEAN:
            # Normalize the entire distance matrix to the range [0, 1] for consistency and comparability
            distance_matrix = MinMaxScaler().fit_transform(distance_matrix)

        for n in N_OF_CLUSTERS:
            print(f'K-Medoids Clustering with n={n}... ({metric.value} distance)')
            kmedoids_model = KMedoids(n_clusters=n, metric='precomputed', method='pam', init='k-medoids++',
                                      random_state=42)
            clusters = kmedoids_model.fit_predict(distance_matrix)  # Pass the precomputed distance matrix
            user_coordinates_df['cluster'] = clusters

            plot_clusters(user_coordinates_df,
                          title=f'User Clusters with {metric.value} Distance (clusters={n}) - K-Medoids, '
                                f'Visualized with PCA',
                          filename=f'clusters_{metric.value.lower()}_n{n}_kmedoids_pca.png')
