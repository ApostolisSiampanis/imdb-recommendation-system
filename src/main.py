import os

import numpy as np
import pandas as pd

from src.clustering.kmedoids_clustering import kmedoids_clustering
from src.constants import *
from src.plot import plot_histogram


def check_and_setup_directories():
    # Directory setup
    os.makedirs(DATAFILES_DIR_PATH, exist_ok=True)
    os.makedirs(HISTOGRAMS_DIR_PATH, exist_ok=True)
    os.makedirs(CLUSTERS_DIR_PATH, exist_ok=True)

    # Check if a dataset file with the name 'Dataset.npy' exists
    dataset_path = os.path.join(DATAFILES_DIR_PATH, DATASET_FILE_NAME)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset file {dataset_path} does not exist or is not a file.')


def get_dataframe():
    pickle_file = os.path.join(DATAFILES_DIR_PATH, 'dataframe.pkl')

    # Load dataset
    if os.path.exists(pickle_file):
        df = pd.read_pickle(pickle_file)
    else:
        dataset_path = os.path.join(DATAFILES_DIR_PATH, DATASET_FILE_NAME)
        raw_data = np.load(dataset_path)
        data = np.array([x.split(',') for x in raw_data])
        df = pd.DataFrame(data=data, columns=['user', 'item', 'rating', 'date'])
        df['user'] = df['user'].str.replace('ur', '').astype(int)
        df['item'] = df['item'].str.replace('tt', '').astype(int)
        df['rating'] = df['rating'].astype(int)
        df['date'] = pd.to_datetime(df['date'])
        df.to_pickle(pickle_file)

    return df


def get_unique_users(df):
    unique_users = df['user'].unique()
    return unique_users, len(unique_users)


def get_unique_items(df):
    unique_items = df['item'].unique()
    return unique_items, len(unique_items)


def get_num_of_ratings_per_user(df):
    pickle_file = os.path.join(DATAFILES_DIR_PATH, 'ratings_per_user.pkl')
    if os.path.isfile(pickle_file):
        num_of_ratings_per_user_df = pd.read_pickle(pickle_file)
    else:
        num_of_ratings_per_user_df = (df.groupby('user')['rating']
                                      .count()
                                      .reset_index(name='ratings_count')
                                      .sort_values(by='ratings_count', ascending=False))
        num_of_ratings_per_user_df.to_pickle(pickle_file)

    return num_of_ratings_per_user_df


def get_time_span_per_user(df):
    pickle_file = os.path.join(DATAFILES_DIR_PATH, 'time_span_per_user.pkl')
    if os.path.exists(pickle_file):
        ratings_span_per_user_df = pd.read_pickle(pickle_file)
    else:
        ratings_span_per_user_df = (df.groupby('user')['date']
                                    .apply(lambda date: max(date) - min(date))
                                    .sort_values(ascending=False)
                                    .reset_index(name='time_span'))
        ratings_span_per_user_df.to_pickle(pickle_file)
    return ratings_span_per_user_df


def join_dataframes_on_user(num_of_ratings_per_user_df, time_span_per_user_df):
    joined_df = num_of_ratings_per_user_df.join(other=time_span_per_user_df.set_index('user'), on='user')
    joined_df['time_span'] = joined_df['time_span'].dt.days
    return joined_df


def create_user_item_matrix(df, unique_users_list, unique_items_list):
    pickle_file = os.path.join(DATAFILES_DIR_PATH, 'user_item_matrix.pkl')

    if os.path.exists(pickle_file):
        user_item_matrix = pd.read_pickle(pickle_file)
    else:
        user_item_matrix = pd.DataFrame(data=0, index=unique_users_list, columns=unique_items_list)

        for _, row in df.iterrows():
            user = row['user']
            item = row['item']
            rating = row['rating']
            user_item_matrix.at[user, item] = rating

        user_item_matrix.to_pickle(pickle_file)

    return user_item_matrix


if __name__ == '__main__':
    check_and_setup_directories()

    # Load the dataset
    df = get_dataframe()

    # Get unique user and item counts
    unique_users, unique_users_count = get_unique_users(df)
    unique_items, unique_items_count = get_unique_items(df)

    num_of_ratings_per_user_df = get_num_of_ratings_per_user(df)

    # Get the time span per user
    time_span_per_user_df = get_time_span_per_user(df)

    # Create a new dataframe by joining the num_of_ratings_per_user_df and time_span_per_user_df dataframes.
    joined_df = join_dataframes_on_user(num_of_ratings_per_user_df, time_span_per_user_df)

    # Discard users with too few or too many ratings
    reduced_joined_df = joined_df[(joined_df['ratings_count'] >= MINIMUM_RATINGS) &
                                  (joined_df['ratings_count'] <= MAXIMUM_RATINGS)]

    # Plot histograms
    plot_histogram(series=reduced_joined_df['ratings_count'],
                   title='Number of Users per Rating Count',
                   xlabel='Rating Count',
                   ylabel='Number of Users',
                   filename='rating_count_hist.png')

    plot_histogram(series=reduced_joined_df['time_span'],
                   title='Number of Users per Rating Time Span',
                   xlabel='Rating Time Span (days)',
                   ylabel='Number of Users',
                   filename='time_span_hist.png')

    # Filter df to keep the users from reduced_joined_df
    final_df = df.loc[df['user'].isin(reduced_joined_df['user'])].reset_index(drop=True)

    # Get unique user and item counts from the final_df
    final_unique_users, final_unique_users_count = get_unique_users(final_df)
    final_unique_items, final_unique_items_count = get_unique_items(final_df)

    # Sort final_unique_users and final_unique_items
    sorted_final_users = np.sort(final_unique_users)
    sorted_final_items = np.sort(final_unique_items)

    final_unique_users_indices = list(range(final_unique_users_count))
    final_unique_items_indices = list(range(final_unique_items_count))

    # Create mappings from original user/item IDs to consecutive integer indices
    final_users_dict = dict(zip(sorted_final_users, final_unique_users_indices))
    final_items_dict = dict(zip(sorted_final_items, final_unique_items_indices))

    # Replace original user/item IDs in final_df with their corresponding integer indices
    final_df['user'] = final_df['user'].map(final_users_dict)
    final_df['item'] = final_df['item'].map(final_items_dict)

    # Create user-item matrix (dataframe)
    user_item_matrix = create_user_item_matrix(final_df,
                                               final_unique_users_indices,
                                               final_unique_items_indices)

    # Perform KMedoids clustering
    kmedoids_clustering(user_item_matrix=user_item_matrix, unique_users_indices=final_unique_users_indices)
