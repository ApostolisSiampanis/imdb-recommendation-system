import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = "datafiles/Dataset.npy"
DATA_PATH = "datafiles"
PICTURES_PATH = "pictures"
MINIMUM_RATINGS = 100
MAXIMUM_RATINGS = 300

def check_and_setup_directories():
    # Directory setup
    os.makedirs(DATA_PATH, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Directory {DATASET_PATH} does not exist.")

    os.makedirs(PICTURES_PATH, exist_ok=True)

    if not os.path.exists(PICTURES_PATH):
        raise FileNotFoundError(f"Directory {PICTURES_PATH} does not exist.")

def get_dataframe():
    pickle_file = os.path.join(DATA_PATH, "dataframe.pkl")

    # Load dataset
    if os.path.exists(pickle_file):
        df = pd.read_pickle(pickle_file)
    else:
        raw_data = np.load(DATASET_PATH)
        data = np.array([x.split(",") for x in raw_data])
        df = pd.DataFrame(data, columns=["user", "item", "rating", "date"])
        df["user"] = df["user"].str.replace("ur", "").astype(int)
        df["item"] = df["item"].str.replace("tt", "").astype(int)
        df["rating"] = df["rating"].astype(int)
        df["date"] = pd.to_datetime(df["date"])
        df.to_pickle(pickle_file)

    return df

def get_unique_users(df):
    unique_users = df["user"].unique()
    return unique_users, len(unique_users)

def get_unique_items(df):
    unique_items = df["item"].unique()
    return unique_items, len(unique_items)

def get_ratings_per_user(df):
    pickle_file = os.path.join(DATA_PATH, "ratings_per_user.pkl")

    if os.path.exists(pickle_file):
        ratings_per_user = pd.read_pickle(pickle_file)
    else:
        ratings_per_user = df.groupby("user")["rating"].count().sort_values(ascending=False).reset_index(
            name="ratings_count")
        ratings_per_user.to_pickle(pickle_file)

    return ratings_per_user

def get_ratings_span(df):
    pickle_file = os.path.join(DATA_PATH, "ratings_span.pkl")
    if os.path.exists(pickle_file):
        ratings_span_df = pd.read_pickle(pickle_file)
    else:
        ratings_span_df = (df.groupby("user")["date"].apply(lambda date:
                                                           max(date) - min(date)).sort_values(ascending=False)
                           .reset_index(name="ratings_span"))
        ratings_span_df.to_pickle(pickle_file)
    return ratings_span_df

def join_ratings_dataframes(ratings_per_user, ratings_span_df):
    ratings_df = ratings_per_user.join(ratings_span_df.set_index('user'), on='user')
    ratings_df["ratings_span"] = ratings_df["ratings_span"].dt.days
    return ratings_df

def plot_histogram(series, title):
    n, bins, patches = plt.hist(series, bins="auto", color="blue",
                                alpha=0.7, rwidth=0.75)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(series.name)
    plt.ylabel("Number of Users") #Frequency
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    filename = series.name + ".png"
    figure_url = os.path.join(PICTURES_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    check_and_setup_directories()

    # Load the dataset
    df = get_dataframe()

    # Get unique user and item counts
    unique_users, unique_users_count = get_unique_users(df)
    unique_items, unique_items_count = get_unique_items(df)

    #print(f"Number of unique users: {unique_users_count}")
    #print(f"Number of unique items: {unique_items_count}")

    ratings_per_user = get_ratings_per_user(df)
    #print(ratings_per_user)

    # Get the ratings span per user
    ratings_span_df = get_ratings_span(df)

    # Create a new ratings dataframe by joining the previously defined dataframes.
    ratings_df = join_ratings_dataframes(ratings_per_user, ratings_span_df)

    #print(ratings_df)

    # Discard users with too few or too many ratings
    reduced_ratings_df = ratings_df[(ratings_df["ratings_count"] >= MINIMUM_RATINGS) &
                                    (ratings_df["ratings_count"] <= MAXIMUM_RATINGS)]

    plot_histogram(reduced_ratings_df["ratings_count"], "Number of Ratings per User")
    plot_histogram(reduced_ratings_df["ratings_span"], "Time span of Ratings per User")

    # Save the reduced ratings dataframe and reset the index
    final_df = df.loc[df["user"].isin(reduced_ratings_df["user"])].reset_index()

    final_df = final_df.drop("index", axis=1)

    #print(final_df)

    final_unique_users, final_unique_users_count = get_unique_users(final_df)
    final_unique_items, final_unique_items_count = get_unique_items(final_df)
    final_ratings_count = len(final_df)