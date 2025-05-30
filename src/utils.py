import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import HISTOGRAMS_DIR_PATH, CLUSTERS_DIR_PATH


def non_zero_ratings(rating_u, rating_v):
    # In this context, 0 explicitly means "no rating", so we check for non-zero ratings.
    return int((rating_u != 0) & (rating_v != 0))


def plot_histogram(series, title, xlabel, ylabel, filename):
    n, bins, patches = plt.hist(series, bins="auto", color="blue",
                                alpha=0.7, rwidth=0.75)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)  # Frequency
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    figure_url = os.path.join(HISTOGRAMS_DIR_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()


def plot_clusters(user_coordinates_df, title, filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(hue='cluster', data=user_coordinates_df, palette='viridis', legend='full', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    figure_url = os.path.join(CLUSTERS_DIR_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()
    print(f"Clusters visualized and saved to {figure_url}")
