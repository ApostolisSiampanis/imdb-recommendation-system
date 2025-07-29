import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import HISTOGRAMS_DIR_PATH, CLUSTERS_DIR_PATH


def plot_histogram(series, title, xlabel, ylabel, filename):
    """
    Plots and saves a histogram of the given data series.

    Args:
        series (pd.Series or np.ndarray): The numerical data to plot.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis (e.g., "Frequency").
        filename (str): Name of the PNG file to save the plot under (saved in HISTOGRAMS_DIR_PATH).

    Saves:
        A PNG image of the histogram in the HISTOGRAMS_DIR_PATH directory.

    Displays:
        The histogram in a pop-up window (using matplotlib).
    """
    n, bins, patches = plt.hist(series, bins='auto', color='blue',
                                alpha=0.7, rwidth=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)  # Frequency
    plt.title(title)
    max_freq = n.max()
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    figure_url = os.path.join(HISTOGRAMS_DIR_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()


def plot_clusters(user_coordinates_df, title, filename):
    """
    Plots and saves a 2D scatter plot of clustered users based on PCA-reduced coordinates.

    Args:
        user_coordinates_df (pd.DataFrame): DataFrame containing user coordinates and cluster labels.
            Expected columns: ['component_1', 'component_2', 'cluster'].
        title (str): Title of the scatter plot.
        filename (str): Name of the PNG file to save the plot under (saved in CLUSTERS_DIR_PATH).

    Saves:
        A PNG image of the plotted clusters in the CLUSTERS_DIR_PATH directory.

    Displays:
        The scatter plot in a pop-up window (using matplotlib).
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='component_1', y='component_2', hue='cluster', data=user_coordinates_df, palette='viridis',
                    legend='full', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    figure_url = os.path.join(CLUSTERS_DIR_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()
    print(f'Clusters visualized and saved to {figure_url}')
