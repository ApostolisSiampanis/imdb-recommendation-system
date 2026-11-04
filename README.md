# IMDb Recommendation System

## [University of Piraeus](https://www.unipi.gr/en/home/) | [Department of Informatics](https://cs.unipi.gr/en/)
**MSc course**: Pattern Recognition and Machine Learning

**Semester**: 1

**Project Completion Year**: 2025

## Project Overview

This project builds a movie recommendation system on IMDb user ratings. It first groups users with similar tastes using K-Medoids clustering and then trains an Artificial Neural Network (ANN) per cluster that predicts a user's ratings from the ratings of their nearest neighbors.

The pipeline is split into three stages:

1. **Data preparation:** Load the raw IMDb ratings, keep only users with 100 to 300 ratings, and build a user-item rating matrix.
2. **User clustering:** Cluster users with K-Medoids using custom Euclidean, Cosine and Jaccard distance matrices, and visualise the clusters in 2D with PCA.
3. **Rating prediction:** For every cluster, train an ANN that takes the rating vectors of a user's 5 nearest neighbors as input and predicts the user's own ratings.

## Key Features

- **Exploratory analysis:** Histograms of the number of ratings per user and of the time span between each user's first and last rating.
- **Custom distance metrics:** Euclidean (normalised to [0, 1]), Cosine and Jaccard distances computed over the user-item matrix.
- **K-Medoids clustering:** PAM with `k-medoids++` initialisation on precomputed distance matrices, run for 2, 3 and 4 clusters.
- **Cluster visualisation:** PCA projection of users to 2D, with one plot per metric and number of clusters.
- **Neural network recommender:** A fully connected network (512 → 256 → 128 units, ReLU, Dropout) trained with a masked MAE loss, so only movies the user actually rated count towards the error.
- **Evaluation:** Mean Absolute Error (MAE) on the training and test sets (67% / 33% split) for every cluster.
- **Caching:** Intermediate DataFrames and matrices are stored as pickle files, so later runs start much faster.

## Technologies Used

- Python
- **Libraries:**
  - `pandas`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `scikit-learn-extra` (K-Medoids)
  - `tensorflow` / `keras`
  - `matplotlib`
  - `seaborn`
  - `tqdm`

## Project Structure

```
/src
    main.py                         # Entry point: data preparation, clustering and ANN pipeline
    constants.py                    # Paths and parameters (rating limits, clusters, neighbors)
    kmedoids_clustering.py          # K-Medoids clustering and PCA visualisation
    ann_recommendation_system.py    # Neural network training and evaluation per cluster
    plot.py                         # Histogram and cluster plots
    /distance
        distance_matrix.py          # Distance matrix calculation for each metric
        euclidean_distance.py
        cosine_distance.py
        jaccard_distance.py
requirements.txt
```

The following folders are created when the program runs and are not tracked in Git:

- **`/datafiles`**: Input dataset and cached pickle files.
- **`/histograms`**: Generated histogram plots.
- **`/clusters`**: Generated cluster plots.

## Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/ApostolisSiampanis/imdb-recommendation-system.git
    cd imdb-recommendation-system
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Place the IMDb ratings dataset at `datafiles/Dataset.npy`. Each entry is a comma-separated string in the form `user,item,rating,date` (for example `ur1234567,tt0111161,9,2015-06-01`).

4. Run the program from the repository root:

    ```bash
    python -m src.main
    ```

## Configuration

The main parameters can be changed in `src/constants.py`:

| Parameter | Default | Description |
|---|---|---|
| `MINIMUM_RATINGS` | `100` | Minimum number of ratings a user needs to be included |
| `MAXIMUM_RATINGS` | `300` | Maximum number of ratings a user can have to be included |
| `N_OF_CLUSTERS_LIST` | `[2, 3, 4]` | Cluster counts tried with Euclidean and Cosine distance |
| `N_OF_CLUSTERS` | `3` | Number of clusters used by the neural network recommender (Jaccard distance) |
| `N_OF_NEIGHBORS` | `5` | Number of nearest neighbors used as input for each user |

## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/ApostolisSiampanis"><img src="https://avatars.githubusercontent.com/u/75365398?v=4" width="100px;" alt="Apostolis Siampanis"/><br /><sub><b>Apostolis Siampanis</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/dimitrisstyl7"><img src="https://avatars.githubusercontent.com/u/75742419?v=4" width="100px;" alt="Dimitris Stylianou"/><br /><sub><b>Dimitris Stylianou</b></sub></a><br /></td>
  </tr>
</table>

## Acknowledgments
This project was developed as part of the "Pattern Recognition and Machine Learning" MSc course at the University of Piraeus. Contributions and feedback are always welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
