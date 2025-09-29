DATAFILES_DIR_PATH = 'datafiles'
HISTOGRAMS_DIR_PATH = 'histograms'
CLUSTERS_DIR_PATH = 'clusters'
DATASET_FILE_NAME = 'Dataset.npy'
MINIMUM_RATINGS = 100
MAXIMUM_RATINGS = 300
N_OF_CLUSTERS_LIST = [2, 3, 4]  # a list with different number of clusters
N_OF_CLUSTERS = 3  # number of clusters
N_OF_NEIGHBORS = 5  # number of the nearest neighbors (excluding the user themself)

# ANN Hyperparameters
EPOCHS = 15
BATCH_SIZE = 64

# Deployment / export settings
EXPORT_DIR = 'deployment_artifacts'
POPULARITY_FALLBACK_SIZE = 50  # number of popular items to keep for cold-start when watchlist empty
