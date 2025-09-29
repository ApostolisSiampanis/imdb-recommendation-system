import os
import json
from functools import lru_cache
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.constants import EXPORT_DIR, N_OF_NEIGHBORS

# Determine project root (src/ parent)
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / EXPORT_DIR

# ----------------------------------------------------------------------------------
# Cold-start / new user inference utilities
# ----------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _artifact_dir() -> Path:
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"Artifacts directory '{ARTIFACTS_DIR}' does not exist. Run main.py first to generate artifacts.")
    return ARTIFACTS_DIR


def _artifact_path(name: str) -> Path:
    p = _artifact_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"Required artifact '{p}' not found. Ensure training export completed.")
    return p


def validate_artifacts(required: Optional[List[str]] = None) -> List[str]:
    default_required = [
        'item_mapping.json', 'reverse_item_mapping.json', 'cluster_labels.npy',
        'neighbors_indices.npy', 'user_item_matrix.npy', 'metadata.json', 'popular_items.json'
    ]
    targets = required or default_required
    missing = [name for name in targets if not (ARTIFACTS_DIR / name).exists()]
    return missing


@lru_cache(maxsize=1)
def _load_item_mapping():
    with open(_artifact_path('item_mapping.json')) as f:
        item_mapping = json.load(f)
    return {int(k): int(v) for k, v in item_mapping.items()}


@lru_cache(maxsize=1)
def _load_reverse_item_mapping():
    with open(_artifact_path('reverse_item_mapping.json')) as f:
        rim = json.load(f)
    return {int(k): int(v) for k, v in rim.items()}


@lru_cache(maxsize=1)
def _load_cluster_labels():
    return np.load(_artifact_path('cluster_labels.npy'))


@lru_cache(maxsize=1)
def _load_neighbors_indices():
    return np.load(_artifact_path('neighbors_indices.npy'))


@lru_cache(maxsize=1)
def _load_user_item_matrix():
    return np.load(_artifact_path('user_item_matrix.npy'))


@lru_cache(maxsize=1)
def _load_metadata():
    with open(_artifact_path('metadata.json')) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _load_cluster_model(cluster_zero_based: int):
    model_path = _artifact_dir() / 'models' / f'cluster_{cluster_zero_based + 1}.keras'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for cluster {cluster_zero_based}: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


@lru_cache(maxsize=1)
def _load_popularity():
    path = ARTIFACTS_DIR / 'popular_items.json'
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _vectorized_jaccard_dist(new_user_binary: np.ndarray, existing_binary: np.ndarray) -> np.ndarray:
    intersection = (existing_binary & new_user_binary).sum(axis=1)
    union = (existing_binary | new_user_binary).sum(axis=1)
    union = np.where(union == 0, 1, union)
    similarity = intersection / union
    return 1.0 - similarity


def _infer_cluster_for_new_user(neighbor_user_indices: np.ndarray) -> int:
    cluster_labels = _load_cluster_labels()
    neighbor_clusters = cluster_labels[neighbor_user_indices]
    unique, counts = np.unique(neighbor_clusters, return_counts=True)
    return int(unique[np.argmax(counts)])


def recommend_for_new_user(
        watchlist_items: List[int],
        watchlist_ratings: Optional[Dict[int, int]] = None,
        top_k: int = 10,
        min_rating_pred: float = 1.0,
        default_watchlist_rating: int = 8,
        return_internal_ids: bool = False,
        validate: bool = True
) -> Dict:
    """Generate recommendations for a brand-new user.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    if validate:
        missing = validate_artifacts()
        if missing:
            raise FileNotFoundError(f"Missing artifacts: {missing}. Run main.py to generate them in '{ARTIFACTS_DIR}'.")

    item_mapping = _load_item_mapping()
    reverse_item_mapping = _load_reverse_item_mapping()
    user_item_matrix = _load_user_item_matrix()
    metadata = _load_metadata()

    num_items = user_item_matrix.shape[1]

    if not watchlist_items:
        popular = _load_popularity()[:top_k]
        return {
            'strategy': 'cold_start_popularity',
            'input_watchlist_count': 0,
            'recommendations': popular,
            'artifact_dir': str(ARTIFACTS_DIR)
        }

    new_user_vector = np.zeros((num_items,), dtype=np.float32)
    missing_items = []
    for original_item_id in watchlist_items:
        try:
            internal = item_mapping[int(original_item_id)]
        except (KeyError, ValueError, TypeError):
            missing_items.append(original_item_id)
            continue
        rating = default_watchlist_rating
        if watchlist_ratings and original_item_id in watchlist_ratings:
            rating = watchlist_ratings[original_item_id]
        new_user_vector[internal] = rating

    existing_binary = user_item_matrix != 0
    new_user_binary = new_user_vector != 0

    distances = _vectorized_jaccard_dist(new_user_binary, existing_binary)
    neighbor_indices = np.argsort(distances)[:N_OF_NEIGHBORS]

    cluster = _infer_cluster_for_new_user(neighbor_indices)
    model = _load_cluster_model(cluster)

    neighbor_vectors = user_item_matrix[neighbor_indices]
    actual_neighbors = neighbor_vectors.shape[0]
    padded = False
    if actual_neighbors < N_OF_NEIGHBORS:
        pad_rows = N_OF_NEIGHBORS - actual_neighbors
        pad_block = np.zeros((pad_rows, neighbor_vectors.shape[1]), dtype=neighbor_vectors.dtype)
        neighbor_vectors = np.vstack([neighbor_vectors, pad_block])
        padded = True

    model_input = neighbor_vectors.reshape(1, -1)
    pred_vector = model.predict(model_input, verbose=0)[0]

    mask_watched = new_user_vector != 0
    pred_vector[mask_watched] = -1e9

    candidate_indices = np.where(pred_vector >= min_rating_pred)[0]
    watched_count = int(np.count_nonzero(mask_watched))
    if candidate_indices.size == 0:
        watched_internal = set(np.where(mask_watched)[0])
        popular = [p for p in _load_popularity() if p.get('internal_item_id') not in watched_internal]
        return {
            'strategy': 'popularity_fallback_after_filter',
            'input_watchlist_count': watched_count,
            'recommendations': popular[:top_k],
            'missing_items_ignored': missing_items,
            'neighbors_used': neighbor_indices.tolist(),
            'neighbors_padded': padded,
            'artifact_dir': str(ARTIFACTS_DIR)
        }

    candidate_scores = pred_vector[candidate_indices]
    order = np.argsort(candidate_scores)[-top_k:][::-1]
    top_indices = candidate_indices[order]

    recommendations = []
    for idx in top_indices:
        original_id = reverse_item_mapping.get(int(idx))
        rec = {
            'original_item_id': original_id,
            'predicted_rating': float(pred_vector[idx])
        }
        if return_internal_ids:
            rec['internal_item_id'] = int(idx)
        recommendations.append(rec)

    return {
        'strategy': 'neighbor_ann_inference',
        'cluster': int(cluster),
        'input_watchlist_count': watched_count,
        'neighbors_used': neighbor_indices.tolist(),
        'neighbors_padded': padded,
        'top_k': top_k,
        'recommendations': recommendations,
        'missing_items_ignored': missing_items,
        'model_input_features': int(model_input.shape[1]),
        'model_output_items': int(pred_vector.shape[0]),
        'epochs_trained': metadata.get('epochs'),
        'artifact_dir': str(ARTIFACTS_DIR)
    }


if __name__ == '__main__':
    # Quick manual check (will fall back to popularity if IDs missing)
    example_watchlist = [123456, 345678, 456789]
    try:
        result = recommend_for_new_user(example_watchlist, watchlist_ratings=None, top_k=5, return_internal_ids=True)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        print(f"Artifact error: {e}")
