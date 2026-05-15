import numpy as np
from scipy.spatial.distance import cdist


def pairwise_hamming(E: np.ndarray) -> np.ndarray:
    """Full pairwise Hamming distance matrix from embedding.

    E: (n, L) int64 — embedding returned by compute_embedding
    Returns: (n, n) float32
    """
    return cdist(E, E, metric="hamming").astype(np.float32)


def pairwise_hamming_chunked(E: np.ndarray, chunk_size: int = 2_000) -> np.ndarray:
    """Memory-efficient version for larger n (builds full matrix row-by-row)."""
    n = E.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D[start:end] = cdist(E[start:end], E, metric="hamming").astype(np.float32)
    return D


def cross_hamming(E_X: np.ndarray, E_Y: np.ndarray) -> np.ndarray:
    """Pairwise Hamming between two embedding matrices. Returns (n_X, n_Y) float32."""
    return cdist(E_X, E_Y, metric="hamming").astype(np.float32)
