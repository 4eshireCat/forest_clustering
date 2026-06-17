"""Weighted Hamming distance computation for partition embeddings."""

import numpy as np


# ---------------------------------------------------------------------------
# Numba support check and module-level kernels
# ---------------------------------------------------------------------------

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _pairwise_weighted_hamming_numba(E, w):
        n, L = E.shape
        D = np.empty((n, n), dtype=np.float32)
        for i in numba.prange(n):
            for j in range(n):
                s = 0.0
                for l in range(L):
                    wl = w[l]
                    if wl == 0.0:
                        continue
                    if E[i, l] != E[j, l]:
                        s += wl
                D[i, j] = s
        return D

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _weighted_cross_hamming_numba(E_X, E_Y, w):
        n_X, L = E_X.shape
        n_Y = E_Y.shape[0]
        D = np.empty((n_X, n_Y), dtype=np.float32)
        for i in numba.prange(n_X):
            for j in range(n_Y):
                s = 0.0
                for l in range(L):
                    wl = w[l]
                    if wl == 0.0:
                        continue
                    if E_X[i, l] != E_Y[j, l]:
                        s += wl
                D[i, j] = s
        return D


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_n_jobs(n_jobs):
    """Resolve n_jobs parameter."""
    import os
    if n_jobs == -1:
        return int(os.environ.get("FOREST_CLUSTERING_N_JOBS",
                                  os.cpu_count() or 1))
    elif n_jobs <= 0:
        return 1
    return n_jobs


# ---------------------------------------------------------------------------
# Reference (serial) implementations
# ---------------------------------------------------------------------------

def pairwise_weighted_hamming(E: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pairwise Hamming distance.

    D[i,j] = sum_l w[l] * [E[i,l] != E[j,l]] / sum_l w[l]

    Parameters
    ----------
    E : np.ndarray, shape (n, L), dtype int64
        Embedding matrix.
    weights : np.ndarray, shape (L,), dtype float64
        Per-iteration weights. Must be 1-D, non-negative, not all zero.

    Returns
    -------
    D : np.ndarray, shape (n, n), dtype float32
        Weighted Hamming distance matrix.
    """
    weights = np.asarray(weights)
    _validate_weighted_distance_inputs(E, weights)
    n, L = E.shape
    w = weights / weights.sum()

    D = np.zeros((n, n), dtype=np.float64)
    for l in range(L):
        if w[l] == 0:
            continue
        col_diff = (E[:, l][:, None] != E[:, l][None, :]).astype(np.float64)
        D += w[l] * col_diff

    return D.astype(np.float32)


def weighted_cross_hamming(E_X: np.ndarray, E_Y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted cross Hamming distance between two embedding matrices.

    Parameters
    ----------
    E_X : np.ndarray, shape (n_X, L), dtype int64
        First embedding matrix.
    E_Y : np.ndarray, shape (n_Y, L), dtype int64
        Second embedding matrix.
    weights : np.ndarray, shape (L,), dtype float64
        Per-iteration weights. Must be 1-D, non-negative, not all zero.

    Returns
    -------
    D : np.ndarray, shape (n_X, n_Y), dtype float32
        Weighted Hamming cross-distance matrix.
    """
    weights = np.asarray(weights)
    _validate_weighted_distance_inputs(E_X, weights)
    _validate_weighted_distance_inputs(E_Y, weights)

    if E_X.shape[1] != E_Y.shape[1]:
        raise ValueError(
            f"E_X and E_Y must have the same number of columns, "
            f"got {E_X.shape[1]} and {E_Y.shape[1]}"
        )

    n_X, L = E_X.shape
    n_Y = E_Y.shape[0]
    w = weights / weights.sum()

    D = np.zeros((n_X, n_Y), dtype=np.float64)
    for l in range(L):
        if w[l] == 0:
            continue
        col_diff = (E_X[:, l][:, None] != E_Y[:, l][None, :]).astype(np.float64)
        D += w[l] * col_diff

    return D.astype(np.float32)


def _validate_weighted_distance_inputs(E: np.ndarray, weights: np.ndarray) -> None:
    """Validate inputs for weighted distance computation."""
    E = np.asarray(E)
    weights = np.asarray(weights)

    if not np.issubdtype(E.dtype, np.integer):
        raise TypeError(f"Embedding must be integer type, got {E.dtype}")

    if E.ndim != 2:
        raise ValueError(f"E must be a 2-D array, got ndim={E.ndim}")

    n = E.shape[0]
    if n == 0:
        raise ValueError("E must have at least one row (n > 0)")

    if weights.ndim != 1:
        raise ValueError(f"weights must be a 1-D array, got ndim={weights.ndim}")

    L = E.shape[1]
    if weights.shape[0] != L:
        raise ValueError(
            f"weights length ({weights.shape[0]}) must match "
            f"number of columns in E ({L})"
        )

    if (weights < 0).any():
        raise ValueError("weights must be non-negative")

    if weights.sum() == 0:
        raise ValueError("weights must not sum to zero (all-zero weights)")

    if not np.isfinite(weights).all():
        raise ValueError("weights must be finite (no NaN or Inf values)")


# ---------------------------------------------------------------------------
# Accelerated (parallel + optional numba) implementations
# ---------------------------------------------------------------------------

def pairwise_weighted_hamming_fast(E, weights, n_jobs=-1):
    """Parallel weighted Hamming distance.

    Uses numba if available, otherwise a cache-friendly chunked numpy
    implementation that is faster than the reference for most matrix sizes.
    Returns (n, n) float32.
    """
    E = np.asarray(E)
    weights = np.asarray(weights)
    n, L = E.shape

    # Validate
    if weights.ndim != 1 or weights.shape[0] != L:
        raise ValueError(f"weights shape {weights.shape} != ({L},)")
    if weights.min() < 0:
        raise ValueError("weights must be non-negative")
    if weights.sum() < 1e-15:
        weights = np.ones(L, dtype=np.float64)

    # Normalize weights
    w = weights.astype(np.float64) / weights.sum()

    # Use module-level numba kernel if available
    if HAS_NUMBA:
        return _pairwise_weighted_hamming_numba(E, w)

    # Chunked numpy fallback — cache-friendly, usually faster than reference
    # Adaptive chunk_size tuned for L2 cache
    chunk_size = 128 if n > 200 else 32

    def _compute_chunk(start, end):
        block = np.zeros((end - start, n), dtype=np.float64)
        for l in range(L):
            wl = w[l]
            if wl == 0:
                continue
            diff = (E[start:end, l][:, None] != E[:, l][None, :]).astype(np.float64)
            block += wl * diff
        return block.astype(np.float32)

    n_jobs = _resolve_n_jobs(n_jobs)
    chunks = list(range(0, n, chunk_size)) + [n]

    # For small matrices, serial execution avoids joblib overhead
    if n <= 1000 or n_jobs == 1:
        blocks = [_compute_chunk(chunks[k], chunks[k + 1])
                  for k in range(len(chunks) - 1)]
    else:
        from joblib import Parallel, delayed
        blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_chunk)(chunks[k], chunks[k + 1])
            for k in range(len(chunks) - 1)
        )
    return np.vstack(blocks)


def pairwise_weighted_hamming_chunked(E, weights, chunk_size=2000, n_jobs=-1):
    """Memory-efficient chunked version.
    Computes the matrix row-by-row to limit memory.
    """
    E = np.asarray(E)
    weights = np.asarray(weights)
    n, L = E.shape

    # Validate
    if weights.ndim != 1 or weights.shape[0] != L:
        raise ValueError(f"weights shape {weights.shape} != ({L},)")
    if weights.min() < 0:
        raise ValueError("weights must be non-negative")
    if weights.sum() < 1e-15:
        weights = np.ones(L, dtype=np.float64)

    w = weights.astype(np.float64) / weights.sum()

    # Use module-level numba kernel if available
    if HAS_NUMBA:
        return _pairwise_weighted_hamming_numba(E, w)

    # Chunked numpy fallback — cache-friendly chunked computation
    def _compute_rows(start, end):
        block = np.zeros((end - start, n), dtype=np.float64)
        for l in range(L):
            wl = w[l]
            if wl == 0:
                continue
            diff = (E[start:end, l][:, None] != E[:, l][None, :]).astype(np.float64)
            block += wl * diff
        return block.astype(np.float32)

    n_jobs = _resolve_n_jobs(n_jobs)
    chunks = list(range(0, n, chunk_size)) + [n]

    # For small matrices, serial execution avoids joblib overhead
    if n <= 1000 or n_jobs == 1:
        blocks = [_compute_rows(chunks[k], chunks[k + 1])
                  for k in range(len(chunks) - 1)]
    else:
        from joblib import Parallel, delayed
        blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_rows)(chunks[k], chunks[k + 1])
            for k in range(len(chunks) - 1)
        )
    return np.vstack(blocks)


def weighted_cross_hamming_fast(E_X, E_Y, weights, n_jobs=-1):
    """Fast cross Hamming distance.
    Returns (n_X, n_Y) float32.
    """
    E_X = np.asarray(E_X)
    E_Y = np.asarray(E_Y)
    weights = np.asarray(weights)
    n_X, L = E_X.shape
    n_Y = E_Y.shape[0]

    if weights.ndim != 1 or weights.shape[0] != L:
        raise ValueError(f"weights shape {weights.shape} != ({L},)")
    if weights.min() < 0:
        raise ValueError("weights must be non-negative")
    if weights.sum() < 1e-15:
        weights = np.ones(L, dtype=np.float64)

    w = weights.astype(np.float64) / weights.sum()

    # Use module-level numba kernel if available
    if HAS_NUMBA:
        return _weighted_cross_hamming_numba(E_X, E_Y, w)

    # Chunked numpy fallback — cache-friendly chunked computation
    chunk_size = 128 if n_X > 200 else 32

    def _compute_cross_chunk(start, end):
        block = np.zeros((end - start, n_Y), dtype=np.float64)
        for l in range(L):
            wl = w[l]
            if wl == 0:
                continue
            diff = (E_X[start:end, l][:, None] != E_Y[:, l][None, :]).astype(np.float64)
            block += wl * diff
        return block.astype(np.float32)

    n_jobs = _resolve_n_jobs(n_jobs)
    chunks = list(range(0, n_X, chunk_size)) + [n_X]

    # For small matrices, serial execution avoids joblib overhead
    if n_X <= 1000 or n_jobs == 1:
        blocks = [_compute_cross_chunk(chunks[k], chunks[k + 1])
                  for k in range(len(chunks) - 1)]
    else:
        from joblib import Parallel, delayed
        blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_cross_chunk)(chunks[k], chunks[k + 1])
            for k in range(len(chunks) - 1)
        )
    return np.vstack(blocks)
