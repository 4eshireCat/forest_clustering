"""Sparse weighted one-hot features for the cell-id embedding.

The ``(n, L)`` cell-id embedding activates exactly one cell per iteration, so the
weighted one-hot feature matrix ``phi`` has **exactly ``L`` non-zeros per row**.
Materialising it densely costs ``O(n * sum_l C_l)`` memory, where ``C_l`` is the
number of distinct cells in column ``l``; when per-column cardinality approaches
``n`` (many features per iteration) this degenerates to ``O(n^2)`` and blows up.
Building it as a sparse CSR matrix costs ``O(n * L)`` regardless of cardinality.

The per-block scaling ``sqrt(w_l / 2)`` makes the *squared* Euclidean distance
between two rows equal to the weighted Hamming distance used everywhere else::

    ||phi_i - phi_j||^2 = sum_l (w_l / 2) * 2 * [cell_il != cell_jl]
                        = sum_l w_l * [cell_il != cell_jl]

so KMeans / MiniBatchKMeans / Birch run on ``phi`` optimise a
weighted-Hamming-consistent objective without ever building a distance matrix.
"""

import numpy as np
from scipy import sparse


def _resolve_weights(weights, L):
    """Return normalised, validated 1-D weights of length L (uniform on failure)."""
    if weights is None:
        return np.full(L, 1.0 / L, dtype=np.float64) if L else np.zeros(0, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (L,) or not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(L, dtype=np.float64)
    return w / w.sum()


def weighted_onehot_features(E, weights=None, sparse_output=True):
    """Weighted one-hot encoding of a cell-id embedding.

    Parameters
    ----------
    E : ndarray of shape (n, L)
        Integer cell-id embedding (values are nominal labels per column).
    weights : ndarray of shape (L,) or None
        Per-iteration weights.  ``None`` (or any invalid input) means uniform.
        Internally normalised to sum to 1.
    sparse_output : bool, default True
        If True return a ``scipy.sparse.csr_matrix`` with exactly ``L`` non-zeros
        per row; if False return a dense ``float64`` ndarray.  Use dense only for
        estimators that cannot consume sparse input (e.g. GaussianMixture, Ward).

    Returns
    -------
    phi : csr_matrix or ndarray, shape (n, sum_l C_l)
        Feature matrix with ``||phi_i - phi_j||^2 == weighted_hamming(i, j)``.
    """
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D, got ndim={E.ndim}")
    n, L = E.shape
    w = _resolve_weights(weights, L)

    if n == 0 or L == 0:
        shape = (n, 0)
        if sparse_output:
            return sparse.csr_matrix(shape, dtype=np.float64)
        return np.zeros(shape, dtype=np.float64)

    # Factorise each column to contiguous codes and offset into a global column
    # space.  Each (row, iteration) contributes exactly one non-zero, placed at
    # flat position ``i * L + l`` so that ``row_idx = repeat(arange(n), L)``.
    col_idx = np.empty(n * L, dtype=np.int64)
    data = np.empty(n * L, dtype=np.float64)
    offset = 0
    for l in range(L):
        _, inv = np.unique(E[:, l], return_inverse=True)
        inv = inv.ravel()
        col_idx[l::L] = inv + offset
        data[l::L] = np.sqrt(w[l] / 2.0)
        offset += int(inv.max()) + 1
    width = max(offset, 1)
    row_idx = np.repeat(np.arange(n, dtype=np.int64), L)

    M = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, width))
    return M if sparse_output else M.toarray()


def estimator_supports_sparse(clf):
    """Heuristic: does ``clf`` accept a sparse CSR feature matrix in fit?

    The centroid-based sklearn estimators we care about (KMeans, MiniBatchKMeans,
    Birch) consume sparse input natively; most others (GaussianMixture,
    AgglomerativeClustering, SpectralClustering with default affinity) require a
    dense array, so we fall back to dense for them.
    """
    sparse_friendly = {"KMeans", "MiniBatchKMeans", "Birch"}
    return type(clf).__name__ in sparse_friendly
