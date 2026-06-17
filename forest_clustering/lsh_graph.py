"""lsh_graph.py — Sparse kNN-Graph via Batched Hamming Distance.

Mathematical specification: see LSH_GRAPH_SPEC.md
"""

import numpy as np
from scipy import sparse
from typing import Optional

# ── Popcount LUT Fallback ──────────────────────────────────────────
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Count set bits in a uint64 array.

    Views each uint64 as 8 uint8s, uses a 256-entry lookup table,
    and sums across the 8 bytes.  Works for arrays of any shape.
    """
    flat = arr.ravel()
    chunk = 1_000_000
    total = np.empty(flat.shape, dtype=np.uint16)
    for i in range(0, len(flat), chunk):
        end = min(i + chunk, len(flat))
        chunk_flat = flat[i:end]
        chunk_u8 = chunk_flat.view(np.uint8).reshape(-1, 8)
        total[i:end] = _POPCOUNT_LUT[chunk_u8].sum(axis=1)
    return total.reshape(arr.shape)


def pack_bits(E: np.ndarray) -> np.ndarray:
    """Pack binary matrix E ∈ {0,1}^{n×m} into uint64 words.

    Parameters
    ----------
    E : ndarray of shape (n, m) with values in {0, 1}

    Returns
    -------
    E_packed : ndarray of shape (n, ceil(m/64)), dtype=uint64
    """
    n, m = E.shape
    words = (m + 63) // 64  # ceil(m/64)
    pad = words * 64 - m

    E_u8 = E.astype(np.uint8)
    if pad > 0:
        E_u8 = np.pad(E_u8, ((0, 0), (0, pad)), mode="constant")

    # packbits → (n, words*8) uint8  → view as (n, words) uint64
    return np.packbits(E_u8, axis=1).view(np.uint64).reshape(n, words)


def batched_hamming_knn(
    E: np.ndarray,
    k: int = 15,
    batch_size: int = 1000,
) -> sparse.coo_matrix:
    """Build sparse kNN graph using batched Hamming distance computation.

    Works for arbitrary integer "cell-id" embeddings (the forest-clustering
    embedding has values in ``[0, K-1]``, not just ``{0, 1}``).  The Hamming
    distance counts the number of *columns* (iterations) in which two samples
    fall in different cells.  For a genuinely binary embedding this reduces to
    the bit-Hamming distance and the fast popcount path is used.

    Parameters
    ----------
    E : np.ndarray, shape (n, m)
        Integer embedding matrix (cell ids).  Binary input uses a packed-bit
        popcount fast path; multi-valued input uses column comparison.
    k : int, default 15
        Number of nearest neighbors.
    batch_size : int, default 1000
        Batch size for distance computation.

    Returns
    -------
    G : scipy.sparse.coo_matrix, shape (n, n)
        Directed kNN graph.  G[i,j] = Hamming distance if j is among
        the k-nearest neighbours of i, else 0.
    """
    # ── Input validation ──────────────────────────────────────────
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D, got {E.ndim}D")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n, m = E.shape
    k_eff = min(k, n - 1)

    # Edge cases
    if n == 0:
        return sparse.coo_matrix((0, 0), dtype=np.uint16)
    if k_eff == 0:
        return sparse.coo_matrix((n, n), dtype=np.uint16)

    is_binary = bool(np.all((E == 0) | (E == 1)))

    # ── Bit packing (binary fast path only) ───────────────────────
    if is_binary:
        E_packed = pack_bits(E)  # (n, words) uint64
        words = E_packed.shape[1]
    else:
        E_packed = None
        words = 0

    # ── Batched distance computation ──────────────────────────────
    rows, cols, data = [], [], []

    # Determine distance dtype (Hamming distance ≤ m)
    dist_dtype = np.uint16 if m <= 65535 else np.uint32

    num_batches = (n + batch_size - 1) // batch_size

    for s in range(num_batches):
        i_start = s * batch_size
        i_end = min(i_start + batch_size, n)
        b_s = i_end - i_start

        if is_binary:
            batch = E_packed[i_start:i_end]  # (b_s, words) uint64
            xor_result = np.bitwise_xor(batch[:, None, :], E_packed[None, :, :])
            if words == 1:
                D_batch = _popcount_u64(xor_result[:, :, 0]).astype(dist_dtype)
            else:
                D_batch = (
                    _popcount_u64(xor_result)
                    .reshape(b_s, n, words)
                    .sum(axis=2)
                    .astype(dist_dtype)
                )
        else:
            # Column Hamming: number of differing iterations.
            batch = E[i_start:i_end]  # (b_s, m)
            D_batch = (batch[:, None, :] != E[None, :, :]).sum(axis=2).astype(dist_dtype)

        # Process each row in the batch
        for j in range(b_s):
            idx = i_start + j
            dists = D_batch[j].copy()
            dists[idx] = np.iinfo(dist_dtype).max  # exclude self

            # Find k_eff smallest
            top_k_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
            top_k_dist = dists[top_k_idx]

            # Deterministic sort (stable)
            sort_order = np.argsort(top_k_dist, kind="mergesort")
            top_k_idx = top_k_idx[sort_order]
            top_k_dist = top_k_dist[sort_order]

            rows.extend([idx] * k_eff)
            cols.extend(top_k_idx.tolist())
            data.extend(top_k_dist.tolist())

    return sparse.coo_matrix(
        (
            np.array(data, dtype=dist_dtype),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n, n),
    )


def symmetrize_knn(G: sparse.coo_matrix) -> sparse.coo_matrix:
    """Symmetrize directed kNN graph via element-wise maximum.

    A_bar[i,j] = max(A[i,j], A[j,i])

    Returns an undirected graph with A_bar = A_bar^T.
    """
    if G.nnz == 0:
        return G.copy()
    G_sym = G.maximum(G.T)
    return G_sym.tocoo()


def try_faiss_binary_knn(E: np.ndarray, k: int) -> Optional[sparse.coo_matrix]:
    """Try using FAISS for binary kNN.  Return None if faiss not available.

    Parameters
    ----------
    E : np.ndarray, shape (n, m), uint8
        Binary embedding.
    k : int
        Number of neighbors.

    Returns
    -------
    G : coo_matrix or None
        kNN graph if faiss is available, else None.
    """
    try:
        import faiss
    except ImportError:
        return None

    n, m = E.shape
    k_eff = min(k, n - 1)
    if k_eff == 0:
        return sparse.coo_matrix((n, n), dtype=np.uint16)

    # FAISS binary index uses bytes
    E_faiss = E.astype(np.uint8).copy()
    index = faiss.IndexBinaryFlat(m * 8)
    index.add(E_faiss)

    # Search k_eff + 1 to account for self
    D, I = index.search(E_faiss, min(k_eff + 1, n))

    # Remove self (first column — self has distance 0)
    D = D[:, 1 : k_eff + 1]
    I = I[:, 1 : k_eff + 1]

    rows = np.repeat(np.arange(n, dtype=np.int32), k_eff)
    cols = I.ravel().astype(np.int32)
    data = D.ravel()

    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n))


def build_sparse_knn_graph(
    E: np.ndarray,
    k: int = 15,
    batch_size: int = 1000,
    symmetrize: bool = True,
) -> sparse.coo_matrix:
    """Build sparse kNN graph from an integer embedding with automatic fallback.

    Strategy
    --------
    1. For a genuinely binary embedding, try FAISS binary kNN (fastest).
    2. Otherwise (or if FAISS is unavailable), use the batched Hamming numpy
       implementation, which counts differing iterations and therefore works
       for arbitrary integer cell-id embeddings (values in ``[0, K-1]``).

    Parameters
    ----------
    E : np.ndarray, shape (n, m)
        Integer embedding matrix (cell ids).
    k : int, default 15
        Number of nearest neighbors.
    batch_size : int, default 1000
        Batch size for numpy fallback.
    symmetrize : bool, default True
        Whether to symmetrize the graph.

    Returns
    -------
    G : scipy.sparse.coo_matrix, shape (n, n)
        kNN graph (undirected if symmetrize=True).
    """
    E = np.asarray(E)
    is_binary = bool(np.all((E == 0) | (E == 1)))

    G = None
    if is_binary:
        # uint8 bit-packing / FAISS is only valid for {0,1} embeddings.
        G = try_faiss_binary_knn(E.astype(np.uint8), k)
    if G is None:
        # Pass the raw integer embedding; batched_hamming_knn handles K-ary.
        G = batched_hamming_knn(E, k, batch_size)

    if symmetrize:
        G = symmetrize_knn(G)

    return G
