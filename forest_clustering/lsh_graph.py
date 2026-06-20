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


def _triu_pairs_for_groups(order, sizes, starts):
    """All within-group ``i<j`` row-id pairs, fully vectorised.

    Group ``g`` occupies ``order[starts[g] : starts[g] + sizes[g]]``.  Pairs are
    enumerated via the closed-form inverse of the triangular numbering, so there
    is no Python loop over buckets.
    """
    sizes = sizes.astype(np.int64)
    ppg = sizes * (sizes - 1) // 2                 # pairs per group
    total = int(ppg.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    g = np.repeat(np.arange(len(sizes)), ppg)      # group of each pair
    base = np.repeat(np.cumsum(ppg) - ppg, ppg)
    p = np.arange(total) - base                    # 0-based linear index in group
    sg = sizes[g]
    s = sg.astype(np.float64)
    # i = largest int with i*(2s-1-i)/2 <= p  (quadratic inverse, float seed)
    i = np.floor(((2 * s - 1) - np.sqrt((2 * s - 1) ** 2 - 8 * p)) / 2).astype(np.int64)
    i = np.clip(i, 0, sg - 2)
    f = lambda ii: ii * (2 * sg - 1 - ii) // 2
    i[f(i + 1) <= p] += 1                           # nudge up over float error
    i[f(i) > p] -= 1                                # nudge down
    j = (p - f(i)) + i + 1
    off = starts[g]
    return order[off + i], order[off + j]


def _hamming_pairs(E, lo, hi, chunk=200_000):
    """Exact column-Hamming distance for row pairs ``(lo, hi)``, chunked."""
    out = np.empty(lo.shape[0], dtype=np.uint32)
    for s in range(0, lo.shape[0], chunk):
        e = min(s + chunk, lo.shape[0])
        out[s:e] = (E[lo[s:e]] != E[hi[s:e]]).sum(axis=1)
    return out


def _compact_codes(E):
    """Factorise each column to small contiguous codes for fast equality.

    Equality between rows is preserved, so column-Hamming is identical, but the
    compact dtype (uint8/uint16/uint32 vs int64 hashes) cuts the memory bandwidth
    of the pair-distance comparisons several-fold.
    """
    n, L = E.shape
    max_card = 1
    cols = []
    for l in range(L):
        _, inv = np.unique(E[:, l], return_inverse=True)
        cols.append(inv.ravel())
        max_card = max(max_card, int(inv.max()) + 1 if inv.size else 1)
    dtype = np.uint8 if max_card <= 256 else (np.uint16 if max_card <= 65536 else np.uint32)
    Ec = np.empty((n, L), dtype=dtype)
    for l in range(L):
        Ec[:, l] = cols[l]
    return Ec


def auto_band_size(E, k=15, max_bucket=500, target_load=None, min_coverage=0.7,
                   sample=4096, grid=(2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32),
                   random_state=None):
    """Pick the smallest band size whose collision load is bounded.

    For each candidate band size we estimate, on a random row sample, the mean
    number of same-bucket mates per row (the "load", which drives cost/memory)
    and the coverage (fraction of rows that land in a non-degenerate bucket).
    The smallest band size with ``load <= target_load`` and
    ``coverage >= min_coverage`` is returned — small bands maximise recall, the
    load cap keeps cost bounded, and the coverage floor rejects degenerate bands
    whose buckets are all skipped.

    ``target_load`` defaults to ``8 * k``.
    """
    E = np.asarray(E)
    n, L = E.shape
    if target_load is None:
        target_load = 8 * k
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    Es = E[idx]

    best, best_score = grid[-1], None
    scale = n / len(idx)            # bucket size grows ~linearly with sample size
    for bs in grid:
        if bs > L:
            continue
        loads, covs = [], []
        nb = max(1, min(L // bs, 8))
        for b in range(nb):
            band = np.ascontiguousarray(Es[:, b * bs:(b + 1) * bs])
            _, inv, cnt = np.unique(band, axis=0, return_inverse=True,
                                    return_counts=True)
            size = cnt[inv.ravel()].astype(np.float64) * scale   # est. full bucket size
            eff = np.where(size > max_bucket, 1.0, size)         # skipped -> 0 mates
            loads.append(float((eff - 1).mean()))
            covs.append(float((eff > 1).mean()))
        load, cov = float(np.mean(loads)), float(np.mean(covs))
        if cov >= min_coverage and load <= target_load:
            return bs
        score = (cov >= min_coverage, -abs(load - target_load))
        if best_score is None or score > best_score:
            best, best_score = bs, score
    return best


def lsh_banding_knn(
    E: np.ndarray,
    k: int = 15,
    band_size=6,
    n_bands: Optional[int] = None,
    max_bucket: int = 150,
    max_candidates_per_row: Optional[int] = None,
    random_state: Optional[int] = None,
) -> sparse.coo_matrix:
    """Sparse kNN graph from a K-ary cell-id embedding via LSH banding.

    The embedding columns are themselves LSH hashes (random-partition cell ids),
    so two samples that fall in the *same cell across all ``band_size`` columns of
    at least one band* are candidate neighbours.  Exact column-Hamming distance is
    computed only for candidate pairs, and the ``k`` nearest are kept per row.

    The implementation is fully vectorised: bucket ids come from
    ``np.unique(..., axis=0)``, within-bucket pairs from a closed-form triangular
    inverse (:func:`_triu_pairs_for_groups`), distances in chunks, and per-row
    top-k from a single lexsort — there is no Python per-row or per-bucket loop.
    Memory is ``O(n * c)`` (``c`` = candidates per row), never ``O(n^2)``.

    Parameters
    ----------
    E : np.ndarray, shape (n, L)
        Integer cell-id embedding.
    k : int, default 15
        Neighbours kept per row.
    band_size : int or 'auto', default 6
        Columns per band ``r``.  ``'auto'`` calls :func:`auto_band_size` to pick
        the smallest band size with bounded collision load.  Smaller ``r`` ->
        more candidates / higher recall; low-entropy (categorical / few-bin)
        embeddings collide heavily and need a *larger* ``r``.
    n_bands : int or None
        Number of bands ``b``.  ``None`` uses ``L // band_size``.
    max_bucket : int, default 150
        Buckets larger than this are skipped (degenerate low-entropy bands).
    max_candidates_per_row : int or None
        Cap on candidate pairs incident to a row before distance computation;
        bounds memory/time to ``O(n * cap)``.  ``None`` -> ``max(64, 8 * k)``.
    random_state : int or None
        Permutes the column-to-band assignment (decorrelates bands) and seeds
        the auto band-size sample / candidate subsampling.

    Returns
    -------
    G : scipy.sparse.coo_matrix, shape (n, n)
        Directed kNN graph with column-Hamming distances — a drop-in replacement
        for :func:`batched_hamming_knn`.
    """
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D, got {E.ndim}D")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    n, L = E.shape
    k_eff = min(k, n - 1)
    dist_dtype = np.uint16 if L <= 65535 else np.uint32

    if n == 0:
        return sparse.coo_matrix((0, 0), dtype=dist_dtype)
    if k_eff == 0:
        return sparse.coo_matrix((n, n), dtype=dist_dtype)

    if band_size == "auto":
        band_size = auto_band_size(E, k=k_eff, max_bucket=max_bucket,
                                   random_state=random_state)
    if band_size < 1:
        raise ValueError(f"band_size must be >= 1 or 'auto', got {band_size}")

    if n_bands is None:
        n_bands = max(1, L // band_size)
    if max_candidates_per_row is None:
        max_candidates_per_row = max(64, 16 * k)

    rng_sub = np.random.default_rng(random_state)
    # Per-band pair cap so the running buffer stays within O(n * cap) memory.
    per_band_cap = max(n, n * max_candidates_per_row // max(1, n_bands))
    rng_sub = np.random.default_rng(random_state)

    # Factorise to compact codes once: preserves equality (hence Hamming) while
    # making bucketing and pair-distance comparisons several-fold cheaper.
    Ec = _compact_codes(E)

    col_order = np.arange(L)
    if random_state is not None:
        col_order = np.random.default_rng(random_state).permutation(L)

    # ── Candidate pairs: vectorised bucketing + triangular pair enumeration ──
    pair_lo, pair_hi = [], []
    for b in range(n_bands):
        band_cols = col_order[b * band_size:(b + 1) * band_size]
        if band_cols.size == 0:
            continue
        band = np.ascontiguousarray(Ec[:, band_cols])
        # Group rows by identical band tuple with a single sort over a void-typed
        # key (cheaper than unique(axis=0) followed by a second argsort).
        vkey = band.view([("", band.dtype)] * band.shape[1]).ravel()
        order = np.argsort(vkey, kind="stable")
        vs = vkey[order]
        bounds = np.flatnonzero(vs[1:] != vs[:-1]) + 1
        starts = np.concatenate(([0], bounds))
        sizes = np.concatenate((bounds, [n])) - starts
        keep = (sizes >= 2) & (sizes <= max_bucket)
        if not keep.any():
            continue
        a, c = _triu_pairs_for_groups(order, sizes[keep], starts[keep])
        if a.shape[0] > per_band_cap:
            s = rng_sub.integers(0, a.shape[0], size=per_band_cap)
            a, c = a[s], c[s]
        pair_lo.append(np.minimum(a, c))
        pair_hi.append(np.maximum(a, c))

    if not pair_lo:
        return sparse.coo_matrix((n, n), dtype=dist_dtype)

    lo = np.concatenate(pair_lo).astype(np.int64)
    hi = np.concatenate(pair_hi).astype(np.int64)
    dist = _hamming_pairs(Ec, lo, hi).astype(np.int64)

    # ── Per-row top-k via a single packed-key argsort (cheaper than a 3-key
    #    lexsort).  Pack (src, dist, dst) into one int64 so sorting orders by
    #    src, then distance, then dst; duplicate (src,dst) pairs from cross-band
    #    collisions become adjacent runs and are dropped — no global unique. ──
    src = np.concatenate([lo, hi])
    dst = np.concatenate([hi, lo])
    dd = np.concatenate([dist, dist])
    id_bits = max(1, int(np.ceil(np.log2(max(n, 2)))))
    d_bits = max(1, int(np.ceil(np.log2(L + 2))))
    if 2 * id_bits + d_bits <= 62:
        packed = (src << (d_bits + id_bits)) | (dd << id_bits) | dst
        o = np.argsort(packed, kind="stable")
    else:
        o = np.lexsort((dst, dd, src))   # very large n: fall back to lexsort
    src, dst, dd = src[o], dst[o], dd[o]
    # drop consecutive duplicate (src, dst)
    dup = np.zeros(src.shape[0], dtype=bool)
    np.logical_and(src[1:] == src[:-1], dst[1:] == dst[:-1], out=dup[1:])
    keep0 = ~dup
    src, dst, dd = src[keep0], dst[keep0], dd[keep0]
    # rank within each src group (already distance-sorted within group)
    first = np.empty(src.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(src[1:], src[:-1], out=first[1:])
    grp_start = np.flatnonzero(first)
    rank = np.arange(src.shape[0]) - np.repeat(grp_start, np.diff(
        np.concatenate((grp_start, [src.shape[0]]))))
    sel = rank < k_eff

    return sparse.coo_matrix(
        (dd[sel].astype(dist_dtype),
         (src[sel].astype(np.int32), dst[sel].astype(np.int32))),
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
