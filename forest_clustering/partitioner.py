"""Random-partition engine: builds IterationSpecs and computes embeddings."""

import numpy as np
from dataclasses import dataclass, field
from joblib import Parallel, delayed


@dataclass
class BinSpec:
    col_idx: int
    type: str  # 'numerical' | 'categorical'
    edges: np.ndarray | None = None      # numerical: (K-1,) sorted cut-points
    cat_map: np.ndarray | None = None    # categorical: (n_unique,) → bin_id in [0, K-1]
    n_unique: int = 0
    K: int = 0                           # per-feature bin count (0 = use spec default)


@dataclass
class IterationSpec:
    bin_specs: list[BinSpec]
    K: int


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_col_stats(
    X: np.ndarray,
    feature_types: list[str],
    quantile_sample: int = 10_000,
    quantile_cuts: bool = False,
    cut_strategy: str = "uniform",
    kde_params: dict | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Compute per-column statistics needed to generate random cut-points."""
    if rng is None:
        rng = np.random.default_rng()
    n = X.shape[0]
    stats = []
    for i, ftype in enumerate(feature_types):
        col = X[:, i]
        if ftype == "numerical":
            finite = col[np.isfinite(col)]
            lo = float(finite.min()) if len(finite) else 0.0
            hi = float(finite.max()) if len(finite) else 1.0
            n_unique = len(np.unique(finite)) if len(finite) else 0
            q25 = float(np.percentile(finite, 25)) if len(finite) else 0.0
            q75 = float(np.percentile(finite, 75)) if len(finite) else 0.0
            std = float(np.std(finite)) if len(finite) else 0.0
            if (quantile_cuts or cut_strategy == "quantile") and len(finite) > 1:
                # Store a sorted empirical sample and draw random probabilities at
                # spec-construction time.  The previous implementation sampled raw
                # observed values and later re-sampled from those points; that is a
                # bootstrap of values, not random empirical quantiles, and it
                # over-represented duplicate-heavy / high-density regions.
                k = min(quantile_sample, len(finite))
                if k < len(finite):
                    q_sample = rng.choice(finite, size=k, replace=False)
                else:
                    q_sample = finite
                qpts = np.sort(np.asarray(q_sample, dtype=np.float64))
                stats.append({"type": "numerical", "quantile_pts": qpts, "min": lo, "max": hi, "n_unique": n_unique, "q25": q25, "q75": q75, "std": std})
            else:
                stats.append({"type": "numerical", "min": lo, "max": hi, "n_unique": n_unique, "q25": q25, "q75": q75, "std": std})
        else:
            # categoricals are label-encoded ints in [0, n_unique-1]; -1 = unknown
            valid = col[col >= 0].astype(np.int32)
            n_unique = int(valid.max()) + 1 if len(valid) else 1
            stats.append({"type": "categorical", "n_unique": n_unique})
    return stats


def build_iteration_specs(
    n_iterations: int,
    col_stats: list[dict],
    n_features_per_iter: int,
    n_bins: int,
    feature_weights: np.ndarray,
    rng: np.random.Generator,
    cut_strategy: str = "uniform",
    kde_params: dict | None = None,
    X: np.ndarray | None = None,
    adaptive_bins_map: dict[int, int] | None = None,
    correlation_groups: list[list[int]] | None = None,
    correlation_aware: bool = False,
) -> list[IterationSpec]:
    d = len(col_stats)
    probs = feature_weights / feature_weights.sum()
    n_sel = min(n_features_per_iter, d)
    specs = []

    for _ in range(n_iterations):
        # Feature selection: correlation-aware or standard
        if correlation_aware and correlation_groups is not None:
            from .correlation_aware import select_features_correlation_aware
            feat_idx = np.array(
                select_features_correlation_aware(
                    correlation_groups, feature_weights, n_sel, rng
                ),
                dtype=np.int64,
            )
        else:
            feat_idx = _weighted_choice_no_replace(probs, n_sel, rng)

        bin_specs = []
        max_K = 0
        for ci in feat_idx:
            s = col_stats[ci]
            # Per-feature bin count from adaptive map, or default
            K_eff = adaptive_bins_map.get(ci, n_bins) if adaptive_bins_map else n_bins
            max_K = max(max_K, K_eff)

            if s["type"] == "numerical":
                col_data = X[:, ci] if X is not None and cut_strategy == "kde_peaks" else None
                edges = _make_num_edges(
                    s, K_eff, rng, cut_strategy=cut_strategy, kde_params=kde_params, col_data=col_data
                )
                bin_specs.append(BinSpec(col_idx=ci, type="numerical", edges=edges, K=K_eff))
            else:
                cat_map = _make_cat_map(s["n_unique"], K_eff, rng)
                bin_specs.append(
                    BinSpec(
                        col_idx=ci, type="categorical", cat_map=cat_map, n_unique=s["n_unique"], K=K_eff
                    )
                )
        specs.append(IterationSpec(bin_specs=bin_specs, K=max_K if adaptive_bins_map else n_bins))

    return specs


def _weighted_choice_no_replace(probs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement via Gumbel-max trick."""
    if size <= 0:
        return np.array([], dtype=np.int64)
    log_p = np.log(np.clip(probs, 1e-300, None))
    keys = log_p + rng.gumbel(size=len(probs))
    return np.argpartition(keys, -size)[-size:]


def _make_num_edges(s: dict, K: int, rng: np.random.Generator, cut_strategy: str = "uniform", kde_params: dict | None = None, col_data: np.ndarray | None = None) -> np.ndarray:
    # Guard: K=0 is invalid — treat as K=1 (no edges needed, everything → single bin)
    K_eff = max(K, 1)
    n_edges = K_eff - 1
    if n_edges == 0:
        return np.array([], dtype=np.float64)
    if cut_strategy == "kde_peaks" and col_data is not None:
        from .kde_cuts import kde_peaks_cut_points
        cuts, _ = kde_peaks_cut_points(col_data, n_edges, rng, kde_params)
        return np.sort(cuts)
    if "quantile_pts" in s:
        pts = np.asarray(s["quantile_pts"], dtype=np.float64)
        if len(pts) == 0:
            return np.array([], dtype=np.float64)
        if pts[0] == pts[-1]:
            return np.full(n_edges, pts[0], dtype=np.float64)
        # True random empirical quantile cuts: draw probabilities uniformly on
        # (0, 1), then interpolate against the sorted empirical sample.
        probs = rng.uniform(0.0, 1.0, size=n_edges)
        chosen = np.quantile(pts, probs, method="linear")
        return np.sort(np.asarray(chosen, dtype=np.float64))
    lo, hi = s["min"], s["max"]
    if lo == hi:
        return np.full(n_edges, lo)
    return np.sort(rng.uniform(lo, hi, size=n_edges))


def _make_cat_map(n_unique: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign each category to a bin in [0, K-1]."""
    if n_unique == 0:
        return np.array([], dtype=np.int32)
    # Guard: K=0 would cause division by zero; treat as K=1 (all → bin 0)
    K_eff = max(K, 1)
    shuffled = rng.permutation(n_unique)
    cat_map = np.empty(n_unique, dtype=np.int32)
    for rank, orig in enumerate(shuffled):
        cat_map[orig] = rank % K_eff
    return cat_map


def apply_categorical_bin_cap(
    col_stats: list[dict],
    n_bins: int,
    min_bins: int,
    max_bins: int,
    n: int,
) -> dict[int, int]:
    """Compute effective bin count per feature, applying categorical auto-capping.

    For categorical features, returns ``clip(n_unique, min_bins, B_max)`` where
    ``B_max = min(max_bins, Sturges(n))``.  For numerical features, no entry is
    added to the returned dict so that the caller's default ``n_bins`` is used.

    Parameters
    ----------
    col_stats : list[dict]
        Per-column statistics from :func:`build_col_stats`.
    n_bins : int
        Fixed ``n_bins`` parameter (used for numerical features, not stored).
    min_bins : int
        Minimum allowed bins.
    max_bins : int
        Maximum allowed bins.
    n : int
        Number of samples (for Sturges rule).

    Returns
    -------
    dict[int, int]
        Mapping ``column_index -> effective_bin_count`` for categorical columns.
        May be empty ``{}`` if no categorical features exist.
    """
    if min_bins > max_bins:
        raise ValueError(f"min_bins ({min_bins}) > max_bins ({max_bins})")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    sturges_cap = int(np.ceil(np.log2(max(n, 2)) + 1))
    B_max = min(max_bins, sturges_cap)

    bins_map: dict[int, int] = {}
    for j, s in enumerate(col_stats):
        if s["type"] == "categorical":
            n_cat = s.get("n_unique")
            if n_cat is None:
                n_cat = s.get("n_categories")
            if n_cat is None:
                raise ValueError(
                    f"Categorical col_stats must have 'n_unique' or 'n_categories', got {s}"
                )
            n_bins_eff = max(min(n_cat, B_max), min_bins)
            bins_map[j] = n_bins_eff
        # numerical features: leave unset → caller uses default n_bins

    return bins_map


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embedding(
    X: np.ndarray,
    specs: list[IterationSpec],
    n_jobs: int = -1,
) -> np.ndarray:
    """Returns (n, L) int64 embedding matrix."""
    if n_jobs == 1 or len(specs) < 4:
        cols = [_cell_ids(X, spec) for spec in specs]
    else:
        cols = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_cell_ids)(X, spec) for spec in specs
        )
    return np.column_stack(cols)  # (n, L) int64


# 64-bit hash-combine constants (deterministic, platform-independent).
# Folding per-feature bins into a hash rather than a positional mixed-radix
# code avoids int64 overflow when n_features_per_iter * log2(K) > 63 bits
# (which silently collapsed distinct cells onto the same id and corrupted the
# Hamming distance).  The hash is a pure function of the per-feature bins, so
# it stays out-of-sample-consistent (the same bin pattern maps to the same id
# on training and new data), and it preserves the per-column equality relation
# that the Hamming distance depends on.
_HASH_INIT = np.uint64(1469598103934665603)   # FNV-1a 64-bit offset basis
_HASH_GOLDEN = np.uint64(0x9E3779B97F4A7C15)  # 2^64 / golden ratio
_HASH_S1 = np.uint64(6)
_HASH_S2 = np.uint64(2)
# Final ids are masked to 52 bits so they are < 2**53 and therefore exactly
# representable as float64.  This matters because the Hamming distance is
# computed with scipy.cdist, which casts the integer embedding to double;
# full-range int64 ids (magnitude ~2**62) would collide under that cast and
# silently zero out the distance matrix.  52 bits keeps collisions between
# distinct cells astronomically unlikely while staying float64-exact.
_HASH_MASK = np.uint64((1 << 52) - 1)


def _cell_ids(X: np.ndarray, spec: IterationSpec) -> np.ndarray:
    """Compute a collision-free cell ID for one iteration. Returns (n,) int64.

    Each sample's per-feature bin assignments are folded into a 64-bit hash
    (overflow-safe, order-sensitive, out-of-sample-consistent).  Two samples
    share an id iff they fall in the same cell for this iteration.

    NaN values in numerical columns are explicitly mapped to bin 0.
    Categorical columns with n_unique=0 map all values to bin 0.
    Features with K=0 are treated as having K=1 (single bin).
    """
    n = X.shape[0]
    cell = np.full(n, _HASH_INIT, dtype=np.uint64)

    for bs in spec.bin_specs:
        col = X[:, bs.col_idx]
        K_j = bs.K if bs.K > 0 else spec.K
        # Guard: K=0 is invalid — treat as single-bin
        if K_j == 0:
            K_j = 1
        if bs.type == "numerical":
            # Explicitly map NaN/inf to bin 0 (searchsorted places NaN at len(edges))
            finite_mask = np.isfinite(col)
            b = np.searchsorted(bs.edges, col, side="right").astype(np.int64)
            b = np.clip(b, 0, K_j - 1)
            b[~finite_mask] = 0  # NaN/inf → bin 0
        else:
            # Guard: n_unique=0 means no valid categories at all
            if bs.n_unique == 0:
                b = np.zeros(n, dtype=np.int64)
            else:
                col_int = col.astype(np.int64)
                valid = (col_int >= 0) & (col_int < bs.n_unique)
                mapped = bs.cat_map[np.clip(col_int, 0, bs.n_unique - 1)]
                b = np.where(valid, mapped, np.int64(K_j - 1)).astype(np.int64)
        # boost-style hash_combine in uint64 (wraparound is intentional mixing)
        b_u = b.astype(np.uint64)
        cell = cell ^ (b_u + _HASH_GOLDEN + (cell << _HASH_S1) + (cell >> _HASH_S2))

    # Mask to 52 bits (float64-exact) and return as int64.  Non-negative,
    # bijective on the masked range, and preserves the equality / Hamming
    # relation the rest of the library depends on.
    return (cell & _HASH_MASK).astype(np.int64)
