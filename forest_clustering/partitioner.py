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
            if quantile_cuts and len(finite) > 1:
                k = min(quantile_sample, len(finite))
                qpts = np.sort(rng.choice(finite, size=k, replace=False))
                stats.append({"type": "numerical", "quantile_pts": qpts, "min": lo, "max": hi})
            else:
                stats.append({"type": "numerical", "min": lo, "max": hi})
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
) -> list[IterationSpec]:
    d = len(col_stats)
    probs = feature_weights / feature_weights.sum()
    n_sel = min(n_features_per_iter, d)
    specs = []

    for _ in range(n_iterations):
        feat_idx = _weighted_choice_no_replace(probs, n_sel, rng)
        bin_specs = []
        for ci in feat_idx:
            s = col_stats[ci]
            if s["type"] == "numerical":
                edges = _make_num_edges(s, n_bins, rng)
                bin_specs.append(BinSpec(col_idx=ci, type="numerical", edges=edges))
            else:
                cat_map = _make_cat_map(s["n_unique"], n_bins, rng)
                bin_specs.append(
                    BinSpec(col_idx=ci, type="categorical", cat_map=cat_map, n_unique=s["n_unique"])
                )
        specs.append(IterationSpec(bin_specs=bin_specs, K=n_bins))

    return specs


def _weighted_choice_no_replace(probs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement via Gumbel-max trick."""
    log_p = np.log(np.clip(probs, 1e-300, None))
    keys = log_p + rng.gumbel(size=len(probs))
    return np.argpartition(keys, -size)[-size:]


def _make_num_edges(s: dict, K: int, rng: np.random.Generator) -> np.ndarray:
    n_edges = K - 1
    if n_edges == 0:
        return np.array([], dtype=np.float64)
    if "quantile_pts" in s:
        pts = s["quantile_pts"]
        if len(pts) <= n_edges:
            return pts
        chosen = rng.choice(pts, size=n_edges, replace=False)
        return np.sort(chosen)
    lo, hi = s["min"], s["max"]
    if lo == hi:
        return np.full(n_edges, lo)
    return np.sort(rng.uniform(lo, hi, size=n_edges))


def _make_cat_map(n_unique: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign each category to a bin in [0, K-1]."""
    shuffled = rng.permutation(n_unique)
    cat_map = np.empty(n_unique, dtype=np.int32)
    for rank, orig in enumerate(shuffled):
        cat_map[orig] = rank % K
    return cat_map


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


def _cell_ids(X: np.ndarray, spec: IterationSpec) -> np.ndarray:
    """Compute mixed-radix cell ID for one iteration. Returns (n,) int64."""
    n = X.shape[0]
    K = spec.K
    cell = np.zeros(n, dtype=np.int64)
    power = np.int64(1)

    for bs in spec.bin_specs:
        col = X[:, bs.col_idx]
        if bs.type == "numerical":
            b = np.searchsorted(bs.edges, col, side="right").astype(np.int64)
            b = np.clip(b, 0, K - 1)
        else:
            col_int = col.astype(np.int64)
            valid = (col_int >= 0) & (col_int < bs.n_unique)
            mapped = bs.cat_map[np.clip(col_int, 0, bs.n_unique - 1)]
            b = np.where(valid, mapped, np.int64(K - 1)).astype(np.int64)
        cell += b * power
        power *= K

    return cell
