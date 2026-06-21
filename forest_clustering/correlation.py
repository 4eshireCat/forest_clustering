import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_feature_weights(
    X: np.ndarray,
    threshold: float = 0.7,
    sample_size: int = 10_000,
    rng: np.random.Generator | None = None,
    feature_types: list[str] | None = None,
) -> np.ndarray:
    """Return per-feature weight array of shape (d,).

    Features belonging to a strongly-correlated group of size G get weight 1/G.
    Correlation is estimated on a random sample via Spearman rank correlation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    if d < 2:
        return np.ones(d)

    if feature_types is None:
        candidate_mask = np.ones(d, dtype=bool)
    else:
        if len(feature_types) != d:
            raise ValueError(f"feature_types length {len(feature_types)} != number of columns {d}")
        # Spearman on label-encoded categoricals is not mathematically meaningful:
        # category codes are arbitrary ordinals.  Correlation weights therefore
        # only use numerical columns unless a future categorical association
        # measure is added.
        candidate_mask = np.array([ft != "categorical" for ft in feature_types], dtype=bool)

    if candidate_mask.sum() < 2:
        return np.ones(d)

    # Sample rows
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_s_all = X[idx].astype(np.float64)
    X_s = X_s_all[:, candidate_mask]
    candidate_indices = np.where(candidate_mask)[0]

    # Drop columns that are constant in the sample (Spearman undefined)
    stds = X_s.std(axis=0)
    variable_mask_local = stds > 0
    if variable_mask_local.sum() < 2:
        return np.ones(d)

    X_var = X_s[:, variable_mask_local]
    var_indices = candidate_indices[variable_mask_local]

    # Spearman correlation matrix on variable columns
    result = stats.spearmanr(X_var, nan_policy="omit")
    if X_var.shape[1] == 2:
        corr_val = float(result.statistic) if np.isscalar(result.statistic) else float(result.statistic[0, 1])
        corr = np.array([[1.0, corr_val], [corr_val, 1.0]])
    else:
        corr = np.array(result.statistic)

    corr = np.abs(np.nan_to_num(corr))

    # Adjacency: connected if |corr| > threshold (no self-loops)
    adj = (corr > threshold).astype(np.uint8)
    np.fill_diagonal(adj, 0)

    # Connected components on variable features
    _, labels = connected_components(csr_matrix(adj), directed=False)

    weights = np.ones(d)
    for comp_id in np.unique(labels):
        members_local = np.where(labels == comp_id)[0]
        if len(members_local) < 2:
            continue
        members_global = var_indices[members_local]
        weights[members_global] = 1.0 / len(members_local)

    return weights
