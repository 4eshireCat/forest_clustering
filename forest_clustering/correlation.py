import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_feature_weights(
    X: np.ndarray,
    threshold: float = 0.7,
    sample_size: int = 10_000,
    rng: np.random.Generator | None = None,
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

    # Sample rows
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_s = X[idx].astype(np.float64)

    # Drop columns that are constant in the sample (Spearman undefined)
    stds = X_s.std(axis=0)
    variable_mask = stds > 0
    if variable_mask.sum() < 2:
        return np.ones(d)

    X_var = X_s[:, variable_mask]
    var_indices = np.where(variable_mask)[0]

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
