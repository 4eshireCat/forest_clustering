"""Permutation feature importance for forest-clustering."""
import numpy as np
import pandas as pd


def compute_permutation_importance(clusterer, X, metric='silhouette',
                                     n_repeats=5, random_state=None):
    """Compute permutation importance for each feature.

    For each feature j:
        IMP_j = mean(Q_base - Q_perm) over n_repeats

    Parameters
    ----------
    clusterer : ForestClusterer
        Fitted clusterer.
    X : DataFrame or ndarray
        Input data (same as used for fit).
    metric : str
        'silhouette' (default).
    n_repeats : int
        Number of permutation repeats per feature.
    random_state : int or None

    Returns
    -------
    importance : pd.DataFrame
        Columns: feature, importance, raw_importance, std, raw_std
    """
    from .partitioner import compute_embedding
    from .distance import pairwise_hamming, pairwise_hamming_chunked
    from .weighted_distance import pairwise_weighted_hamming

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    rng = np.random.default_rng(random_state)

    # Convert X to DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    feature_names = list(X.columns)
    n_features = len(feature_names)

    # Check clusterer is fitted
    if getattr(clusterer, 'labels_', None) is None:
        raise ValueError("Clusterer has not been fitted yet. Call fit() before computing permutation importance.")

    # Baseline: compute score on original data with original labels
    labels = clusterer.labels_
    D = clusterer.pairwise_distance()

    if metric == 'silhouette':
        # For silhouette, we need at least 2 clusters
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            # Return NaN if only one cluster
            return pd.DataFrame({
                'feature': feature_names,
                'importance': np.full(n_features, np.nan),
                'raw_importance': np.full(n_features, np.nan),
                'std': np.full(n_features, np.nan),
                'raw_std': np.full(n_features, np.nan),
            })
        baseline_score = clusterer._safe_silhouette_score(D, labels, metric='precomputed')
        if np.isnan(baseline_score):
            baseline_score = 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute importance for each feature
    importances = np.zeros(n_features)
    stds = np.zeros(n_features)

    for j, col_name in enumerate(feature_names):
        deltas = np.zeros(n_repeats)
        for r in range(n_repeats):
            # Permute column j
            X_perm = X.copy()
            col_values = np.asarray(X_perm.iloc[:, j]).copy()
            rng.shuffle(col_values)
            X_perm.iloc[:, j] = col_values

            # Transform permuted data using original encoder and specs
            X_perm_enc = clusterer.encoder_.transform(X_perm)
            E_perm = compute_embedding(X_perm_enc, clusterer.specs_, n_jobs=clusterer.n_jobs)

            # Compute distance matrix on permuted embedding
            n = E_perm.shape[0]

            # Use weighted Hamming when weights differ from uniform
            use_weighted = (
                hasattr(clusterer, "iteration_weights_")
                and not np.allclose(clusterer.iteration_weights_, 1.0)
            )
            if use_weighted:
                weights = clusterer.iteration_weights_
                if weights.sum() < 1e-15:
                    weights = np.ones(clusterer.n_iterations, dtype=np.float64)
                D_perm = pairwise_weighted_hamming(E_perm, weights)
            else:
                chunk_size = 2_000
                if n <= chunk_size:
                    D_perm = pairwise_hamming(E_perm)
                else:
                    D_perm = pairwise_hamming_chunked(E_perm, chunk_size=chunk_size)

            # Compute score using ORIGINAL labels
            perm_score = clusterer._safe_silhouette_score(D_perm, labels, metric='precomputed')
            if np.isnan(perm_score):
                perm_score = 0.0

            deltas[r] = baseline_score - perm_score

        importances[j] = np.mean(deltas)
        stds[j] = np.std(deltas, ddof=1) if n_repeats > 1 else 0.0

    # Normalize: clip to [0, max], then divide by max
    raw_importances = importances.copy()
    raw_stds = stds.copy()
    importances = np.clip(importances, 0, None)
    max_imp = importances.max()
    if max_imp > 1e-15:
        importances_norm = importances / max_imp
    else:
        importances_norm = np.zeros(n_features)

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances_norm,
        'raw_importance': raw_importances,
        'std': stds / max_imp if max_imp > 1e-15 else stds,
        'raw_std': raw_stds,
    })
