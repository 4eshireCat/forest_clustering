"""Correlation-aware feature grouping and selection."""
import numpy as np


def build_correlation_groups(feature_weights, corr_matrix, threshold=0.7):
    """Group correlated features using graph connected components.

    Parameters
    ----------
    feature_weights : ndarray of shape (d,)
        Feature weights (used for validation only).
    corr_matrix : ndarray of shape (d, d)
        Correlation matrix.
    threshold : float
        Group features with |corr| > threshold.

    Returns
    -------
    groups : list of list of int
        Feature index groups.
    """
    d = len(feature_weights)
    if corr_matrix.shape != (d, d):
        raise ValueError(f"corr_matrix shape {corr_matrix.shape} != ({d}, {d})")
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    # Build adjacency: edge if |corr| > threshold (excluding diagonal)
    adj = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj, False)

    # Find connected components via BFS/DFS
    visited = [False] * d
    groups = []

    for start in range(d):
        if visited[start]:
            continue
        # BFS
        group = []
        queue = [int(start)]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            group.append(int(node))
            neighbors = np.where(adj[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
        groups.append(sorted(group))

    return groups


def select_features_correlation_aware(groups, feature_weights, n_select, rng):
    """Select n_select features, at most one per group.

    Strategy:
    1. Compute group importance = max weight in group
    2. Select n_select groups (weighted by importance, without replacement)
    3. From each selected group, pick the highest-weight feature

    If n_select > n_groups: fill remaining with highest-weight features
    from any group (may pick second from a group as fallback).

    Returns
    -------
    selected : list of int
        Feature indices.
    """
    if n_select <= 0:
        return []

    # Filter out empty groups for selection
    groups = [g for g in groups if len(g) > 0]

    # Group importance = max weight
    group_importance = np.array(
        [max((feature_weights[j] for j in g), default=0.0) for g in groups]
    )

    n_groups = len(groups)
    if n_select >= n_groups:
        # Select all groups, then fill remaining
        selected_groups = list(range(n_groups))
        n_remaining = n_select - n_groups
    else:
        # Weighted random selection of groups without replacement
        group_imp_sum = group_importance.sum()
        if group_imp_sum < 1e-15:
            # All weights zero → uniform selection
            probs = np.ones(n_groups) / n_groups
        else:
            probs = group_importance / group_imp_sum
        selected_groups = sorted(
            rng.choice(n_groups, size=n_select, replace=False, p=probs)
        )
        n_remaining = 0

    # Pick highest-weight feature from each selected group
    selected = []
    used_features = set()
    for gi in selected_groups:
        best = max(groups[gi], key=lambda j: feature_weights[j])
        selected.append(best)
        used_features.add(best)

    # Fill remaining if needed
    if n_remaining > 0:
        remaining = sorted(
            [j for j in range(len(feature_weights)) if j not in used_features],
            key=lambda j: feature_weights[j],
            reverse=True,
        )
        if len(remaining) >= n_remaining:
            selected.extend(remaining[:n_remaining])
        else:
            # All features already used — fall back to random selection
            # from all features to reach the requested count
            all_features = list(range(len(feature_weights)))
            n_extra = n_remaining - len(remaining)
            selected.extend(remaining)
            # Sample with replacement if we need more than total features
            if n_extra > 0:
                extra = rng.choice(
                    all_features, size=n_extra, replace=True
                ).tolist()
                selected.extend(extra)

    return selected
