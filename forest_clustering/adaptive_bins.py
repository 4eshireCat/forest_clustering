"""Adaptive bin count computation per feature."""
import numpy as np


def compute_adaptive_bins(col_stats, n, min_bins=2, max_bins=10):
    """Compute optimal n_bins for each feature.

    Parameters
    ----------
    col_stats : list of dict
        Column statistics from build_col_stats.
    n : int
        Total number of samples.
    min_bins : int
        Minimum number of bins (default 2).
    max_bins : int
        Maximum number of bins (default 10).

    Returns
    -------
    bins_map : dict[int, int]
        Mapping col_idx -> n_bins.
    """
    if min_bins > max_bins:
        raise ValueError(f"min_bins ({min_bins}) > max_bins ({max_bins})")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    sturges_cap = int(np.ceil(np.log2(max(n, 2)) + 1))
    effective_max = min(max_bins, sturges_cap)

    bins_map = {}
    for j, s in enumerate(col_stats):
        if s["type"] == "categorical":
            n_cat = s.get("n_categories")
            if n_cat is None:
                n_uni = s.get("n_unique")
                if n_uni is None:
                    raise ValueError(
                        f"Categorical col_stats must have 'n_categories' or 'n_unique', got {s}"
                    )
                n_cat = n_uni
            bins_map[j] = max(min(n_cat, effective_max), min_bins)
            continue

        # Numerical feature
        n_unique = s.get("n_unique", n)
        sigma = s.get("std", 0.0)
        if "min" not in s or "max" not in s:
            raise ValueError(
                f"Numerical col_stats must have 'min' and 'max' keys, got keys: {list(s.keys())}"
            )
        range_val = s["max"] - s["min"]

        # Discrete short-circuit: few unique values -> exact match
        # Note: this creates a boundary where features with n_unique just below
        # discrete_cap get exact-match bins, while those just above use the
        # continuous scoring formula — this is intentional for stability.
        discrete_cap = int(np.sqrt(n)) + 1
        if n_unique <= discrete_cap and n_unique >= 2:
            bins_map[j] = max(min(n_unique, effective_max), min_bins)
            continue

        # Estimate std from range if not available
        if sigma < 1e-12:
            sigma = range_val / 4.0

        # Constant feature
        if range_val < 1e-12:
            bins_map[j] = min_bins
            continue

        # Component 1: spread (IQR-based, fallback to sigma)
        q25 = s.get("q25")
        q75 = s.get("q75")
        if q25 is not None and q75 is not None and q75 > q25 and range_val > 1e-12:
            iqr = q75 - q25
            c_spread = min(iqr / range_val * 2.0, 1.0)
        elif sigma > 0 and range_val > 1e-12:
            c_spread = min(sigma / range_val * 2.0, 1.0)  # fallback to sigma
        else:
            c_spread = 0.0

        # Component 2: unique-value ratio
        c_unique = min(n_unique / (np.sqrt(n) * 2), 1.0)

        # Composite score (robust to zero-mean — avoids c_cv blowup)
        C = 0.5 * c_spread + 0.5 * c_unique
        C = np.clip(C, 0.0, 1.0)

        # Map to bins
        n_bins = int(round(min_bins + (effective_max - min_bins) * C))
        n_bins = max(min_bins, min(n_bins, effective_max))

        bins_map[j] = n_bins

    return bins_map
