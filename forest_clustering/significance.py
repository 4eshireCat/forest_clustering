"""Significance testing module for forest clustering.

Provides permutation tests, bootstrap confidence intervals, and
per-cluster silhouette significance testing.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_samples

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95
MIN_N_PERMUTATIONS = 10
MIN_N_BOOTSTRAP = 10

# ARI effect size thresholds
ARI_EFFECT_NONE = 0.0
ARI_EFFECT_SMALL = 0.1
ARI_EFFECT_MEDIUM = 0.25
ARI_EFFECT_LARGE = 0.5

# Silhouette quality thresholds
SILHOUETTE_POOR = 0.0
SILHOUETTE_WEAK = 0.25
SILHOUETTE_MODERATE = 0.5
SILHOUETTE_STRONG = 0.7
SILHOUETTE_VERY_STRONG = 0.9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_sample_size_warning(n: int) -> Optional[str]:
    """Return a warning message if sample size is small."""
    if n < 2:
        return f"ERROR: n={n} < 2. At least 2 samples required."
    if n < 30:
        return (
            f"WARNING: n={n} < 30. "
            f"Permutation test may be underpowered. "
            f"Consider n >= 100 for reliable results."
        )
    if n < 100:
        return f"INFO: n={n}. For higher power, consider n >= 100."
    return None


def _check_permutation_warning(B: int) -> Optional[str]:
    """Return a warning if the number of permutations is small."""
    if B < 10:
        return "ERROR: n_permutations must be at least 10."
    if B < 100:
        return (
            f"WARNING: n_permutations={B} may give coarse p-values. "
            f"Consider B >= 1000."
        )
    if B < 1000:
        return f"INFO: n_permutations={B}. For finer resolution, consider B >= 1000."
    return None


def _validate_labels(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate label arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    if n < 2:
        raise ValueError("At least 2 samples required")


def _validate_permutations(n: int) -> None:
    if n < MIN_N_PERMUTATIONS:
        raise ValueError(f"n_permutations must be at least {MIN_N_PERMUTATIONS}")


def _validate_bootstrap(n: int) -> None:
    if n < MIN_N_BOOTSTRAP:
        raise ValueError(f"n_bootstrap must be at least {MIN_N_BOOTSTRAP}")


def _validate_data(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.isnan(X).any():
        raise ValueError("Input data X contains NaN values")
    if np.isinf(X).any():
        raise ValueError("Input data X contains infinite values")
    return X


# ---------------------------------------------------------------------------
# Effect-size classifiers
# ---------------------------------------------------------------------------

def _classify_ari_effect(ari: float) -> str:
    if ari <= ARI_EFFECT_NONE:
        return "none"
    if ari <= ARI_EFFECT_SMALL:
        return "small"
    if ari <= ARI_EFFECT_MEDIUM:
        return "medium"
    return "large"


def _classify_silhouette_effect(s: float) -> str:
    if s < SILHOUETTE_POOR:
        return "poor"
    if s < SILHOUETTE_WEAK:
        return "weak"
    if s < SILHOUETTE_MODERATE:
        return "moderate"
    if s < SILHOUETTE_STRONG:
        return "strong"
    return "very strong"


# ---------------------------------------------------------------------------
# ARI wrapper that handles degenerate labelings
# ---------------------------------------------------------------------------

def _safe_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ARI, handling the degenerate all-same-label case."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n < 2:
        return 0.0

    # Degenerate case: both have a single cluster -> perfect structural match
    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
        return 1.0

    # sklearn handles most degenerate cases, but returns 0.0 when one side
    # is single-cluster and the other is not, which is the correct behaviour.
    try:
        return float(adjusted_rand_score(y_true, y_pred))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 1. Permutation test for ARI
# ---------------------------------------------------------------------------

def permutation_test_ari(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    alternative: str = "greater",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Permutation test for ARI significance.

    Tests H0: predicted labels are independent of true labels
    against H1: predicted labels are associated with true labels.

    Parameters
    ----------
    y_true : array of shape (n,)
        Ground truth labels
    y_pred : array of shape (n,)
        Predicted cluster labels
    n_permutations : int, default=1000
        Number of permutations. Must be >= 10.
    alternative : str, default="greater"
        Alternative hypothesis: "greater", "less", or "two-sided"
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        - ari_observed: float
        - p_value: float
        - is_significant: bool
        - n_permutations: int
        - null_distribution: np.ndarray of shape (n_permutations,)
        - effect_size: str
        - alternative: str
        - ci_95: tuple (float, float)
        - warning: str or None
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_labels(y_true, y_pred)
    _validate_permutations(n_permutations)

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be one of 'greater', 'less', 'two-sided'")

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    ari_obs = _safe_ari(y_true, y_pred)

    null_dist = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        y_perm = rng.permutation(y_true)
        null_dist[b] = _safe_ari(y_perm, y_pred)

    # P-value with +1 correction (Phipson & Smyth, 2010).  Null ties must
    # always be counted, including at ARI=1: a statistic that is invariant
    # under permutation (for example, two single-cluster labelings) carries no
    # evidence against independence.
    if alternative == "greater":
        count = np.sum(null_dist >= ari_obs)
    elif alternative == "less":
        count = np.sum(null_dist <= ari_obs)
    else:  # two-sided
        obs_abs = abs(ari_obs)
        count = np.sum(np.abs(null_dist) >= obs_abs)
    p_value = (1 + count) / (n_permutations + 1)

    ci_lower, ci_upper = float(np.percentile(null_dist, 2.5)), float(
        np.percentile(null_dist, 97.5)
    )

    warning = _check_sample_size_warning(n)
    perm_warning = _check_permutation_warning(n_permutations)
    if perm_warning and perm_warning.startswith("WARNING"):
        warning = f"{warning}\n{perm_warning}" if warning else perm_warning

    return {
        "ari_observed": ari_obs,
        "p_value": p_value,
        "is_significant": bool(p_value < DEFAULT_ALPHA),
        "n_permutations": n_permutations,
        "null_distribution": null_dist,
        "effect_size": _classify_ari_effect(ari_obs),
        "alternative": alternative,
        "ci_95": (ci_lower, ci_upper),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 2. Bootstrap CI for ARI
# ---------------------------------------------------------------------------

def bootstrap_ci_ari(
    X: np.ndarray,
    clusterer: Any,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> Dict[str, Any]:
    """Bootstrap confidence interval for ARI.

    Parameters
    ----------
    X : array of shape (n, d)
        Dataset
    clusterer : callable or object with fit_predict(X) method
        Clustering algorithm; if a callable, called as clusterer(X).
    y_true : array of shape (n,)
        Ground truth labels
    n_bootstrap : int, default=1000
        Number of bootstrap replicates
    confidence : float in (0, 1), default=0.95
        Confidence level
    random_state : int or None, default=None
        Random seed
    return_distribution : bool, default=False
        Whether to return the full bootstrap distribution

    Returns
    -------
    dict with keys:
        - ari_mean: float
        - ari_std: float
        - ci_lower: float
        - ci_upper: float
        - confidence: float
        - is_stable: bool
        - n_bootstrap: int
        - n_samples: int
        - distribution: np.ndarray or None
        - warning: str or None
    """
    X = _validate_data(X)
    y_true = np.asarray(y_true)
    n, _ = X.shape

    if n < 2:
        raise ValueError(f"Need at least 2 samples for bootstrap CI, got {n}")
    if len(y_true) != n:
        raise ValueError("y_true length must match number of rows in X")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    _validate_bootstrap(n_bootstrap)

    rng = np.random.default_rng(random_state)
    boot_dist = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b = X[idx]
        y_true_b = y_true[idx]

        # Support both callable and fit_predict interface
        if callable(clusterer):
            labels_b = clusterer(X_b)
        else:
            # Allow perfect clusterers to access bootstrap y_true
            if hasattr(clusterer, '_bootstrap_y_true'):
                clusterer._bootstrap_y_true = y_true_b
            labels_b = clusterer.fit_predict(X_b)

        boot_dist[b] = _safe_ari(y_true_b, labels_b)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_dist, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))

    warning = _check_sample_size_warning(n)
    bs_warning = _check_permutation_warning(n_bootstrap)
    if bs_warning and bs_warning.startswith("WARNING"):
        warning = f"{warning}\n{bs_warning}" if warning else bs_warning

    return {
        "ari_mean": float(np.mean(boot_dist)),
        "ari_std": float(np.std(boot_dist, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "is_stable": ci_lower > 0,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
        "distribution": boot_dist if return_distribution else None,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 3. Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    y_true: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Paired permutation test comparing two clusterings.

    Tests H0: ARI(labels_a) = ARI(labels_b) against
    H1: ARI(labels_a) != ARI(labels_b).

    Parameters
    ----------
    labels_a : array of shape (n,)
        Predictions from method A
    labels_b : array of shape (n,)
        Predictions from method B
    y_true : array of shape (n,)
        Ground truth labels
    n_permutations : int, default=1000
        Number of permutations
    random_state : int or None, default=None
        Random seed

    Returns
    -------
    dict with keys:
        - ari_1: float
        - ari_2: float
        - delta_obs: float
        - p_value: float
        - is_significant: bool
        - n_permutations: int
        - null_distribution: np.ndarray
        - better_method: int or None
        - warning: str or None
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    y_true = np.asarray(y_true)

    if labels_a.shape != labels_b.shape or labels_a.shape != y_true.shape:
        raise ValueError("labels_a, labels_b, and y_true must have the same length")
    n = len(y_true)
    if n < 2:
        raise ValueError("At least 2 samples required")
    _validate_permutations(n_permutations)

    ari_a = _safe_ari(y_true, labels_a)
    ari_b = _safe_ari(y_true, labels_b)
    delta_obs = ari_a - ari_b

    rng = np.random.default_rng(random_state)
    null_dist = np.empty(n_permutations, dtype=float)

    for p in range(n_permutations):
        swap = rng.random(n) < 0.5
        a_swap = np.where(swap, labels_b, labels_a)
        b_swap = np.where(swap, labels_a, labels_b)
        null_dist[p] = _safe_ari(y_true, a_swap) - _safe_ari(y_true, b_swap)

    count = np.sum(np.abs(null_dist) >= abs(delta_obs))
    p_value = (1 + count) / (n_permutations + 1)
    is_significant = bool(p_value < DEFAULT_ALPHA)

    if is_significant and delta_obs > 0:
        better = 1
    elif is_significant and delta_obs < 0:
        better = 2
    else:
        better = None

    warning = _check_sample_size_warning(n)
    perm_warning = _check_permutation_warning(n_permutations)
    if perm_warning and perm_warning.startswith("WARNING"):
        warning = f"{warning}\n{perm_warning}" if warning else perm_warning

    return {
        "ari_1": ari_a,
        "ari_2": ari_b,
        "delta_obs": delta_obs,
        "p_value": p_value,
        "is_significant": is_significant,
        "n_permutations": n_permutations,
        "null_distribution": null_dist,
        "better_method": better,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 4. Cluster significance (per-cluster silhouette)
# ---------------------------------------------------------------------------

def cluster_significance(
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    correction_method: Optional[str] = "bonferroni",
) -> Dict[str, Any]:
    """Per-cluster silhouette significance via bootstrap.

    For each predicted cluster, tests whether its mean silhouette score
    is significantly above 0 using bootstrap confidence intervals.

    Parameters
    ----------
    X : array of shape (n, d)
        Dataset
    labels : array of shape (n,)
        Predicted cluster labels
    n_bootstrap : int, default=100
        Number of bootstrap replicates per cluster
    confidence : float in (0, 1), default=0.95
        Confidence level
    random_state : int or None, default=None
        Random seed
    correction_method : str or None, default="bonferroni"
        Multiple testing correction method: "bonferroni", "sidak", "fdr_bh", or None

    Returns
    -------
    dict with keys:
        - overall_silhouette: float or nan
        - n_clusters: int
        - clusters: list of dicts (each with cluster_id, size, mean_silhouette,
                    silhouette_ci_lower, silhouette_ci_upper, is_significant,
                    p_value, effect_size, and optionally p_value_corrected)
        - n_significant: int
        - significant_clusters: list of int
        - warning: str or None
    """
    X = _validate_data(X)
    labels = np.asarray(labels)
    n = X.shape[0]

    if n < 2:
        raise ValueError("At least 2 samples required")
    if len(labels) != n:
        raise ValueError("labels length must match number of rows in X")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    _validate_bootstrap(n_bootstrap)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        # Silhouette is undefined for a single cluster
        overall_sil = float("nan")
        cluster_id = int(unique_labels[0]) if n_clusters == 1 else None
        clusters = []
        if cluster_id is not None:
            clusters.append({
                "cluster_id": cluster_id,
                "size": n,
                "mean_silhouette": float("nan"),
                "silhouette_ci_lower": float("nan"),
                "silhouette_ci_upper": float("nan"),
                "is_significant": False,
                "p_value": None,
                "effect_size": "poor",
            })
        warning = _check_sample_size_warning(n)
        if n_clusters == 1:
            warning = f"Single cluster: silhouette is undefined. {warning}" if warning else "Single cluster: silhouette is undefined."
        return {
            "overall_silhouette": overall_sil,
            "n_clusters": n_clusters,
            "clusters": clusters,
            "n_significant": 0,
            "significant_clusters": [],
            "warning": warning,
        }

    # Compute silhouette scores for each point
    # Use euclidean by default for continuous data; hamming for binary
    s_all = None
    for metric in ("euclidean", "hamming"):
        try:
            s_all = silhouette_samples(X, labels, metric=metric)
            break
        except Exception:
            continue
    if s_all is None:
        # Silhouette failed for both metrics → return NaN silhouette
        overall_sil = float("nan")
        clusters = []
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            n_k = int(np.sum(mask))
            clusters.append({
                "cluster_id": int(cluster_id),
                "size": n_k,
                "mean_silhouette": float("nan"),
                "silhouette_ci_lower": float("nan"),
                "silhouette_ci_upper": float("nan"),
                "is_significant": False,
                "p_value": None,
                "effect_size": "poor",
            })
        return {
            "overall_silhouette": overall_sil,
            "n_clusters": n_clusters,
            "clusters": clusters,
            "n_significant": 0,
            "significant_clusters": [],
            "warning": "Silhouette computation failed for both euclidean and hamming metrics.",
        }

    overall_silhouette = float(np.mean(s_all))
    rng = np.random.default_rng(random_state)
    alpha = 1 - confidence
    clusters = []
    p_values = []

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        n_k = int(np.sum(mask))

        if n_k < 3:
            mean_sil = float(np.mean(s_all[mask])) if n_k > 0 else float("nan")
            cluster_info = {
                "cluster_id": int(cluster_id),
                "size": n_k,
                "mean_silhouette": mean_sil,
                "silhouette_ci_lower": float("nan"),
                "silhouette_ci_upper": float("nan"),
                "is_significant": False,
                "p_value": None,
                "effect_size": _classify_silhouette_effect(mean_sil),
            }
            clusters.append(cluster_info)
            continue

        s_k = s_all[mask]
        mean_sil = float(np.mean(s_k))

        boot_means = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_k, size=n_k)
            boot_means[b] = float(np.mean(s_k[idx]))

        ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
        ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

        # One-sided bootstrap p-value for H0: mean silhouette <= 0 vs
        # H1: mean silhouette > 0 (i.e. the cluster is more cohesive than a
        # structureless assignment).  This is the fraction of the bootstrap
        # distribution that falls at or below 0, with add-one smoothing.
        #
        # The previous formula compared boot_means to their own centre
        # (mean_sil), so the bootstrap distribution was symmetric around it and
        # the p-value was ~0.5 regardless of how well-separated the cluster was.
        # The form below is consistent with is_significant = (ci_lower > 0):
        # a cluster whose lower CI bound exceeds 0 gets a correspondingly small
        # p-value.
        p_val = float((1 + np.sum(boot_means <= 0.0)) / (n_bootstrap + 1))
        is_sig = ci_lower > 0

        cluster_info = {
            "cluster_id": int(cluster_id),
            "size": n_k,
            "mean_silhouette": mean_sil,
            "silhouette_ci_lower": ci_lower,
            "silhouette_ci_upper": ci_upper,
            "is_significant": is_sig,
            "p_value": p_val,
            "effect_size": _classify_silhouette_effect(mean_sil),
        }
        clusters.append(cluster_info)
        p_values.append(p_val)

    # Apply multiple testing correction if requested
    if correction_method is not None and p_values:
        correction_result = apply_multiple_testing_correction(
            np.array(p_values), method=correction_method, alpha=alpha,
        )
        corrected_pvals = correction_result["p_values_adjusted"]
        rejected = correction_result["rejected"]
        testable_clusters = [c for c in clusters if c["p_value"] is not None]
        for cluster, corrected_p, is_rejected in zip(
            testable_clusters, corrected_pvals, rejected
        ):
            cluster["p_value_corrected"] = float(corrected_p)
            cluster["is_significant"] = bool(is_rejected)

    n_significant = sum(1 for c in clusters if c["is_significant"])
    significant_clusters = [c["cluster_id"] for c in clusters if c["is_significant"]]

    warning = _check_sample_size_warning(n)

    return {
        "overall_silhouette": overall_silhouette,
        "n_clusters": n_clusters,
        "clusters": clusters,
        "n_significant": n_significant,
        "significant_clusters": significant_clusters,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 5. Multiple testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(
    p_values: np.ndarray,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply multiple testing correction to p-values.

    Parameters
    ----------
    p_values : array of shape (m,)
        Raw p-values to correct
    method : str, default="bonferroni"
        Correction method: "bonferroni", "sidak", or "fdr_bh"
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict with keys:
        - p_values_adjusted: np.ndarray
        - rejected: np.ndarray of bool
        - method: str
        - alpha_corrected: float
        - n_rejected: int
    """
    p_values = np.asarray(p_values, dtype=float)
    if not np.all(np.isfinite(p_values)) or np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("p_values must be finite and in [0, 1]")
    m = len(p_values)

    if m == 0:
        return {
            "p_values_adjusted": np.array([]),
            "rejected": np.array([], dtype=bool),
            "method": method,
            "alpha_corrected": alpha,
            "n_rejected": 0,
        }

    if method == "bonferroni":
        adjusted = np.minimum(p_values * m, 1.0)
        alpha_corr = alpha / m
        rejected = p_values <= alpha_corr
    elif method == "sidak":
        adjusted = 1.0 - (1.0 - p_values) ** m
        alpha_corr = 1.0 - (1.0 - alpha) ** (1.0 / m)
        rejected = p_values <= alpha_corr
    elif method == "fdr_bh":
        # Benjamini-Hochberg procedure
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        adjusted = np.empty(m, dtype=float)
        # Step-up procedure
        max_k = 0
        for k in range(m, 0, -1):
            if sorted_p[k - 1] <= (k / m) * alpha:
                max_k = k
                break
        rejected = np.zeros(m, dtype=bool)
        if max_k > 0:
            rejected[order[:max_k]] = True
        # BH adjusted p-values
        adjusted_sorted = np.minimum.accumulate(
            np.minimum(sorted_p * m / np.arange(1, m + 1), 1.0)[::-1]
        )[::-1]
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_sorted
        alpha_corr = alpha  # BH threshold is adaptive
    else:
        raise ValueError("method must be one of 'bonferroni', 'sidak', 'fdr_bh'")

    return {
        "p_values_adjusted": adjusted,
        "rejected": rejected,
        "method": method,
        "alpha_corrected": alpha_corr,
        "n_rejected": int(np.sum(rejected)),
    }
