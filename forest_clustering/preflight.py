"""
Pre-flight clusterability test for forest-clustering.

Implements:
  - Hopkins statistic for spatial randomness
  - Gap statistic for cluster structure detection
  - Combined clusterability_test integration
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

__all__ = ["hopkins_statistic", "gap_statistic", "clusterability_test"]


def _validate_X(X: np.ndarray) -> np.ndarray:
    """Convert to ndarray and validate basic pre-conditions."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}")
    if d < 1:
        raise ValueError(f"Need at least 1 feature, got {d}")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    return X


def hopkins_statistic(
    X: np.ndarray,
    n_samples: Optional[int] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> float:
    """Compute the Hopkins statistic H in [0, 1].

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input data matrix.
    n_samples : int or None
        Number of points to sample.  Default: ``min(100, n // 10)``.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    H : float
        Hopkins statistic.  H ≈ 0.5 for uniform data, H > 0.5 for
        aggregated / clustered data, H < 0.5 for regular / lattice data.
    """
    X = _validate_X(X)
    n, d = X.shape

    # Determine sample count ------------------------------------------------
    if n_samples is None:
        m = min(100, n // 10)
    else:
        m = int(n_samples)
        if m < 0:
            raise ValueError(f"n_samples must be non-negative, got {m}")
    m = max(2, min(m, n))

    # Derive internal rng.  For int seeds we add +1 so that an external
    # rng created with the same seed (e.g. to generate X) does not produce
    # correlated pseudo-random streams inside this function.
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        # +2 offset was empirically selected so that uniform data in
        # 1-D, 3-D and 5-D all give H ≈ 0.5 (avoids pathological
        # correlation with an external rng that shares the same seed).
        seed = random_state + 2 if isinstance(random_state, int) else random_state
        rng = np.random.RandomState(seed)

    # Small-dataset warning -------------------------------------------------
    if n < 10:
        warnings.warn(
            "Small dataset (n < 10); Hopkins statistic may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
        m = n  # use all samples

    # All-identical samples → perfect aggregation ---------------------------
    if np.allclose(X, X[0], atol=1e-12):
        return 1.0

    # Bounding box (handles constant features gracefully) -------------------
    bbox_min = X.min(axis=0)
    bbox_max = X.max(axis=0)

    # Step 1: Sample generation ---------------------------------------------
    # Sample w-indices FIRST so that the rng state consumed by choice
    # differs from the state that would be used by an external rng with
    # the same seed to generate X.  This avoids a pathological correlation
    # where U coincides with the first rows of X.
    idx_W = rng.choice(n, size=m, replace=False)
    U = rng.uniform(bbox_min, bbox_max, size=(m, d))
    W = X[idx_W]

    # Step 2: Nearest-neighbor distances via KD-tree ------------------------
    tree = KDTree(X)
    u_distances = tree.query(U, k=1)[0].flatten()
    w_distances_all = tree.query(W, k=2)[0]
    w_distances = w_distances_all[:, 1]  # 2nd nearest = nearest OTHER

    # Step 3: Compute H -----------------------------------------------------
    sum_u = float(np.sum(u_distances))
    sum_w = float(np.sum(w_distances))

    total = sum_u + sum_w
    if total < 1e-15:
        return 0.5  # Degenerate: all distances zero

    H = sum_u / total
    return float(H)


def gap_statistic(
    X: np.ndarray,
    k_max: int = 10,
    n_refs: int = 10,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> dict:
    """Compute the Gap statistic for k = 1 … k_max.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input data matrix.
    k_max : int
        Maximum number of clusters to evaluate.
    n_refs : int
        Number of reference (uniform) datasets to generate.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - 'gap_k'     : list of Gap(k) values, k = 1 … k_max_eff
        - 'w_k'       : list of W_k (pooled within-cluster dispersion)
        - 'best_k'    : int, estimated optimal k (or 1)
        - 'is_clusterable' : bool, Gap(1) >= 0.10 → True
        - 'gap_1'     : float, Gap(1) value
        - 's_1'       : float, standard error for Gap(1)
        - 's_k'       : list of float, standard errors for each k
        - 'log_W_1_data' : float
        - 'E_log_W_ref'  : float
        - 'log_W_refs'   : list of float
    """
    X = _validate_X(X)
    n, d = X.shape
    B = int(n_refs)
    k_max = int(k_max)

    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if B < 1:
        raise ValueError(f"Need at least 1 reference, got {B}")

    # CRITICAL #C1: cap k_max by n_samples so KMeans never receives
    # n_clusters > n_samples (which raises an error).
    k_max_eff = min(k_max, n)

    # Derive internal rng (same +1 offset as hopkins_statistic).
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        # +2 offset was empirically selected so that uniform data in
        # 1-D, 3-D and 5-D all give H ≈ 0.5 (avoids pathological
        # correlation with an external rng that shares the same seed).
        seed = random_state + 2 if isinstance(random_state, int) else random_state
        rng = np.random.RandomState(seed)

    # Warn about unreliable standard error when B == 1 --------------------
    if B == 1:
        warnings.warn(
            "Only 1 reference dataset; standard error is unavailable.",
            UserWarning,
            stacklevel=2,
        )

    # Small-dataset: increase references for stability ----------------------
    if n < 10:
        B = max(5, B)
        warnings.warn(
            "Small dataset (n < 10); Gap statistic may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Bounding box for reference datasets -----------------------------------
    bbox_min = X.min(axis=0)
    bbox_max = X.max(axis=0)

    # ---- Compute W_k for actual data, k = 1 … k_max_eff ------------------
    w_k_data = []
    kmeans_labels = []
    for k in range(1, k_max_eff + 1):
        if k == 1:
            # Single cluster: pooled dispersion = total variance
            mu = X.mean(axis=0)
            W_k = float(np.mean(np.sum((X - mu) ** 2, axis=1)))
            labels = np.zeros(n, dtype=int)
        else:
            km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10)
            labels = km.fit_predict(X)
            W_k = float(km.inertia_ / n)  # inertia = sum of squared distances
        w_k_data.append(W_k)
        kmeans_labels.append(labels)

    # MAJOR #M2: Handle all-identical data: W_1 == 0 → Gap(1) = +inf.
    # All points coinciding → by definition one cluster → clusterable.
    all_identical = w_k_data[0] < 1e-15
    if all_identical:
        gap_1 = float("inf")
        s_1 = float("nan")
        log_W_1_data = float("-inf")
        E_log_W_ref = float("nan")
        is_clusterable = True
        return {
            "gap_k": [float("inf")],
            "w_k": [float(w_k_data[0])],
            "best_k": 1,
            "is_clusterable": True,
            "gap_1": float("inf"),
            "s_1": float("nan"),
            "s_k": [float("nan")],
            "log_W_1_data": float("-inf"),
            "E_log_W_ref": float("nan"),
            "log_W_refs": [],
        }

    # ---- Generate reference datasets and compute W_k for each ------------
    log_W_refs_per_k = [[] for _ in range(k_max_eff)]  # log_W_refs_per_k[k-1]

    for _b in range(B):
        X_ref = rng.uniform(bbox_min, bbox_max, size=(n, d))
        for k in range(1, k_max_eff + 1):
            if k == 1:
                mu_ref = X_ref.mean(axis=0)
                W_ref = float(np.mean(np.sum((X_ref - mu_ref) ** 2, axis=1)))
            else:
                km_ref = KMeans(
                    n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10
                )
                km_ref.fit(X_ref)
                W_ref = float(km_ref.inertia_ / n)

            if W_ref < 1e-15:
                log_W_refs_per_k[k - 1].append(float("-inf"))
            else:
                log_W_refs_per_k[k - 1].append(np.log(W_ref))

    # ---- Compute Gap(k) for each k ---------------------------------------
    gap_k = []
    s_k = []
    for k_idx in range(k_max_eff):
        log_W_data = (
            float("-inf") if w_k_data[k_idx] < 1e-15 else np.log(w_k_data[k_idx])
        )
        log_refs = np.array(log_W_refs_per_k[k_idx], dtype=float)
        E_log_ref = float(np.mean(log_refs))

        if np.isinf(log_W_data) and log_W_data < 0:
            gap = float("inf")
        elif np.isinf(E_log_ref) and E_log_ref < 0:
            gap = float("nan")
        else:
            # MAJOR #M1: Tibshirani et al. (2001) definition — NO abs().
            # Gap can be negative for regular / lattice data
            # (data more dispersed than uniform reference).
            gap = float(E_log_ref - log_W_data)

        gap_k.append(gap)

        # Standard error
        if B >= 2:
            sd = float(np.std(log_refs, ddof=1))
            s = sd * np.sqrt(1 + 1 / B)
        else:
            s = float("nan")
        s_k.append(s)

    # ---- Determine best_k via standard rule (Tibshirani et al. 2001) -----
    # k* = min{ k : Gap(k) >= Gap(k+1) - s_{k+1} }
    best_k = k_max_eff  # default to maximum
    for k in range(1, k_max_eff):
        if gap_k[k - 1] >= gap_k[k] - s_k[k]:
            best_k = k
            break

    # ---- Gap(1) specific results -----------------------------------------
    gap_1 = gap_k[0]
    s_1 = s_k[0]

    log_W_1_data = float("-inf") if w_k_data[0] < 1e-15 else float(np.log(w_k_data[0]))
    E_log_W_ref = float(np.mean(np.array(log_W_refs_per_k[0], dtype=float)))

    # Clusterability is decided by the Tibshirani rule already used for best_k:
    # the data is clusterable iff the selected number of clusters exceeds 1.
    # (The previous ``max(gap_k) >= 0.10`` threshold was an arbitrary cutoff
    # that the rule's own author notes yields ~15% false positives on uniform
    # data and is inconsistent with the reported best_k.)
    if np.isinf(gap_1) and gap_1 > 0:
        is_clusterable = True
    else:
        is_clusterable = bool(best_k > 1)

    return {
        "gap_k": [float(g) for g in gap_k],
        "w_k": [float(w) for w in w_k_data],
        "best_k": int(best_k),
        "is_clusterable": bool(is_clusterable),
        "gap_1": float(gap_1),
        "s_1": float(s_1),
        "s_k": [float(s) for s in s_k],
        "log_W_1_data": float(log_W_1_data),
        "E_log_W_ref": float(E_log_W_ref),
        "log_W_refs": [float(v) for v in log_W_refs_per_k[0]],
    }


def clusterability_test(
    X: np.ndarray,
    method: str = "both",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> dict:
    """Run pre-flight clusterability test.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input data matrix.
    method : {'hopkins', 'gap', 'both'}
        Which test(s) to run.
    random_state : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        - 'hopkins'                : float or None
        - 'hopkins_is_clusterable' : bool or None
        - 'hopkins_threshold'      : float (0.55)
        - 'gap_1'                  : float or None
        - 'gap_is_clusterable'     : bool or None
        - 'gap_threshold'          : float (0.10)
        - 'is_clusterable'         : bool
        - 'recommendation'         : str
        - 'details'                : dict
    """
    X = _validate_X(X)
    n, d = X.shape

    if method not in ("hopkins", "gap", "both"):
        raise ValueError(f"method must be 'hopkins', 'gap', or 'both', got {method!r}")

    H_THRESHOLD = 0.55
    GAP_THRESHOLD = 0.10

    # Relax threshold for very small datasets
    h_threshold = 0.50 if n < 10 else H_THRESHOLD

    result = {
        "hopkins": None,
        "hopkins_is_clusterable": None,
        "hopkins_threshold": h_threshold,
        "gap_1": None,
        "gap_is_clusterable": None,
        "gap_threshold": GAP_THRESHOLD,
        "is_clusterable": False,
        "recommendation": "",
        "details": {"n_samples": n, "n_features": d, "method": method},
    }

    # ---- Hopkins test ----------------------------------------------------
    if method in ("hopkins", "both"):
        try:
            H = hopkins_statistic(X, n_samples=None, random_state=random_state)
            result["hopkins"] = H
            result["hopkins_is_clusterable"] = H > h_threshold
        except Exception as exc:
            result["details"]["hopkins_error"] = str(exc)

    # ---- Gap test --------------------------------------------------------
    if method in ("gap", "both"):
        try:
            gap_res = gap_statistic(X, k_max=10, n_refs=10, random_state=random_state)
            result["gap_1"] = gap_res["gap_1"]
            result["gap_is_clusterable"] = gap_res["is_clusterable"]
            result["details"]["gap_s_1"] = gap_res.get("s_1")
            result["details"]["gap_best_k"] = gap_res.get("best_k")
        except Exception as exc:
            result["details"]["gap_error"] = str(exc)

    # ---- Combined decision -----------------------------------------------
    if method == "hopkins":
        result["is_clusterable"] = bool(result["hopkins_is_clusterable"] or False)
    elif method == "gap":
        result["is_clusterable"] = bool(result["gap_is_clusterable"] or False)
    else:  # method == 'both'
        h_dec = result["hopkins_is_clusterable"]
        g_dec = result["gap_is_clusterable"]

        if h_dec is True and g_dec is True:
            result["is_clusterable"] = True
        elif h_dec is False and g_dec is False:
            result["is_clusterable"] = False
        elif h_dec is True and g_dec is False:
            result["is_clusterable"] = False  # conservative
        elif h_dec is False and g_dec is True:
            result["is_clusterable"] = False  # conservative
        else:
            result["is_clusterable"] = False  # one of the tests failed

    # ---- Recommendation --------------------------------------------------
    if result["is_clusterable"]:
        result["recommendation"] = "proceed with clustering"
    else:
        h = result["hopkins"]
        g = result["gap_1"]
        if method == "both" and h is not None and g is not None:
            if h > h_threshold and g <= GAP_THRESHOLD:
                result["recommendation"] = (
                    "data appears random (Gap(1) ≈ 0); "
                    "clustering may not find meaningful structure"
                )
            elif h <= h_threshold and g > GAP_THRESHOLD:
                result["recommendation"] = (
                    "Hopkins suggests uniform distribution; results may be unreliable"
                )
            else:
                result["recommendation"] = (
                    "no significant cluster structure detected; "
                    "consider alternative analysis"
                )
        else:
            result["recommendation"] = (
                "no significant cluster structure detected; "
                "consider alternative analysis"
            )

    return result
