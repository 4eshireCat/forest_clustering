"""Per-iteration weight computation based on cell-size distribution."""

import numpy as np


def compute_iteration_weights(E: np.ndarray, strategy: str = "uniform", weight_temperature: float = None) -> np.ndarray:
    """Compute per-iteration weights based on cell-size distribution.

    Parameters
    ----------
    E : np.ndarray, shape (n, L), dtype int64
        Embedding matrix.
    strategy : {"uniform", "entropy", "inverse_gini"}
        Weighting strategy.
    weight_temperature : float or None, default None
        Temperature for weight scaling. Values < 1 sharpen (exaggerate
        differences), values > 1 soften (reduce differences). Must be > 0.
        When None (default), behaves identically to 1.0 but preserves
        backward-compatible fallback of returning zeros when all raw
        weights are zero.

    Returns
    -------
    weights : np.ndarray, shape (L,), dtype float64
        Per-iteration weights with mean = 1.0.
    """
    # Input validation
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"E must be a 2-D array, got ndim={E.ndim}")
    n, L = E.shape
    if n == 0:
        raise ValueError("E must have at least one row (n > 0)")
    if not np.issubdtype(E.dtype, np.integer):
        raise TypeError(f"E must have an integer dtype, got {E.dtype}")

    if strategy == "uniform":
        return np.ones(L, dtype=np.float64)

    # Resolve temperature and whether it was explicitly passed
    if weight_temperature is None:
        weight_temperature = 1.0
        explicit_temperature = False
    else:
        explicit_temperature = True

    if strategy == "entropy":
        return _entropy_weights(E, n, L, weight_temperature, explicit_temperature)

    if strategy == "inverse_gini":
        return _inverse_gini_weights(E, n, L, weight_temperature, explicit_temperature)

    raise ValueError(f"Unknown strategy: {strategy!r}. Expected one of "
                     f"'uniform', 'entropy', 'inverse_gini'.")


def _entropy_weights(E: np.ndarray, n: int, L: int, weight_temperature: float, explicit_temperature: bool) -> np.ndarray:
    """Compute entropy-based per-iteration weights."""
    raw_weights = np.empty(L, dtype=np.float64)

    for l in range(L):
        col = E[:, l]
        unique_vals, counts = np.unique(col, return_counts=True)
        p = counts / n
        entropy = -np.sum(p * np.log(p))

        n_unique = len(unique_vals)

        if n == 1:
            # Single sample: avoid division by zero, assign neutral weight
            max_entropy = 0.0
        else:
            max_entropy = np.log(n_unique) if n_unique > 1 else 0.0

        if n == 1:
            raw_weights[l] = 1.0
        elif max_entropy == 0.0:
            raw_weights[l] = 0.0
        else:
            raw_weights[l] = entropy / max_entropy

    return _apply_temperature_and_normalize(raw_weights, weight_temperature, explicit_temperature)


def _inverse_gini_weights(E: np.ndarray, n: int, L: int, weight_temperature: float, explicit_temperature: bool) -> np.ndarray:
    """Compute inverse-Gini-based per-iteration weights."""
    raw_weights = np.empty(L, dtype=np.float64)

    for l in range(L):
        col = E[:, l]
        unique_vals, counts = np.unique(col, return_counts=True)
        p = counts / n
        gini = 1.0 - np.sum(p * p)

        n_unique = len(unique_vals)

        if n == 1:
            # Single sample: avoid division by zero, assign neutral weight
            max_gini = 0.0
        elif n_unique == 1:
            max_gini = 0.0
        else:
            max_gini = 1.0 - 1.0 / min(n, n_unique)

        if n == 1:
            raw_weights[l] = 1.0
        elif max_gini == 0.0:
            raw_weights[l] = 0.0
        else:
            raw_weights[l] = gini / max_gini

    return _apply_temperature_and_normalize(raw_weights, weight_temperature, explicit_temperature)


def _apply_temperature_and_normalize(raw_weights: np.ndarray, weight_temperature: float, explicit_temperature: bool) -> np.ndarray:
    """Apply temperature scaling and normalize so mean = 1.0.

    If all raw weights are zero:
    - When temperature was explicitly passed: return ones (new behaviour).
    - When temperature was omitted (default): return zeros (backward compat).
    """
    # Validate temperature
    if np.isnan(weight_temperature):
        raise ValueError(f"weight_temperature must not be NaN")
    if weight_temperature <= 0:
        raise ValueError(f"weight_temperature must be positive, got {weight_temperature}")
    # MED-10: clamp extreme temperatures to avoid numerical issues
    if weight_temperature < 0.1:
        import warnings
        warnings.warn(f"weight_temperature={weight_temperature} clamped to 0.1", UserWarning)
        weight_temperature = 0.1
    elif weight_temperature > 10.0:
        import warnings
        warnings.warn(f"weight_temperature={weight_temperature} clamped to 10.0", UserWarning)
        weight_temperature = 10.0

    # Temperature scaling
    if weight_temperature != 1.0:
        scaled = raw_weights ** (1.0 / weight_temperature)
        # MED-10: guard against non-finite results from extreme temperatures
        if not np.all(np.isfinite(scaled)):
            import warnings
            warnings.warn(
                f"weight_temperature={weight_temperature} produced non-finite "
                f"weights; falling back to unscaled raw weights.",
                UserWarning,
                stacklevel=3,
            )
            scaled = raw_weights
        if weight_temperature > 1.0:
            # Soften: linearly blend toward uniform weights.
            # blend ∈ (0, 1); as t → ∞, blend → 1 (uniform).
            # This preserves ordering (convex combination of ordered
            # quantities) and gives correct asymptotic behaviour.
            blend = 1.0 - 1.0 / weight_temperature
            scaled = scaled * (1.0 - blend) + 1.0 * blend
    else:
        scaled = raw_weights

    # Normalize: mean = 1.0
    L = len(scaled)
    if scaled.mean() > 0:
        weights = scaled / scaled.mean()
    else:
        # MED-4: warn about zero-weight fallback
        import warnings
        warnings.warn(
            f"All {L} raw iteration weights are zero (all iterations may be "
            f"identical or perfectly uniform). Falling back to uniform weights. "
            f"Consider increasing n_iterations or checking data diversity.",
            UserWarning,
            stacklevel=3,
        )
        # Backward compatibility: when temperature not explicitly passed,
        # preserve old behaviour of returning zeros.
        if explicit_temperature:
            weights = np.ones(L, dtype=np.float64)
        else:
            weights = np.zeros(L, dtype=np.float64)

    return weights


def _normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
    """Normalize raw weights so mean = 1.0.

    If all raw weights are zero, return zeros.
    """
    raw_mean = raw_weights.mean()
    if raw_mean == 0.0:
        return np.zeros_like(raw_weights)
    return raw_weights / raw_mean
