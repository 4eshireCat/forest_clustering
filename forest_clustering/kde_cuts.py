"""Density-aware cut-point generation using KDE peak-valley detection."""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

KDE_PEAKS_DEFAULTS = {
    "bandwidth_rule": "silverman",
    "grid_resolution": 512,
    "grid_extension": 3.0,
    "min_prominence": 1e-6,
    "min_bandwidth_frac": 1e-4,
    "max_peaks": 50,
    "subsample_threshold": 10000,
    "subsample_size": 5000,
}


def kde_peaks_cut_points(data, n_cuts, rng, kde_params=None):
    """Generate cut-points at valleys between KDE peaks.

    Parameters
    ----------
    data : ndarray of shape (n,)
        Numerical column data.
    n_cuts : int
        Number of cut-points needed (K - 1).
    rng : np.random.Generator
        Random state for reproducible fallbacks.
    kde_params : dict or None
        Override default KDE parameters.

    Returns
    -------
    cuts : ndarray of shape (n_cuts,), sorted
        Cut-point positions.
    strategy : str
        One of "kde", "kde+uniform", "uniform", "quantile".
    """
    if n_cuts < 0:
        raise ValueError("n_cuts must be non-negative")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        data = data.ravel()
    n = data.shape[0]
    if n == 0:
        raise ValueError("data must not be empty")

    if n_cuts == 0:
        return np.array([], dtype=np.float64), "kde"

    # 1. PREPROCESS
    finite = data[np.isfinite(data)]
    n_eff = finite.shape[0]
    if n_eff == 0:
        raise ValueError("data contains no finite values")

    # Edge cases: all identical or too few points
    if n_eff < 2 or np.allclose(finite, finite[0]):
        return _uniform_cuts(finite, n_cuts, rng), "uniform"

    n_unique = len(np.unique(finite))
    if n_eff <= 5 or n_unique <= n_cuts + 1:
        return _quantile_cuts(finite, n_cuts, rng), "quantile"

    # Parameters
    params = {**KDE_PEAKS_DEFAULTS}
    if kde_params is not None:
        params.update(kde_params)

    # 2. SUBSAMPLE if large
    finite_full = finite
    if n_eff > params["subsample_threshold"]:
        subsample_size_eff = min(params["subsample_size"], n_eff)
        idx = rng.choice(n_eff, size=subsample_size_eff, replace=False)
        finite = finite[idx]
        n_eff = finite.shape[0]

    # 3. COMPUTE BANDWIDTH
    if "bandwidth" in params:
        h = float(params["bandwidth"])
        range_data = float(finite.max() - finite.min())
        if range_data > 0:
            h = max(h, range_data * params["min_bandwidth_frac"])
    else:
        h = _compute_bandwidth(finite, params["bandwidth_rule"], params["min_bandwidth_frac"])

    # 4. BUILD GRID
    x_min = float(finite.min())
    x_max = float(finite.max())
    G = max(256, min(2048, params["grid_resolution"]))
    pad = params["grid_extension"] * h
    grid = np.linspace(x_min - pad, x_max + pad, G)

    # 5. EVALUATE KDE
    sigma = float(np.std(finite, ddof=1))
    if sigma == 0.0:
        sigma = 1.0
    bw_method = h / sigma
    if bw_method <= 0.0:
        bw_method = 1.0
    kde = stats.gaussian_kde(finite, bw_method=bw_method)
    f_grid = kde.evaluate(grid)

    # 6. FIND PEAKS & VALLEYS
    peaks, valleys = _find_peaks_valleys(
        grid, f_grid, params["min_prominence"], params["max_peaks"], n
    )

    # Exclude boundary valleys (within h of data range) when interior valleys exist
    interior_valleys = [v for v in valleys if x_min + h <= grid[v[0]] <= x_max - h]
    if interior_valleys:
        valleys = interior_valleys
    # If no interior valleys, keep boundary valleys as fallback

    # 7. SELECT CUTS FROM VALLEYS
    if len(valleys) >= n_cuts:
        cuts = _select_valleys(valleys, peaks, grid, f_grid, finite_full, n_cuts)
        return np.sort(np.asarray(cuts, dtype=np.float64)), "kde"
    elif len(valleys) > 0:
        valley_cuts = grid[np.array([v[0] for v in valleys])]
        n_uniform = n_cuts - len(valleys)
        uniform_cuts = _uniform_cuts(finite_full, n_uniform, rng)
        cuts = _merge_cuts(valley_cuts, uniform_cuts, finite_full)
        return np.sort(np.asarray(cuts, dtype=np.float64)), "kde+uniform"
    else:
        return _uniform_cuts(finite_full, n_cuts, rng), "uniform"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_bandwidth(data, method="silverman", min_frac=1e-4):
    """Compute KDE bandwidth."""
    n = data.shape[0]
    sigma = float(np.std(data, ddof=1))
    q25, q75 = np.percentile(data, [25.0, 75.0])
    iqr = q75 - q25
    if method == "silverman":
        if iqr == 0.0:
            h = 0.9 * sigma * n ** (-1.0 / 5.0)
        else:
            h = 0.9 * min(sigma, iqr / 1.34) * n ** (-1.0 / 5.0)
    elif method == "scott":
        h = sigma * n ** (-1.0 / 5.0)
    else:
        # fallback to silverman
        if iqr == 0.0:
            h = 0.9 * sigma * n ** (-1.0 / 5.0)
        else:
            h = 0.9 * min(sigma, iqr / 1.34) * n ** (-1.0 / 5.0)
    range_data = float(data.max() - data.min())
    h = max(h, range_data * min_frac)
    return float(h)


def _find_peaks_valleys(grid, f_grid, min_prominence, max_peaks, n):
    """Detect peaks and valleys in KDE grid using scipy.

    Parameters
    ----------
    grid, f_grid : ndarray
        KDE evaluation grid and densities.
    min_prominence : float
        Minimum peak prominence.
    max_peaks : int
        Maximum number of peaks to retain.
    n : int
        Original sample size (for adaptive flat-guard threshold).
    """
    peaks_idx, pprops = find_peaks(f_grid, prominence=min_prominence)
    valleys_idx, vprops = find_peaks(-f_grid, prominence=min_prominence)
    peaks = [(int(i), float(f_grid[i])) for i in peaks_idx]
    valleys = [(int(i), float(f_grid[i])) for i in valleys_idx]

    max_f = float(np.max(f_grid))

    # Guard: reject approximately flat KDEs (uniform-like distributions)
    # where sampling noise produces spurious ripples.  Multimodal KDEs
    # have max_f >> mean_f; flat ones have max_f ≈ mean_f.
    # Adaptive threshold: relaxes for larger n (less sampling noise).
    mean_f = float(np.mean(f_grid))
    flat_threshold = 1.0 + 2.0 * (n ** (-0.2))
    if max_f > 0 and max_f / mean_f < flat_threshold:
        return [], []

    # Filter peaks by relative prominence to ignore noise-induced ripples
    if max_f > 0 and len(peaks) > 0:
        peak_proms = pprops.get("prominences", np.zeros(len(peaks)))
        # Require at least 2 % prominence relative to global max
        # (prom / local_peak_height would let tiny noise peaks through)
        peaks = [p for p, prom in zip(peaks, peak_proms) if prom / max_f >= 0.02]

    # Fewer than 2 meaningful peaks → no interior valleys
    if len(peaks) < 2:
        return peaks, []

    # Enforce alternation
    peaks, valleys = _enforce_alternation(peaks, valleys, f_grid)

    # Keep only valleys that lie strictly between the first and last peak
    if peaks and valleys:
        first_peak_idx = peaks[0][0]
        last_peak_idx = peaks[-1][0]
        valleys = [v for v in valleys if first_peak_idx < v[0] < last_peak_idx]

    # Cap peaks
    if len(peaks) > max_peaks:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:max_peaks]
        peaks, valleys = _enforce_alternation(peaks, valleys, f_grid)

    return peaks, valleys


def _enforce_alternation(peaks, valleys, f_grid):
    """Ensure peaks and valleys alternate by index."""
    ext = [(idx, "p", val) for idx, val in peaks] + [(idx, "v", val) for idx, val in valleys]
    ext.sort(key=lambda x: x[0])
    filtered = []
    for item in ext:
        idx, typ, val = item
        if not filtered:
            filtered.append(item)
            continue
        last_idx, last_typ, last_val = filtered[-1]
        if typ == last_typ:
            if typ == "p":
                if val > last_val:
                    filtered[-1] = item
            else:
                if val < last_val:
                    filtered[-1] = item
        else:
            filtered.append(item)
    peaks_out = [(idx, val) for idx, typ, val in filtered if typ == "p"]
    valleys_out = [(idx, val) for idx, typ, val in filtered if typ == "v"]
    return peaks_out, valleys_out


def _select_valleys(valleys, peaks, grid, f_grid, data, n_cuts):
    """Score valleys by depth and balance, return top n_cuts."""
    if len(valleys) <= n_cuts:
        return grid[np.array([v[0] for v in valleys])]
    # Compute depth (prominence) for each valley using nearest peaks
    peak_indices = sorted([p[0] for p in peaks])
    prominences = []
    for idx, val in valleys:
        left = None
        for p_idx in reversed(peak_indices):
            if p_idx < idx:
                left = p_idx
                break
        right = None
        for p_idx in peak_indices:
            if p_idx > idx:
                right = p_idx
                break
        if left is None or right is None:
            prom = 0.0
        else:
            prom = min(f_grid[left], f_grid[right]) - f_grid[idx]
        prominences.append(max(prom, 0.0))
    max_prom = max(prominences) if prominences else 1.0
    n = len(data)
    scores = []
    for (idx, val), prom in zip(valleys, prominences):
        depth_score = prom / max_prom if max_prom > 0 else 0.0
        g_v = grid[idx]
        n_left = int(np.sum(data < g_v))
        coverage_score = 1.0 - abs(n_left / n - 0.5) * 2.0
        score = 0.7 * depth_score + 0.3 * coverage_score
        scores.append(score)
    order = np.argsort(scores)[::-1]
    selected = order[:n_cuts]
    selected_indices = np.array([valleys[i][0] for i in selected])
    return grid[selected_indices]


def _uniform_cuts(data, n_cuts, rng):
    """Return n_cuts uniformly random cut points in [min, max] of data."""
    if n_cuts == 0:
        return np.array([], dtype=np.float64)
    lo = float(data.min())
    hi = float(data.max())
    if lo == hi:
        return np.full(n_cuts, lo, dtype=np.float64)
    return np.sort(rng.uniform(lo, hi, size=n_cuts).astype(np.float64))


def _quantile_cuts(data, n_cuts, rng):
    """Return cut points based on midpoints between unique values."""
    if n_cuts == 0:
        return np.array([], dtype=np.float64)
    unique_sorted = np.unique(data)
    if unique_sorted.shape[0] == 1:
        return np.full(n_cuts, unique_sorted[0], dtype=np.float64)
    mids = (unique_sorted[:-1] + unique_sorted[1:]) / 2.0
    if mids.shape[0] >= n_cuts:
        idx = np.linspace(0, mids.shape[0] - 1, n_cuts).astype(int)
        return np.sort(mids[idx].astype(np.float64))
    else:
        repeats = int(np.ceil(n_cuts / mids.shape[0]))
        full = np.tile(mids, repeats)[:n_cuts]
        return np.sort(full.astype(np.float64))


def _merge_cuts(valley_cuts, uniform_cuts, data):
    """Merge valley and uniform cuts."""
    all_cuts = np.concatenate([valley_cuts, uniform_cuts])
    return np.sort(all_cuts.astype(np.float64))
