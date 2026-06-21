"""Shared utilities for tree-proximity clustering estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


_RARE_SENTINEL_BASE = "__forest_clustering_rare__"


def to_frame(X) -> pd.DataFrame:
    """Convert array-like input to a DataFrame and normalize missing sentinels."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        arr = np.asarray(X)
        if arr.ndim == 0:
            raise ValueError("X must be a 1D or 2D array-like object")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be 1D or 2D array-like")
        df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
    return df.where(pd.notna(df), np.nan)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _numeric_string_fraction(s: pd.Series) -> float:
    vals = s.dropna()
    if len(vals) == 0:
        return 0.0
    parsed = pd.to_numeric(vals, errors="coerce")
    return float(np.isfinite(parsed.to_numpy(dtype=np.float64)).mean())


def _should_coerce_numeric_string(s: pd.Series, enabled: bool, min_fraction: float) -> bool:
    if not enabled or is_numeric_series(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return False
    return _numeric_string_fraction(s) >= float(min_fraction)


def _rare_sentinel(values) -> str:
    sentinel = _RARE_SENTINEL_BASE
    existing = set(values)
    if sentinel not in existing:
        return sentinel
    k = 1
    while f"{sentinel}_{k}" in existing:
        k += 1
    return f"{sentinel}_{k}"


class TabularFrameNormalizer(BaseEstimator, TransformerMixin):
    """Lightweight DataFrame normalizer before sklearn tree preprocessing.

    It handles three quality fixes consistently across URF, ExtraTrees and the
    greedy binary-tree estimator: numeric-string coercion, rare-category grouping
    and explicit missingness indicators.
    """

    def __init__(
        self,
        add_missing_indicators: bool = False,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction

    def fit(self, X, y=None):
        X_df = to_frame(X)
        self.columns_ = list(X_df.columns)
        self.numeric_string_cols_ = []
        self.missing_indicator_cols_ = []
        self.rare_maps_ = {}
        self.output_columns_ = list(X_df.columns)

        for col in X_df.columns:
            s = X_df[col]
            if _should_coerce_numeric_string(s, self.coerce_numeric_strings, self.numeric_string_min_fraction):
                self.numeric_string_cols_.append(col)
            if self.add_missing_indicators and s.isna().any():
                ind = f"{col}__missing"
                self.missing_indicator_cols_.append((col, ind))
                self.output_columns_.append(ind)

        norm = self._base_transform(X_df)
        for col in norm.columns:
            if is_numeric_series(norm[col]):
                continue
            non_missing = norm[col].dropna()
            if len(non_missing) == 0:
                continue
            values = list(non_missing.unique())
            if self.rare_category_min_count is None and self.rare_category_min_freq is None:
                continue
            counts = non_missing.value_counts(dropna=True)
            min_count = 1 if self.rare_category_min_count is None else int(self.rare_category_min_count)
            min_freq = 0.0 if self.rare_category_min_freq is None else float(self.rare_category_min_freq)
            denom = max(len(non_missing), 1)
            keep = {v for v in values if counts.get(v, 0) >= min_count and counts.get(v, 0) / denom >= min_freq}
            if len(keep) < len(values):
                self.rare_maps_[col] = {"keep": keep, "rare": _rare_sentinel(values)}
        return self

    def transform(self, X):
        X_df = to_frame(X)
        if list(X_df.columns) != list(self.columns_):
            raise ValueError(f"Column names do not match fit() columns. Expected {list(self.columns_)}, got {list(X_df.columns)}")
        out = self._base_transform(X_df)
        for col, cfg in getattr(self, "rare_maps_", {}).items():
            keep = cfg["keep"]
            rare = cfg["rare"]
            s = out[col]
            out[col] = s.where(s.isna() | s.isin(keep), rare)
        for col, ind in getattr(self, "missing_indicator_cols_", []):
            out[ind] = X_df[col].isna().astype(float)
        return out[self.output_columns_]

    def _base_transform(self, X_df):
        out = X_df.copy()
        for col in getattr(self, "numeric_string_cols_", []):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "output_columns_", []), dtype=object)


def make_onehot(**kwargs):
    """Create OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(sparse=True, **kwargs)


def make_numeric_imputer(add_indicator: bool = False):
    """Median imputer that keeps all-missing numeric columns when available."""
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=add_indicator)
    except TypeError:  # sklearn < 1.2
        return SimpleImputer(strategy="median", add_indicator=add_indicator)


def make_categorical_imputer(add_indicator: bool = False):
    """Categorical imputer that preserves all-missing columns as a real bucket."""
    try:
        return SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True, add_indicator=add_indicator)
    except TypeError:  # sklearn < 1.2
        return SimpleImputer(strategy="constant", fill_value="__missing__", add_indicator=add_indicator)


def build_tree_preprocessor(
    X_df: pd.DataFrame,
    add_missing_indicators: bool = False,
    rare_category_min_count: int | None = None,
    rare_category_min_freq: float | None = None,
    coerce_numeric_strings: bool = False,
    numeric_string_min_fraction: float = 0.90,
):
    normalizer_probe = TabularFrameNormalizer(
        add_missing_indicators=add_missing_indicators,
        rare_category_min_count=rare_category_min_count,
        rare_category_min_freq=rare_category_min_freq,
        coerce_numeric_strings=coerce_numeric_strings,
        numeric_string_min_fraction=numeric_string_min_fraction,
    ).fit(X_df)
    X_norm = normalizer_probe.transform(X_df)
    numeric_cols = [c for c in X_norm.columns if is_numeric_series(X_norm[c])]
    categorical_cols = [c for c in X_norm.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", make_numeric_imputer()),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", make_categorical_imputer()),
                ("onehot", make_onehot(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    return Pipeline([
        ("normalize", TabularFrameNormalizer(
            add_missing_indicators=add_missing_indicators,
            rare_category_min_count=rare_category_min_count,
            rare_category_min_freq=rare_category_min_freq,
            coerce_numeric_strings=coerce_numeric_strings,
            numeric_string_min_fraction=numeric_string_min_fraction,
        )),
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)),
    ])


def make_leaf_encoder(E: np.ndarray):
    categories = [np.unique(E[:, j]) for j in range(E.shape[1])]
    return make_onehot(categories=categories, handle_unknown="ignore", dtype=np.float32)


def leaf_proximity(E: np.ndarray) -> np.ndarray:
    """Same-leaf co-occurrence probabilities for a leaf-id matrix."""
    E = np.asarray(E)
    n = E.shape[0]
    P = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        P[i, :] = np.mean(E[i] == E, axis=1)
    return P


def leaf_cross_proximity(E_X: np.ndarray, E_Y: np.ndarray) -> np.ndarray:
    E_X = np.asarray(E_X)
    E_Y = np.asarray(E_Y)
    P = np.empty((E_X.shape[0], E_Y.shape[0]), dtype=np.float32)
    for i in range(E_X.shape[0]):
        P[i, :] = np.mean(E_X[i] == E_Y, axis=1)
    return P


def is_precomputed_clusterer(estimator) -> bool:
    return getattr(estimator, "metric", None) == "precomputed" or getattr(estimator, "affinity", None) == "precomputed"


def _is_spectral_precomputed(estimator) -> bool:
    return type(estimator).__name__ == "SpectralClustering" and getattr(estimator, "affinity", None) == "precomputed"


def default_agglomerative_labels(P: np.ndarray, n_clusters: int | None) -> np.ndarray:
    n = P.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    k = 3 if n_clusters is None else int(n_clusters)
    k = max(1, min(k, n))
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    D = (1.0 - P).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    try:
        clf = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:  # sklearn < 1.2
        clf = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return clf.fit_predict(D)


def run_leaf_clusterer(clusterer, n_clusters, E: np.ndarray, P: np.ndarray, Z=None, cluster_input: str = "auto") -> np.ndarray:
    """Run downstream clusterer on a mathematically appropriate representation.

    ``cluster_input`` controls what user-supplied downstream estimators receive:
    ``auto`` chooses similarity for spectral precomputed estimators, distance for
    other precomputed estimators, raw leaf ids for Hamming, and one-hot leaf
    features otherwise.
    """
    if cluster_input not in {"auto", "embedding", "onehot", "distance", "similarity"}:
        raise ValueError("cluster_input must be one of 'auto', 'embedding', 'onehot', 'distance', 'similarity'")

    if clusterer is None:
        return default_agglomerative_labels(P, n_clusters)

    clf = clone(clusterer)

    mode = cluster_input
    if mode == "auto":
        if _is_spectral_precomputed(clf):
            mode = "similarity"
        elif getattr(clf, "metric", None) == "precomputed" or getattr(clf, "affinity", None) == "precomputed":
            mode = "distance"
        elif getattr(clf, "metric", None) == "hamming":
            mode = "embedding"
        else:
            mode = "onehot"

    if mode == "distance":
        X_in = (1.0 - P).astype(np.float64)
        np.fill_diagonal(X_in, 0.0)
    elif mode == "similarity":
        X_in = P.astype(np.float64, copy=True)
        np.fill_diagonal(X_in, 1.0)
    elif mode == "embedding":
        X_in = E
    else:
        if Z is None:
            encoder = make_leaf_encoder(E)
            Z = encoder.fit_transform(E)
        X_in = Z

    if hasattr(clf, "fit_predict"):
        return clf.fit_predict(X_in)
    clf.fit(X_in)
    return clf.labels_
