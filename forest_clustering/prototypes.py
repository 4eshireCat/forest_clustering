"""Prototype-based sampling and subsampled clustering utilities.

The module provides two sklearn-style estimators:

``PrototypeSampler`` builds a compact weighted set of representative rows before
clustering.  It is deliberately conservative: the original rows are not lost,
because ``inverse_assignment_`` can expand prototype labels back to the full
training set.

``SubsampledClusterer`` combines a sampler with any sklearn-compatible clusterer
and exposes full-data ``labels_`` while fitting the expensive clustering step on
prototypes only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted

from .transformer import ForestTransformer
from ._tree_common import build_tree_preprocessor, to_frame


_ALLOWED_METHODS = {"leaf_signature", "birch"}
_ALLOWED_REPRESENTATIVES = {"first", "medoid", "centroid"}
_ALLOWED_ASSIGNMENTS = {"prototype", "nearest_prototype", "classifier"}


@dataclass(frozen=True)
class CompressionReport:
    """Human-readable diagnostics for a fitted ``PrototypeSampler``."""

    method: str
    n_samples: int
    n_prototypes: int
    compression_ratio: float
    min_weight: float
    max_weight: float
    mean_weight: float
    rare_bucket_count: int = 0
    reconstruction_error_mean: float | None = None
    reconstruction_error_max: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "n_prototypes": self.n_prototypes,
            "compression_ratio": self.compression_ratio,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "mean_weight": self.mean_weight,
            "rare_bucket_count": self.rare_bucket_count,
            "reconstruction_error_mean": self.reconstruction_error_mean,
            "reconstruction_error_max": self.reconstruction_error_max,
        }


class PrototypeSampler(BaseEstimator, TransformerMixin):
    """Build weighted prototypes for large tabular clustering tasks.

    Parameters
    ----------
    method : {"leaf_signature", "birch"}, default="leaf_signature"
        ``"leaf_signature"`` uses the library's random-partition embedding and
        groups rows with identical coarse signatures.  It works with mixed-type
        tabular data.  ``"birch"`` uses sklearn's BIRCH on a numeric encoded
        feature space and is best for dense numeric data.
    n_prototypes : int, "auto" or None, default="auto"
        Target maximum number of prototypes.  ``"auto"`` derives it from
        ``compression``.  ``None`` disables the target and keeps all buckets.
    compression : float or None, default=0.2
        Target retained fraction when ``n_prototypes="auto"``.  ``0.2`` means
        roughly at most 20% of the original row count, subject to rare-bucket
        preservation and natural bucket structure.
    representative : {"first", "medoid", "centroid"}, default="medoid"
        How to choose a prototype row for every bucket.  ``"centroid"`` returns
        numeric centroids for BIRCH; for mixed leaf signatures it safely falls
        back to medoids/first rows because categorical centroids are not valid
        original observations.
    preserve_rare : bool, default=True
        If true, buckets smaller than ``rare_bucket_min_size`` are kept as
        individual rows rather than collapsed.  This protects rare clusters and
        anomalies from being erased by compression.
    rare_bucket_min_size : int, default=3
        Buckets below this size are considered rare.
    random_state : int or None, default=None
        Reproducibility seed.

    Leaf-signature parameters
    -------------------------
    n_partitions : int, default=128
        Number of random partitions used to build signatures.
    n_features, n_bins, feature_types, cat_threshold, cut_strategy,
    add_missing_indicators, rare_category_min_count, rare_category_min_freq,
    coerce_numeric_strings, numeric_string_min_fraction
        Forwarded to ``ForestTransformer``.
    signature_depth : {"auto", "full"} or int, default="auto"
        Number of partition columns used for bucketing.  ``"auto"`` searches
        for a prefix length that reaches the prototype budget without using an
        overly coarse single-column signature.

    BIRCH parameters
    ----------------
    birch_threshold : float, default=0.5
        Radius threshold forwarded to sklearn ``Birch``.
    birch_branching_factor : int, default=50
        Branching factor forwarded to sklearn ``Birch``.

    Attributes
    ----------
    prototype_indices_ : ndarray of shape (n_prototypes,)
        Indices of representative training rows.  For centroid-only BIRCH this
        is the nearest original row to every centroid.
    sample_weight_ : ndarray of shape (n_prototypes,)
        Number of original samples represented by each prototype.
    inverse_assignment_ : ndarray of shape (n_samples,)
        Prototype index for each original training row.
    compression_report_ : dict
        Diagnostics about compression and reconstruction.
    """

    def __init__(
        self,
        method: str = "leaf_signature",
        n_prototypes: int | str | None = "auto",
        compression: float | None = 0.2,
        representative: str = "medoid",
        preserve_rare: bool = True,
        rare_bucket_min_size: int = 3,
        random_state: int | None = None,
        n_partitions: int = 128,
        n_features="sqrt",
        n_bins="auto",
        signature_depth: str | int = "auto",
        feature_types: dict | None = None,
        cat_threshold: int = 10,
        cut_strategy: str = "uniform",
        quantile_cuts: bool = False,
        n_jobs: int = -1,
        add_missing_indicators: bool = True,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = True,
        numeric_string_min_fraction: float = 0.90,
        birch_threshold: float = 0.5,
        birch_branching_factor: int = 50,
    ):
        self.method = method
        self.n_prototypes = n_prototypes
        self.compression = compression
        self.representative = representative
        self.preserve_rare = preserve_rare
        self.rare_bucket_min_size = rare_bucket_min_size
        self.random_state = random_state
        self.n_partitions = n_partitions
        self.n_features = n_features
        self.n_bins = n_bins
        self.signature_depth = signature_depth
        self.feature_types = feature_types
        self.cat_threshold = cat_threshold
        self.cut_strategy = cut_strategy
        self.quantile_cuts = quantile_cuts
        self.n_jobs = n_jobs
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction
        self.birch_threshold = birch_threshold
        self.birch_branching_factor = birch_branching_factor

    def fit(self, X, y=None):
        X_df = to_frame(X)
        n = len(X_df)
        if n < 1:
            raise ValueError("X must contain at least one sample")
        self._validate_params(n)
        self.input_columns_ = list(X_df.columns)
        self._is_dataframe_input_ = isinstance(X, pd.DataFrame)

        if self.method == "leaf_signature":
            self._fit_leaf_signature(X_df)
        elif self.method == "birch":
            self._fit_birch(X_df)
        else:  # pragma: no cover, validated above
            raise ValueError(f"Unsupported method: {self.method!r}")

        self.n_features_in_ = X_df.shape[1]
        return self

    def fit_resample(self, X, y=None):
        """Fit the sampler and return ``(X_prototypes, sample_weight)``."""
        self.fit(X, y=y)
        return self.X_resampled_, self.sample_weight_.copy()

    def transform(self, X):
        """Assign rows in ``X`` to fitted prototypes.

        Returns
        -------
        assignment : ndarray of shape (n_samples,)
            Index of the nearest/exact prototype for every input row.
        """
        check_is_fitted(self, "prototype_indices_")
        X_df = self._check_columns(X)
        if self.method == "leaf_signature":
            E = self.transformer_.transform(X_df)
            return self._assign_signatures(E)
        Z = self.preprocessor_.transform(X_df)
        return self._assign_feature_matrix(Z)

    def expand_labels(self, prototype_labels):
        """Expand prototype labels back to the training rows."""
        check_is_fitted(self, "inverse_assignment_")
        labels = np.asarray(prototype_labels)
        if labels.shape[0] != len(self.prototype_indices_):
            raise ValueError(
                "prototype_labels must have length equal to n_prototypes_ "
                f"({len(self.prototype_indices_)})"
            )
        return labels[self.inverse_assignment_]

    def assign_labels(self, X, prototype_labels):
        """Assign labels to new rows through the nearest fitted prototype."""
        assignment = self.transform(X)
        labels = np.asarray(prototype_labels)
        if labels.shape[0] != len(self.prototype_indices_):
            raise ValueError("prototype_labels length does not match fitted prototypes")
        return labels[assignment]

    def compression_summary(self) -> dict[str, Any]:
        """Return a copy of the fitted compression diagnostics."""
        check_is_fitted(self, "compression_report_")
        return dict(self.compression_report_)

    def plot_compression(self, ax=None):
        """Bar chart comparing original rows and retained prototypes."""
        check_is_fitted(self, "compression_report_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        report = self.compression_report_
        ax.bar(["Original", "Prototypes"], [report["n_samples"], report["n_prototypes"]])
        ax.set_ylabel("Rows")
        ax.set_title("Prototype compression")
        return ax

    def plot_prototype_weights(self, ax=None, bins: int = 30):
        """Histogram of how many original rows every prototype represents."""
        check_is_fitted(self, "sample_weight_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))
        ax.hist(self.sample_weight_, bins=min(int(bins), max(1, len(self.sample_weight_))))
        ax.set_xlabel("Prototype weight")
        ax.set_ylabel("Count")
        ax.set_title("Prototype weight distribution")
        return ax

    def plot_reconstruction_error(self, ax=None, bins: int = 30):
        """Histogram of feature-space reconstruction distances when available."""
        check_is_fitted(self, "reconstruction_distances_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))
        ax.hist(self.reconstruction_distances_, bins=min(int(bins), max(1, len(self.reconstruction_distances_))))
        ax.set_xlabel("Distance to assigned prototype")
        ax.set_ylabel("Count")
        ax.set_title("Reconstruction error")
        return ax

    # ------------------------------------------------------------------
    # Fitting implementations
    # ------------------------------------------------------------------

    def _fit_leaf_signature(self, X_df: pd.DataFrame):
        self.transformer_ = ForestTransformer(
            n_iterations=int(self.n_partitions),
            n_features=self.n_features,
            n_bins=self.n_bins,
            corr_threshold=None,
            feature_types=self.feature_types,
            cat_threshold=self.cat_threshold,
            quantile_cuts=self.quantile_cuts,
            cut_strategy=self.cut_strategy,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        ).fit(X_df)
        E = self.transformer_.get_embedding()
        self.full_signature_ = E
        depth = self._choose_signature_depth(E)
        self.signature_depth_ = depth
        E_key = E[:, :depth]
        self.prototype_signatures_ = None  # filled after groups are built
        self._build_groups_from_keys(X_df, E_key, feature_matrix=E.astype(float))
        self.prototype_signatures_ = E[self.prototype_indices_, :]
        self.reconstruction_distances_ = self._signature_reconstruction_distances(E)
        self._finalize_report(reconstruction_distances=self.reconstruction_distances_)

    def _fit_birch(self, X_df: pd.DataFrame):
        if self.representative == "centroid" and not all(pd.api.types.is_numeric_dtype(X_df[c]) for c in X_df.columns):
            warnings.warn("representative='centroid' on mixed data returns encoded centroids, not original rows; using medoid rows instead", RuntimeWarning)
        self.preprocessor_ = build_tree_preprocessor(
            X_df,
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        )
        Z = self.preprocessor_.fit_transform(X_df)
        if hasattr(Z, "toarray"):
            Z_dense = Z.toarray()
        else:
            Z_dense = np.asarray(Z)
        self.feature_matrix_ = Z_dense
        model = Birch(
            threshold=float(self.birch_threshold),
            branching_factor=int(self.birch_branching_factor),
            n_clusters=None,
        )
        model.fit(Z_dense)
        self.birch_ = model
        labels = np.asarray(model.labels_, dtype=np.int64)
        unique = np.unique(labels)
        if len(unique) == 0:
            labels = np.zeros(len(X_df), dtype=np.int64)
            unique = np.array([0])
        # If BIRCH creates too many subclusters, merge by clustering its own
        # centers with a second BIRCH using a larger effective threshold.  This
        # keeps the public target n_prototypes meaningful without inventing a
        # lossy random sample.
        target = self._target_n_prototypes(len(X_df))
        if target is not None and len(unique) > target:
            # Greedy merge to nearest selected centers.  Deterministic and cheap.
            centers = np.vstack([Z_dense[labels == u].mean(axis=0) for u in unique])
            chosen = self._farthest_first_indices(centers, target)
            d = pairwise_distances(centers, centers[chosen], metric="euclidean")
            remap = {u: int(np.argmin(d[i])) for i, u in enumerate(unique)}
            labels = np.asarray([remap[v] for v in labels], dtype=np.int64)
        keys = labels.reshape(-1, 1)
        self._build_groups_from_keys(X_df, keys, feature_matrix=Z_dense)
        self.prototype_features_ = Z_dense[self.prototype_indices_]
        self.reconstruction_distances_ = self._feature_reconstruction_distances(Z_dense)
        self._finalize_report(reconstruction_distances=self.reconstruction_distances_)

    # ------------------------------------------------------------------
    # Grouping / representative selection
    # ------------------------------------------------------------------

    def _build_groups_from_keys(self, X_df: pd.DataFrame, keys: np.ndarray, feature_matrix=None):
        keys = np.asarray(keys)
        if keys.ndim == 1:
            keys = keys.reshape(-1, 1)
        groups: dict[tuple[Any, ...], list[int]] = {}
        for i, row in enumerate(keys):
            groups.setdefault(tuple(row.tolist()), []).append(i)

        proto_indices: list[int] = []
        inverse = np.empty(len(X_df), dtype=np.int64)
        weights: list[float] = []
        proto_to_indices: list[np.ndarray] = []
        rare_count = 0

        for _, indices in groups.items():
            idx_arr = np.asarray(indices, dtype=np.int64)
            if self.preserve_rare and len(idx_arr) < int(self.rare_bucket_min_size):
                rare_count += 1
                for idx in idx_arr:
                    proto_id = len(proto_indices)
                    proto_indices.append(int(idx))
                    inverse[idx] = proto_id
                    weights.append(1.0)
                    proto_to_indices.append(np.array([idx], dtype=np.int64))
                continue

            proto_id = len(proto_indices)
            rep = self._representative_index(idx_arr, feature_matrix)
            proto_indices.append(int(rep))
            inverse[idx_arr] = proto_id
            weights.append(float(len(idx_arr)))
            proto_to_indices.append(idx_arr)

        self.prototype_indices_ = np.asarray(proto_indices, dtype=np.int64)
        self.inverse_assignment_ = inverse
        self.sample_weight_ = np.asarray(weights, dtype=np.float64)
        self.prototype_to_indices_ = proto_to_indices
        self.rare_bucket_count_ = int(rare_count)
        self.X_resampled_ = self._take_rows(X_df, self.prototype_indices_)
        self.n_prototypes_ = len(self.prototype_indices_)

    def _representative_index(self, idx_arr: np.ndarray, feature_matrix=None) -> int:
        if len(idx_arr) == 1 or self.representative == "first" or feature_matrix is None:
            return int(idx_arr[0])
        if self.representative == "centroid" and self.method == "birch":
            # Return nearest row to centroid so that X_resampled keeps the same
            # schema as X.  True encoded centroids are exposed only internally.
            pass
        Z = np.asarray(feature_matrix[idx_arr], dtype=np.float64)
        if Z.ndim != 2 or Z.shape[0] == 1:
            return int(idx_arr[0])
        center = Z.mean(axis=0)
        d = np.sum((Z - center) ** 2, axis=1)
        return int(idx_arr[int(np.argmin(d))])

    def _take_rows(self, X_df: pd.DataFrame, indices: np.ndarray):
        out = X_df.iloc[indices].copy()
        out.index = pd.RangeIndex(len(out))
        return out if self._is_dataframe_input_ else out.to_numpy()

    # ------------------------------------------------------------------
    # Assignment and diagnostics
    # ------------------------------------------------------------------

    def _assign_signatures(self, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E)
        P = np.asarray(self.prototype_signatures_)
        # Exact full-signature lookup first for training-like rows.
        lookup = {tuple(row.tolist()): i for i, row in enumerate(P)}
        out = np.empty(E.shape[0], dtype=np.int64)
        miss = []
        for i, row in enumerate(E):
            key = tuple(row.tolist())
            j = lookup.get(key)
            if j is None:
                miss.append(i)
            else:
                out[i] = j
        if miss:
            d = pairwise_distances(E[miss].astype(float), P.astype(float), metric="hamming")
            out[np.asarray(miss, dtype=np.int64)] = np.argmin(d, axis=1)
        return out

    def _assign_feature_matrix(self, Z) -> np.ndarray:
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        Z = np.asarray(Z, dtype=np.float64)
        d = pairwise_distances(Z, np.asarray(self.prototype_features_, dtype=np.float64), metric="euclidean")
        return np.argmin(d, axis=1).astype(np.int64)

    def _signature_reconstruction_distances(self, E: np.ndarray) -> np.ndarray:
        P = E[self.prototype_indices_]
        assigned = P[self.inverse_assignment_]
        return np.mean(E != assigned, axis=1).astype(float)

    def _feature_reconstruction_distances(self, Z: np.ndarray) -> np.ndarray:
        P = Z[self.prototype_indices_]
        assigned = P[self.inverse_assignment_]
        return np.sqrt(np.sum((Z - assigned) ** 2, axis=1)).astype(float)

    def _finalize_report(self, reconstruction_distances=None):
        n = len(self.inverse_assignment_)
        p = len(self.prototype_indices_)
        rec_mean = None
        rec_max = None
        if reconstruction_distances is not None and len(reconstruction_distances):
            rec_mean = float(np.mean(reconstruction_distances))
            rec_max = float(np.max(reconstruction_distances))
        report = CompressionReport(
            method=self.method,
            n_samples=int(n),
            n_prototypes=int(p),
            compression_ratio=float(p / max(n, 1)),
            min_weight=float(np.min(self.sample_weight_)) if p else 0.0,
            max_weight=float(np.max(self.sample_weight_)) if p else 0.0,
            mean_weight=float(np.mean(self.sample_weight_)) if p else 0.0,
            rare_bucket_count=int(getattr(self, "rare_bucket_count_", 0)),
            reconstruction_error_mean=rec_mean,
            reconstruction_error_max=rec_max,
        )
        self.compression_report_ = report.as_dict()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _validate_params(self, n: int):
        if self.method not in _ALLOWED_METHODS:
            raise ValueError(f"method must be one of {sorted(_ALLOWED_METHODS)}, got {self.method!r}")
        if self.representative not in _ALLOWED_REPRESENTATIVES:
            raise ValueError(f"representative must be one of {sorted(_ALLOWED_REPRESENTATIVES)}")
        if self.compression is not None and not (0 < float(self.compression) <= 1):
            raise ValueError("compression must be in (0, 1] or None")
        if int(self.rare_bucket_min_size) < 1:
            raise ValueError("rare_bucket_min_size must be >= 1")
        target = self._target_n_prototypes(n)
        if target is not None and target < 1:
            raise ValueError("n_prototypes target must be >= 1")

    def _target_n_prototypes(self, n: int) -> int | None:
        if self.n_prototypes is None:
            return None
        if self.n_prototypes == "auto":
            if self.compression is None:
                return None
            return max(1, min(n, int(np.ceil(float(self.compression) * n))))
        return max(1, min(n, int(self.n_prototypes)))

    def _choose_signature_depth(self, E: np.ndarray) -> int:
        L = E.shape[1]
        if self.signature_depth == "full":
            return L
        if isinstance(self.signature_depth, str) and self.signature_depth != "auto":
            raise ValueError("signature_depth must be 'auto', 'full', or an integer")
        if isinstance(self.signature_depth, int):
            return max(1, min(L, int(self.signature_depth)))
        target = self._target_n_prototypes(E.shape[0])
        if target is None:
            return L
        best_depth = L
        best_gap = float("inf")
        # Try a small deterministic grid of prefix lengths.  We prefer the
        # deepest signature that remains within budget, because deeper signatures
        # preserve more detail.  If none fits, choose the closest one.
        candidates = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, L]))
        candidates = [d for d in candidates if 1 <= d <= L]
        within = []
        for d in candidates:
            n_unique = np.unique(E[:, :d], axis=0).shape[0]
            gap = abs(n_unique - target)
            if n_unique <= target:
                within.append(d)
            if gap < best_gap:
                best_gap = gap
                best_depth = d
        return max(within) if within else best_depth

    def _check_columns(self, X) -> pd.DataFrame:
        X_df = to_frame(X)
        if list(X_df.columns) != list(self.input_columns_):
            raise ValueError(f"Column names do not match fit() columns. Expected {self.input_columns_}, got {list(X_df.columns)}")
        return X_df

    def _farthest_first_indices(self, centers: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(int(k), len(centers)))
        rng = np.random.default_rng(self.random_state)
        first = int(rng.integers(0, len(centers)))
        chosen = [first]
        min_dist = pairwise_distances(centers, centers[[first]], metric="euclidean").ravel()
        while len(chosen) < k:
            idx = int(np.argmax(min_dist))
            chosen.append(idx)
            d = pairwise_distances(centers, centers[[idx]], metric="euclidean").ravel()
            min_dist = np.minimum(min_dist, d)
        return np.asarray(chosen, dtype=np.int64)


class SubsampledClusterer(BaseEstimator, ClusterMixin):
    """Fit a clusterer on prototypes and expand labels to all training rows.

    Parameters
    ----------
    sampler : PrototypeSampler or None, default=None
        Sampler used to build prototypes.  ``None`` creates a conservative
        ``PrototypeSampler(method="leaf_signature")``.
    clusterer : estimator or None, default=None
        sklearn-style clusterer fitted on the prototypes.  The estimator should
        expose ``fit_predict`` or ``fit`` + ``labels_``.  ``None`` uses
        ``AutoTreeClusterer`` when available.
    assignment : {"prototype", "nearest_prototype", "classifier"}, default="nearest_prototype"
        How to label new rows in ``predict``.  ``"nearest_prototype"`` uses the
        sampler's own assignment logic.  ``"classifier"`` trains a classifier
        to mimic expanded labels if ``ClusterLabelClassifier`` is available.
        ``"prototype"`` is an alias for nearest prototype.
    """

    def __init__(self, sampler=None, clusterer=None, assignment: str = "nearest_prototype", random_state: int | None = None):
        self.sampler = sampler
        self.clusterer = clusterer
        self.assignment = assignment
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.assignment not in _ALLOWED_ASSIGNMENTS:
            raise ValueError(f"assignment must be one of {sorted(_ALLOWED_ASSIGNMENTS)}")
        self.sampler_ = clone(self.sampler) if self.sampler is not None else PrototypeSampler(random_state=self.random_state)
        X_proto, weights = self.sampler_.fit_resample(X)
        self.sample_weight_ = weights
        self.X_prototypes_ = X_proto

        if self.clusterer is None:
            from .auto import AutoTreeClusterer
            clusterer = AutoTreeClusterer(random_state=self.random_state, n_restarts=1, k_range=(2, 3, 4))
        else:
            clusterer = clone(self.clusterer)
        self.clusterer_ = clusterer
        try:
            labels_proto = clusterer.fit_predict(X_proto, sample_weight=weights)
        except TypeError:
            labels_proto = clusterer.fit_predict(X_proto) if hasattr(clusterer, "fit_predict") else None
        if labels_proto is None:
            try:
                clusterer.fit(X_proto, sample_weight=weights)
            except TypeError:
                clusterer.fit(X_proto)
            labels_proto = getattr(clusterer, "labels_")
        self.prototype_labels_ = np.asarray(labels_proto)
        self.labels_ = self.sampler_.expand_labels(self.prototype_labels_)
        self.compression_report_ = self.sampler_.compression_summary()

        if self.assignment == "classifier":
            from .explain import ClusterLabelClassifier
            self.assignment_model_ = ClusterLabelClassifier(
                clusterer=None,
                random_state=self.random_state,
            ).fit(X, y=self.labels_)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def predict(self, X):
        check_is_fitted(self, "prototype_labels_")
        if self.assignment == "classifier" and hasattr(self, "assignment_model_"):
            return self.assignment_model_.predict(X)
        return self.sampler_.assign_labels(X, self.prototype_labels_)

    def compression_summary(self) -> dict[str, Any]:
        check_is_fitted(self, "compression_report_")
        return dict(self.compression_report_)

    def plot_prototypes(self, ax=None):
        """Plot sampler compression chart."""
        check_is_fitted(self, "sampler_")
        return self.sampler_.plot_compression(ax=ax)

    def plot_prototype_weights(self, ax=None, bins: int = 30):
        check_is_fitted(self, "sampler_")
        return self.sampler_.plot_prototype_weights(ax=ax, bins=bins)


class _FixedLabelClusterer(BaseEstimator, ClusterMixin):
    """Tiny internal clusterer that returns precomputed labels."""

    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def fit(self, X, y=None):
        if len(X) != len(self.labels):
            raise ValueError("Fixed labels length does not match X")
        self.labels_ = np.asarray(self.labels)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_
