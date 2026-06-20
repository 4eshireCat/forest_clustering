import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.utils.validation import check_is_fitted

from .feature_encoder import DataEncoder
from .correlation import compute_feature_weights
from .partitioner import build_col_stats, build_iteration_specs, compute_embedding, apply_categorical_bin_cap
from .distance import pairwise_hamming, pairwise_hamming_chunked, cross_hamming
from .weighted_distance import pairwise_weighted_hamming, weighted_cross_hamming


# Below this many samples the graph clusterers use the exact dense distance
# matrix (cheap at this scale, and reproduces classic tie-breaking exactly);
# above it they switch to the matrix-free LSH-banding kNN graph.
_DENSE_GRAPH_MAX_N = 12000


def _resolve_n_features(n_features, d: int) -> int:
    if n_features == "sqrt":
        return max(1, int(np.ceil(np.sqrt(d))))
    if n_features == "log2":
        return max(1, int(np.ceil(np.log2(max(d, 2)))))
    if isinstance(n_features, float) and 0 < n_features <= 1.0:
        return max(1, int(np.ceil(n_features * d)))
    return max(1, int(n_features))


class ForestClusterer(BaseEstimator, ClusterMixin):
    """Clustering via random-partition similarity embeddings.

    Parameters
    ----------
    n_iterations : int
        Number of random partitioning iterations (L). More → more stable embeddings.
    n_features : int | float | "sqrt" | "log2"
        Features selected per iteration. Float = fraction, "sqrt" = ceil(sqrt(d)).
    n_bins : int
        Number of bins per feature per iteration (K).
    clusterer : estimator, str, or None
        Downstream clustering algorithm.

        - An sklearn-compatible estimator supporting ``fit_predict``.  Centroid
          estimators (KMeans / MiniBatchKMeans / Birch) cluster on a sparse
          weighted one-hot feature matrix (no distance matrix).  If
          ``metric='precomputed'`` it receives the pairwise distance matrix; with
          ``metric='hamming'`` or unset it receives the ``(n, L)`` embedding.
        - The string ``'louvain'`` or ``'leiden'`` selects graph community
          detection on a sparse kNN graph built directly from the embedding
          (no dense ``O(n^2)`` matrix).  Optional ``':'`` parameters are
          supported, e.g. ``'leiden:k=20,resolution=1.5'`` (``k`` = neighbours,
          ``resolution``/``gamma`` = resolution).
        - Default: ``KMeans(n_clusters=3)`` when neither ``clusterer`` nor
          ``n_clusters`` is specified.
    corr_threshold : float or None
        Spearman |corr| threshold for grouping correlated features (1/G weighting).
        None disables correlation-based weighting.
    corr_sample_size : int
        Number of rows to sample when computing feature correlations.
    feature_types : dict or None
        Override detected feature types: {col_name_or_idx: "numerical"|"categorical"}.
    cat_threshold : int
        Numerical columns with ≤ this many unique values are treated as categorical.
    quantile_cuts : bool
        If True, cut-points for numerical features are sampled from empirical quantiles
        instead of uniform [min, max].
    n_jobs : int
        Parallelism for embedding computation (passed to joblib).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_iterations: int = 200,
        n_features="sqrt",
        n_bins: int = 3,
        n_clusters: int | None = None,
        clusterer=None,
        corr_threshold: float | None = 0.7,
        corr_sample_size: int = 10_000,
        feature_types: dict | None = None,
        cat_threshold: int = 10,
        quantile_cuts: bool = False,
        cut_strategy: str = "uniform",
        kde_params: dict | None = None,
        n_jobs: int = -1,
        random_state: int | None = None,
        iteration_weighting: str = "uniform",
        weight_temperature: float = 1.0,
        partial_fit_strategy: str = "drift",
        partial_fit_rebuild_threshold: float = 0.3,
        partial_fit_max_samples: int = 100_000,
        partial_fit_drift_threshold: float = 0.3,
        adaptive_bins: bool = False,
        min_bins: int = 2,
        max_bins: int = 10,
        correlation_aware: bool = False,
        corr_group_threshold: float = 0.7,
        compute_importance: bool = False,
        importance_repeats: int = 5,
        importance_metric: str = 'silhouette',
        auto_feature_types: str = 'naive',  # 'naive' | 'smart'
        contrastive: bool = False,
    ):
        self.n_iterations = n_iterations
        self.n_features = n_features
        self.n_bins = n_bins
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        self.n_clusters = n_clusters
        self.clusterer = clusterer
        if self.n_clusters is not None and self.clusterer is not None:
            raise ValueError(
                f"Both n_clusters={self.n_clusters} and clusterer={self.clusterer!r} "
                f"were provided. n_clusters is ignored; the explicit clusterer is used. "
                f"To use n_clusters, set clusterer=None."
            )
        self.corr_threshold = corr_threshold
        self.corr_sample_size = corr_sample_size
        self.feature_types = feature_types
        self.cat_threshold = cat_threshold
        self.quantile_cuts = quantile_cuts
        self.cut_strategy = cut_strategy
        self.kde_params = kde_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.iteration_weighting = iteration_weighting
        self.weight_temperature = weight_temperature
        self.partial_fit_strategy = partial_fit_strategy
        self.partial_fit_rebuild_threshold = partial_fit_rebuild_threshold
        self.partial_fit_max_samples = partial_fit_max_samples
        self.partial_fit_drift_threshold = partial_fit_drift_threshold
        self.adaptive_bins = adaptive_bins
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.correlation_aware = correlation_aware
        self.corr_group_threshold = corr_group_threshold
        self.compute_importance = compute_importance
        self.importance_repeats = importance_repeats
        self.importance_metric = importance_metric
        self.auto_feature_types = auto_feature_types
        self.contrastive = contrastive

    # ------------------------------------------------------------------
    # Core sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the model: build the random-partition embedding and cluster it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Mixed-type tabular data (numerical and/or categorical).
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : ForestClusterer
            Fitted estimator with ``embedding_`` and ``labels_`` set.
        """
        rng = np.random.default_rng(self.random_state)

        if self.cut_strategy not in ("uniform", "quantile", "kde_peaks"):
            raise ValueError(f"Invalid cut_strategy: {self.cut_strategy!r}")

        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")

        if self.feature_types is None:
            detected = DataEncoder.detect_feature_types(
                X, strategy=self.auto_feature_types
            )
            self.encoder_ = DataEncoder(detected, self.cat_threshold)
        else:
            self.encoder_ = DataEncoder(self.feature_types, self.cat_threshold)
        X_enc = self.encoder_.fit_transform(X)
        n, d = X_enc.shape

        n_feat = _resolve_n_features(self.n_features, d)

        # Feature weights from correlation
        if self.corr_threshold is not None and d > 1:
            self.feature_weights_ = compute_feature_weights(
                X_enc,
                threshold=self.corr_threshold,
                sample_size=self.corr_sample_size,
                rng=rng,
            )
        else:
            self.feature_weights_ = np.ones(d)

        # Column statistics for cut-point generation
        self.col_stats_ = build_col_stats(
            X_enc,
            self.encoder_.feature_types_,
            quantile_cuts=self.quantile_cuts,
            cut_strategy=self.cut_strategy,
            kde_params=self.kde_params,
            rng=rng,
        )

        # Compute per-feature adaptive bins
        if self.adaptive_bins:
            from .adaptive_bins import compute_adaptive_bins
            self.adaptive_bins_map_ = compute_adaptive_bins(
                self.col_stats_, len(X_enc), self.min_bins, self.max_bins
            )
        else:
            # NEW: Apply categorical bin cap even when adaptive_bins=False
            self.adaptive_bins_map_ = apply_categorical_bin_cap(
                self.col_stats_,
                n_bins=self.n_bins,
                min_bins=self.min_bins,
                max_bins=self.max_bins,
                n=len(X_enc),
            )
            # NOTE: bins_map may be {} if no categoricals present.
            #       build_iteration_specs treats empty dict same as None.

        # Compute correlation groups for correlation-aware selection
        self.correlation_groups_ = None
        if self.correlation_aware:
            from .correlation_aware import build_correlation_groups

            num_mask = np.array(
                [ft != "categorical" for ft in self.encoder_.feature_types_]
            )
            if num_mask.sum() >= 2:
                X_num = X_enc[:, num_mask].astype(np.float64)
                # Handle NaN
                col_medians = np.nanmedian(X_num, axis=0)
                X_num_clean = np.where(np.isfinite(X_num), X_num, col_medians)
                corr = np.corrcoef(X_num_clean.T)
                np.fill_diagonal(corr, 1.0)
                # Replace NaN with 0 (constant features)
                corr = np.nan_to_num(corr, nan=0.0)
                # Map back to full feature index space
                full_corr = np.eye(d)
                num_indices = np.where(num_mask)[0]
                for i_idx, i in enumerate(num_indices):
                    for j_idx, j in enumerate(num_indices):
                        full_corr[i, j] = corr[i_idx, j_idx]
                self.correlation_groups_ = build_correlation_groups(
                    self.feature_weights_, full_corr, self.corr_group_threshold
                )
            else:
                # Not enough numerical features -> singleton groups
                self.correlation_groups_ = [[j] for j in range(len(self.feature_weights_))]

        # Build all iteration specs
        self.specs_ = build_iteration_specs(
            n_iterations=self.n_iterations,
            col_stats=self.col_stats_,
            n_features_per_iter=n_feat,
            n_bins=self.n_bins,
            feature_weights=self.feature_weights_,
            rng=rng,
            cut_strategy=self.cut_strategy,
            kde_params=self.kde_params,
            X=X_enc,
            adaptive_bins_map=self.adaptive_bins_map_,
            correlation_groups=self.correlation_groups_,
            correlation_aware=self.correlation_aware,
        )

        # Compute training embedding
        if self.contrastive:
            from .contrastive_splits import fit_contrastive_tree
            self.contrastive_trees_ = []
            self.embedding_ = np.zeros((n, self.n_iterations), dtype=np.int64)
            for it in range(self.n_iterations):
                tree = fit_contrastive_tree(
                    X_enc, max_depth=max(3, self.n_bins),
                    n_pairs=min(20, max(2, n // 2)),
                    temperature=0.5,
                    random_state=(self.random_state + it) if self.random_state is not None else it,
                )
                self.contrastive_trees_.append(tree)
                self.embedding_[:, it] = tree.apply(X_enc)
        else:
            self.contrastive_trees_ = None
            self.embedding_ = compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)

        # CRIT-3: warn if temperature has no effect with uniform
        if self.iteration_weighting == "uniform" and abs(self.weight_temperature - 1.0) > 1e-12:
            import warnings
            warnings.warn(
                f"weight_temperature={self.weight_temperature} has no effect when "
                f"iteration_weighting='uniform' because all raw uniform weights are 1.0 "
                f"and 1.0**(1/T) == 1.0 for any T. Use iteration_weighting='entropy' or "
                f"'inverse_gini' to enable temperature scaling.",
                UserWarning,
                stacklevel=2,
            )

        # MED-3: warn if kde_peaks has no effect on categorical features
        if self.cut_strategy == 'kde_peaks':
            cat_features = [j for j, s in enumerate(self.col_stats_) if s['type'] == 'categorical']
            if cat_features:
                import warnings
                warnings.warn(
                    f"cut_strategy='kde_peaks' has no effect on categorical features "
                    f"(columns {cat_features}). Using uniform cuts for categorical.",
                    UserWarning,
                    stacklevel=2,
                )

        # Compute per-iteration weights
        if self.iteration_weighting != "uniform":
            from .iteration_weights import compute_iteration_weights
            self.iteration_weights_ = compute_iteration_weights(
                self.embedding_, self.iteration_weighting, self.weight_temperature
            )
        else:
            self.iteration_weights_ = np.ones(self.n_iterations, dtype=np.float64)

        # Reset online state
        self._total_samples_seen_ = n
        self._col_stats_accum_ = {}
        self._update_accumulated_stats_from_col_stats()
        self._n_rebuilds_ = 0

        # Store encoded data for potential rebuilds (skip for "never" strategy)
        if self.partial_fit_strategy != "never":
            self._all_seen_data_enc_ = X_enc
        else:
            self._all_seen_data_enc_ = None

        # Run clustering and store labels
        self.labels_ = self._run_clusterer(self.embedding_)

        # Compute permutation importance if requested
        if self.compute_importance:
            from .permutation_importance import compute_permutation_importance
            self._importance_df = compute_permutation_importance(
                self, X, self.importance_metric, self.importance_repeats, self.random_state
            )

        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit the model and return cluster labels for ``X``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Integer cluster labels (``-1`` denotes noise where applicable).
        """
        self.fit(X)
        return self.labels_

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return the (n, L) embedding matrix."""
        self.fit(X)
        return self.embedding_

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self):
        """Normalized permutation importance scores."""
        if not hasattr(self, '_importance_df') or self._importance_df is None:
            raise AttributeError("Importances not computed. Set compute_importance=True in __init__.")
        return self._importance_df['importance'].values

    def get_feature_importances(self, detailed=False):
        """Return importance DataFrame."""
        if getattr(self, '_importance_dirty_', False):
            raise AttributeError(
                "Importances were invalidated by partial_fit(). "
                "Recompute by calling fit(X_full) with compute_importance=True, "
                "or call fit() on the accumulated data."
            )
        if not hasattr(self, '_importance_df') or self._importance_df is None:
            raise AttributeError("Importances not computed. Set compute_importance=True in __init__.")
        if detailed:
            return self._importance_df.copy()
        cols = ['feature', 'importance']
        if 'raw_importance' in self._importance_df.columns:
            cols.append('raw_importance')
        return self._importance_df[cols].copy()

    # ------------------------------------------------------------------
    # Online / incremental fit
    # ------------------------------------------------------------------

    def partial_fit(self, X, y=None):
        """Incremental/online fit.

        First call: equivalent to fit(X).
        Subsequent calls: process X with existing specs,
        optionally rebuild if drift detected.

        Returns self.
        """
        # Validate strategy first (before any fit logic)
        if self.partial_fit_strategy not in ("drift", "periodic", "never"):
            raise ValueError(f"Unknown strategy: {self.partial_fit_strategy}")

        # Handle empty input
        if hasattr(X, '__len__') and len(X) == 0:
            raise ValueError("X must not be empty")

        # If not fitted yet → full fit
        if not hasattr(self, "encoder_"):
            return self.fit(X, y)

        n_new = len(X)

        # Track total samples seen
        self._total_samples_seen_ = getattr(self, '_total_samples_seen_', 0) + n_new

        # Encode new data with existing encoder
        X_enc = self.encoder_.transform(X)
        X_enc = np.asarray(X_enc)

        # Accumulate encoded data for potential rebuilds (skip for "never" strategy,
        # cap to partial_fit_max_samples for other strategies to prevent O(N^2) memory)
        if self.partial_fit_strategy != "never":
            if not hasattr(self, '_all_seen_data_enc_') or self._all_seen_data_enc_ is None:
                self._all_seen_data_enc_ = X_enc
            else:
                self._all_seen_data_enc_ = np.concatenate([self._all_seen_data_enc_, X_enc], axis=0)
                # Cap stored data to prevent unbounded growth
                if self._all_seen_data_enc_.shape[0] > self.partial_fit_max_samples:
                    self._all_seen_data_enc_ = self._all_seen_data_enc_[-self.partial_fit_max_samples:]

        # Detect drift
        drift_detected = self._detect_drift(X_enc)

        # Decide whether to rebuild
        should_rebuild = False
        if self.partial_fit_strategy == "drift":
            should_rebuild = drift_detected
        elif self.partial_fit_strategy == "periodic":
            should_rebuild = self._total_samples_seen_ >= self.partial_fit_max_samples
        # "never" → never rebuild

        # Compute embedding for new data (contrastive trees or specs)
        new_embedding = self._embed_encoded(X_enc)

        # Accumulate embedding
        if hasattr(self, "embedding_"):
            self.embedding_ = np.concatenate([self.embedding_, new_embedding], axis=0)
        else:
            self.embedding_ = new_embedding

        # Update col_stats with new data info
        if not hasattr(self, '_col_stats_accum_'):
            self._col_stats_accum_ = {}
        self._update_accumulated_stats(X_enc)

        # Rebuild if needed
        if should_rebuild:
            self._rebuild_specs(X_enc)
            if self.partial_fit_strategy == "periodic":
                self._total_samples_seen_ = 0  # reset for next interval

        # CRITICAL-2: Recompute labels on the accumulated embedding
        self.labels_ = self._run_clusterer(self.embedding_)

        # HIGH-2 / MED-2: Invalidate cached importances; can't recompute without all historical X
        if hasattr(self, '_importance_df'):
            self._importance_df = None
            self._importance_dirty_ = True

        return self

    def _detect_drift(self, X_enc):
        """Detect if new data has drifted from training distribution.
        Returns True if drift detected (exceeds threshold).
        """
        if not hasattr(self, '_col_stats_accum_'):
            return False

        n_features = X_enc.shape[1]
        drifted_features = 0

        for j in range(n_features):
            col = X_enc[:, j]
            col_finite = col[np.isfinite(col)]
            if len(col_finite) == 0:
                continue

            # Get reference stats
            ref = self._col_stats_accum_.get(j, {})
            if not ref:
                continue

            # Compute drift: max of mean-drift, range-drift, std-drift
            ref_mean = ref.get('mean', 0)
            ref_std = ref.get('std', 1)

            if ref_std < 1e-10:
                continue  # constant column

            new_mean = np.mean(col_finite)
            new_std = np.std(col_finite)

            mean_drift = abs(new_mean - ref_mean) / ref_std
            std_drift = abs(new_std - ref_std) / max(ref_std, 1e-10)

            feature_drift = max(mean_drift, std_drift)
            if feature_drift > self.partial_fit_drift_threshold:
                drifted_features += 1

        drift_fraction = drifted_features / max(n_features, 1)
        return drift_fraction >= self.partial_fit_rebuild_threshold

    def _update_accumulated_stats(self, X_enc):
        """Update accumulated statistics for drift detection."""
        n_features = X_enc.shape[1]
        for j in range(n_features):
            col = X_enc[:, j]
            col_finite = col[np.isfinite(col)]
            if len(col_finite) == 0:
                continue
            self._col_stats_accum_[j] = {
                'mean': np.mean(col_finite),
                'std': np.std(col_finite),
                'min': np.min(col_finite),
                'max': np.max(col_finite),
            }

    def _update_accumulated_stats_from_col_stats(self):
        """Populate _col_stats_accum_ from fitted col_stats."""
        if not hasattr(self, 'col_stats_') or self.col_stats_ is None:
            return
        for j, s in enumerate(self.col_stats_):
            if s['type'] == 'numerical':
                self._col_stats_accum_[j] = {
                    'mean': s.get('mean', (s['max'] + s['min']) / 2),
                    'std': s.get('std', max((s['max'] - s['min']) / 4, 1e-6)),
                    'min': s['min'],
                    'max': s['max'],
                }

    def _rebuild_specs(self, X_enc=None):
        """Rebuild iteration specs using updated statistics."""
        # Update col_stats_ from accumulated stats
        if hasattr(self, '_col_stats_accum_') and self._col_stats_accum_:
            for j, stats in self._col_stats_accum_.items():
                if j < len(self.col_stats_) and self.col_stats_[j]['type'] == 'numerical':
                    self.col_stats_[j]['min'] = stats['min']
                    self.col_stats_[j]['max'] = stats['max']

        # Use unique RNG seed per rebuild
        self._n_rebuilds_ = getattr(self, '_n_rebuilds_', 0) + 1
        rs = self.random_state if self.random_state is not None else 42
        rng = np.random.default_rng(rs + self._n_rebuilds_)
        n_feat = _resolve_n_features(self.n_features, self.encoder_.d_)

        # Recompute adaptive bins from accumulated data
        n_current = (
            self._all_seen_data_enc_.shape[0]
            if hasattr(self, '_all_seen_data_enc_') and self._all_seen_data_enc_ is not None
            else getattr(self, '_total_samples_seen_', 0)
        )
        if self.adaptive_bins and n_current > 0:
            from .adaptive_bins import compute_adaptive_bins
            self.adaptive_bins_map_ = compute_adaptive_bins(
                self.col_stats_, n_current, self.min_bins, self.max_bins
            )
        else:
            # NEW: Recompute categorical caps with potentially updated stats
            self.adaptive_bins_map_ = apply_categorical_bin_cap(
                self.col_stats_,
                n_bins=self.n_bins,
                min_bins=self.min_bins,
                max_bins=self.max_bins,
                n=max(n_current, 1),
            )

        # Build new specs
        self.specs_ = build_iteration_specs(
            n_iterations=self.n_iterations,
            col_stats=self.col_stats_,
            n_features_per_iter=n_feat,
            n_bins=self.n_bins,
            feature_weights=self.feature_weights_,
            rng=rng,
            cut_strategy=self.cut_strategy,
            kde_params=self.kde_params,
            X=self._all_seen_data_enc_ if X_enc is None else X_enc,
            adaptive_bins_map=self.adaptive_bins_map_,
            correlation_groups=self.correlation_groups_,
            correlation_aware=self.correlation_aware,
        )

        # Recompute ALL embeddings with new specs
        self._recompute_all_embeddings()

    def _recompute_all_embeddings(self):
        """Recompute embeddings for all seen data with current specs."""
        if not hasattr(self, '_all_seen_data_enc_') or self._all_seen_data_enc_ is None:
            return  # no stored data
        self.embedding_ = self._embed_encoded(self._all_seen_data_enc_)

    def get_drift_report(self):
        """Return drift status per feature."""
        if not hasattr(self, '_col_stats_accum_'):
            return {}
        report = {}
        for j, stats in self._col_stats_accum_.items():
            report[j] = {
                'mean': stats.get('mean', None),
                'std': stats.get('std', None),
                'drift_detected': False,  # simplified
            }
        return report

    # ------------------------------------------------------------------
    # Transform / distance
    # ------------------------------------------------------------------

    def _embed_encoded(self, X_enc) -> np.ndarray:
        """Embed already-encoded data, using the stored contrastive trees when
        contrastive=True (so the embedding is consistent with the one produced
        at fit time) and the partition specs otherwise."""
        X_enc = np.asarray(X_enc)
        if getattr(self, "contrastive_trees_", None):
            cols = [tree.apply(X_enc) for tree in self.contrastive_trees_]
            return np.column_stack(cols).astype(np.int64)
        return compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)

    def transform(self, X) -> np.ndarray:
        """Apply fitted partition specs / contrastive trees to new data.

        Returns (n, L) embedding consistent with the training embedding.
        """
        check_is_fitted(self, "specs_")
        X_enc = self.encoder_.transform(X)
        return self._embed_encoded(X_enc)

    def get_embedding(self) -> np.ndarray:
        """Return the fitted ``(n_samples, n_iterations)`` cell-id embedding."""
        check_is_fitted(self, "embedding_")
        return self.embedding_

    def pairwise_distance(
        self,
        X=None,
        Y=None,
        chunk_size: int = 2_000,
    ) -> np.ndarray:
        """Hamming distance matrix from embeddings.

        X=None → use training embedding.
        Y=None → square matrix D[i,j] = d(X[i], X[j]).
        X,Y provided → rectangular matrix D[i,j] = d(X[i], Y[j]).
        """
        check_is_fitted(self, "embedding_")

        E_X = self.embedding_ if X is None else self.transform(X)

        # Use weighted Hamming when weights differ from uniform
        use_weighted = (
            hasattr(self, "iteration_weights_")
            and not np.allclose(self.iteration_weights_, 1.0)
        )

        # Resolve effective weights; fall back to uniform if all weights
        # are (near-)zero to avoid "weights must not sum to zero" crashes.
        if use_weighted:
            weights = self.iteration_weights_
            if weights.sum() < 1e-15:
                weights = np.ones(self.n_iterations, dtype=np.float64)
        else:
            weights = np.ones(self.n_iterations, dtype=np.float64)

        if Y is not None:
            E_Y = self.transform(Y)
            if use_weighted:
                return weighted_cross_hamming(E_X, E_Y, weights)
            return cross_hamming(E_X, E_Y)

        n = E_X.shape[0]
        if use_weighted:
            return pairwise_weighted_hamming(E_X, weights)

        if n <= chunk_size:
            return pairwise_hamming(E_X)
        return pairwise_hamming_chunked(E_X, chunk_size=chunk_size)

    def get_iteration_weights(self):
        """Return the per-iteration weights computed during fit().

        Returns
        -------
        weights : np.ndarray, shape (L,), dtype float64
            Per-iteration weights with mean = 1.0.
        """
        check_is_fitted(self, "iteration_weights_")
        return self.iteration_weights_

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_silhouette_score(D, labels, metric='precomputed'):
        """Compute silhouette score, return NaN for single cluster."""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < 2:
            return np.nan
        try:
            return silhouette_score(D, labels, metric=metric)
        except ValueError:
            return np.nan

    def _embedding_as_weighted_features(self, E: np.ndarray, sparse_output: bool = False):
        """Map the (n, L) nominal cell-id embedding to weighted one-hot features.

        The cell ids produced per iteration are *nominal* labels (their integer
        magnitude is arbitrary), so feeding them straight into a Euclidean
        algorithm such as KMeans imposes a meaningless ordinal geometry and
        makes the result depend on the arbitrary label codes.  Instead we
        one-hot encode each iteration and scale block ``l`` by ``sqrt(w_l / 2)``.

        With weights normalised to ``sum_l w_l = 1`` this makes the *squared*
        Euclidean distance between two rows equal to the weighted Hamming
        distance used everywhere else::

            ||phi_i - phi_j||^2 = sum_l (w_l / 2) * 2 * [cell_il != cell_jl]
                                = sum_l w_l * [cell_il != cell_jl]

        so KMeans/Ward on these features optimise a weighted-Hamming-consistent
        objective and are invariant to relabelling of the cell ids.

        The matrix has exactly ``L`` non-zeros per row, so it is built as a sparse
        CSR matrix (``O(n * L)`` memory, independent of per-column cardinality).
        ``sparse_output=False`` densifies it for estimators that cannot consume
        sparse input.
        """
        from .sparse_features import weighted_onehot_features
        w = getattr(self, "iteration_weights_", None)
        return weighted_onehot_features(E, weights=w, sparse_output=sparse_output)

    def _run_graph_clusterer(self, clf, E: np.ndarray):
        """Run a graph community clusterer, choosing the graph builder by size.

        For ``n <= _DENSE_GRAPH_MAX_N`` the exact dense Hamming distance matrix is
        used (cheap at this scale and identical to the classic behaviour, with
        exact tie-breaking); above it the matrix-free LSH-banding kNN graph is
        used so the dense ``O(n^2)`` matrix is never materialised.

        Returns the fitted ``clf`` (with ``labels_`` set).
        """
        n = E.shape[0]
        if n <= _DENSE_GRAPH_MAX_N:
            D = self.pairwise_distance().astype(np.float64)
            clf.fit(D)
        else:
            clf.fit_embedding(E)
        return clf

    def _run_clusterer(self, E: np.ndarray) -> np.ndarray:
        clf = self.clusterer

        # Default clusterer: KMeans on the *weighted one-hot* features.  The raw
        # cell ids are nominal, so KMeans-on-cell-ids (the previous default) was
        # not invariant to relabelling and ignored iteration_weighting entirely.
        # One-hot encoding with per-iteration sqrt(w_l/2) scaling makes squared
        # Euclidean equal the weighted Hamming distance, so the default is now
        # relabel-invariant and weight-aware while keeping KMeans' balanced,
        # well-separated clusters (a precomputed average-linkage default tends to
        # chain into one giant cluster on the saturated Hamming matrix).
        if clf is None:
            from sklearn.cluster import KMeans
            n = E.shape[0]
            k = self.n_clusters if self.n_clusters is not None else 3
            k = max(1, min(int(k), n))
            if k <= 1 or n <= 1:
                return np.zeros(n, dtype=np.int64)
            clf = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)

        # Decide how a user-supplied estimator should consume the embedding.
        #   - "precomputed" / Louvain  -> weighted Hamming distance matrix
        #   - "hamming"                -> raw cell ids (Hamming is already the
        #                                 correct nominal metric, relabel-invariant)
        #   - anything Euclidean (KMeans, Ward, GMM, ...) -> weighted one-hot
        metric = getattr(clf, "metric", None) if not isinstance(clf, str) else None
        if isinstance(clf, str) or metric in ("precomputed", "hamming") or \
                hasattr(clf, "_is_louvain_clusterer"):
            E_input = E  # raw cell ids; precomputed/louvain branches recompute D
        else:
            # Centroid estimators (KMeans/MiniBatchKMeans/Birch) consume the
            # weighted one-hot features.  For large n use a sparse CSR matrix
            # (O(n*L) memory, no dense blow-up); for small n use the dense
            # matrix, which is cheap and reproduces the classic results exactly
            # (sklearn's sparse and dense KMeans code paths differ slightly).
            from .sparse_features import estimator_supports_sparse
            use_sparse = estimator_supports_sparse(clf) and E.shape[0] > _DENSE_GRAPH_MAX_N
            E_input = self._embedding_as_weighted_features(E, sparse_output=use_sparse)

        # Handle string shortcuts for graph clustering
        if isinstance(clf, str):
            if clf in ('louvain', 'leiden') or clf.startswith(('louvain:', 'leiden:')):
                from .graph_clustering import GraphLouvainClusterer
                method = clf.split(':', 1)[0]
                params = {'community_method': method}
                if ':' in clf:
                    param_str = clf.split(':', 1)[1]
                    for pair in param_str.split(','):
                        pair = pair.strip()
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            k = k.strip()
                            v = v.strip()
                            if k == 'k':
                                params['n_neighbors'] = int(v)
                            elif k in ('gamma', 'resolution'):
                                params['resolution'] = float(v)
                            else:
                                raise ValueError(f"Unknown {method} parameter: {k!r}. "
                                                 f"Supported: k, gamma, resolution. "
                                                 f"Got: {clf!r}")
                self._clusterer_instance = GraphLouvainClusterer(**params, random_state=self.random_state)
                # Small n: exact dense distance matrix (classic behaviour);
                # large n: matrix-free LSH-banding kNN graph.
                return self._run_graph_clusterer(self._clusterer_instance, E).labels_
            else:
                raise ValueError(f"Unknown clusterer string shortcut: {clf!r}")

        # Handle GraphLouvainClusterer or any clusterer that expects
        # a precomputed distance matrix
        if hasattr(clf, 'fit_predict'):
            # GraphLouvainClusterer works on a sparse kNN graph built directly
            # from the embedding — no dense O(n^2) distance matrix.
            if hasattr(clf, '_is_louvain_clusterer'):
                # Small n: exact dense distance matrix (classic behaviour);
                # large n: matrix-free LSH-banding kNN graph.
                return self._run_graph_clusterer(clf, E).labels_
            # For other clusterers, check if they want precomputed metric
            metric = getattr(clf, "metric", None)
            if metric == "precomputed":
                D = self.pairwise_distance().astype(np.float64)
                return clf.fit_predict(D)
            # Otherwise pass embedding directly
            labels = clf.fit_predict(E_input)
        elif hasattr(clf, 'fit'):
            clf.fit(E_input)
            labels = clf.labels_
        else:
            raise ValueError(f"clusterer must have fit_predict or fit method, got {type(clf).__name__}")

        # If DBSCAN finds only 1 cluster, try smaller eps values.
        # Retry on the raw cell ids with Hamming, the correct nominal metric.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters <= 1 and isinstance(clf, DBSCAN):
            for eps in [0.3, 0.2, 0.1]:
                clf_retry = DBSCAN(metric="hamming", eps=eps, n_jobs=self.n_jobs)
                labels = clf_retry.fit_predict(E)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters >= 2:
                    break

        return labels
