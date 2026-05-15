import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.utils.validation import check_is_fitted

from .feature_encoder import DataEncoder
from .correlation import compute_feature_weights
from .partitioner import build_col_stats, build_iteration_specs, compute_embedding
from .distance import pairwise_hamming, pairwise_hamming_chunked, cross_hamming


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
    clusterer : sklearn-compatible estimator or None
        Downstream clustering algorithm. Must support fit_predict().
        If metric="precomputed", receives the pairwise distance matrix.
        If metric="hamming" or not set, receives the (n, L) embedding directly.
        Default: DBSCAN(metric="hamming").
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
        clusterer=None,
        corr_threshold: float | None = 0.7,
        corr_sample_size: int = 10_000,
        feature_types: dict | None = None,
        cat_threshold: int = 10,
        quantile_cuts: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        self.n_iterations = n_iterations
        self.n_features = n_features
        self.n_bins = n_bins
        self.clusterer = clusterer
        self.corr_threshold = corr_threshold
        self.corr_sample_size = corr_sample_size
        self.feature_types = feature_types
        self.cat_threshold = cat_threshold
        self.quantile_cuts = quantile_cuts
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Core sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)

        self.encoder_ = DataEncoder(
            feature_types_override=self.feature_types,
            cat_threshold=self.cat_threshold,
        )
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
            rng=rng,
        )

        # Build all iteration specs
        self.specs_ = build_iteration_specs(
            n_iterations=self.n_iterations,
            col_stats=self.col_stats_,
            n_features_per_iter=n_feat,
            n_bins=self.n_bins,
            feature_weights=self.feature_weights_,
            rng=rng,
        )

        # Compute training embedding
        self.embedding_ = compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X)
        return self._run_clusterer(self.embedding_)

    # ------------------------------------------------------------------
    # Transform / distance
    # ------------------------------------------------------------------

    def transform(self, X) -> np.ndarray:
        """Apply fitted partition specs to new data. Returns (n, L) embedding."""
        check_is_fitted(self, "specs_")
        X_enc = self.encoder_.transform(X)
        return compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)

    def get_embedding(self) -> np.ndarray:
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

        if Y is not None:
            E_Y = self.transform(Y)
            return cross_hamming(E_X, E_Y)

        n = E_X.shape[0]
        if n <= chunk_size:
            return pairwise_hamming(E_X)
        return pairwise_hamming_chunked(E_X, chunk_size=chunk_size)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_clusterer(self, E: np.ndarray) -> np.ndarray:
        clf = self.clusterer
        if clf is None:
            clf = DBSCAN(metric="hamming", n_jobs=self.n_jobs)

        metric = getattr(clf, "metric", None)
        if metric == "precomputed":
            D = self.pairwise_distance().astype(np.float64)
            return clf.fit_predict(D)

        # Pass embedding directly; works for DBSCAN(metric='hamming'),
        # HDBSCAN, KMeans, Agglomerative, etc.
        return clf.fit_predict(E)
