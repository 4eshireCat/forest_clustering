"""sklearn-compatible transformer that produces the random-partition embedding."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .clusterer import _resolve_n_features
from .feature_encoder import DataEncoder
from .correlation import compute_feature_weights
from .partitioner import build_col_stats, build_iteration_specs, compute_embedding


class ForestTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that produces the random-partition embedding.

    Parameters mirror ForestClusterer.

    Parameters
    ----------
    n_iterations : int
        Number of random partitioning iterations (L).
    n_features : int | float | "sqrt" | "log2"
        Features selected per iteration.
    n_bins : int
        Number of bins per feature per iteration.
    corr_threshold : float or None
        Spearman |corr| threshold for grouping correlated features.
    corr_sample_size : int
        Number of rows to sample when computing feature correlations.
    feature_types : dict or None
        Override detected feature types.
    cat_threshold : int
        Numerical columns with <= this many unique values treated as categorical.
    quantile_cuts : bool
        If True, cut-points are sampled from empirical quantiles.
    n_jobs : int
        Parallelism for embedding computation.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_iterations: int = 200,
        n_features="sqrt",
        n_bins: int = 3,
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
    ):
        self.n_iterations = n_iterations
        self.n_features = n_features
        self.n_bins = n_bins
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

    def fit(self, X, y=None):
        """Fit the transformer: encode data, build specs, compute embedding.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n, d)
            Input data.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)

        if self.cut_strategy not in ("uniform", "quantile", "kde_peaks"):
            raise ValueError(f"Invalid cut_strategy: {self.cut_strategy!r}")

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
            cut_strategy=self.cut_strategy,
            kde_params=self.kde_params,
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
            cut_strategy=self.cut_strategy,
            kde_params=self.kde_params,
            X=X_enc,
        )

        # Compute training embedding
        self.embedding_ = compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)

        # Compute per-iteration weights if non-uniform weighting requested
        if self.iteration_weighting != "uniform":
            from .iteration_weights import compute_iteration_weights
            self.iteration_weights_ = compute_iteration_weights(
                self.embedding_, self.iteration_weighting, self.weight_temperature
            )
        else:
            self.iteration_weights_ = np.ones(self.n_iterations, dtype=np.float64)

        return self

    def transform(self, X):
        """Apply fitted partition specs to new data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_new, d)
            New input data.

        Returns
        -------
        E : np.ndarray, shape (n_new, L), dtype int64
            Embedding matrix.
        """
        check_is_fitted(self, "specs_")
        X_enc = self.encoder_.transform(X)
        return compute_embedding(X_enc, self.specs_, n_jobs=self.n_jobs)

    def get_embedding(self):
        """Return the fitted embedding.

        Returns
        -------
        E : np.ndarray, shape (n, L), dtype int64
            The embedding computed during fit().
        """
        check_is_fitted(self, "embedding_")
        return self.embedding_
