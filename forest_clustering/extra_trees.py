"""ExtraTrees-based unsupervised proximity clustering.

This estimator is the extremely-randomized-tree sibling of the Breiman-style
unsupervised random forest estimator: it learns to separate observed samples
from a synthetic null sample, then clusters observed samples by same-leaf
co-occurrence across ExtraTrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._tree_common import (
    build_tree_preprocessor,
    is_numeric_series,
    leaf_cross_proximity,
    leaf_proximity,
    make_leaf_encoder,
    run_leaf_clusterer,
    to_frame,
)


class ExtraTreesProximityClusterer(BaseEstimator, ClusterMixin):
    """Unsupervised proximity clustering with ExtraTrees leaves.

    The model creates a synthetic null sample, trains an ``ExtraTreesClassifier``
    to distinguish real rows from synthetic rows, converts real rows to leaf ids,
    and defines proximity as the fraction of trees in which two samples land in
    the same leaf.  Non-precomputed downstream clusterers receive a one-hot leaf
    embedding, because raw leaf-id integers are arbitrary labels.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_clusters: int | None = 3,
        synthetic: str = "permute_marginals",
        clusterer=None,
        max_depth=None,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap: bool = False,
        n_jobs=None,
        random_state: int | None = None,
        cluster_input: str = "auto",
        add_missing_indicators: bool = False,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.n_estimators = n_estimators
        self.n_clusters = n_clusters
        self.synthetic = synthetic
        self.clusterer = clusterer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.cluster_input = cluster_input
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction

    def fit(self, X, y=None):
        X_df = to_frame(X)
        n, d = X_df.shape
        if n < 1:
            raise ValueError("X must contain at least one sample")
        if d < 1:
            raise ValueError("X must contain at least one feature")
        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        rng = np.random.default_rng(self.random_state)
        X_syn = self._make_synthetic(X_df, rng)
        X_all = pd.concat([X_df, X_syn], ignore_index=True)
        y_all = np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = d
        if self.cluster_input not in {"auto", "embedding", "onehot", "distance", "similarity"}:
            raise ValueError("cluster_input must be one of 'auto', 'embedding', 'onehot', 'distance', 'similarity'")
        self.preprocessor_ = build_tree_preprocessor(
            X_df,
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        )
        self.forest_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.pipeline_ = Pipeline([
            ("preprocess", self.preprocessor_),
            ("forest", self.forest_),
        ])
        self.pipeline_.fit(X_all, y_all)
        self.forest_ = self.pipeline_.named_steps["forest"]

        self.leaf_embedding_ = self._apply_leaves(X_df)
        self.leaf_encoder_ = make_leaf_encoder(self.leaf_embedding_)
        self.leaf_onehot_embedding_ = self.leaf_encoder_.fit_transform(self.leaf_embedding_)
        self.proximity_ = leaf_proximity(self.leaf_embedding_)
        self.labels_ = run_leaf_clusterer(
            self.clusterer,
            self.n_clusters,
            self.leaf_embedding_,
            self.proximity_,
            self.leaf_onehot_embedding_,
            cluster_input=self.cluster_input,
        )
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def transform(self, X):
        """Return integer leaf ids with shape ``(n_samples, n_estimators)``."""
        check_is_fitted(self, "pipeline_")
        return self._apply_leaves(to_frame(X))

    def transform_onehot(self, X):
        """Return one-hot leaf embedding suitable for Euclidean clusterers."""
        check_is_fitted(self, "leaf_encoder_")
        return self.leaf_encoder_.transform(self.transform(X))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.leaf_embedding_

    def proximity_matrix(self, X=None, Y=None):
        check_is_fitted(self, "leaf_embedding_")
        if X is None and Y is None:
            return self.proximity_.copy()
        E_X = self.leaf_embedding_ if X is None else self.transform(X)
        if Y is None:
            return leaf_proximity(E_X)
        E_Y = self.transform(Y)
        return leaf_cross_proximity(E_X, E_Y)

    def similarity_matrix(self, X=None, Y=None):
        return self.proximity_matrix(X=X, Y=Y)

    def pairwise_distance(self, X=None, Y=None):
        P = self.proximity_matrix(X=X, Y=Y)
        return (1.0 - P).astype(np.float32, copy=False)

    def _apply_leaves(self, X_df):
        Xt = self.pipeline_.named_steps["preprocess"].transform(X_df)
        return self.pipeline_.named_steps["forest"].apply(Xt).astype(np.int64, copy=False)

    def _make_synthetic(self, X_df, rng):
        if self.synthetic == "permute_marginals":
            out = X_df.copy(deep=True)
            for col in out.columns:
                out[col] = rng.permutation(out[col].to_numpy(dtype=object))
            return out
        if self.synthetic == "uniform_box":
            data = {}
            for col in X_df.columns:
                s = X_df[col]
                if is_numeric_series(s):
                    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                    finite = vals[np.isfinite(vals)]
                    if finite.size == 0:
                        data[col] = np.full(len(s), np.nan)
                    elif np.nanmin(finite) == np.nanmax(finite):
                        data[col] = np.full(len(s), float(finite[0]))
                    else:
                        data[col] = rng.uniform(np.nanmin(finite), np.nanmax(finite), len(s))
                else:
                    cats = s.dropna().to_numpy(dtype=object)
                    if cats.size == 0:
                        data[col] = np.array([np.nan] * len(s), dtype=object)
                    else:
                        data[col] = rng.choice(cats, size=len(s), replace=True)
            return pd.DataFrame(data, columns=X_df.columns)
        raise ValueError("synthetic must be 'permute_marginals' or 'uniform_box'")
