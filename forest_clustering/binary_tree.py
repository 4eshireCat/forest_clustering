"""Greedy unsupervised binary tree clustering.

The estimator builds one interpretable binary partition tree by greedily choosing
axis-aligned splits that reduce within-node squared error in a preprocessed
numeric feature space. Leaves are the clusters; samples in the same leaf have
proximity 1 and samples in different leaves have proximity 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from ._tree_common import build_tree_preprocessor, to_frame


@dataclass
class _Node:
    id: int
    depth: int
    sample_indices: np.ndarray
    prediction: int | None = None
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: Any = None
    right: Any = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class UnsupervisedBinaryTreeClusterer(BaseEstimator, ClusterMixin):
    """Greedy top-down binary tree clustering with a sklearn-like API.

    The algorithm recursively partitions samples with axis-aligned binary splits.
    Each candidate split is scored by reduction in total within-cluster squared
    error across the transformed feature matrix. This is a CART-like unsupervised
    objective and produces directly interpretable leaf clusters.

    Parameters
    ----------
    n_clusters : int, default=3
        Target number of leaf clusters. The tree stops earlier if no valid split
        remains.
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=4
        Minimum samples required in a leaf candidate before it can be split.
    min_samples_leaf : int, default=2
        Minimum samples per child after a split.
    max_features : {"sqrt", "log2"}, int, float or None, default=None
        Number of transformed features considered per node. None means all.
    n_thresholds : int or None, default=32
        Maximum candidate thresholds per feature, drawn from empirical quantiles.
        None considers all unique midpoint thresholds.
    random_state : int or None, default=None
        Reproducibility seed used when subsampling features.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_depth=None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features=None,
        n_thresholds: int | None = 32,
        random_state: int | None = None,
        add_missing_indicators: bool = False,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
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
        if int(self.n_clusters) < 1:
            raise ValueError("n_clusters must be >= 1")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if int(self.min_samples_split) < 2 * int(self.min_samples_leaf):
            raise ValueError("min_samples_split must be at least 2 * min_samples_leaf")
        if self.n_thresholds is not None and int(self.n_thresholds) < 1:
            raise ValueError("n_thresholds must be >= 1 or None")
        if self.max_depth is not None and int(self.max_depth) < 0:
            raise ValueError("max_depth must be >= 0 or None")

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = d
        self.preprocessor_ = build_tree_preprocessor(
            X_df,
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        )
        Xt = self.preprocessor_.fit_transform(X_df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        self.X_transformed_ = np.asarray(Xt, dtype=np.float64)
        self.transformed_feature_names_ = self._get_transformed_feature_names()
        self._rng_ = np.random.default_rng(self.random_state)
        self._next_node_id_ = 0

        self.tree_ = self._new_node(np.arange(n, dtype=np.int64), depth=0)
        leaves = [self.tree_]
        heap = []
        self._push_candidate(self.tree_, heap)

        target_leaves = min(int(self.n_clusters), n)
        while len(leaves) < target_leaves and heap:
            neg_gain, _, node, split = heapq.heappop(heap)
            if not node.is_leaf:
                continue
            gain = -neg_gain
            if split is None or gain <= 0:
                continue
            feature, threshold, left_idx, right_idx = split
            node.feature = int(feature)
            node.threshold = float(threshold)
            node.gain = float(gain)
            node.left = self._new_node(left_idx, node.depth + 1)
            node.right = self._new_node(right_idx, node.depth + 1)
            leaves.remove(node)
            leaves.extend([node.left, node.right])
            self._push_candidate(node.left, heap)
            self._push_candidate(node.right, heap)

        self.leaves_ = self._collect_leaves(self.tree_)
        for label, leaf in enumerate(self.leaves_):
            leaf.prediction = label
        self.labels_ = self._predict_transformed(self.X_transformed_)
        self.leaf_embedding_ = self.labels_.reshape(-1, 1).astype(np.int64)
        self.proximity_ = self._leaf_proximity(self.labels_)
        self.n_leaves_ = len(self.leaves_)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        check_is_fitted(self, "tree_")
        Xt = self._transform_frame(to_frame(X))
        return self._predict_transformed(Xt)

    def transform(self, X):
        return self.predict(X).reshape(-1, 1).astype(np.int64)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.leaf_embedding_

    def proximity_matrix(self, X=None, Y=None):
        check_is_fitted(self, "labels_")
        if X is None and Y is None:
            return self.proximity_.copy()
        labels_x = self.labels_ if X is None else self.predict(X)
        if Y is None:
            return self._leaf_proximity(labels_x)
        labels_y = self.predict(Y)
        return (labels_x[:, None] == labels_y[None, :]).astype(np.float32)

    def similarity_matrix(self, X=None, Y=None):
        return self.proximity_matrix(X=X, Y=Y)

    def pairwise_distance(self, X=None, Y=None):
        P = self.proximity_matrix(X=X, Y=Y)
        return (1.0 - P).astype(np.float32, copy=False)

    def rules(self):
        """Return human-readable root-to-leaf rules for the fitted tree."""
        check_is_fitted(self, "tree_")
        out = []
        def walk(node, clauses):
            if node.is_leaf:
                out.append({
                    "leaf": node.prediction,
                    "n_samples": int(node.sample_indices.size),
                    "rule": " and ".join(clauses) if clauses else "all samples",
                })
                return
            name = self.transformed_feature_names_[node.feature]
            thr = node.threshold
            walk(node.left, clauses + [f"{name} <= {thr:.6g}"])
            walk(node.right, clauses + [f"{name} > {thr:.6g}"])
        walk(self.tree_, [])
        return out

    def _new_node(self, indices, depth):
        node = _Node(id=self._next_node_id_, depth=int(depth), sample_indices=np.asarray(indices, dtype=np.int64))
        self._next_node_id_ += 1
        return node

    def _push_candidate(self, node, heap):
        split = self._best_split(node.sample_indices, node.depth)
        gain = split[4] if split is not None else 0.0
        payload = None if split is None else (split[0], split[1], split[2], split[3])
        heapq.heappush(heap, (-float(gain), node.id, node, payload))

    def _best_split(self, idx, depth):
        if idx.size < self.min_samples_split:
            return None
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        Xn = self.X_transformed_[idx]
        parent_sse = self._sse(Xn)
        if parent_sse <= 1e-12:
            return None
        best = None
        features = self._feature_candidates(Xn.shape[1])
        for j in features:
            values = Xn[:, j]
            thresholds = self._thresholds(values)
            for thr in thresholds:
                mask = values <= thr
                n_left = int(mask.sum())
                n_right = int(idx.size - n_left)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_idx = idx[mask]
                right_idx = idx[~mask]
                gain = parent_sse - self._sse(self.X_transformed_[left_idx]) - self._sse(self.X_transformed_[right_idx])
                if best is None or gain > best[4]:
                    best = (j, float(thr), left_idx, right_idx, float(gain))
        return best

    def _feature_candidates(self, p):
        mf = self.max_features
        if mf is None:
            k = p
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(p)))
        elif mf == "log2":
            k = max(1, int(np.log2(p)))
        elif isinstance(mf, float):
            if not (0 < mf <= 1):
                raise ValueError("float max_features must be in (0, 1]")
            k = max(1, int(np.ceil(mf * p)))
        else:
            k = int(mf)
            if k < 1:
                raise ValueError("max_features must select at least one feature")
        k = min(k, p)
        if k == p:
            return np.arange(p)
        return self._rng_.choice(p, size=k, replace=False)

    def _thresholds(self, values):
        uniq = np.unique(values[np.isfinite(values)])
        if uniq.size <= 1:
            return np.array([], dtype=float)
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if self.n_thresholds is None or mids.size <= self.n_thresholds:
            return mids
        q = np.linspace(0, 1, int(self.n_thresholds) + 2)[1:-1]
        return np.unique(np.quantile(mids, q))

    @staticmethod
    def _sse(X):
        if X.shape[0] <= 1:
            return 0.0
        centered = X - X.mean(axis=0, keepdims=True)
        return float(np.sum(centered * centered))

    def _predict_transformed(self, Xt):
        labels = np.empty(Xt.shape[0], dtype=np.int64)
        for i, row in enumerate(Xt):
            node = self.tree_
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            labels[i] = int(node.prediction)
        return labels

    def _transform_frame(self, X_df):
        Xt = self.preprocessor_.transform(X_df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return np.asarray(Xt, dtype=np.float64)

    @staticmethod
    def _leaf_proximity(labels):
        labels = np.asarray(labels)
        return (labels[:, None] == labels[None, :]).astype(np.float32)

    @staticmethod
    def _collect_leaves(node):
        if node.is_leaf:
            return [node]
        return UnsupervisedBinaryTreeClusterer._collect_leaves(node.left) + UnsupervisedBinaryTreeClusterer._collect_leaves(node.right)

    def _get_transformed_feature_names(self):
        try:
            return np.asarray(self.preprocessor_.get_feature_names_out(), dtype=object)
        except Exception:
            return np.asarray([f"z{i}" for i in range(self.X_transformed_.shape[1])], dtype=object)
