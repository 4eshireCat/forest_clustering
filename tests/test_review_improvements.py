import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin

from forest_clustering import (
    ForestClusterer,
    UnsupervisedRandomForestClusterer,
    ExtraTreesProximityClusterer,
    UnsupervisedBinaryTreeClusterer,
    __version__,
)


class RecordingClusterer(BaseEstimator, ClusterMixin):
    last_shape = None
    last_sparse = None

    def fit_predict(self, X, y=None):
        RecordingClusterer.last_shape = X.shape
        RecordingClusterer.last_sparse = sparse.issparse(X)
        return np.zeros(X.shape[0], dtype=np.int64)


def _mixed_small():
    return pd.DataFrame({
        "num": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
        "cat": ["a", "a", None, "b", "b", None],
    })


def test_non_precomputed_downstream_clusterer_gets_onehot_leaf_embedding_urf():
    X = _mixed_small()
    RecordingClusterer.last_shape = None
    est = UnsupervisedRandomForestClusterer(
        n_estimators=6,
        clusterer=RecordingClusterer(),
        random_state=0,
        min_samples_leaf=1,
    ).fit(X)
    assert RecordingClusterer.last_shape[0] == len(X)
    assert RecordingClusterer.last_sparse is True
    assert RecordingClusterer.last_shape == est.leaf_onehot_embedding_.shape
    assert est.transform_onehot(X).shape == est.leaf_onehot_embedding_.shape


def test_non_precomputed_downstream_clusterer_gets_onehot_leaf_embedding_extra_trees():
    X = _mixed_small()
    RecordingClusterer.last_shape = None
    est = ExtraTreesProximityClusterer(
        n_estimators=6,
        clusterer=RecordingClusterer(),
        random_state=0,
        min_samples_leaf=1,
    ).fit(X)
    assert RecordingClusterer.last_shape[0] == len(X)
    assert RecordingClusterer.last_sparse is True
    assert RecordingClusterer.last_shape == est.leaf_onehot_embedding_.shape
    assert est.transform_onehot(X).shape == est.leaf_onehot_embedding_.shape


def test_tree_estimators_handle_all_missing_columns_without_dropping_all_features():
    X = pd.DataFrame({"num": [np.nan] * 12, "cat": [None] * 12})
    for Est in (UnsupervisedRandomForestClusterer, ExtraTreesProximityClusterer):
        est = Est(n_estimators=5, n_clusters=2, random_state=1, n_jobs=1).fit(X)
        assert est.leaf_embedding_.shape == (len(X), 5)
        assert np.allclose(np.diag(est.proximity_matrix()), 1.0)
        assert est.transform(X).shape == (len(X), 5)
    bt = UnsupervisedBinaryTreeClusterer(n_clusters=2, min_samples_leaf=2, min_samples_split=4).fit(X)
    assert bt.labels_.shape == (len(X),)
    assert np.allclose(np.diag(bt.proximity_matrix()), 1.0)


def test_forest_clusterer_keeps_sklearn_init_lightweight_and_validates_on_fit():
    est = ForestClusterer(n_bins=0, n_iterations=2, n_jobs=1)
    try:
        est.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    except ValueError as e:
        assert "n_bins" in str(e)
    else:
        raise AssertionError("fit should reject n_bins=0")


def test_package_version_matches_pyproject_version_prefix():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project_version = pyproject["project"]["version"]
    assert __version__.startswith(project_version)
