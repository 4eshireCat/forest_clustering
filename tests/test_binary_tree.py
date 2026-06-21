import numpy as np
import pandas as pd
import pytest

from forest_clustering import UnsupervisedBinaryTreeClusterer


def _toy_data():
    return pd.DataFrame({
        "x": [0, 0.1, 0.2, 4.8, 5.0, 5.2, 9.8, 10.0, 10.1],
        "cat": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    })


def test_binary_tree_sklearn_style_api():
    X = _toy_data()
    est = UnsupervisedBinaryTreeClusterer(n_clusters=3, random_state=42, min_samples_leaf=2)
    labels = est.fit_predict(X)
    assert labels.shape == (len(X),)
    assert est.fit_transform(X).shape == (len(X), 1)
    assert est.transform(X).shape == (len(X), 1)
    assert est.predict(X).shape == (len(X),)
    assert est.n_features_in_ == 2
    assert len(est.rules()) == est.n_leaves_


def test_binary_tree_proximity_is_leaf_equality():
    X = _toy_data()
    est = UnsupervisedBinaryTreeClusterer(n_clusters=3, random_state=0, min_samples_leaf=2).fit(X)
    P = est.proximity_matrix()
    assert P.shape == (len(X), len(X))
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 1.0)
    assert set(np.unique(P)).issubset({0.0, 1.0})
    D = est.pairwise_distance()
    assert np.allclose(D, 1.0 - P)


def test_binary_tree_reproducible_with_feature_subsampling():
    X = _toy_data()
    a = UnsupervisedBinaryTreeClusterer(n_clusters=3, max_features="sqrt", random_state=123, min_samples_leaf=2).fit_predict(X)
    b = UnsupervisedBinaryTreeClusterer(n_clusters=3, max_features="sqrt", random_state=123, min_samples_leaf=2).fit_predict(X)
    assert np.array_equal(a, b)


def test_binary_tree_cross_proximity_shape():
    X = _toy_data()
    est = UnsupervisedBinaryTreeClusterer(n_clusters=3, random_state=0, min_samples_leaf=2).fit(X.iloc[:6])
    P = est.proximity_matrix(X.iloc[:2], X.iloc[2:5])
    assert P.shape == (2, 3)


def test_binary_tree_input_validation():
    with pytest.raises(ValueError):
        UnsupervisedBinaryTreeClusterer(n_clusters=0).fit(_toy_data())
    with pytest.raises(ValueError):
        UnsupervisedBinaryTreeClusterer(min_samples_leaf=2, min_samples_split=3).fit(_toy_data())
