import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from forest_clustering import UnsupervisedRandomForestClusterer


def _toy_mixed_data(n=40):
    rng = np.random.default_rng(123)
    left = pd.DataFrame({
        "age": rng.normal(25, 2, n // 2),
        "fare": rng.normal(20, 3, n // 2),
        "sex": ["female"] * (n // 2),
    })
    right = pd.DataFrame({
        "age": rng.normal(60, 2, n // 2),
        "fare": rng.normal(90, 3, n // 2),
        "sex": ["male"] * (n // 2),
    })
    return pd.concat([left, right], ignore_index=True)


def test_urf_sklearn_clone_and_fit_predict_shape():
    X = _toy_mixed_data()
    est = UnsupervisedRandomForestClusterer(
        n_estimators=25,
        n_clusters=2,
        random_state=42,
        synthetic="permute_marginals",
    )
    cloned = clone(est)
    labels = cloned.fit_predict(X)
    assert labels.shape == (len(X),)
    assert hasattr(cloned, "forest_")
    assert hasattr(cloned, "labels_")
    assert cloned.leaf_embedding_.shape[0] == len(X)


def test_urf_proximity_is_symmetric_unit_diagonal_and_distance_complement():
    X = _toy_mixed_data()
    est = UnsupervisedRandomForestClusterer(n_estimators=20, n_clusters=2, random_state=7).fit(X)
    P = est.proximity_matrix()
    D = est.pairwise_distance()
    assert P.shape == (len(X), len(X))
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 1.0)
    assert np.all((P >= 0.0) & (P <= 1.0))
    assert np.allclose(D, 1.0 - P)


def test_urf_reproducible_with_fixed_random_state():
    X = _toy_mixed_data()
    a = UnsupervisedRandomForestClusterer(n_estimators=30, n_clusters=2, random_state=11).fit(X)
    b = UnsupervisedRandomForestClusterer(n_estimators=30, n_clusters=2, random_state=11).fit(X)
    assert np.array_equal(a.leaf_embedding_, b.leaf_embedding_)
    assert np.array_equal(a.labels_, b.labels_)


def test_urf_transform_new_samples_and_cross_proximity():
    X = _toy_mixed_data()
    est = UnsupervisedRandomForestClusterer(n_estimators=15, n_clusters=2, random_state=5).fit(X.iloc[:30])
    E = est.transform(X.iloc[30:])
    P = est.proximity_matrix(X.iloc[:3], X.iloc[30:35])
    assert E.shape == (len(X.iloc[30:]), est.n_estimators)
    assert P.shape == (3, 5)
    assert np.all((P >= 0.0) & (P <= 1.0))


def test_urf_rejects_unknown_synthetic_mode():
    X = _toy_mixed_data()
    with pytest.raises(ValueError, match="synthetic"):
        UnsupervisedRandomForestClusterer(synthetic="bad-mode").fit(X)
