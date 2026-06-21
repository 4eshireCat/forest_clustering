import numpy as np
import pandas as pd
import pytest

from forest_clustering import ForestClusterer
from forest_clustering.correlation import compute_feature_weights
from forest_clustering.partitioner import build_col_stats, _make_num_edges


def test_fit_rejects_zero_feature_dataframe():
    X = pd.DataFrame(index=range(3))
    with pytest.raises(ValueError, match="at least one feature"):
        ForestClusterer(n_iterations=2, n_jobs=1).fit(X)


def test_similarity_matrix_is_one_minus_distance():
    X = pd.DataFrame({"x": [0.0, 0.1, 10.0, 10.1], "c": ["a", "a", "b", "b"]})
    fc = ForestClusterer(n_iterations=12, n_clusters=2, random_state=0, n_jobs=1).fit(X)
    D = fc.pairwise_distance()
    S = fc.similarity_matrix()
    assert S.dtype == np.float32
    assert np.allclose(S, 1.0 - D)
    assert np.allclose(np.diag(S), 1.0)


def test_spearman_weights_ignore_categorical_codes():
    X = np.array([
        [0, 0.0],
        [1, 1.0],
        [2, 2.0],
        [3, 3.0],
        [4, 4.0],
    ])
    weights = compute_feature_weights(
        X,
        threshold=0.1,
        rng=np.random.default_rng(0),
        feature_types=["categorical", "numerical"],
    )
    assert np.allclose(weights, np.ones(2))


def test_quantile_edges_are_drawn_from_uniform_probability_scale():
    rng = np.random.default_rng(42)
    X = np.arange(1000, dtype=float).reshape(-1, 1)
    stats = build_col_stats(
        X,
        ["numerical"],
        quantile_cuts=True,
        rng=np.random.default_rng(1),
    )[0]
    edges = np.concatenate([_make_num_edges(stats, 5, rng) for _ in range(200)])
    # For an arithmetic sample, uniform quantile probabilities imply a mean near
    # the sample midpoint. This catches value-bootstrap regressions on skewed use.
    assert 450.0 < float(edges.mean()) < 550.0
    assert np.all(np.diff(np.sort(_make_num_edges(stats, 10, np.random.default_rng(3)))) >= 0)
