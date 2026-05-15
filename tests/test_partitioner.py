import numpy as np
import pytest
from forest_clustering.partitioner import (
    build_col_stats,
    build_iteration_specs,
    compute_embedding,
    _cell_ids,
)


def make_X(n=100, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def test_col_stats_numerical():
    X = make_X()
    stats = build_col_stats(X, ["numerical"] * 5)
    assert all(s["type"] == "numerical" for s in stats)
    assert all("min" in s and "max" in s for s in stats)


def test_col_stats_categorical():
    X = np.column_stack([np.random.randint(0, 5, 100)] * 3).astype(float)
    stats = build_col_stats(X, ["categorical"] * 3)
    assert all(s["type"] == "categorical" for s in stats)
    assert all(s["n_unique"] == 5 for s in stats)


def test_embedding_shape():
    X = make_X(n=80, d=6)
    stats = build_col_stats(X, ["numerical"] * 6)
    rng = np.random.default_rng(42)
    specs = build_iteration_specs(50, stats, 3, 3, np.ones(6), rng)
    E = compute_embedding(X, specs, n_jobs=1)
    assert E.shape == (80, 50)
    assert E.dtype == np.int64


def test_cell_ids_deterministic():
    X = make_X(n=20, d=4)
    stats = build_col_stats(X, ["numerical"] * 4)
    rng = np.random.default_rng(7)
    specs = build_iteration_specs(1, stats, 2, 2, np.ones(4), rng)
    c1 = _cell_ids(X, specs[0])
    c2 = _cell_ids(X, specs[0])
    np.testing.assert_array_equal(c1, c2)


def test_cell_ids_in_range():
    X = make_X(n=30, d=3)
    stats = build_col_stats(X, ["numerical"] * 3)
    rng = np.random.default_rng(1)
    K = 4
    specs = build_iteration_specs(1, stats, 3, K, np.ones(3), rng)
    c = _cell_ids(X, specs[0])
    # max possible cell_id for M=3 features with K=4: K^3 - 1 = 63
    assert c.min() >= 0
    assert c.max() < K**3


def test_embedding_parallel_matches_sequential():
    X = make_X(n=50, d=5)
    stats = build_col_stats(X, ["numerical"] * 5)
    rng = np.random.default_rng(99)
    specs = build_iteration_specs(20, stats, 3, 3, np.ones(5), rng)
    E_seq = compute_embedding(X, specs, n_jobs=1)
    E_par = compute_embedding(X, specs, n_jobs=2)
    np.testing.assert_array_equal(E_seq, E_par)


def test_quantile_cuts():
    X = np.random.exponential(1, (200, 4))  # skewed distribution
    rng = np.random.default_rng(42)
    stats = build_col_stats(X, ["numerical"] * 4, quantile_cuts=True, rng=rng)
    assert all("quantile_pts" in s for s in stats)
    rng2 = np.random.default_rng(42)
    specs = build_iteration_specs(10, stats, 2, 3, np.ones(4), rng2)
    E = compute_embedding(X, specs, n_jobs=1)
    assert E.shape == (200, 10)


def test_categorical_binning():
    # categorical column with 6 unique values encoded as 0..5
    X = np.random.randint(0, 6, (100, 2)).astype(float)
    stats = build_col_stats(X, ["categorical", "categorical"])
    rng = np.random.default_rng(0)
    specs = build_iteration_specs(10, stats, 2, 3, np.ones(2), rng)
    E = compute_embedding(X, specs, n_jobs=1)
    assert E.shape == (100, 10)
