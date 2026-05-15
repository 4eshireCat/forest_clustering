import numpy as np
import pytest
from forest_clustering.correlation import compute_feature_weights


def test_uncorrelated_features_get_weight_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5))
    weights = compute_feature_weights(X, threshold=0.7, sample_size=500, rng=rng)
    assert weights.shape == (5,)
    np.testing.assert_allclose(weights, 1.0, atol=1e-10)


def test_perfectly_correlated_pair_gets_half_weight():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(1000)
    X = np.column_stack([base, base, rng.standard_normal(1000)])
    weights = compute_feature_weights(X, threshold=0.7, sample_size=1000, rng=rng)
    # First two features form a group of size 2 → weight 0.5
    assert weights[0] == pytest.approx(0.5, abs=1e-6)
    assert weights[1] == pytest.approx(0.5, abs=1e-6)
    assert weights[2] == pytest.approx(1.0, abs=1e-6)


def test_three_correlated_features():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(2000)
    X = np.column_stack([base, base * 0.99, base * 0.98, rng.standard_normal(2000)])
    weights = compute_feature_weights(X, threshold=0.9, sample_size=2000, rng=rng)
    # First three should form a group
    assert weights[:3].max() <= 1.0 / 3 + 1e-6
    assert weights[3] == pytest.approx(1.0, abs=1e-6)


def test_single_feature():
    X = np.random.randn(100, 1)
    weights = compute_feature_weights(X, threshold=0.7)
    np.testing.assert_allclose(weights, 1.0)


def test_constant_column_handled():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.standard_normal(200), np.ones(200)])
    # constant column has std=0, should not crash
    weights = compute_feature_weights(X, threshold=0.7, sample_size=200, rng=rng)
    assert weights.shape == (2,)


def test_sampling_reduces_cost():
    """Sampling must make large-n correlation complete in reasonable time."""
    import time
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100_000, 20))
    t0 = time.perf_counter()
    compute_feature_weights(X, threshold=0.7, sample_size=5_000, rng=rng)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Took {elapsed:.1f}s — too slow"
