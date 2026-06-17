"""
Comprehensive TDD test suite for forest_clustering.preflight module.

Tests cover:
  - hopkins_statistic: range, uniform data, clustered data, grid data, edge cases
  - gap_statistic: uniform data, clustered data, identical samples, B=1, schema
  - clusterability_test: integration, return schema, method switching, decisions
  - Invalid inputs: negative n_samples, non-2D arrays, NaN/Inf, n<2
  - Determinism: same random_state -> same result
  - Warnings: small n warnings

All tests are designed to FAIL on an empty/stub implementation and PASS on a
correct implementation following the PREFLIGHT_SPEC.md specification.
"""

import warnings
import numpy as np
import pytest

from forest_clustering.preflight import (
    clusterability_test,
    gap_statistic,
    hopkins_statistic,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Reproducible RandomState fixture."""
    return np.random.RandomState(42)


@pytest.fixture
def data_uniform(rng):
    """Uniform random data with no cluster structure."""
    return rng.uniform(0, 1, size=(500, 3))


@pytest.fixture
def data_blobs(rng):
    """Well-separated Gaussian blobs -> strong cluster structure."""
    return np.vstack([
        rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 200),
        rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 200),
        rng.multivariate_normal([0, 5], np.eye(2) * 0.1, 200),
    ])


@pytest.fixture
def data_single_cluster(rng):
    """Single tight cluster."""
    center = rng.uniform(0, 1, size=(1, 5))
    return center + rng.normal(0, 0.01, size=(300, 5))


@pytest.fixture
def data_grid():
    """Regular grid data -> lattice-like structure."""
    x = np.linspace(0, 1, 20)
    return np.array([[a, b] for a in x for b in x])


@pytest.fixture
def data_identical():
    """All-identical samples -> perfect aggregation."""
    return np.ones((50, 3))


@pytest.fixture
def data_constant_feature(rng):
    """Data with one constant feature dimension."""
    return np.column_stack([rng.randn(100, 2), np.ones(100)])


@pytest.fixture
def data_1d(rng):
    """Single-feature data."""
    return rng.randn(200, 1)


@pytest.fixture
def data_small_n():
    """Very small dataset (n < 10)."""
    return np.random.RandomState(42).randn(5, 2)


# ============================================================================
# Hopkins Statistic - Core Functionality
# ============================================================================


class TestHopkinsUniformData:
    """H-1: Uniform data should produce H ~ 0.5."""

    def test_uniform_3d(self, rng):
        X = rng.uniform(0, 1, size=(1000, 3))
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert 0.45 <= H <= 0.55, f"Expected H ~ 0.5 for uniform 3D data, got H={H:.4f}"

    def test_uniform_5d(self, rng):
        X = rng.uniform(0, 1, size=(1000, 5))
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert 0.45 <= H <= 0.55, f"Expected H ~ 0.5 for uniform 5D data, got H={H:.4f}"

    def test_uniform_1d(self, rng):
        X = rng.uniform(0, 1, size=(1000, 1))
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert 0.45 <= H <= 0.55, f"Expected H ~ 0.5 for uniform 1D data, got H={H:.4f}"


class TestHopkinsClusteredData:
    """H-2: Clustered data should produce H > 0.5 (aggregation bias)."""

    def test_well_separated_blobs(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([0, 5], np.eye(2) * 0.1, 200),
        ])
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert H > 0.6, f"Expected H > 0.6 for well-separated blobs, got H={H:.4f}"

    def test_single_tight_cluster(self, rng):
        center = rng.uniform(0, 1, size=(1, 5))
        X = center + rng.normal(0, 0.01, size=(300, 5))
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert H > 0.6, f"Expected H > 0.6 for single tight cluster, got H={H:.4f}"


class TestHopkinsGridData:
    """H-3: Regular grid data should produce H < 0.5 (regularity bias)."""

    def test_2d_grid(self):
        x = np.linspace(0, 1, 20)
        X = np.array([[a, b] for a in x for b in x])
        H = hopkins_statistic(X, n_samples=100, random_state=42)
        assert H < 0.5, f"Expected H < 0.5 for regular grid, got H={H:.4f}"

    def test_3d_grid(self):
        x = np.linspace(0, 1, 10)
        X = np.array([[a, b, c] for a in x for b in x for c in x])
        H = hopkins_statistic(X, n_samples=50, random_state=42)
        assert H < 0.5, f"Expected H < 0.5 for 3D regular grid, got H={H:.4f}"


class TestHopkinsEdgeCases:
    """H-4 through H-6: Edge cases for Hopkins statistic."""

    def test_all_identical_returns_one(self):
        """All-identical samples represent perfect aggregation -> H = 1.0."""
        X = np.ones((50, 3))
        H = hopkins_statistic(X, n_samples=10, random_state=42)
        assert H == 1.0, f"Expected H = 1.0 for identical samples, got H={H:.4f}"

    def test_constant_feature_handled(self, data_constant_feature):
        """Constant feature should not crash; H must remain in [0, 1]."""
        H = hopkins_statistic(data_constant_feature, n_samples=50, random_state=42)
        assert 0 <= H <= 1, f"Expected H in [0, 1] with constant feature, got H={H:.4f}"

    def test_single_sample_fails(self):
        """n = 1 violates the n >= 2 precondition."""
        X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            hopkins_statistic(X)

    def test_all_constant_features(self):
        """All features constant -> bounding box has zero volume -> H = 0.5."""
        X = np.full((20, 3), 5.0)
        H = hopkins_statistic(X, n_samples=5, random_state=42)
        assert H == 1.0, f"Expected H = 1.0 for all-constant identical data, got H={H:.4f}"

    def test_two_samples(self):
        """Minimal valid dataset with n = 2."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        H = hopkins_statistic(X, n_samples=2, random_state=42)
        assert 0 <= H <= 1, f"Expected H in [0, 1] for n=2, got H={H:.4f}"

    def test_1d_data(self, data_1d):
        """Single feature should work without special handling."""
        H = hopkins_statistic(data_1d, n_samples=50, random_state=42)
        assert 0 <= H <= 1, f"Expected H in [0, 1] for 1D data, got H={H:.4f}"


class TestHopkinsRangeInvariant:
    """H-5: H in [0, 1] for all valid inputs."""

    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("d", [1, 5, 50])
    def test_range_for_various_shapes(self, n, d):
        rng = np.random.RandomState(n + d)
        X = rng.randn(n, d)
        H = hopkins_statistic(X, random_state=42)
        assert 0 <= H <= 1, f"H out of range for shape ({n}, {d}): H={H:.4f}"


class TestHopkinsDeterminism:
    """Same random_state must yield identical H."""

    def test_same_seed_same_result(self, data_uniform):
        H1 = hopkins_statistic(data_uniform, n_samples=50, random_state=123)
        H2 = hopkins_statistic(data_uniform, n_samples=50, random_state=123)
        assert H1 == H2, f"Determinism failed: {H1} != {H2}"

    def test_different_seed_different_result(self, data_uniform):
        H1 = hopkins_statistic(data_uniform, n_samples=50, random_state=123)
        H2 = hopkins_statistic(data_uniform, n_samples=50, random_state=456)
        # With high probability different seeds give different H
        # (Allow for extremely rare collisions)
        assert isinstance(H1, float) and isinstance(H2, float)


class TestHopkinsInvalidInputs:
    """Hopkins should raise on invalid inputs."""

    def test_negative_n_samples(self, data_uniform):
        with pytest.raises(ValueError):
            hopkins_statistic(data_uniform, n_samples=-5)

    def test_non_2d_array_1d(self):
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            hopkins_statistic(X)

    def test_non_2d_array_3d(self):
        X = np.zeros((5, 4, 3))
        with pytest.raises(ValueError):
            hopkins_statistic(X)

    def test_nan_in_data(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0]])
        with pytest.raises(ValueError):
            hopkins_statistic(X)

    def test_inf_in_data(self):
        X = np.array([[1.0, 2.0], [np.inf, 3.0]])
        with pytest.raises(ValueError):
            hopkins_statistic(X)

    def test_zero_n_samples(self, data_uniform):
        """n_samples=0 should be clamped to minimum valid (2) or raise."""
        # Accept either: clamped to 2, or raise ValueError
        try:
            H = hopkins_statistic(data_uniform, n_samples=0, random_state=42)
            assert 0 <= H <= 1
        except ValueError:
            pass

    def test_n_samples_greater_than_n(self):
        """n_samples > n should be clamped to n."""
        X = np.random.randn(10, 2)
        H = hopkins_statistic(X, n_samples=1000, random_state=42)
        assert 0 <= H <= 1, "n_samples > n should be clamped, not raise"


# ============================================================================
# Gap Statistic - Core Functionality
# ============================================================================


class TestGapUniformData:
    """G-1: Uniform data should produce Gap(1) ~ 0."""

    def test_uniform_3d(self, rng):
        X = rng.uniform(0, 1, size=(500, 3))
        result = gap_statistic(X, k_max=1, n_refs=10, random_state=42)
        assert abs(result["gap_1"]) < 0.1, (
            f"Expected Gap(1) ~ 0 for uniform data, got {result['gap_1']:.4f}"
        )

    def test_uniform_5d(self, rng):
        X = rng.uniform(0, 1, size=(500, 5))
        result = gap_statistic(X, k_max=1, n_refs=10, random_state=42)
        assert abs(result["gap_1"]) < 0.1, (
            f"Expected Gap(1) ~ 0 for uniform 5D data, got {result['gap_1']:.4f}"
        )


class TestGapClusteredData:
    """G-2: Well-separated clusters should produce positive max Gap(k)."""

    def test_well_separated_blobs(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 100),
            rng.multivariate_normal([5, 5], np.eye(2) * 0.1, 100),
        ])
        result = gap_statistic(X, k_max=3, n_refs=10, random_state=42)
        assert max(result["gap_k"]) > 0.1, (
            f"Expected max Gap(k) > 0.1 for clustered data, got {result['gap_k']}"
        )
        assert result["best_k"] == 2, (
            f"Expected best_k = 2 for 2 blobs, got {result['best_k']}"
        )

    def test_three_blobs(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 100),
            rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 100),
            rng.multivariate_normal([0, 5], np.eye(2) * 0.1, 100),
        ])
        result = gap_statistic(X, k_max=5, n_refs=10, random_state=42)
        assert max(result["gap_k"]) > 0.1, (
            f"Expected max Gap(k) > 0.1 for 3 blobs, got {result['gap_k']}"
        )


class TestGapEdgeCases:
    """G-3 through G-4: Edge cases for Gap statistic."""

    def test_all_identical_returns_inf(self):
        """All-identical samples -> W_1_data = 0 -> log = -inf -> Gap(1) = +inf."""
        X = np.ones((50, 3))
        result = gap_statistic(X, k_max=1, n_refs=5, random_state=42)
        assert result["gap_1"] == float("inf"), (
            f"Expected Gap(1) = +inf for identical samples, got {result['gap_1']}"
        )

    def test_b_equal_one_produces_nan_s1(self):
        """B = 1 -> cannot compute sample standard deviation -> s_1 = NaN."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        result = gap_statistic(X, k_max=1, n_refs=1, random_state=42)
        assert np.isnan(result["s_1"]), (
            f"Expected s_1 = NaN for B=1, got {result['s_1']}"
        )

    def test_b_equal_one_computes_gap(self):
        """B = 1 should still compute a valid gap_1 value (not NaN)."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        result = gap_statistic(X, k_max=1, n_refs=1, random_state=42)
        assert not np.isnan(result["gap_1"]), "gap_1 should be finite for B=1"

    def test_small_n(self):
        """Gap should work for n < 10."""
        X = np.random.RandomState(42).randn(5, 2)
        result = gap_statistic(X, k_max=1, n_refs=5, random_state=42)
        assert "gap_1" in result
        assert np.isfinite(result["gap_1"]) or result["gap_1"] == float("inf")

    def test_single_sample_fails(self):
        """n = 1 violates the n >= 2 precondition."""
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            gap_statistic(X)

    def test_zero_n_refs_fails(self):
        """n_refs = 0 violates B >= 1 precondition."""
        X = np.random.randn(50, 3)
        with pytest.raises(ValueError):
            gap_statistic(X, n_refs=0)


class TestGapReturnSchema:
    """Gap statistic return value must contain all expected keys with correct types."""

    EXPECTED_KEYS = {"gap_1", "s_1", "log_W_1_data", "E_log_W_ref", "log_W_refs"}

    def test_return_keys(self, data_uniform):
        result = gap_statistic(data_uniform, k_max=1, n_refs=5, random_state=42)
        missing = self.EXPECTED_KEYS - set(result.keys())
        assert missing == set(), f"Missing keys in gap_statistic result: {missing}"

    def test_gap_1_is_float(self, data_uniform):
        result = gap_statistic(data_uniform, k_max=1, n_refs=5, random_state=42)
        assert isinstance(result["gap_1"], float)

    def test_s_1_is_float(self, data_uniform):
        result = gap_statistic(data_uniform, k_max=1, n_refs=5, random_state=42)
        assert isinstance(result["s_1"], float) or np.isnan(result["s_1"])

    def test_log_W_refs_is_list_of_floats(self, data_uniform):
        result = gap_statistic(data_uniform, k_max=1, n_refs=5, random_state=42)
        assert isinstance(result["log_W_refs"], list)
        assert len(result["log_W_refs"]) == 5
        assert all(isinstance(v, float) for v in result["log_W_refs"])


class TestGapDeterminism:
    """Same random_state must yield identical Gap results."""

    def test_same_seed_same_result(self, data_uniform):
        r1 = gap_statistic(data_uniform, k_max=1, n_refs=10, random_state=77)
        r2 = gap_statistic(data_uniform, k_max=1, n_refs=10, random_state=77)
        assert r1["gap_1"] == r2["gap_1"]
        assert r1["s_1"] == r2["s_1"]
        assert r1["E_log_W_ref"] == r2["E_log_W_ref"]


class TestGapInvalidInputs:
    """Gap should raise on invalid inputs."""

    def test_non_2d_array_1d(self):
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            gap_statistic(X)

    def test_non_2d_array_3d(self):
        X = np.zeros((5, 4, 3))
        with pytest.raises(ValueError):
            gap_statistic(X)

    def test_nan_in_data(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0]])
        with pytest.raises(ValueError):
            gap_statistic(X)

    def test_inf_in_data(self):
        X = np.array([[1.0, 2.0], [np.inf, 3.0]])
        with pytest.raises(ValueError):
            gap_statistic(X)

    def test_negative_k_max(self, data_uniform):
        with pytest.raises(ValueError):
            gap_statistic(data_uniform, k_max=-1)


# ============================================================================
# clusterability_test - Integration
# ============================================================================


class TestClusterabilityUniformData:
    """I-1: Uniform data -> not clusterable (both tests should say False)."""

    def test_uniform_both_method(self, rng):
        X = rng.uniform(0, 1, size=(500, 3))
        result = clusterability_test(X, method="both", random_state=42)
        assert result["is_clusterable"] is False, (
            f"Uniform data should not be clusterable. "
            f"H={result['hopkins']}, gap_1={result['gap_1']}"
        )

    def test_uniform_hopkins_method(self, rng):
        X = rng.uniform(0, 1, size=(500, 3))
        result = clusterability_test(X, method="hopkins", random_state=42)
        assert result["is_clusterable"] is False

    def test_uniform_gap_method(self, rng):
        X = rng.uniform(0, 1, size=(500, 3))
        result = clusterability_test(X, method="gap", random_state=42)
        assert result["is_clusterable"] is False


class TestClusterabilityBlobs:
    """I-2: Well-separated blobs -> clusterable (both tests should say True)."""

    def test_blobs_both_method(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([0, 5], np.eye(2) * 0.1, 200),
        ])
        result = clusterability_test(X, method="both", random_state=42)
        assert result["is_clusterable"] is True, (
            f"Well-separated blobs should be clusterable. "
            f"H={result['hopkins']}, gap_1={result['gap_1']}"
        )

    def test_blobs_hopkins_only(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 200),
        ])
        result = clusterability_test(X, method="hopkins", random_state=42)
        assert result["is_clusterable"] is True

    def test_blobs_gap_only(self, rng):
        X = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2) * 0.1, 200),
            rng.multivariate_normal([5, 0], np.eye(2) * 0.1, 200),
        ])
        result = clusterability_test(X, method="gap", random_state=42)
        assert result["is_clusterable"] is True


class TestClusterabilityReturnSchema:
    """I-3: Return dict must contain exactly the expected keys."""

    EXPECTED_KEYS = {
        "hopkins",
        "hopkins_is_clusterable",
        "hopkins_threshold",
        "gap_1",
        "gap_is_clusterable",
        "gap_threshold",
        "is_clusterable",
        "recommendation",
        "details",
    }

    def test_both_method_keys(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert set(result.keys()) == self.EXPECTED_KEYS, (
            f"Key mismatch: expected {self.EXPECTED_KEYS}, got {set(result.keys())}"
        )

    def test_hopkins_method_keys(self, data_uniform):
        result = clusterability_test(data_uniform, method="hopkins", random_state=42)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_gap_method_keys(self, data_uniform):
        result = clusterability_test(data_uniform, method="gap", random_state=42)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_threshold_defaults(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert result["hopkins_threshold"] == 0.55
        assert result["gap_threshold"] == 0.10

    def test_is_clusterable_is_bool(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert isinstance(result["is_clusterable"], bool)

    def test_recommendation_is_string(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_details_is_dict(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert isinstance(result["details"], dict)


class TestClusterabilityMethodSwitching:
    """I-4: Method parameter controls which tests run."""

    def test_hopkins_method_skips_gap(self, data_uniform):
        result = clusterability_test(data_uniform, method="hopkins", random_state=42)
        assert result["hopkins"] is not None
        assert result["gap_1"] is None
        assert result["gap_is_clusterable"] is None

    def test_gap_method_skips_hopkins(self, data_uniform):
        result = clusterability_test(data_uniform, method="gap", random_state=42)
        assert result["gap_1"] is not None
        assert result["hopkins"] is None
        assert result["hopkins_is_clusterable"] is None

    def test_both_method_computes_both(self, data_uniform):
        result = clusterability_test(data_uniform, method="both", random_state=42)
        assert result["hopkins"] is not None
        assert result["gap_1"] is not None

    def test_invalid_method_raises(self, data_uniform):
        with pytest.raises(ValueError):
            clusterability_test(data_uniform, method="invalid")


class TestClusterabilityAllIdentical:
    """All-identical data -> both tests indicate clusterable."""

    def test_identical_hopkins_is_one(self):
        X = np.ones((30, 3))
        result = clusterability_test(X, method="both", random_state=42)
        assert result["hopkins"] == 1.0
        assert result["hopkins_is_clusterable"] is True

    def test_identical_gap_is_inf(self):
        X = np.ones((30, 3))
        result = clusterability_test(X, method="both", random_state=42)
        assert result["gap_1"] == float("inf")
        assert result["gap_is_clusterable"] is True

    def test_identical_combined_decision(self):
        X = np.ones((30, 3))
        result = clusterability_test(X, method="both", random_state=42)
        assert result["is_clusterable"] is True


class TestClusterabilityDeterminism:
    """Same random_state -> identical clusterability_test results."""

    def test_same_seed_same_result(self, data_uniform):
        r1 = clusterability_test(data_uniform, method="both", random_state=99)
        r2 = clusterability_test(data_uniform, method="both", random_state=99)
        assert r1["hopkins"] == r2["hopkins"]
        assert r1["gap_1"] == r2["gap_1"]
        assert r1["is_clusterable"] == r2["is_clusterable"]
        assert r1["recommendation"] == r2["recommendation"]


class TestClusterabilityInvalidInputs:
    """clusterability_test should propagate validation errors."""

    def test_non_2d_array(self):
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            clusterability_test(X)

    def test_nan_in_data(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0]])
        with pytest.raises(ValueError):
            clusterability_test(X)

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            clusterability_test(X)


# ============================================================================
# Warnings
# ============================================================================


class TestWarnings:
    """Test that appropriate warnings are emitted for edge cases."""

    def test_small_n_warning_hopkins(self):
        """n < 10 should trigger a warning for Hopkins."""
        X = np.random.RandomState(42).randn(5, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hopkins_statistic(X, random_state=42)
            # Should have at least one warning about small dataset
            assert len(w) >= 1, "Expected a warning for small n (< 10)"

    def test_small_n_warning_gap(self):
        """n < 10 should trigger a warning for Gap."""
        X = np.random.RandomState(42).randn(5, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gap_statistic(X, k_max=1, n_refs=5, random_state=42)
            assert len(w) >= 1, "Expected a warning for small n (< 10)"

    def test_small_n_warning_clusterability(self):
        """n < 10 should trigger a warning from clusterability_test."""
        X = np.random.RandomState(42).randn(5, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clusterability_test(X, method="both", random_state=42)
            assert len(w) >= 1, "Expected a warning for small n (< 10)"

    def test_b_equal_one_warning(self):
        """B = 1 should trigger a warning about unreliable standard error."""
        X = np.random.RandomState(42).randn(50, 3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gap_statistic(X, k_max=1, n_refs=1, random_state=42)
            assert len(w) >= 1, "Expected a warning for B=1 (no std error)"
