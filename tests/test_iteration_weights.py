"""Tests for forest_clustering.iteration_weights.compute_iteration_weights.

These tests define the expected API and behavior BEFORE production code exists.
All tests should fail (ImportError/AttributeError) until the module is implemented.
"""

import numpy as np
import pytest


class TestImport:
    """Module and function importability."""

    def test_module_importable(self):
        """The iteration_weights module can be imported."""
        from forest_clustering import iteration_weights  # noqa: F401

    def test_function_importable(self):
        """compute_iteration_weights can be imported directly."""
        from forest_clustering.iteration_weights import compute_iteration_weights  # noqa: F401

    def test_function_available_from_package(self):
        """compute_iteration_weights is re-exported from forest_clustering."""
        from forest_clustering import compute_iteration_weights  # noqa: F401


class TestUniformStrategy:
    """Tests for strategy='uniform'."""

    def test_uniform_returns_ones(self, sample_embedding):
        """Uniform strategy must return an array of ones."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = sample_embedding
        L = E.shape[1]
        weights = compute_iteration_weights(E, strategy="uniform")
        expected = np.ones(L, dtype=np.float64)
        np.testing.assert_array_equal(weights, expected)

    def test_uniform_sum_equals_L(self, sample_embedding):
        """Uniform weights must sum to L (since each weight = 1)."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = sample_embedding
        L = E.shape[1]
        weights = compute_iteration_weights(E, strategy="uniform")
        assert weights.sum() == pytest.approx(L)

    def test_uniform_dtype(self, sample_embedding):
        """Uniform weights must be float64."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="uniform")
        assert weights.dtype == np.float64


class TestEntropyStrategy:
    """Tests for strategy='entropy'."""

    def test_entropy_returns_in_unit_interval(self, sample_embedding):
        """Entropy-normalized weights must lie in [0, 1]."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="entropy")
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_entropy_non_negative(self, sample_embedding):
        """All entropy weights must be non-negative."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="entropy")
        assert (weights >= 0).all()

    def test_entropy_perfect_separation_higher_weights(self, perfect_separation_embedding):
        """Perfect separation (max entropy) should give higher weights than no separation."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        w_perfect = compute_iteration_weights(perfect_separation_embedding, strategy="entropy")
        # Perfect separation has max entropy → weights should be high (near 1.0)
        assert w_perfect.mean() == pytest.approx(1.0, abs=0.1)

    def test_entropy_mean_is_one(self, sample_embedding):
        """After normalization, mean weight must be 1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="entropy")
        assert weights.mean() == pytest.approx(1.0)

    def test_entropy_all_same_cell(self, no_separation_embedding):
        """When all samples fall in the same cell, entropy = 0 → weight = 0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(no_separation_embedding, strategy="entropy")
        np.testing.assert_array_almost_equal(weights, np.zeros(3))

    def test_entropy_mixed_quality_ordering(self, mixed_quality_embedding):
        """Iteration with better separation should get higher or equal weight."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(mixed_quality_embedding, strategy="entropy")
        # Column 0: 5 unique cells (best), Column 1: 3 unique (moderate), Column 2: 1 unique (worst)
        assert weights[0] >= weights[1] >= weights[2]


class TestInverseGiniStrategy:
    """Tests for strategy='inverse_gini'."""

    def test_inverse_gini_returns_in_unit_interval(self, sample_embedding):
        """Inverse-Gini-normalized weights must lie in [0, 1]."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="inverse_gini")
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_inverse_gini_non_negative(self, sample_embedding):
        """All inverse-Gini weights must be non-negative."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="inverse_gini")
        assert (weights >= 0).all()

    def test_inverse_gini_mean_is_one(self, sample_embedding):
        """After normalization, mean weight must be 1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="inverse_gini")
        assert weights.mean() == pytest.approx(1.0)

    def test_inverse_gini_all_same_cell(self, no_separation_embedding):
        """When all samples fall in the same cell, gini = 0 → weight = 0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(no_separation_embedding, strategy="inverse_gini")
        np.testing.assert_array_almost_equal(weights, np.zeros(3))

    def test_inverse_gini_perfect_separation(self, perfect_separation_embedding):
        """Perfect separation should give high inverse-Gini weights."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(perfect_separation_embedding, strategy="inverse_gini")
        assert weights.mean() == pytest.approx(1.0, abs=0.1)


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_single_sample(self):
        """With n=1, every iteration has a single cell → entropy = 0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.array([[0, 1, 2]], dtype=np.int64)  # 1 sample, 3 iterations
        weights_uniform = compute_iteration_weights(E, strategy="uniform")
        weights_entropy = compute_iteration_weights(E, strategy="entropy")
        weights_gini = compute_iteration_weights(E, strategy="inverse_gini")

        np.testing.assert_array_equal(weights_uniform, np.ones(3))
        # Single sample → single cell → entropy = 0, but max_entropy also = 0
        # Should handle division by zero gracefully (return 1.0 per spec)
        assert weights_entropy.mean() == pytest.approx(1.0)
        assert weights_gini.mean() == pytest.approx(1.0)

    def test_single_iteration(self, sample_embedding):
        """L=1: a single iteration should produce a scalar-like weight array."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = sample_embedding[:, 0:1]  # (5, 1)
        for strategy in ("uniform", "entropy", "inverse_gini"):
            weights = compute_iteration_weights(E, strategy=strategy)
            assert weights.shape == (1,)
            assert weights.mean() == pytest.approx(1.0)

    def test_all_samples_same_cell_all_strategies(self):
        """All same cell across all strategies should behave consistently."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.zeros((10, 5), dtype=np.int64)

        w_uni = compute_iteration_weights(E, strategy="uniform")
        w_ent = compute_iteration_weights(E, strategy="entropy")
        w_gini = compute_iteration_weights(E, strategy="inverse_gini")

        np.testing.assert_array_equal(w_uni, np.ones(5))
        np.testing.assert_array_almost_equal(w_ent, np.zeros(5))
        np.testing.assert_array_almost_equal(w_gini, np.zeros(5))

    def test_max_entropy_division_by_zero(self):
        """If max_entropy == 0 (e.g., n=1), weight should be 1.0 as per spec."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.array([[7]], dtype=np.int64)  # n=1, L=1
        weights = compute_iteration_weights(E, strategy="entropy")
        assert weights[0] == pytest.approx(1.0)

    def test_large_cell_counts(self):
        """Embedding with many unique cell IDs should still work."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        rng = np.random.default_rng(42)
        E = rng.integers(0, 100, size=(50, 10), dtype=np.int64)
        for strategy in ("uniform", "entropy", "inverse_gini"):
            weights = compute_iteration_weights(E, strategy=strategy)
            assert weights.shape == (10,)
            assert (weights >= 0).all()
            assert weights.mean() == pytest.approx(1.0)


class TestInvalidInput:
    """Input validation tests."""

    def test_invalid_strategy_raises_value_error(self, sample_embedding):
        """An unknown strategy name must raise ValueError."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        with pytest.raises(ValueError):
            compute_iteration_weights(sample_embedding, strategy="unknown_strategy")

    def test_empty_embedding_raises(self):
        """An empty embedding (n=0) should raise an appropriate error."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.empty((0, 5), dtype=np.int64)
        with pytest.raises(ValueError):
            compute_iteration_weights(E, strategy="uniform")

    def test_non_integer_embedding_raises(self):
        """A non-integer embedding should raise TypeError or ValueError."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.array([[0.5, 1.2], [2.3, 3.4]])
        with pytest.raises((TypeError, ValueError)):
            compute_iteration_weights(E, strategy="uniform")


class TestWeightNormalization:
    """Tests specific to the mean=1.0 normalization step."""

    def test_uniform_mean_is_one(self, sample_embedding):
        """Uniform weights always have mean 1.0 by construction."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(sample_embedding, strategy="uniform")
        assert weights.mean() == pytest.approx(1.0)

    def test_entropy_mean_is_one_even_with_zeros(self, no_separation_embedding):
        """If all weights are zero (all same cell), result is all zeros."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        weights = compute_iteration_weights(no_separation_embedding, strategy="entropy")
        np.testing.assert_array_almost_equal(weights, np.zeros(3))

    def test_all_strategies_same_shape(self, sample_embedding):
        """All strategies must return a 1-D array of length L."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        L = sample_embedding.shape[1]
        for strategy in ("uniform", "entropy", "inverse_gini"):
            weights = compute_iteration_weights(sample_embedding, strategy=strategy)
            assert weights.ndim == 1
            assert weights.shape[0] == L
