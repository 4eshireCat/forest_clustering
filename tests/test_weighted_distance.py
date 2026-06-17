"""Tests for forest_clustering.weighted_distance module.

Defines expected behavior for:
  - pairwise_weighted_hamming(E, weights) -> (n, n) float32
  - weighted_cross_hamming(E_X, E_Y, weights) -> (n_X, n_Y) float32

All tests should fail (ImportError / AttributeError) until production code exists.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist


class TestImport:
    """Module and function importability."""

    def test_module_importable(self):
        """The weighted_distance module can be imported."""
        from forest_clustering import weighted_distance  # noqa: F401

    def test_pairwise_weighted_hamming_importable(self):
        """pairwise_weighted_hamming can be imported."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming  # noqa: F401

    def test_weighted_cross_hamming_importable(self):
        """weighted_cross_hamming can be imported."""
        from forest_clustering.weighted_distance import weighted_cross_hamming  # noqa: F401

    def test_functions_available_from_package(self):
        """Both functions are re-exported from the package root."""
        from forest_clustering import pairwise_weighted_hamming, weighted_cross_hamming  # noqa: F401


class TestPairwiseWeightedHammingShape:
    """Output shape tests."""

    def test_output_shape_n_n(self, sample_embedding, uniform_weights):
        """pairwise_weighted_hamming must return (n, n) for n samples."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        n = sample_embedding.shape[0]
        assert D.shape == (n, n)

    def test_output_dtype_float32(self, sample_embedding, uniform_weights):
        """Distance matrix must be float32."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        assert D.dtype == np.float32

    def test_output_shape_single_sample(self):
        """With n=1, output must be (1, 1)."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        E = np.array([[0, 1, 2]], dtype=np.int64)
        w = np.ones(3, dtype=np.float64)
        D = pairwise_weighted_hamming(E, w)
        assert D.shape == (1, 1)

    def test_output_shape_many_samples(self):
        """With many samples, output shape is still (n, n)."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        rng = np.random.default_rng(42)
        E = rng.integers(0, 10, size=(100, 20), dtype=np.int64)
        w = np.ones(20, dtype=np.float64)
        D = pairwise_weighted_hamming(E, w)
        assert D.shape == (100, 100)


class TestPairwiseWeightedHammingProperties:
    """Mathematical property tests."""

    def test_diagonal_is_zero(self, sample_embedding, uniform_weights):
        """The diagonal of a distance matrix must be all zeros."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(sample_embedding.shape[0]))

    def test_symmetry(self, sample_embedding, uniform_weights):
        """D[i, j] must equal D[j, i] for all i, j."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_range_zero_to_one(self, sample_embedding, uniform_weights):
        """All entries must lie in [0, 1]."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        assert D.min() >= 0.0
        assert D.max() <= 1.0

    def test_triangle_inequality(self, sample_embedding, uniform_weights):
        """Weighted Hamming is a metric — must satisfy triangle inequality."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        n = D.shape[0]
        # Check D[i,j] <= D[i,k] + D[k,j] for a few triplets
        rng = np.random.default_rng(42)
        for _ in range(20):
            i, j, k = rng.choice(n, size=3, replace=False)
            assert D[i, j] <= D[i, k] + D[k, j] + 1e-6

    def test_self_distances_are_zero(self, sample_embedding, uniform_weights):
        """Distance from a sample to itself is exactly zero."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        D = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        assert (np.diag(D) == 0).all()


class TestUniformWeightsEquivalence:
    """Uniform weights should give the same result as standard Hamming."""

    def test_uniform_equals_standard_hamming(self, sample_embedding):
        """pairwise_weighted_hamming with uniform weights must equal pairwise_hamming."""
        from forest_clustering.distance import pairwise_hamming
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w = np.ones(sample_embedding.shape[1], dtype=np.float64)
        D_weighted = pairwise_weighted_hamming(sample_embedding, w)
        D_standard = pairwise_hamming(sample_embedding)
        np.testing.assert_array_almost_equal(D_weighted, D_standard)

    def test_uniform_equals_scipy_hamming(self, sample_embedding):
        """Verify against scipy's cdist with metric='hamming'."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w = np.ones(sample_embedding.shape[1], dtype=np.float64)
        D_weighted = pairwise_weighted_hamming(sample_embedding, w)
        D_scipy = cdist(sample_embedding, sample_embedding, metric="hamming").astype(np.float32)
        np.testing.assert_array_almost_equal(D_weighted, D_scipy)


class TestNonUniformWeights:
    """Tests with non-uniform per-iteration weights."""

    def test_zero_weights_effect(self, sample_embedding):
        """Setting weight=0 on an iteration should make that iteration not contribute.

        If we zero out one column's weight, the distance should equal Hamming
        computed on the remaining columns only.
        """
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        n, L = sample_embedding.shape
        w = np.ones(L, dtype=np.float64)
        w[0] = 0.0  # zero out first iteration

        D = pairwise_weighted_hamming(sample_embedding, w)

        # Manually compute expected: Hamming on columns 1..L-1 only
        E_sub = sample_embedding[:, 1:]
        D_expected = cdist(E_sub, E_sub, metric="hamming").astype(np.float32)
        np.testing.assert_array_almost_equal(D, D_expected)

    def test_higher_weight_on_bad_iteration_changes_distances(self, sample_embedding):
        """When one iteration has a higher weight, disagreements there count more."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        n, L = sample_embedding.shape
        w1 = np.ones(L, dtype=np.float64)
        w2 = np.ones(L, dtype=np.float64)
        w2[0] = 5.0  # give first iteration much more weight

        D1 = pairwise_weighted_hamming(sample_embedding, w1)
        D2 = pairwise_weighted_hamming(sample_embedding, w2)

        # At least some off-diagonal pair should differ
        mask = ~np.eye(n, dtype=bool)
        assert not np.allclose(D1[mask], D2[mask])

    def test_weight_ordering_effect(self, sample_embedding):
        """Swapping weights should change the distance matrix predictably."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        L = sample_embedding.shape[1]
        w = np.arange(1, L + 1, dtype=np.float64)  # 1, 2, 3, ...
        D1 = pairwise_weighted_hamming(sample_embedding, w)

        # Reverse the weights
        w_rev = w[::-1]
        D2 = pairwise_weighted_hamming(sample_embedding, w_rev)

        # In general D1 != D2 unless the embedding has special symmetry
        # Just assert both are valid distance matrices
        assert D1.shape == D2.shape
        assert (np.diag(D1) == 0).all()
        assert (np.diag(D2) == 0).all()

    def test_all_zero_weights_raises(self, sample_embedding):
        """All-zero weights would cause division by zero and should raise."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w = np.zeros(sample_embedding.shape[1], dtype=np.float64)
        with pytest.raises((ValueError, ZeroDivisionError)):
            pairwise_weighted_hamming(sample_embedding, w)


class TestWeightedCrossHamming:
    """Tests for weighted_cross_hamming (train/test cross-distance)."""

    def test_cross_shape(self, sample_embedding, uniform_weights):
        """weighted_cross_hamming must return (n_X, n_Y)."""
        from forest_clustering.weighted_distance import weighted_cross_hamming

        E_X = sample_embedding[:3]   # 3 samples
        E_Y = sample_embedding[3:]   # 2 samples
        D_cross = weighted_cross_hamming(E_X, E_Y, uniform_weights)
        assert D_cross.shape == (3, 2)

    def test_cross_dtype(self, sample_embedding, uniform_weights):
        """Cross-distance matrix must be float32."""
        from forest_clustering.weighted_distance import weighted_cross_hamming

        D_cross = weighted_cross_hamming(sample_embedding[:3], sample_embedding[3:], uniform_weights)
        assert D_cross.dtype == np.float32

    def test_cross_symmetry_with_self(self, sample_embedding, uniform_weights):
        """cross_hamming(E, E) should equal pairwise_hamming(E)."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            weighted_cross_hamming,
        )

        D_pairwise = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        D_cross = weighted_cross_hamming(sample_embedding, sample_embedding, uniform_weights)
        np.testing.assert_array_almost_equal(D_pairwise, D_cross)

    def test_cross_range_zero_to_one(self, sample_embedding, uniform_weights):
        """Cross-distance entries must lie in [0, 1]."""
        from forest_clustering.weighted_distance import weighted_cross_hamming

        D_cross = weighted_cross_hamming(sample_embedding[:3], sample_embedding[3:], uniform_weights)
        assert D_cross.min() >= 0.0
        assert D_cross.max() <= 1.0

    def test_cross_zero_weights_effect(self, sample_embedding):
        """Zero-weight columns should not affect cross-distance."""
        from forest_clustering.weighted_distance import weighted_cross_hamming

        L = sample_embedding.shape[1]
        w = np.ones(L, dtype=np.float64)
        w[0] = 0.0

        E_X = sample_embedding[:3]
        E_Y = sample_embedding[3:]

        D_cross = weighted_cross_hamming(E_X, E_Y, w)
        D_expected = cdist(E_X[:, 1:], E_Y[:, 1:], metric="hamming").astype(np.float32)
        np.testing.assert_array_almost_equal(D_cross, D_expected)

    def test_cross_mismatched_weights_length_raises(self, sample_embedding):
        """Weights length must match number of iterations (columns) in E."""
        from forest_clustering.weighted_distance import weighted_cross_hamming

        w_wrong = np.ones(sample_embedding.shape[1] + 3, dtype=np.float64)
        with pytest.raises((ValueError, IndexError)):
            weighted_cross_hamming(sample_embedding[:2], sample_embedding[2:], w_wrong)

    def test_cross_vs_pairwise_consistency(self, sample_embedding):
        """Cross and pairwise should agree on overlapping subsets."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            weighted_cross_hamming,
        )

        rng = np.random.default_rng(42)
        E = rng.integers(0, 5, size=(20, 6), dtype=np.int64)
        w = np.array([1.0, 2.0, 1.0, 0.5, 2.0, 1.5], dtype=np.float64)

        D_pairwise = pairwise_weighted_hamming(E, w)
        D_cross = weighted_cross_hamming(E, E, w)

        np.testing.assert_array_almost_equal(D_pairwise, D_cross)


class TestInputValidation:
    """Input validation and error handling."""

    def test_mismatched_weights_length_raises(self, sample_embedding):
        """If weights length != L, must raise ValueError or IndexError."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w_wrong = np.ones(sample_embedding.shape[1] + 2, dtype=np.float64)
        with pytest.raises((ValueError, IndexError)):
            pairwise_weighted_hamming(sample_embedding, w_wrong)

    def test_negative_weights_raise(self, sample_embedding):
        """Negative weights should raise ValueError."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w = np.ones(sample_embedding.shape[1], dtype=np.float64)
        w[0] = -1.0
        with pytest.raises(ValueError):
            pairwise_weighted_hamming(sample_embedding, w)

    def test_empty_embedding_raises(self):
        """Empty embedding should raise."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        E = np.empty((0, 3), dtype=np.int64)
        w = np.ones(3, dtype=np.float64)
        with pytest.raises((ValueError, IndexError)):
            pairwise_weighted_hamming(E, w)

    def test_weights_1d_only(self, sample_embedding):
        """Weights must be 1-D array."""
        from forest_clustering.weighted_distance import pairwise_weighted_hamming

        w_2d = np.ones((1, sample_embedding.shape[1]), dtype=np.float64)
        with pytest.raises((ValueError, TypeError)):
            pairwise_weighted_hamming(sample_embedding, w_2d)
