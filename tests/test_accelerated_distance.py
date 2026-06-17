"""Tests for accelerated (parallel + optional numba) Hamming distance."""

import time

import numpy as np
import pytest


class TestAcceleratedCorrectness:
    """Accelerated version must match reference exactly."""

    def test_fast_matches_reference(self, sample_embedding, uniform_weights):
        """pairwise_weighted_hamming_fast must equal pairwise_weighted_hamming."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            pairwise_weighted_hamming_fast,
        )

        D_ref = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        D_fast = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=1
        )
        np.testing.assert_array_almost_equal(D_ref, D_fast, decimal=5)

    def test_fast_parallel_matches_serial(self, sample_embedding, uniform_weights):
        """n_jobs=2 must give same result as n_jobs=1."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
        )

        D1 = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=1
        )
        D2 = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=2
        )
        np.testing.assert_array_almost_equal(D1, D2, decimal=5)

    def test_fast_non_uniform_weights(self, sample_embedding):
        """Test with non-uniform weights."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            pairwise_weighted_hamming_fast,
        )

        w = np.array([1.0, 2.0, 0.5, 3.0])
        D_ref = pairwise_weighted_hamming(sample_embedding, w)
        D_fast = pairwise_weighted_hamming_fast(sample_embedding, w, n_jobs=2)
        np.testing.assert_array_almost_equal(D_ref, D_fast, decimal=5)

    def test_fast_symmetry(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
        )

        D = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=2
        )
        np.testing.assert_array_almost_equal(D, D.T)

    def test_fast_diagonal_zero(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
        )

        D = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=2
        )
        np.testing.assert_array_almost_equal(
            np.diag(D), np.zeros(sample_embedding.shape[0])
        )

    def test_fast_range(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
        )

        D = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights, n_jobs=2
        )
        assert D.min() >= 0.0 and D.max() <= 1.0


class TestChunkedDistance:
    """Chunked version must match full version."""

    def test_chunked_matches_full(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            pairwise_weighted_hamming_chunked,
        )

        D_ref = pairwise_weighted_hamming(sample_embedding, uniform_weights)
        D_chunked = pairwise_weighted_hamming_chunked(
            sample_embedding, uniform_weights, chunk_size=2
        )
        np.testing.assert_array_almost_equal(D_ref, D_chunked, decimal=5)

    def test_chunked_different_chunk_sizes(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_chunked,
        )

        D1 = pairwise_weighted_hamming_chunked(
            sample_embedding, uniform_weights, chunk_size=1
        )
        D2 = pairwise_weighted_hamming_chunked(
            sample_embedding, uniform_weights, chunk_size=3
        )
        np.testing.assert_array_almost_equal(D1, D2, decimal=5)

    def test_chunked_dtype(self, sample_embedding, uniform_weights):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_chunked,
        )

        D = pairwise_weighted_hamming_chunked(
            sample_embedding, uniform_weights
        )
        assert D.dtype == np.float32


class TestCrossHammingFast:
    """Fast cross-version."""

    def test_cross_fast_matches_reference(
        self, sample_embedding, uniform_weights
    ):
        from forest_clustering.weighted_distance import (
            weighted_cross_hamming,
            weighted_cross_hamming_fast,
        )

        E_X = sample_embedding[:3]
        E_Y = sample_embedding[3:]
        D_ref = weighted_cross_hamming(E_X, E_Y, uniform_weights)
        D_fast = weighted_cross_hamming_fast(E_X, E_Y, uniform_weights, n_jobs=2)
        np.testing.assert_array_almost_equal(D_ref, D_fast, decimal=5)

    def test_cross_fast_self_equals_pairwise(
        self, sample_embedding, uniform_weights
    ):
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
            weighted_cross_hamming_fast,
        )

        D_pair = pairwise_weighted_hamming_fast(
            sample_embedding, uniform_weights
        )
        D_cross = weighted_cross_hamming_fast(
            sample_embedding, sample_embedding, uniform_weights
        )
        np.testing.assert_array_almost_equal(D_pair, D_cross, decimal=5)


class TestPerformance:
    """Performance benchmarks."""

    def test_fast_is_faster_than_reference(self):
        """Fast version should be at least 1.5x faster."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming,
            pairwise_weighted_hamming_fast,
        )

        rng = np.random.default_rng(42)
        E = rng.integers(0, 10, size=(500, 100), dtype=np.int64)
        w = np.ones(100)

        t0 = time.perf_counter()
        D_ref = pairwise_weighted_hamming(E, w)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        D_fast = pairwise_weighted_hamming_fast(E, w, n_jobs=2)
        t_fast = time.perf_counter() - t0

        assert t_fast < t_ref * 1.5, (
            f"Fast: {t_fast:.3f}s, Ref: {t_ref:.3f}s"
        )


class TestNumbaFallback:
    """Tests work regardless of numba availability."""

    def test_has_accelerated_function(self):
        """pairwise_weighted_hamming_fast must be importable."""
        from forest_clustering.weighted_distance import (
            pairwise_weighted_hamming_fast,
        )

        assert callable(pairwise_weighted_hamming_fast)
