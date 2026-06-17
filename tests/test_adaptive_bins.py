"""Tests for adaptive bin count computation.

These tests verify the compute_adaptive_bins() utility and its integration
into ForestClusterer via the adaptive_bins=True parameter.
"""

import numpy as np
import pytest


class TestAdaptiveBinsFunction:
    """Test compute_adaptive_bins directly."""

    def test_binary_feature(self):
        """Binary feature → exactly 2 bins (discrete short-circuit)."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.5,
                "n_unique": 2,
            }
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        assert bins[0] == 2

    def test_constant_feature(self):
        """Constant feature → min_bins via discrete short-circuit."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": 5.0,
                "max": 5.0,
                "mean": 5.0,
                "std": 0.0,
                "n_unique": 1,
            }
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        assert bins[0] == 2

    def test_uniform_high_variance(self):
        """Uniform U[0,100] → near max_bins due to high complexity."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": 0.0,
                "max": 100.0,
                "mean": 50.0,
                "std": 28.87,
                "n_unique": 1000,
            }
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        # High spread + high cardinality → many bins
        assert bins[0] >= 7

    def test_gaussian_moderate(self):
        """Gaussian N(0,1) → moderate bins."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": -3.5,
                "max": 3.5,
                "mean": 0.0,
                "std": 1.0,
                "n_unique": 998,
            }
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        # Gaussian has moderate spread and cardinality
        assert 4 <= bins[0] <= 10

    def test_sturges_cap_small_n(self):
        """Small dataset: Sturges cap should limit bins."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": 0.0,
                "max": 100.0,
                "mean": 50.0,
                "std": 20.0,
                "n_unique": 50,
            }
        ]
        bins = compute_adaptive_bins(col_stats, n=20, min_bins=2, max_bins=10)
        sturges = int(np.ceil(np.log2(20) + 1))  # 6
        assert bins[0] <= sturges

    def test_multiple_features_different_bins(self):
        """Different features get different bin counts."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {
                "type": "numerical",
                "min": 0.0,
                "max": 100.0,
                "mean": 50.0,
                "std": 30.0,
                "n_unique": 100,
            },
            {
                "type": "numerical",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.5,
                "n_unique": 2,
            },
            {"type": "categorical", "n_categories": 5},
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        assert bins[0] > bins[1]  # high-variance > binary
        assert bins[1] == 2  # binary = 2
        assert bins[2] == 5  # categorical uses n_categories

    def test_all_categorical(self):
        """All categorical features → use n_categories."""
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        col_stats = [
            {"type": "categorical", "n_categories": 3},
            {"type": "categorical", "n_categories": 7},
        ]
        bins = compute_adaptive_bins(col_stats, n=1000, min_bins=2, max_bins=10)
        assert bins[0] == 3
        assert bins[1] == 7


class TestClustererAdaptiveBins:
    """Test ForestClusterer with adaptive_bins."""

    def test_clusterer_accepts_adaptive_bins(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, adaptive_bins=True, random_state=42
        )
        labels = fc.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)

    def test_adaptive_bins_false_is_default(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.fit(sample_data)
        assert fc.adaptive_bins == False

    def test_adaptive_produces_different_bins(self, sample_data):
        """Adaptive should produce different bin counts per feature."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, adaptive_bins=True, random_state=42
        )
        fc.fit(sample_data)
        # Check that different features have different n_bins where expected
        all_bins = set()
        for spec in fc.specs_:
            for bs in spec.bin_specs:
                if hasattr(bs, "K"):
                    all_bins.add(bs.K)
        # With adaptive bins we expect some variation in K across specs
        assert len(all_bins) > 1  # not all the same

    def test_adaptive_vs_fixed_quality(self, sample_data):
        """Adaptive should produce different clustering than fixed."""
        from forest_clustering import ForestClusterer

        fc1 = ForestClusterer(
            n_iterations=10, adaptive_bins=False, random_state=42
        )
        fc2 = ForestClusterer(
            n_iterations=10, adaptive_bins=True, random_state=42
        )
        fc1.fit(sample_data)
        fc2.fit(sample_data)
        # Embeddings should differ due to different per-feature bin counts
        assert not np.array_equal(fc1.embedding_, fc2.embedding_)


class TestInputValidation:
    """Input validation for adaptive bins."""

    def test_min_bins_greater_than_max_raises(self):
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        with pytest.raises(ValueError):
            compute_adaptive_bins([], n=100, min_bins=10, max_bins=2)

    def test_negative_n_raises(self):
        from forest_clustering.adaptive_bins import compute_adaptive_bins

        with pytest.raises(ValueError):
            compute_adaptive_bins([], n=-1)
