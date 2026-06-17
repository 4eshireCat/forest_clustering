"""Tests for correlation-aware feature grouping.

These tests verify the build_correlation_groups() and
select_features_correlation_aware() utilities and their integration
into ForestClusterer via the correlation_aware=True parameter.
"""

import numpy as np
import pytest


class TestCorrelationGroups:
    """Test build_correlation_groups."""

    def test_basic_grouping(self):
        """5 features: 0~1 (0.9), 2~3 (0.8), 4 independent."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.array(
            [
                [1.0, 0.9, 0.1, 0.2, 0.0],
                [0.9, 1.0, 0.1, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.8, 0.0],
                [0.2, 0.1, 0.8, 1.0, 0.1],
                [0.0, 0.1, 0.0, 0.1, 1.0],
            ]
        )
        weights = np.ones(5)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        # Should be [[0,1], [2,3], [4]] or similar
        assert len(groups) == 3
        group_sets = [set(g) for g in groups]
        assert {0, 1} in group_sets
        assert {2, 3} in group_sets
        assert {4} in group_sets

    def test_all_independent(self):
        """All features independent → singleton groups."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.eye(5)
        weights = np.ones(5)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        assert len(groups) == 5
        assert all(len(g) == 1 for g in groups)

    def test_all_correlated(self):
        """All features correlated → single group."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.full((5, 5), 0.9)
        np.fill_diagonal(corr, 1.0)
        weights = np.ones(5)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2, 3, 4}

    def test_transitive_chain(self):
        """0~1 (0.8), 1~2 (0.8), 0 not ~2 (0.5) → all in one group."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.array(
            [
                [1.0, 0.8, 0.5],
                [0.8, 1.0, 0.8],
                [0.5, 0.8, 1.0],
            ]
        )
        weights = np.ones(3)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_negative_correlation(self):
        """Negative correlation with |rho| > threshold → grouped."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.array([[1.0, -0.9], [-0.9, 1.0]])
        weights = np.ones(2)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        assert len(groups) == 1

    def test_threshold_boundary(self):
        """corr = 0.69, threshold = 0.7 → NOT grouped."""
        from forest_clustering.correlation_aware import build_correlation_groups

        corr = np.array([[1.0, 0.69], [0.69, 1.0]])
        weights = np.ones(2)
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        assert len(groups) == 2


class TestCorrelationAwareSelection:
    """Test select_features_correlation_aware."""

    def test_selects_at_most_one_per_group(self):
        """Selected features must be from different groups when possible."""
        from forest_clustering.correlation_aware import (
            build_correlation_groups,
            select_features_correlation_aware,
        )

        corr = np.array(
            [
                [1.0, 0.9, 0.0],
                [0.9, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        weights = np.array([1.0, 0.5, 1.0])
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        rng = np.random.default_rng(42)
        selected = select_features_correlation_aware(
            groups, weights, n_select=2, rng=rng
        )
        # selected[0] in {0,1}, selected[1] == 2
        assert len(selected) == 2
        assert selected[1] == 2  # group {2} is always selected

    def test_selects_highest_weight_from_group(self):
        """When group is selected, pick highest-weight feature."""
        from forest_clustering.correlation_aware import (
            build_correlation_groups,
            select_features_correlation_aware,
        )

        corr = np.array(
            [
                [1.0, 0.9, 0.0],
                [0.9, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        weights = np.array([1.0, 0.5, 1.0])
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        rng = np.random.default_rng(42)
        selected = select_features_correlation_aware(
            groups, weights, n_select=2, rng=rng
        )
        assert 0 in selected  # weight 1.0 > 0.5, so feature 0 wins in group {0,1}

    def test_more_groups_than_select(self):
        """n_select < n_groups → select top n_select groups."""
        from forest_clustering.correlation_aware import (
            build_correlation_groups,
            select_features_correlation_aware,
        )

        corr = np.eye(5)
        weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        groups = build_correlation_groups(weights, corr, threshold=0.7)
        rng = np.random.default_rng(42)
        selected = select_features_correlation_aware(
            groups, weights, n_select=3, rng=rng
        )
        assert len(selected) == 3
        assert set(selected).issubset({0, 1, 2, 3, 4})


class TestClustererCorrelationAware:
    """Test ForestClusterer with correlation_aware."""

    def test_clusterer_accepts_correlation_aware(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, correlation_aware=True, random_state=42
        )
        labels = fc.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)

    def test_correlation_groups_stored(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, correlation_aware=True, random_state=42
        )
        fc.fit(sample_data)
        assert hasattr(fc, "correlation_groups_")
        assert len(fc.correlation_groups_) > 0

    def test_correlation_aware_vs_random(self, sample_data):
        """Correlation-aware should produce valid specs."""
        from forest_clustering import ForestClusterer

        fc1 = ForestClusterer(
            n_iterations=10, correlation_aware=False, random_state=42
        )
        fc2 = ForestClusterer(
            n_iterations=10, correlation_aware=True, random_state=42
        )
        fc1.fit(sample_data)
        fc2.fit(sample_data)
        # Compare feature selections in first spec
        feat1 = set(bs.col_idx for bs in fc1.specs_[0].bin_specs)
        feat2 = set(bs.col_idx for bs in fc2.specs_[0].bin_specs)
        # May differ or not, but should be valid
        assert len(feat1) > 0
        assert len(feat2) > 0


class TestInputValidation:
    """Input validation."""

    def test_invalid_threshold_raises(self):
        from forest_clustering.correlation_aware import build_correlation_groups

        with pytest.raises(ValueError):
            build_correlation_groups(np.ones(3), np.eye(3), threshold=1.5)

    def test_mismatched_weights_corr_shape_raises(self):
        from forest_clustering.correlation_aware import build_correlation_groups

        with pytest.raises(ValueError):
            build_correlation_groups(np.ones(3), np.eye(5), threshold=0.7)
