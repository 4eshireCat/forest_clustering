"""Tests for permutation feature importance."""

import numpy as np
import pandas as pd
import pytest


class TestPermutationImportance:
    """Test compute_permutation_importance function."""

    def test_known_important_feature(self):
        """A feature with clear cluster structure should have high importance."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        # Feature 'a' has clear 3-cluster structure
        df = pd.DataFrame({
            'important': np.concatenate([rng.normal(-5, 0.5, 100),
                                          rng.normal(0, 0.5, 100),
                                          rng.normal(5, 0.5, 100)]),
            'noise': rng.normal(0, 1, 300),
        })
        fc = ForestClusterer(n_iterations=20, random_state=42)
        fc.fit(df)
        imp = compute_permutation_importance(fc, df, n_repeats=3, random_state=42)

        # 'important' should have much higher importance than 'noise'
        important_idx = list(df.columns).index('important')
        noise_idx = list(df.columns).index('noise')
        assert imp['importance'][important_idx] > imp['importance'][noise_idx]

    def test_noise_feature_near_zero(self):
        """Pure noise feature should have lower importance than structured feature."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'a': np.concatenate([rng.normal(-5, 1, 100), rng.normal(5, 1, 100)]),
            'noise': rng.normal(0, 10, 200),
        })
        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.fit(df)
        imp = compute_permutation_importance(fc, df, n_repeats=2, random_state=42)

        important_idx = list(df.columns).index('a')
        noise_idx = list(df.columns).index('noise')
        # Noise should have lower importance than the clearly structured feature
        assert imp['importance'][noise_idx] < imp['importance'][important_idx]

    def test_returns_dataframe(self):
        """Result should be a DataFrame."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3.], 'b': ['x', 'y', 'x']})
        fc = ForestClusterer(n_iterations=5, random_state=42)
        fc.fit(df)
        imp = compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
        assert isinstance(imp, pd.DataFrame)

    def test_importance_shape(self):
        """Importance array should have one score per feature."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': ['x', 'y', 'x', 'y']})
        fc = ForestClusterer(n_iterations=5, random_state=42)
        fc.fit(df)
        imp = compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
        assert len(imp) == len(df.columns)

    def test_reproducibility(self):
        """Same random_state → same results."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        df = pd.DataFrame({'a': rng.random(50), 'b': rng.random(50)})
        fc1 = ForestClusterer(n_iterations=10, random_state=42)
        fc1.fit(df)
        imp1 = compute_permutation_importance(fc1, df, n_repeats=2, random_state=123)

        fc2 = ForestClusterer(n_iterations=10, random_state=42)
        fc2.fit(df)
        imp2 = compute_permutation_importance(fc2, df, n_repeats=2, random_state=123)

        np.testing.assert_array_almost_equal(
            imp1['importance'].values, imp2['importance'].values, decimal=5
        )

    def test_normalized_to_0_1(self):
        """Normalized importances should be in [0, 1]."""
        from forest_clustering.permutation_importance import compute_permutation_importance
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.], 'b': ['x', 'y', 'x', 'y', 'x']})
        fc = ForestClusterer(n_iterations=5, random_state=42)
        fc.fit(df)
        imp = compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
        assert imp['importance'].min() >= 0.0
        assert imp['importance'].max() <= 1.0

    def test_clusterer_importance_property(self):
        """ForestClusterer should have feature_importances_ after fit."""
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        df = pd.DataFrame({'a': rng.random(50), 'b': rng.random(50)})
        fc = ForestClusterer(n_iterations=10, random_state=42, compute_importance=True)
        fc.fit(df)
        assert hasattr(fc, 'feature_importances_')
        assert fc.feature_importances_.shape[0] == len(df.columns)


class TestInputValidation:
    def test_negative_n_repeats_raises(self):
        from forest_clustering.permutation_importance import compute_permutation_importance
        with pytest.raises(ValueError):
            compute_permutation_importance(None, None, n_repeats=-1)
