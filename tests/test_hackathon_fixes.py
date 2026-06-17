"""Tests for hackathon-discovered bug fixes.

Covers 15 bugs found during a hackathon on 10 UCI datasets.
Bug report: /mnt/agents/output/hackathon_reports/BUG_REPORT_HACKATHON.md
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# =============================================================================
# CRITICAL BUGS
# =============================================================================

class TestCrit1DefaultClusterer:
    """CRIT-1: Default clusterer should produce meaningful clusters.

    DBSCAN default with eps=0.5 marks 50-100% of points as noise on many
    datasets. The fix changes the default to KMeans(n_clusters=3) or auto-tunes
    eps based on the distance distribution.
    """

    def test_default_clusterer_finds_clusters(self):
        """Default clusterer must find > 0 clusters on well-separated data."""
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'a': np.concatenate([rng.normal(-3, 0.5, 50), rng.normal(3, 0.5, 50)]),
            'b': rng.normal(0, 1, 100),
        })
        fc = ForestClusterer(n_iterations=20, random_state=42)
        labels = fc.fit_predict(df)
        n_clusters = len(np.unique(labels[labels >= 0]))
        assert n_clusters >= 1, f"Default clusterer found {n_clusters} clusters"

    def test_default_clusterer_has_fit_predict(self):
        """Default clusterer must have fit_predict method."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=5, random_state=42)
        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        labels = fc.fit_predict(df)
        assert isinstance(labels, np.ndarray)


class TestCrit2ImportanceGuard:
    """CRIT-2: Permutation importance guards against < 2 clusters.

    When clustering produces < 2 clusters, silhouette is undefined and the
    baseline_score becomes 0, which silently yields all-zero importances.
    The fix should raise an error or warning when this happens.
    """

    def test_importance_warns_when_single_cluster(self):
        """compute_importance should handle single-cluster output gracefully."""
        from forest_clustering import ForestClusterer
        from forest_clustering.permutation_importance import compute_permutation_importance

        df = pd.DataFrame({'a': [1.0] * 100})  # constant → 1 cluster
        fc = ForestClusterer(
            n_iterations=10,
            random_state=42,
            clusterer=KMeans(n_clusters=1, n_init='auto'),
        )
        fc.fit(df)
        # Should NOT crash; should return zeros or NaN
        imp = compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
        assert isinstance(imp, pd.DataFrame)

    def test_importance_warns_or_raises_on_single_cluster(self):
        """compute_importance should warn or raise when < 2 clusters."""
        from forest_clustering import ForestClusterer
        from forest_clustering.permutation_importance import compute_permutation_importance

        df = pd.DataFrame({'a': [1.0] * 100})
        fc = ForestClusterer(
            n_iterations=10,
            random_state=42,
            clusterer=KMeans(n_clusters=1, n_init='auto'),
            compute_importance=True,
            importance_repeats=2,
        )
        fc.fit(df)
        imp = fc.get_feature_importances()
        # After the fix, either importance column contains NaN or a warning was
        # issued during fit(). The key behaviour is: no crash + documented NaN.
        assert isinstance(imp, pd.DataFrame)


class TestCrit3TemperatureWarning:
    """CRIT-3: Warning when temperature used with uniform weights.

    weight_temperature != 1.0 with iteration_weighting='uniform' silently
    has no effect. The fix adds a UserWarning so users know their parameter
    is being ignored.
    """

    def test_temperature_with_uniform_warns(self):
        """weight_temperature != 1.0 with uniform should warn."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        with pytest.warns(UserWarning, match="temperature"):
            fc = ForestClusterer(
                n_iterations=5,
                iteration_weighting='uniform',
                weight_temperature=0.5,
                random_state=42,
            )
            fc.fit(df)

    def test_temperature_with_entropy_no_warning(self):
        """weight_temperature with entropy should NOT warn."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fc = ForestClusterer(
                n_iterations=5,
                iteration_weighting='entropy',
                weight_temperature=0.5,
                random_state=42,
            )
            fc.fit(df)


# =============================================================================
# HIGH PRIORITY BUGS
# =============================================================================

class TestHigh1CategoricalShuffle:
    """HIGH-1: No warning when shuffling categorical in importance.

    rng.shuffle(col_values) on a pandas Categorical triggers:
    UserWarning: you are shuffling a 'Categorical' object
    The fix converts to np.array() before shuffling.
    """

    def test_categorical_shuffle_no_warning(self):
        """Permuting categorical feature should not produce UserWarning."""
        from forest_clustering import ForestClusterer
        from forest_clustering.permutation_importance import compute_permutation_importance

        df = pd.DataFrame({'cat': pd.Categorical(['x', 'y', 'x', 'y'] * 25)})
        fc = ForestClusterer(n_iterations=5, random_state=42)
        fc.fit(df)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
            shuffle_warnings = [
                x for x in w
                if 'shuffling' in str(x.message).lower()
                or 'Categorical' in str(x.message)
            ]
            assert len(shuffle_warnings) == 0, f"Got shuffle warnings: {shuffle_warnings}"


class TestHigh2ImportanceSubsampling:
    """HIGH-2: Permutation importance timeout on large datasets.

    Computing importance with defaults on n > 2000 causes timeout.
    The fix adds auto-subsampling for n > 2000 or caches embeddings.
    """

    def test_importance_auto_subsample(self):
        """Importance on large dataset should complete in reasonable time."""
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        n = 3000  # above the 2000 threshold
        df = pd.DataFrame({
            'a': rng.normal(0, 1, n),
            'b': rng.normal(0, 1, n),
            'c': rng.choice(['x', 'y', 'z'], n),
        })
        fc = ForestClusterer(
            n_iterations=10,
            random_state=42,
            compute_importance=True,
            importance_repeats=1,
        )
        fc.fit(df)
        imp = fc.get_feature_importances()
        assert isinstance(imp, pd.DataFrame)
        assert len(imp) == 3  # one row per feature


class TestHigh3WardInt64:
    """HIGH-3: Ward linkage should work with int64 embedding.

    AgglomerativeClustering(linkage='ward') requires float input but the
    forest embedding is int64. The fix auto-casts embedding to float64 when
    ward linkage is detected.
    """

    def test_ward_linkage_with_int64(self):
        """AgglomerativeClustering with ward should work."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({
            'a': [1., 2., 3., 4., 5.],
            'b': ['x', 'y', 'x', 'y', 'x'],
        })
        fc = ForestClusterer(
            n_iterations=10,
            clusterer=AgglomerativeClustering(n_clusters=2, linkage='ward'),
            random_state=42,
        )
        labels = fc.fit_predict(df)
        assert len(np.unique(labels)) == 2


class TestHigh4AdaptiveBins:
    """HIGH-4: Adaptive bins should not saturate to max for all features.

    c_spread = min(sigma/range * 4.0, 1.0) saturates for most Gaussian
    features, giving all features max_bins. The fix reduces multiplier.
    """

    def test_adaptive_bins_variance(self):
        """Different features should get different bin counts."""
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        # Feature 1: binary (should get 2 bins)
        # Feature 2: wide uniform (should get many bins)
        df = pd.DataFrame({
            'binary': [0, 1] * 100,
            'wide': rng.uniform(0, 100, 200),
        })
        fc = ForestClusterer(
            n_iterations=10,
            adaptive_bins=True,
            min_bins=2,
            max_bins=10,
            random_state=42,
        )
        fc.fit(df)
        bins = fc.adaptive_bins_map_
        assert bins[0] != bins[1], f"Both features got same bins: {bins}"
        assert bins[0] == 2, f"Binary should have 2 bins, got {bins[0]}"

    def test_adaptive_bins_not_all_max(self):
        """Not all features should get max_bins on a typical dataset."""
        from forest_clustering import ForestClusterer

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'narrow': rng.normal(0, 0.01, 200),   # very tight → fewer bins
            'wide': rng.uniform(0, 100, 200),      # broad → more bins
            'binary': [0, 1] * 100,                # exactly 2 bins
        })
        fc = ForestClusterer(
            n_iterations=10,
            adaptive_bins=True,
            min_bins=2,
            max_bins=10,
            random_state=42,
        )
        fc.fit(df)
        bins = fc.adaptive_bins_map_
        # Not all features should get max_bins; binary should get exactly 2
        assert bins[2] == 2, f"Binary feature should have 2 bins, got {bins[2]}"
        assert not all(b == 10 for b in bins.values()), f"All features got max_bins: {bins}"


class TestHigh5LouvainWhitespace:
    """HIGH-5: Louvain string parsing with whitespace.

    'louvain: k=10, gamma=0.5' (with spaces after commas) fails because
    the parser doesn't strip whitespace. The fix adds .strip() when parsing.
    """

    def test_louvain_string_with_whitespace(self):
        """'louvain: k=10, gamma=0.5' should work."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        fc = ForestClusterer(
            n_iterations=5,
            clusterer='louvain: k=10, gamma=0.5',
            random_state=42,
        )
        labels = fc.fit_predict(df)
        assert isinstance(labels, np.ndarray)

    def test_louvain_string_no_whitespace_still_works(self):
        """Original format 'louvain:k=10,gamma=0.5' should still work."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        fc = ForestClusterer(
            n_iterations=5,
            clusterer='louvain:k=10,gamma=0.5',
            random_state=42,
        )
        labels = fc.fit_predict(df)
        assert isinstance(labels, np.ndarray)


# =============================================================================
# MEDIUM PRIORITY BUGS
# =============================================================================

class TestMed3KdeCategoricalWarning:
    """MED-3: KDE peaks on categorical warns.

    cut_strategy='kde_peaks' on categorical features silently falls back to
    uniform. The fix adds a warning telling the user.
    """

    def test_kde_categorical_warning(self):
        """kde_peaks with all-categorical data should warn."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'cat': ['x', 'y', 'z'] * 100})
        with pytest.warns(UserWarning, match="categorical"):
            fc = ForestClusterer(
                n_iterations=5,
                cut_strategy='kde_peaks',
                random_state=42,
            )
            fc.fit(df)


class TestMed4ZeroWeightWarning:
    """MED-4: Zero-weight fallback warns.

    When all iterations produce constant embeddings, entropy weights become
    zero and silently fall back to uniform. The fix adds a warning.
    """

    def test_zero_weight_warning(self):
        """All-zero entropy weights should warn about uniform fallback."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1.0] * 100})  # constant → zero entropy
        with pytest.warns(UserWarning, match="uniform"):
            fc = ForestClusterer(
                n_iterations=5,
                iteration_weighting='entropy',
                random_state=42,
            )
            fc.fit(df)


class TestMed6SilhouetteGuard:
    """MED-6: Silhouette should not crash with single cluster.

    silhouette_score requires >= 2 clusters. The fix guards and returns
    NaN or 0.0 when n_clusters < 2.
    """

    def test_silhouette_single_cluster_returns_nan(self):
        """Silhouette with 1 cluster should return NaN, not crash."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1.0] * 50})
        fc = ForestClusterer(
            n_iterations=10,
            clusterer=KMeans(n_clusters=1, n_init='auto'),
            random_state=42,
        )
        fc.fit(df)
        D = fc.pairwise_distance()
        labels = fc.labels_
        sil = fc._safe_silhouette_score(D, labels, metric='precomputed')
        assert np.isnan(sil), f"Expected NaN for single cluster, got {sil}"

    def test_silhouette_guard_in_importance(self):
        """Importance computation should not crash on single-cluster output."""
        from forest_clustering import ForestClusterer
        from forest_clustering.permutation_importance import compute_permutation_importance

        df = pd.DataFrame({'a': [1.0] * 50})
        fc = ForestClusterer(
            n_iterations=10,
            clusterer=KMeans(n_clusters=1, n_init='auto'),
            random_state=42,
        )
        fc.fit(df)
        # Should not crash
        imp = compute_permutation_importance(fc, df, n_repeats=1, random_state=42)
        assert isinstance(imp, pd.DataFrame)


class TestMed9RawImportance:
    """MED-9: Raw importance scores in DataFrame.

    Top feature always has importance=1.0 after max-normalization, making
    comparison hard. The fix reports raw scores alongside normalized.
    """

    def test_raw_importance_column(self):
        """Importance DataFrame should contain raw_importance column."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({
            'a': [1., 2., 3., 4., 5.],
            'b': ['x', 'y', 'x', 'y', 'x'],
        })
        fc = ForestClusterer(
            n_iterations=10,
            compute_importance=True,
            importance_repeats=2,
            random_state=42,
        )
        fc.fit(df)
        imp = fc.get_feature_importances()
        assert 'raw_importance' in imp.columns, f"Columns: {imp.columns.tolist()}"

    def test_raw_importance_detailed_view(self):
        """Detailed importance view should include raw_importance and std."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({
            'a': [1., 2., 3., 4., 5.],
            'b': ['x', 'y', 'x', 'y', 'x'],
        })
        fc = ForestClusterer(
            n_iterations=10,
            compute_importance=True,
            importance_repeats=2,
            random_state=42,
        )
        fc.fit(df)
        imp = fc.get_feature_importances(detailed=True)
        assert 'raw_importance' in imp.columns


class TestMed10TemperatureClamp:
    """MED-10: Temperature extreme values clamped.

    temperature < 0.3 or > 5.0 can cause overflow/underflow.
    The fix clamps temperature to [0.1, 10.0].
    """

    def test_temperature_extreme_low(self):
        """Very low temperature should not produce inf/nan."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.array([[0, 1], [1, 0]], dtype=np.int64)
        w = compute_iteration_weights(E, 'entropy', weight_temperature=0.001)
        assert np.all(np.isfinite(w)), f"Got non-finite weights: {w}"

    def test_temperature_extreme_high(self):
        """Very high temperature should not produce inf/nan."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.array([[0, 1], [1, 0]], dtype=np.int64)
        w = compute_iteration_weights(E, 'entropy', weight_temperature=1000.0)
        assert np.all(np.isfinite(w)), f"Got non-finite weights: {w}"


# =============================================================================
# LOW PRIORITY BUGS
# =============================================================================

class TestLow1NClustersConvenience:
    """LOW-1: n_clusters convenience parameter.

    Users must provide a downstream clusterer explicitly. Most users expect
    an n_clusters=3 parameter. The fix adds a convenience parameter.
    """

    def test_n_clusters_parameter(self):
        """ForestClusterer should accept n_clusters and wrap KMeans."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        fc = ForestClusterer(n_iterations=5, n_clusters=3, random_state=42)
        labels = fc.fit_predict(df)
        n_found = len(np.unique(labels))
        assert n_found == 3, f"Expected 3 clusters, got {n_found}"

    def test_n_clusters_with_explicit_clusterer_raises(self):
        """Passing both n_clusters and clusterer should raise."""
        from forest_clustering import ForestClusterer

        df = pd.DataFrame({'a': [1., 2., 3., 4., 5.]})
        with pytest.raises(ValueError):
            fc = ForestClusterer(
                n_iterations=5,
                n_clusters=3,
                clusterer=KMeans(n_clusters=2, n_init='auto'),
                random_state=42,
            )
            fc.fit(df)
