"""Tests for graph-based clustering with Louvain community detection.

All tests should fail (ImportError / AttributeError) until the production
module ``forest_clustering.graph_clustering`` exists.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestGraphLouvainBasic:
    """Basic functionality."""

    def test_class_importable(self):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        assert hasattr(GraphLouvainClusterer, 'fit')
        assert hasattr(GraphLouvainClusterer, 'labels_')

    def test_finds_clusters_on_blobs(self):
        """Well-separated Gaussian blobs -> Louvain should find multiple clusters."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from sklearn.datasets import make_blobs
        from scipy.spatial.distance import cdist

        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6,
                          random_state=42, center_box=(-5, 5))
        D = cdist(X, X, metric='euclidean')

        clusterer = GraphLouvainClusterer(n_neighbors=15, random_state=42)
        clusterer.fit(D)

        n_found = len(np.unique(clusterer.labels_[clusterer.labels_ >= 0]))
        assert n_found >= 2, f"Expected >=2 clusters, got {n_found}"
        assert len(clusterer.labels_) == 300

    def test_labels_are_integers(self):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        D = cdist(X, X)

        clusterer = GraphLouvainClusterer(n_neighbors=10, random_state=42)
        clusterer.fit(D)

        assert clusterer.labels_.dtype in (np.int32, np.int64, int)

    def test_noise_marking(self):
        """noise_strategy='mark' should produce -1 labels for outliers."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((100, 5))
        D = cdist(X, X)

        clusterer = GraphLouvainClusterer(n_neighbors=5, noise_strategy='mark', random_state=42)
        clusterer.fit(D)

        # Some noise labels may exist (or may not -- depends on graph structure)
        # Just verify no crash
        assert len(clusterer.labels_) == 100

    def test_all_noise_strategy(self):
        """Test all three noise strategies work without crash."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        D = cdist(X, X)

        for strategy in ['mark', 'merge', 'singleton']:
            clusterer = GraphLouvainClusterer(n_neighbors=10, noise_strategy=strategy, random_state=42)
            clusterer.fit(D)
            assert len(clusterer.labels_) == 50

    def test_deterministic_with_same_seed(self):
        """Same random_state -> same labels."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        D = cdist(X, X)

        c1 = GraphLouvainClusterer(n_neighbors=10, random_state=42)
        c1.fit(D)
        c2 = GraphLouvainClusterer(n_neighbors=10, random_state=42)
        c2.fit(D)

        np.testing.assert_array_equal(c1.labels_, c2.labels_)


# ---------------------------------------------------------------------------
# Resolution parameter
# ---------------------------------------------------------------------------

class TestResolutionParameter:
    """Resolution parameter gamma controls granularity."""

    def test_higher_resolution_more_clusters(self):
        """Higher resolution -> more communities (generally)."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from sklearn.datasets import make_blobs
        from scipy.spatial.distance import cdist

        X, _ = make_blobs(n_samples=200, centers=5, cluster_std=1.0, random_state=42)
        D = cdist(X, X)

        c_low = GraphLouvainClusterer(n_neighbors=15, resolution=0.5, random_state=42)
        c_low.fit(D)
        n_low = len(np.unique(c_low.labels_[c_low.labels_ >= 0]))

        c_high = GraphLouvainClusterer(n_neighbors=15, resolution=2.0, random_state=42)
        c_high.fit(D)
        n_high = len(np.unique(c_high.labels_[c_high.labels_ >= 0]))

        assert n_high >= n_low, f"Expected {n_high} >= {n_low} for higher resolution"

    def test_resolution_one_default(self):
        """resolution=1.0 should be default behavior."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        c = GraphLouvainClusterer(n_neighbors=15)
        assert c.resolution == 1.0


# ---------------------------------------------------------------------------
# Weight transforms
# ---------------------------------------------------------------------------

class TestWeightTransforms:
    """Different weight transforms."""

    def test_all_transforms_work(self):
        """All weight transforms should work without crash."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        D = cdist(X, X)

        for transform in ['exp', 'linear', 'inverse']:
            c = GraphLouvainClusterer(n_neighbors=10, weight_transform=transform, random_state=42)
            c.fit(D)
            assert len(c.labels_) == 50
            assert len(np.unique(c.labels_)) >= 1


# ---------------------------------------------------------------------------
# K parameter
# ---------------------------------------------------------------------------

class TestKNeighborsParameter:
    """k parameter affects graph connectivity."""

    def test_low_k_produces_many_clusters(self):
        """Very low k -> sparse graph -> many small communities."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((100, 3))
        D = cdist(X, X)

        c_low = GraphLouvainClusterer(n_neighbors=3, random_state=42)
        c_low.fit(D)
        n_low = len(np.unique(c_low.labels_))

        c_high = GraphLouvainClusterer(n_neighbors=30, random_state=42)
        c_high.fit(D)
        n_high = len(np.unique(c_high.labels_))

        assert n_low >= n_high, f"Expected {n_low} >= {n_high} for lower k"


# ---------------------------------------------------------------------------
# Integration with ForestClusterer
# ---------------------------------------------------------------------------

class TestForestClustererIntegration:
    """Integration with ForestClusterer."""

    def test_louvain_via_clusterer_param(self, sample_data):
        """Pass GraphLouvainClusterer as clusterer parameter."""
        from forest_clustering import ForestClusterer
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        gc = GraphLouvainClusterer(n_neighbors=10, random_state=42)
        fc = ForestClusterer(n_iterations=10, clusterer=gc, random_state=42)
        labels = fc.fit_predict(sample_data)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_louvain_string_shortcut(self, sample_data):
        """Use 'louvain' string as clusterer."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, clusterer='louvain', random_state=42)
        labels = fc.fit_predict(sample_data)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_louvain_string_with_params(self, sample_data):
        """Use 'louvain:k=15,gamma=0.5' string with params."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, clusterer='louvain:k=15,gamma=0.5', random_state=42)
        labels = fc.fit_predict(sample_data)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_louvain_vs_kmeans_different_labels(self, sample_data):
        """Louvain and KMeans should potentially produce different results."""
        from forest_clustering import ForestClusterer
        from sklearn.cluster import KMeans

        fc_louvain = ForestClusterer(n_iterations=10, clusterer='louvain', random_state=42)
        l_louvain = fc_louvain.fit_predict(sample_data)

        fc_kmeans = ForestClusterer(n_iterations=10,
                                     clusterer=KMeans(n_clusters=3, n_init='auto', random_state=42),
                                     random_state=42)
        l_kmeans = fc_kmeans.fit_predict(sample_data)

        n_louvain = len(np.unique(l_louvain[l_louvain >= 0]))
        n_kmeans = len(np.unique(l_kmeans))
        print(f"Louvain: {n_louvain} clusters, KMeans: {n_kmeans} clusters")
        # They may differ -- that's expected
        assert n_louvain >= 1 and n_kmeans >= 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Invalid inputs."""

    def test_invalid_weight_transform_raises(self):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        with pytest.raises(ValueError):
            GraphLouvainClusterer(weight_transform='bogus')

    def test_invalid_noise_strategy_raises(self):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        with pytest.raises(ValueError):
            GraphLouvainClusterer(noise_strategy='bogus')

    def test_n_neighbors_too_large(self):
        """k >= n should be handled gracefully."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        from scipy.spatial.distance import cdist

        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        D = cdist(X, X)

        c = GraphLouvainClusterer(n_neighbors=100, random_state=42)  # k > n
        c.fit(D)  # should not crash
        assert len(c.labels_) == 5
