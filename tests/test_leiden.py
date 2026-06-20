"""Tests for the Leiden community-detection backend (community_method='leiden')."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from forest_clustering import ForestClusterer
from forest_clustering.graph_clustering import GraphLouvainClusterer

leidenalg = pytest.importorskip("leidenalg")


def _blobs(n_per=70, k=4, p=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(k, p))
    X = np.vstack([rng.normal(centers[i], 1.0, size=(n_per, p)) for i in range(k)])
    y = np.repeat(np.arange(k), n_per)
    return X, y


def _embedding(X, seed=0):
    return ForestClusterer(n_iterations=120, n_bins=3, corr_threshold=None,
                           random_state=seed).fit(X).embedding_


class TestLeidenBackend:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            GraphLouvainClusterer(community_method='nope')

    def test_leiden_recovers_blobs(self):
        X, y = _blobs()
        E = _embedding(X)
        gc = GraphLouvainClusterer(n_neighbors=15, community_method='leiden',
                                   random_state=0).fit_embedding(E, method='knn')
        assert gc.labels_.shape == (X.shape[0],)
        assert adjusted_rand_score(y, gc.labels_) > 0.9

    def test_leiden_matches_louvain_on_easy_blobs(self):
        X, y = _blobs()
        E = _embedding(X)
        lou = GraphLouvainClusterer(n_neighbors=15, community_method='louvain',
                                    random_state=0).fit_embedding(E, method='knn')
        lei = GraphLouvainClusterer(n_neighbors=15, community_method='leiden',
                                    random_state=0).fit_embedding(E, method='knn')
        assert abs(adjusted_rand_score(y, lou.labels_)
                   - adjusted_rand_score(y, lei.labels_)) < 0.1

    def test_resolution_increases_communities(self):
        X, _ = _blobs()
        E = _embedding(X)
        lo = GraphLouvainClusterer(community_method='leiden', resolution=0.3,
                                   random_state=0).fit_embedding(E, method='knn')
        hi = GraphLouvainClusterer(community_method='leiden', resolution=3.0,
                                   random_state=0).fit_embedding(E, method='knn')
        n_lo = len(np.unique(lo.labels_[lo.labels_ >= 0]))
        n_hi = len(np.unique(hi.labels_[hi.labels_ >= 0]))
        assert n_hi >= n_lo

    def test_determinism(self):
        X, _ = _blobs()
        E = _embedding(X)
        a = GraphLouvainClusterer(community_method='leiden', random_state=7
                                  ).fit_embedding(E, method='knn').labels_
        b = GraphLouvainClusterer(community_method='leiden', random_state=7
                                  ).fit_embedding(E, method='knn').labels_
        np.testing.assert_array_equal(a, b)


class TestLeidenStringShortcut:
    def test_clusterer_leiden_shortcut(self):
        X, y = _blobs()
        labels = ForestClusterer(n_iterations=120, n_bins=3, clusterer='leiden',
                                 corr_threshold=None, random_state=0).fit_predict(X)
        assert labels.shape == (X.shape[0],)
        assert adjusted_rand_score(y, labels) > 0.9

    def test_leiden_shortcut_with_params(self):
        X, _ = _blobs()
        labels = ForestClusterer(n_iterations=120, n_bins=3,
                                 clusterer='leiden:k=10,resolution=1.5',
                                 corr_threshold=None, random_state=0).fit_predict(X)
        assert labels.shape == (X.shape[0],)
