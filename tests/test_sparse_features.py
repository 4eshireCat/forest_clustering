"""Tests for sparse weighted one-hot features (forest_clustering.sparse_features).

Path 1: the (n, L) cell-id embedding -> weighted one-hot feature matrix with
exactly L non-zeros per row, where squared Euclidean distance equals the
weighted Hamming distance.  These tests pin that equivalence and the sparsity.
"""

import numpy as np
import pytest
from scipy import sparse

from forest_clustering.sparse_features import (
    weighted_onehot_features,
    estimator_supports_sparse,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def E(rng):
    # K-ary cell ids with large, arbitrary magnitudes (like the real hashes)
    return (rng.integers(0, 8, size=(120, 40)).astype(np.int64)
            * rng.integers(1, 10 ** 6))


def _sqdist(P):
    P = P.toarray() if sparse.issparse(P) else P
    G = P @ P.T
    d = np.diag(G)
    return d[:, None] + d[None, :] - 2 * G


def weighted_hamming(E, w):
    w = w / w.sum()
    n = E.shape[0]
    D = np.zeros((n, n))
    for l in range(E.shape[1]):
        D += w[l] * (E[:, l][:, None] != E[:, l][None, :])
    return D


class TestSparsity:
    def test_exactly_L_nonzeros_per_row(self, E):
        P = weighted_onehot_features(E, sparse_output=True)
        assert sparse.issparse(P)
        assert P.nnz == E.shape[0] * E.shape[1]
        assert np.all(P.getnnz(axis=1) == E.shape[1])

    def test_sparse_equals_dense(self, E):
        Ps = weighted_onehot_features(E, sparse_output=True)
        Pd = weighted_onehot_features(E, sparse_output=False)
        assert not sparse.issparse(Pd)
        np.testing.assert_allclose(Ps.toarray(), Pd)

    def test_width_is_sum_of_cardinalities(self, E):
        P = weighted_onehot_features(E, sparse_output=True)
        expected = sum(np.unique(E[:, l]).size for l in range(E.shape[1]))
        assert P.shape == (E.shape[0], expected)


class TestHammingEquivalence:
    def test_sqdist_equals_uniform_hamming(self, E):
        P = weighted_onehot_features(E, sparse_output=True)
        D = weighted_hamming(E, np.ones(E.shape[1]))
        np.testing.assert_allclose(_sqdist(P), D, atol=1e-9)

    def test_sqdist_equals_weighted_hamming(self, E, rng):
        w = rng.uniform(0.1, 5.0, size=E.shape[1])
        P = weighted_onehot_features(E, weights=w, sparse_output=True)
        D = weighted_hamming(E, w)
        np.testing.assert_allclose(_sqdist(P), D, atol=1e-9)

    def test_invalid_weights_fall_back_to_uniform(self, E):
        bad = np.zeros(E.shape[1])  # sums to zero -> uniform fallback
        P = weighted_onehot_features(E, weights=bad, sparse_output=True)
        D = weighted_hamming(E, np.ones(E.shape[1]))
        np.testing.assert_allclose(_sqdist(P), D, atol=1e-9)


class TestEdgeCases:
    def test_empty_rows(self):
        P = weighted_onehot_features(np.empty((0, 5), dtype=np.int64))
        assert P.shape[0] == 0

    def test_single_row(self):
        P = weighted_onehot_features(np.array([[1, 2, 3]], dtype=np.int64))
        assert P.shape[0] == 1
        assert P.nnz == 3

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            weighted_onehot_features(np.arange(5, dtype=np.int64))


class TestEstimatorSupportsSparse:
    def test_kmeans_family_supported(self):
        from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
        assert estimator_supports_sparse(KMeans())
        assert estimator_supports_sparse(MiniBatchKMeans())
        assert estimator_supports_sparse(Birch())

    def test_dense_only_estimators_not_supported(self):
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import AgglomerativeClustering
        assert not estimator_supports_sparse(GaussianMixture())
        assert not estimator_supports_sparse(AgglomerativeClustering())


class TestForestClustererIntegration:
    def test_default_kmeans_uses_sparse_and_clusters(self):
        """Default KMeans path must run via sparse features and recover blobs."""
        from sklearn.metrics import adjusted_rand_score
        from forest_clustering import ForestClusterer
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(c, 1.0, size=(80, 6))
                       for c in rng.normal(0, 6, size=(3, 6))])
        y = np.repeat(np.arange(3), 80)
        fc = ForestClusterer(n_iterations=120, n_bins=3, n_clusters=3,
                             corr_threshold=None, random_state=0)
        labels = fc.fit_predict(X)
        assert labels.shape == (240,)
        assert adjusted_rand_score(y, labels) > 0.7
