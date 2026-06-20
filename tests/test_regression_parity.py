"""Regression guards: the scalable refactor must not change small-`n` results.

- Centroid path: KMeans on the sparse one-hot must equal KMeans on the dense
  one-hot (bit-identical labels) given the same embedding.
- Graph path: for `n <= _DENSE_GRAPH_MAX_N` the ForestClusterer Louvain route
  must use the exact dense distance matrix and match GraphLouvainClusterer.fit(D)
  directly (i.e. the classic behaviour, exact tie-breaking).
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from forest_clustering import ForestClusterer
from forest_clustering.clusterer import _DENSE_GRAPH_MAX_N
from forest_clustering.sparse_features import weighted_onehot_features
from forest_clustering.graph_clustering import GraphLouvainClusterer


@pytest.fixture
def categorical_X():
    rng = np.random.default_rng(0)
    # 4 latent groups expressed through low-cardinality categorical codes
    base = rng.integers(0, 3, size=(4, 8))
    rows, y = [], []
    for g in range(4):
        for _ in range(60):
            r = base[g].copy()
            flip = rng.random(8) < 0.2
            r[flip] = rng.integers(0, 3, size=flip.sum())
            rows.append(r); y.append(g)
    return np.array(rows, dtype=float), np.array(y)


def test_sparse_dense_kmeans_identical(categorical_X):
    X, _ = categorical_X
    E = ForestClusterer(n_iterations=120, n_bins=3, corr_threshold=None,
                        random_state=0).fit(X).embedding_
    Pd = weighted_onehot_features(E, sparse_output=False)
    Ps = weighted_onehot_features(E, sparse_output=True)
    ld = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(Pd)
    ls = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(Ps)
    np.testing.assert_array_equal(ld, ls)


def test_small_n_louvain_matches_dense_matrix(categorical_X):
    X, _ = categorical_X
    assert X.shape[0] <= _DENSE_GRAPH_MAX_N
    fc = ForestClusterer(n_iterations=120, n_bins=3, clusterer="louvain",
                         corr_threshold=None, random_state=0)
    labels_via_clusterer = fc.fit_predict(X)
    # Reproduce the dense route directly
    D = fc.pairwise_distance().astype(np.float64)
    ref = GraphLouvainClusterer(random_state=0).fit_predict(D)
    np.testing.assert_array_equal(labels_via_clusterer, ref)


def test_small_n_uses_dense_not_banding(categorical_X, monkeypatch):
    """For small n the banding builder must not be invoked by the Louvain route."""
    import forest_clustering.graph_clustering as gcmod
    called = {"banding": False}
    orig = gcmod.GraphLouvainClusterer.fit_embedding

    def spy(self, *a, **k):
        called["banding"] = True
        return orig(self, *a, **k)

    monkeypatch.setattr(gcmod.GraphLouvainClusterer, "fit_embedding", spy)
    X, _ = categorical_X
    ForestClusterer(n_iterations=80, n_bins=3, clusterer="louvain",
                    corr_threshold=None, random_state=0).fit_predict(X)
    assert called["banding"] is False
