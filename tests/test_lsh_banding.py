"""Tests for LSH banding kNN graph (forest_clustering.lsh_graph.lsh_banding_knn).

Path 2: build a sparse kNN graph from the cell-id embedding by banding the
columns (LSH), computing exact Hamming only on candidate pairs.  These tests pin
graph validity, recall against brute force on structured data, determinism, and
the GraphLouvainClusterer integration with method='banding'.
"""

import numpy as np
import pytest
from scipy import sparse

from forest_clustering.lsh_graph import lsh_banding_knn, batched_hamming_knn


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def clustered_embedding(n_per=80, n_clusters=4, L=120, K=3, p=12, seed=0):
    """A real forest embedding on well-separated blobs (has neighbour structure)."""
    from forest_clustering.feature_encoder import DataEncoder
    from forest_clustering.partitioner import (
        build_col_stats, build_iteration_specs, compute_embedding,
    )
    r = np.random.default_rng(seed)
    centers = r.normal(0, 6, size=(n_clusters, p))
    X = np.vstack([r.normal(centers[i], 1.0, size=(n_per, p))
                   for i in range(n_clusters)]).astype(np.float64)
    y = np.repeat(np.arange(n_clusters), n_per)
    enc = DataEncoder(); Xe = enc.fit_transform(X)
    stats = build_col_stats(Xe, enc.feature_types_)
    specs = build_iteration_specs(
        n_iterations=L, col_stats=stats,
        n_features_per_iter=max(1, int(np.sqrt(p))), n_bins=K,
        feature_weights=np.ones(len(stats)), rng=np.random.default_rng(seed + 1),
    )
    return compute_embedding(Xe, specs, n_jobs=1), y


def brute_knn_idx(E, k):
    D = (E[:, None, :] != E[None, :, :]).sum(2)
    np.fill_diagonal(D, E.shape[1] + 1)
    return np.argsort(D, axis=1)[:, :k]


class TestGraphValidity:
    def test_shape_and_type(self, rng):
        E = rng.integers(0, 5, size=(60, 40)).astype(np.int64)
        G = lsh_banding_knn(E, k=8, band_size=3)
        assert isinstance(G, sparse.coo_matrix)
        assert G.shape == (60, 60)

    def test_distances_nonneg_integer_bounded(self, rng):
        E = rng.integers(0, 5, size=(60, 40)).astype(np.int64)
        G = lsh_banding_knn(E, k=8, band_size=3)
        assert np.all(G.data >= 0)
        assert np.all(G.data == G.data.astype(int))
        assert np.all(G.data <= E.shape[1])  # column-Hamming <= L

    def test_no_self_loops_and_k_cap(self, rng):
        E = rng.integers(0, 5, size=(60, 40)).astype(np.int64)
        k = 8
        G = lsh_banding_knn(E, k=k, band_size=2).tocsr()
        for i in range(60):
            nbrs = G[i].indices
            assert i not in nbrs
            assert len(nbrs) <= k

    def test_stored_distances_are_exact_column_hamming(self, rng):
        E = rng.integers(0, 4, size=(40, 30)).astype(np.int64)
        G = lsh_banding_knn(E, k=5, band_size=2).tocsr()
        for i in range(40):
            for j, d in zip(G[i].indices, G[i].data):
                assert d == int((E[i] != E[j]).sum())


class TestRecall:
    def test_high_recall_on_structured_data(self):
        """On well-separated blobs banding should recover most true neighbours."""
        E, _ = clustered_embedding()
        k = 15
        true = brute_knn_idx(E, k)
        G = lsh_banding_knn(E, k=k, band_size=4, random_state=0).tocsr()
        recall = np.mean([
            len(set(true[i]) & set(G[i].indices)) / k for i in range(E.shape[0])
        ])
        assert recall > 0.8, f"recall={recall:.3f} too low"

    def test_smaller_bands_increase_recall(self):
        E, _ = clustered_embedding()
        k = 15
        true = brute_knn_idx(E, k)

        def rec(bs):
            G = lsh_banding_knn(E, k=k, band_size=bs, random_state=0).tocsr()
            return np.mean([len(set(true[i]) & set(G[i].indices)) / k
                            for i in range(E.shape[0])])

        assert rec(3) >= rec(8) - 1e-9


class TestAutoBandSize:
    def test_returns_valid_band_size(self):
        from forest_clustering.lsh_graph import auto_band_size
        E, _ = clustered_embedding()
        bs = auto_band_size(E, k=15, random_state=0)
        assert isinstance(bs, int)
        assert 1 <= bs <= E.shape[1]

    def test_auto_string_runs_and_is_valid(self):
        E, _ = clustered_embedding()
        G = lsh_banding_knn(E, k=15, band_size='auto', random_state=0)
        assert isinstance(G, sparse.coo_matrix)
        assert G.shape == (E.shape[0], E.shape[0])
        assert np.all(G.data >= 0)

    def test_auto_high_recall(self):
        E, _ = clustered_embedding()
        k = 15
        true = brute_knn_idx(E, k)
        G = lsh_banding_knn(E, k=k, band_size='auto', random_state=0).tocsr()
        recall = np.mean([len(set(true[i]) & set(G[i].indices)) / k
                          for i in range(E.shape[0])])
        assert recall > 0.8, f"auto recall={recall:.3f}"


class TestCompactCodes:
    def test_preserves_equality(self, rng):
        from forest_clustering.lsh_graph import _compact_codes
        E = (rng.integers(0, 6, size=(50, 20)).astype(np.int64)
             * rng.integers(1, 10 ** 6))
        Ec = _compact_codes(E)
        for _ in range(20):
            i, j = rng.integers(0, 50, size=2)
            assert (E[i] != E[j]).sum() == (Ec[i] != Ec[j]).sum()

    def test_uses_small_dtype(self, rng):
        from forest_clustering.lsh_graph import _compact_codes
        E = rng.integers(0, 5, size=(100, 10)).astype(np.int64)
        assert _compact_codes(E).dtype == np.uint8


class TestDeterminism:
    def test_repeated_runs_identical(self, rng):
        E = rng.integers(0, 5, size=(80, 50)).astype(np.int64)
        a = lsh_banding_knn(E, k=10, band_size=3, random_state=7)
        b = lsh_banding_knn(E, k=10, band_size=3, random_state=7)
        assert (a.tocsr() != b.tocsr()).nnz == 0


class TestEdgeCases:
    def test_empty(self):
        G = lsh_banding_knn(np.empty((0, 10), dtype=np.int64), k=5)
        assert G.shape == (0, 0)

    def test_single_row(self):
        G = lsh_banding_knn(np.array([[1, 2, 3]], dtype=np.int64), k=5)
        assert G.shape == (1, 1)
        assert G.nnz == 0

    def test_invalid_band_size(self, rng):
        E = rng.integers(0, 5, size=(10, 10)).astype(np.int64)
        with pytest.raises(ValueError):
            lsh_banding_knn(E, k=3, band_size=0)


class TestLouvainIntegration:
    def test_fit_embedding_banding_runs(self):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        E, y = clustered_embedding()
        gc = GraphLouvainClusterer(n_neighbors=15, random_state=0)
        gc.fit_embedding(E, method='banding', band_size=4)
        assert gc.labels_.shape == (E.shape[0],)
        assert hasattr(gc, 'knn_graph_')
        # should find a sensible number of communities on 4 blobs
        n_comm = len(np.unique(gc.labels_[gc.labels_ >= 0]))
        assert n_comm >= 2

    def test_banding_quality_matches_exact_knn(self):
        """Banding-based Louvain should match exact-kNN Louvain on easy blobs."""
        from sklearn.metrics import adjusted_rand_score
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        E, y = clustered_embedding()
        knn = GraphLouvainClusterer(n_neighbors=15, random_state=0)
        knn.fit_embedding(E, method='knn')
        band = GraphLouvainClusterer(n_neighbors=15, random_state=0)
        band.fit_embedding(E, method='banding', band_size=4)
        ari_knn = adjusted_rand_score(y, knn.labels_)
        ari_band = adjusted_rand_score(y, band.labels_)
        assert ari_band >= ari_knn - 0.1

    def test_invalid_method_raises(self, rng):
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        E = rng.integers(0, 5, size=(30, 20)).astype(np.int64)
        gc = GraphLouvainClusterer()
        with pytest.raises(ValueError):
            gc.fit_embedding(E, method='nope')
