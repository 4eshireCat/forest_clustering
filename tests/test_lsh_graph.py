"""Comprehensive TDD tests for forest_clustering.lsh_graph module.

These tests cover batched_hamming_knn() and GraphLouvainClusterer.fit_embedding()
with extensive edge cases, correctness invariants, and parametrized scenarios.

All tests are designed to:
- FAIL on an empty / stub implementation (ImportError, AttributeError, or assertion failure)
- PASS on a correct implementation

Spec reference: LSH_GRAPH_SPEC.md (Sections 1-9)
"""

import numpy as np
import pytest
from scipy import sparse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def binary_embedding_5x8():
    """Small deterministic binary embedding: 5 points, 8 dimensions.

    Known Hamming distances (computed manually):
    - Point 0: [1,0,1,1,0,0,1,1]
    - Point 1: [1,0,1,0,0,1,1,0]
    - Point 2: [0,1,0,1,1,0,0,1]
    - Point 3: [0,1,0,0,1,1,0,0]
    - Point 4: [1,0,1,1,0,0,1,0]
    """
    return np.array([
        [1, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
    ], dtype=np.uint8)


@pytest.fixture
def embedding_50x32(rng):
    """Medium-sized random binary embedding: 50 points, 32 dimensions."""
    return rng.randint(0, 2, size=(50, 32), dtype=np.uint8)


@pytest.fixture
def embedding_100x256(rng):
    """Large-dimension random binary embedding: 100 points, 256 dimensions."""
    return rng.randint(0, 2, size=(100, 256), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helper functions for brute-force correctness verification
# ---------------------------------------------------------------------------

def brute_force_hamming_knn(E, k):
    """Brute-force kNN graph using Hamming distance.

    Returns a scipy.sparse.coo_matrix of shape (n, n) with exact kNN edges.
    This is the ground-truth reference implementation.
    """
    n, m = E.shape
    k_eff = min(k, n - 1)

    if n == 0:
        return sparse.coo_matrix((0, 0), dtype=np.uint16)
    if k_eff == 0:
        return sparse.coo_matrix((n, n), dtype=np.uint16)

    # Compute full Hamming distance matrix
    D_full = np.zeros((n, n), dtype=np.uint16)
    for i in range(n):
        for j in range(n):
            D_full[i, j] = np.sum(E[i] != E[j])

    rows, cols, data = [], [], []
    for i in range(n):
        dists = D_full[i].copy()
        dists[i] = np.iinfo(np.uint16).max  # exclude self
        top_k_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_k_dist = dists[top_k_idx]
        # Deterministic stable sort
        sort_order = np.argsort(top_k_dist, kind='mergesort')
        top_k_idx = top_k_idx[sort_order]
        top_k_dist = top_k_dist[sort_order]

        rows.extend([i] * k_eff)
        cols.extend(top_k_idx.tolist())
        data.extend(top_k_dist.tolist())

    return sparse.coo_matrix(
        (np.array(data, dtype=np.uint16),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )


def compute_hamming_distance_matrix(E):
    """Compute the full n x n Hamming distance matrix."""
    n = E.shape[0]
    D = np.zeros((n, n), dtype=np.uint16)
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sum(E[i] != E[j])
    return D


def check_knn_correctness(G, E, k):
    """Verify that sparse matrix G is a correct kNN graph for embedding E.

    Checks:
    1. Output shape = (n, n)
    2. Each row has exactly k_eff non-zero entries
    3. All non-zero entries are among the k_eff smallest Hamming distances
    4. No self-loops
    5. All distances are non-negative integers
    """
    n = E.shape[0]
    k_eff = min(k, n - 1)

    assert G.shape == (n, n), f"Expected shape ({n},{n}), got {G.shape}"

    if k_eff == 0:
        assert G.nnz == 0
        return

    # Invariant 2: entry count
    expected_nnz = n * k_eff
    assert G.nnz == expected_nnz, f"Expected nnz={expected_nnz}, got {G.nnz}"

    # Convert to CSR for efficient row access
    G_csr = G.tocsr()

    # Compute full distance matrix
    D_full = compute_hamming_distance_matrix(E)

    for i in range(n):
        row = G_csr[i]
        neighbors = row.indices
        distances = row.data

        # Invariant 3: no self-loops
        assert i not in neighbors, f"Self-loop found at row {i}"

        # Invariant for row: exactly k_eff neighbors
        assert len(neighbors) == k_eff, \
            f"Row {i}: expected {k_eff} neighbors, got {len(neighbors)}"

        # Invariant 4: non-negative integer distances
        assert np.all(distances >= 0), f"Row {i}: negative distances found"
        assert np.all(distances == distances.astype(int)), \
            f"Row {i}: non-integer distances found"

        # Invariant 5: kNN correctness
        # All returned distances must be <= the k_eff-th smallest distance
        sorted_dists = np.sort(D_full[i])
        sorted_dists_excluding_self = sorted_dists[1:]  # first is self (distance 0)
        if len(sorted_dists_excluding_self) >= k_eff:
            kth_dist = sorted_dists_excluding_self[k_eff - 1]
            assert np.all(distances <= kth_dist), \
                f"Row {i}: some distances > kth smallest ({kth_dist})"


# =============================================================================
# TESTS: Import & API Surface
# =============================================================================

class TestImportAndAPISurface:
    """Module and class must be importable with expected API."""

    def test_batched_hamming_knn_importable(self):
        """batched_hamming_knn function must exist and be callable."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        assert callable(batched_hamming_knn)

    def test_batched_hamming_knn_signature(self):
        """Function signature: batched_hamming_knn(E, k=15, batch_size=1000)."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        import inspect
        sig = inspect.signature(batched_hamming_knn)
        params = list(sig.parameters.keys())
        assert 'E' in params
        assert 'k' in params
        assert 'batch_size' in params

    def test_batched_hamming_knn_default_params(self):
        """Check default parameter values."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        import inspect
        sig = inspect.signature(batched_hamming_knn)
        assert sig.parameters['k'].default == 15
        assert sig.parameters['batch_size'].default == 1000

    def test_fit_embedding_method_exists(self):
        """GraphLouvainClusterer must have fit_embedding method."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        assert hasattr(GraphLouvainClusterer, 'fit_embedding')

    def test_fit_embedding_is_callable(self):
        """fit_embedding must be a callable method."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer
        clusterer = GraphLouvainClusterer()
        assert callable(clusterer.fit_embedding)


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases from spec Section 4."""

    def test_empty_embedding_returns_empty_matrix(self):
        """Empty input (n=0) returns 0x0 empty COO matrix with nnz=0."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.empty((0, 10), dtype=np.uint8)
        G = batched_hamming_knn(E, k=5, batch_size=100)

        assert G.shape == (0, 0)
        assert G.nnz == 0
        assert isinstance(G, sparse.coo_matrix)

    def test_empty_embedding_different_k(self):
        """Empty input works for any k value."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.empty((0, 5), dtype=np.uint8)

        for k in [0, 1, 5, 100]:
            G = batched_hamming_knn(E, k=k, batch_size=10)
            assert G.shape == (0, 0), f"Failed for k={k}"
            assert G.nnz == 0, f"Failed for k={k}"

    def test_single_point_returns_no_edges(self):
        """Single point: k_eff = min(k, 0) = 0, so nnz=0."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.array([[1, 0, 1, 1, 0]], dtype=np.uint8)
        G = batched_hamming_knn(E, k=5, batch_size=100)

        assert G.shape == (1, 1)
        assert G.nnz == 0
        assert G.diagonal()[0] == 0

    def test_single_point_different_k(self):
        """Single point with various k values always gives empty graph."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.array([[0, 1, 0]], dtype=np.uint8)

        for k in [0, 1, 5, 100]:
            G = batched_hamming_knn(E, k=k, batch_size=10)
            assert G.shape == (1, 1), f"Failed for k={k}"
            assert G.nnz == 0, f"Failed for k={k}"

    def test_k_zero_returns_zero_matrix(self):
        """k=0: k_eff=0, return zero matrix regardless of n."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        rng = np.random.RandomState(123)
        E = rng.randint(0, 2, size=(50, 32), dtype=np.uint8)
        G = batched_hamming_knn(E, k=0, batch_size=25)

        assert G.nnz == 0
        assert G.shape == (50, 50)
        assert np.all(G.toarray() == 0)

    @pytest.mark.parametrize("n,k", [
        (5, 10),   # k > n-1
        (3, 3),    # k == n
        (10, 100), # k >> n
        (2, 5),    # minimum n with k>n
    ])
    def test_k_greater_than_n_minus_one_is_capped(self, n, k):
        """k > n-1: k_eff is automatically capped to n-1.

        Each point should connect to all n-1 other points.
        """
        from forest_clustering.lsh_graph import batched_hamming_knn
        rng = np.random.RandomState(42)
        E = rng.randint(0, 2, size=(n, 16), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        k_eff = min(k, n - 1)
        assert G.nnz == n * k_eff, f"n={n}, k={k}: expected nnz={n*k_eff}, got {G.nnz}"

        # Each row should have exactly k_eff non-zero entries
        G_csr = G.tocsr()
        for i in range(n):
            assert len(G_csr[i].indices) == k_eff

    def test_two_points_k_equals_one(self):
        """Minimal non-trivial case: 2 points, k=1.

        Both points should be mutual nearest neighbors (only choice).
        """
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=np.uint8)
        G = batched_hamming_knn(E, k=1, batch_size=10)

        assert G.shape == (2, 2)
        assert G.nnz == 2  # Each of 2 points has 1 neighbor

        # Each point's only neighbor is the other point
        # Hamming distance between [1,0,1,0] and [0,1,0,1] = 4
        G_dense = G.toarray()
        assert G_dense[0, 1] == 4
        assert G_dense[1, 0] == 4
        assert G_dense[0, 0] == 0
        assert G_dense[1, 1] == 0


# =============================================================================
# TESTS: Self-Exclusion (No Self-Loops)
# =============================================================================

class TestSelfExclusion:
    """Diagonal must be all zeros — no self-loops in kNN graph."""

    @pytest.mark.parametrize("n,k", [
        (10, 3),
        (20, 5),
        (50, 10),
        (100, 15),
    ])
    def test_diagonal_is_all_zeros(self, n, k, rng):
        """For any valid input, diagonal entries must be zero."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(n, 32), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=25)

        diag = G.diagonal()
        assert np.all(diag == 0), f"n={n}, k={k}: found non-zero diagonal entries"

    def test_no_diagonal_entries_in_coo(self, rng):
        """COO format should not contain any (i, i) entries."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(30, 16), dtype=np.uint8)
        G = batched_hamming_knn(E, k=5, batch_size=10)

        for row_idx, col_idx in zip(G.row, G.col):
            assert row_idx != col_idx, f"Found self-loop at ({row_idx}, {col_idx})"


# =============================================================================
# TESTS: Output Shape and Entry Count Invariants
# =============================================================================

class TestShapeAndEntryCount:
    """Invariant 1 (Output Shape) and Invariant 2 (Entry Count)."""

    @pytest.mark.parametrize("n,m,k", [
        (5, 8, 2),
        (10, 16, 3),
        (20, 32, 5),
        (50, 64, 10),
        (100, 128, 15),
    ])
    def test_output_shape(self, n, m, k, rng):
        """G.shape must equal (n, n) where n = E.shape[0]."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=max(1, n // 3))

        assert G.shape == (n, n)

    @pytest.mark.parametrize("n,k", [
        (5, 2),
        (10, 3),
        (20, 5),
        (50, 10),
        (100, 15),
    ])
    def test_entry_count(self, n, k, rng):
        """nnz(G) must equal n * k_eff where k_eff = min(k, n-1)."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        m = 32
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=max(1, n // 3))

        k_eff = min(k, n - 1)
        expected_nnz = n * k_eff
        assert G.nnz == expected_nnz, \
            f"n={n}, k={k}: expected nnz={expected_nnz}, got {G.nnz}"

    def test_return_type_is_coo(self, rng):
        """Return type must be scipy.sparse.coo_matrix."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(20, 16), dtype=np.uint8)
        G = batched_hamming_knn(E, k=3, batch_size=10)

        assert isinstance(G, sparse.coo_matrix)


# =============================================================================
# TESTS: Hamming Distance Correctness
# =============================================================================

class TestHammingDistanceCorrectness:
    """Verify that stored distances are correct Hamming distances."""

    def test_known_embedding_exact_distances(self, binary_embedding_5x8):
        """Known embedding: verify exact Hamming distances.

        E = [[1,0,1,1,0,0,1,1],
             [1,0,1,0,0,1,1,0],
             [0,1,0,1,1,0,0,1],
             [0,1,0,0,1,1,0,0],
             [1,0,1,1,0,0,1,0]]

        Hamming distances (excluding self):
        d(0,1) = 3
        d(0,2) = 5
        d(0,3) = 8
        d(0,4) = 1
        d(1,2) = 8
        d(1,3) = 5
        d(1,4) = 2
        d(2,3) = 3
        d(2,4) = 6
        d(3,4) = 7
        """
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = binary_embedding_5x8
        n = E.shape[0]

        # Compute all distances manually
        D_full = compute_hamming_distance_matrix(E)

        # Verify known distances (computed by brute force)
        assert D_full[0, 1] == 3, f"d(0,1) expected 3, got {D_full[0,1]}"
        assert D_full[0, 2] == 5, f"d(0,2) expected 5, got {D_full[0,2]}"
        assert D_full[0, 3] == 8, f"d(0,3) expected 8, got {D_full[0,3]}"
        assert D_full[0, 4] == 1, f"d(0,4) expected 1, got {D_full[0,4]}"
        assert D_full[1, 2] == 8, f"d(1,2) expected 8, got {D_full[1,2]}"
        assert D_full[1, 3] == 5, f"d(1,3) expected 5, got {D_full[1,3]}"
        assert D_full[1, 4] == 2, f"d(1,4) expected 2, got {D_full[1,4]}"
        assert D_full[2, 3] == 3, f"d(2,3) expected 3, got {D_full[2,3]}"
        assert D_full[2, 4] == 6, f"d(2,4) expected 6, got {D_full[2,4]}"
        assert D_full[3, 4] == 7, f"d(3,4) expected 7, got {D_full[3,4]}"

        # Now test k=2
        k = 2
        G = batched_hamming_knn(E, k=k, batch_size=10)

        # Check kNN correctness with brute-force reference
        check_knn_correctness(G, E, k)

    def test_all_zero_embedding(self, rng):
        """All-zero embedding: all pairwise distances are 0.

        With k_eff neighbors, any selection of k_eff neighbors is valid
        (all distances are equal). The output should still have exactly
        k_eff non-zero entries per row with value 0.
        """
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m = 10, 16
        E = np.zeros((n, m), dtype=np.uint8)
        k = 3
        G = batched_hamming_knn(E, k=k, batch_size=5)

        k_eff = min(k, n - 1)
        assert G.nnz == n * k_eff

        # All distances should be 0 (all points are identical)
        assert np.all(G.data == 0), "All distances should be 0 for identical embeddings"

        # Each row should have exactly k_eff neighbors, no self-loops
        G_csr = G.tocsr()
        for i in range(n):
            neighbors = G_csr[i].indices
            assert len(neighbors) == k_eff
            assert i not in neighbors

    def test_all_identical_rows(self, rng):
        """All rows identical: same as all-zero, all distances = 0."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m = 10, 16
        E = np.ones((n, m), dtype=np.uint8)
        k = 3
        G = batched_hamming_knn(E, k=k, batch_size=5)

        k_eff = min(k, n - 1)
        assert G.nnz == n * k_eff
        assert np.all(G.data == 0)

    def test_distances_are_nonnegative_integers(self, rng):
        """All edge weights (distances) must be non-negative integers."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 50, 64, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=25)

        assert len(G.data) > 0
        assert np.all(G.data >= 0), "Negative distances found"
        assert np.all(G.data == G.data.astype(int)), "Non-integer distances found"

    def test_distances_bounded_by_embedding_dim(self, rng):
        """No distance can exceed the embedding dimension m."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 50, 32, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=25)

        assert np.all(G.data <= m), \
            f"Found distance > embedding dimension {m}"


# =============================================================================
# TESTS: kNN Correctness Against Brute Force
# =============================================================================

class TestKNNCorrectness:
    """Compare batched_hamming_knn output against brute-force reference."""

    @pytest.mark.parametrize("n,m,k", [
        (5, 8, 2),
        (10, 16, 3),
        (20, 32, 5),
        (30, 64, 7),
        (50, 32, 10),
    ])
    def test_against_brute_force(self, n, m, k, rng):
        """For small n, compare against exact brute-force implementation."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G_batched = batched_hamming_knn(E, k=k, batch_size=max(1, n // 3))
        G_brute = brute_force_hamming_knn(E, k)

        # Compare as dense arrays (both should produce same result)
        np.testing.assert_array_equal(
            G_batched.toarray(), G_brute.toarray(),
            err_msg=f"Mismatch for n={n}, m={m}, k={k}"
        )

    @pytest.mark.parametrize("n,m,k", [
        (5, 8, 2),
        (10, 16, 3),
        (20, 32, 5),
        (30, 64, 7),
    ])
    def test_knn_invariant(self, n, m, k, rng):
        """Invariant 5: each row has k_eff smallest Hamming distances."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=max(1, n // 3))

        check_knn_correctness(G, E, k)


# =============================================================================
# TESTS: Batch Size Variations
# =============================================================================

class TestBatchSize:
    """Tests for different batch sizes — must produce identical results."""

    def test_batch_size_greater_than_n(self, rng):
        """batch_size > n: single batch, equivalent to non-batched."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 30, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G1 = batched_hamming_knn(E, k=k, batch_size=10)   # 3 batches
        G2 = batched_hamming_knn(E, k=k, batch_size=100)  # 1 batch

        np.testing.assert_array_equal(
            G1.toarray(), G2.toarray(),
            err_msg="batch_size > n should give same result"
        )

    def test_batch_size_equals_one(self, rng):
        """batch_size = 1: one point at a time."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 16, 3
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G1 = batched_hamming_knn(E, k=k, batch_size=5)
        G2 = batched_hamming_knn(E, k=k, batch_size=1)

        np.testing.assert_array_equal(
            G1.toarray(), G2.toarray(),
            err_msg="batch_size=1 should give same result"
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 25, 50, 100])
    def test_various_batch_sizes_same_result(self, batch_size, rng):
        """Multiple batch sizes must produce identical results."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 30, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G_ref = batched_hamming_knn(E, k=k, batch_size=50)
        G_test = batched_hamming_knn(E, k=k, batch_size=batch_size)

        np.testing.assert_array_equal(
            G_ref.toarray(), G_test.toarray(),
            err_msg=f"batch_size={batch_size} gave different result"
        )


# =============================================================================
# TESTS: Large Embedding Dimensions
# =============================================================================

class TestLargeDimensions:
    """Tests for large m (embedding dimensions requiring multi-word packing)."""

    def test_dimension_65_requires_two_words(self, rng):
        """m=65 > 64, requires 2 uint64 words per row."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 65, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        check_knn_correctness(G, E, k)

    def test_dimension_128(self, rng):
        """m=128 = 2*64, exactly 2 uint64 words."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 128, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        check_knn_correctness(G, E, k)

    def test_dimension_129_requires_three_words(self, rng):
        """m=129 > 128, requires 3 uint64 words."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 129, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        check_knn_correctness(G, E, k)

    def test_dimension_256(self, rng):
        """m=256 = 4*64, exactly 4 uint64 words."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 256, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        check_knn_correctness(G, E, k)

    def test_dimension_1000(self, rng):
        """m=1000: large dimension, many uint64 words (16 words)."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 100, 1000, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=50)

        assert G.shape == (n, n)
        k_eff = min(k, n - 1)
        assert G.nnz == n * k_eff

        check_knn_correctness(G, E, k)

    def test_dimension_1(self):
        """m=1: smallest possible embedding dimension."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 10, 1, 3
        E = np.random.RandomState(42).randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=5)

        check_knn_correctness(G, E, k)

    def test_dimension_64_exactly_one_word(self, rng):
        """m=64: exactly fits one uint64 word."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 64, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        check_knn_correctness(G, E, k)


# =============================================================================
# TESTS: Memory / Scaling (no OOM)
# =============================================================================

class TestMemoryScaling:
    """Tests that verify memory efficiency — no OOM on reasonably large inputs."""

    def test_1000_points_256_dims(self, rng):
        """n=1000, m=256: should complete without OOM."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 1000, 256, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        # With batch_size=500, peak memory is ~500 * 1000 * 2 = 1 MB
        G = batched_hamming_knn(E, k=k, batch_size=500)

        assert G.shape == (n, n)
        assert G.nnz == n * k

    def test_2000_points_128_dims(self, rng):
        """n=2000, m=128: larger n with moderate batch size."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 2000, 128, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G = batched_hamming_knn(E, k=k, batch_size=1000)

        assert G.shape == (n, n)
        assert G.nnz == n * k


# =============================================================================
# TESTS: Determinism
# =============================================================================

class TestDeterminism:
    """Same input with same parameters must produce identical output."""

    def test_same_input_same_output(self, rng):
        """Run twice with same input -> identical results."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 50, 64, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G1 = batched_hamming_knn(E, k=k, batch_size=25)
        G2 = batched_hamming_knn(E, k=k, batch_size=25)

        np.testing.assert_array_equal(
            G1.toarray(), G2.toarray(),
            err_msg="Same input produced different outputs"
        )

    def test_same_output_data_row_col(self, rng):
        """COO components (row, col, data) must be identical."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 30, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        G1 = batched_hamming_knn(E, k=k, batch_size=10)
        G2 = batched_hamming_knn(E, k=k, batch_size=10)

        np.testing.assert_array_equal(G1.row, G2.row)
        np.testing.assert_array_equal(G1.col, G2.col)
        np.testing.assert_array_equal(G1.data, G2.data)

    def test_determinism_across_batch_sizes(self, rng):
        """Different batch sizes for same input -> same result."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 30, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        # Multiple runs with different batch sizes should all match
        results = []
        for bs in [1, 5, 10, 15, 30, 50]:
            G = batched_hamming_knn(E, k=k, batch_size=bs)
            results.append(G.toarray())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Determinism failed between batch size comparisons"
            )


# =============================================================================
# TESTS: Input Validation
# =============================================================================

class TestInputValidation:
    """Invalid inputs must raise appropriate errors."""

    def test_k_negative_raises(self, rng):
        """k < 0 must raise ValueError."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(10, 16), dtype=np.uint8)

        with pytest.raises(ValueError):
            batched_hamming_knn(E, k=-1, batch_size=10)

    def test_batch_size_zero_raises(self, rng):
        """batch_size = 0 must raise ValueError."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(10, 16), dtype=np.uint8)

        with pytest.raises(ValueError):
            batched_hamming_knn(E, k=3, batch_size=0)

    def test_batch_size_negative_raises(self, rng):
        """batch_size < 0 must raise ValueError."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(10, 16), dtype=np.uint8)

        with pytest.raises(ValueError):
            batched_hamming_knn(E, k=3, batch_size=-5)

    def test_non_binary_embedding_supported(self, rng):
        """Non-binary (K-ary) cell-id embeddings must be handled via column
        Hamming distance (the forest-clustering embedding is K-ary, not {0,1})."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 5, size=(10, 16), dtype=np.int64)  # values 0-4

        k = 3
        G = batched_hamming_knn(E, k=k, batch_size=10).tocsr()
        assert G.shape == (10, 10)
        # Each row keeps exactly k neighbours.
        assert G.nnz == 10 * k
        # Stored distances equal the column-Hamming distance to those neighbours.
        for i in range(10):
            d = (E[i] != E).sum(axis=1).astype(int)
            d[i] = 10 ** 9
            nn = np.argsort(d)[:k]
            stored = G[i].toarray().ravel()
            stored = sorted(stored[G[i].toarray().ravel() > 0].tolist())
            assert sorted(d[nn].tolist()) == stored

    def test_non_2d_embedding_raises(self, rng):
        """1D or 3D embedding must raise ValueError."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E_1d = rng.randint(0, 2, size=(10,), dtype=np.uint8)
        E_3d = rng.randint(0, 2, size=(2, 5, 8), dtype=np.uint8)

        with pytest.raises(ValueError):
            batched_hamming_knn(E_1d, k=3, batch_size=10)

        with pytest.raises(ValueError):
            batched_hamming_knn(E_3d, k=3, batch_size=10)

    @pytest.mark.parametrize("dtype", [
        np.uint8, np.int8, np.bool_, np.int32, np.float32,
    ])
    def test_binary_values_with_different_dtypes(self, dtype, rng):
        """Any dtype containing only 0/1 values should be accepted."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 10, 16, 3
        E = rng.randint(0, 2, size=(n, m)).astype(dtype)

        G = batched_hamming_knn(E, k=k, batch_size=5)
        assert G.shape == (n, n)


# =============================================================================
# TESTS: GraphLouvainClusterer.fit_embedding Integration
# =============================================================================

class TestFitEmbeddingIntegration:
    """GraphLouvainClusterer.fit_embedding(E, k) builds kNN graph and clusters."""

    def test_fit_embedding_sets_labels(self, rng):
        """fit_embedding must set self.labels_ with correct length."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        n, m, k = 50, 32, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        clusterer = GraphLouvainClusterer(random_state=42)
        result = clusterer.fit_embedding(E, k=k)

        # Returns self
        assert result is clusterer

        # labels_ is set
        assert clusterer.labels_ is not None
        assert isinstance(clusterer.labels_, np.ndarray)
        assert len(clusterer.labels_) == n

    def test_fit_embedding_produces_reasonable_clusters(self, rng):
        """Well-separated binary clusters -> should find multiple clusters."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        # Create 3 well-separated groups in Hamming space
        n_per_group = 30
        # Group 1: all zeros in first half, random in second
        E1 = np.hstack([
            np.zeros((n_per_group, 32), dtype=np.uint8),
            rng.randint(0, 2, size=(n_per_group, 32), dtype=np.uint8)
        ])
        # Group 2: all ones in first half, random in second
        E2 = np.hstack([
            np.ones((n_per_group, 32), dtype=np.uint8),
            rng.randint(0, 2, size=(n_per_group, 32), dtype=np.uint8)
        ])
        # Group 3: alternating in first half, random in second
        E3 = np.hstack([
            np.tile([1, 0], (n_per_group, 16)).astype(np.uint8),
            rng.randint(0, 2, size=(n_per_group, 32), dtype=np.uint8)
        ])
        E = np.vstack([E1, E2, E3])

        clusterer = GraphLouvainClusterer(random_state=42)
        clusterer.fit_embedding(E, k=10)

        n_clusters = len(np.unique(clusterer.labels_[clusterer.labels_ >= 0]))
        assert n_clusters >= 2, f"Expected >=2 clusters for well-separated data, got {n_clusters}"

    def test_fit_embedding_single_point(self):
        """Single point should produce single cluster label."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        E = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        clusterer = GraphLouvainClusterer(random_state=42)
        clusterer.fit_embedding(E, k=5)

        assert len(clusterer.labels_) == 1
        assert clusterer.labels_[0] == 0

    def test_fit_embedding_empty(self):
        """Empty embedding should produce empty labels array."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        E = np.empty((0, 10), dtype=np.uint8)
        clusterer = GraphLouvainClusterer(random_state=42)
        clusterer.fit_embedding(E, k=5)

        assert len(clusterer.labels_) == 0
        assert clusterer.labels_.dtype == np.int64

    def test_fit_embedding_k_none_uses_default(self, rng):
        """k=None should use default k value (from clusterer or spec default)."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        n, m = 50, 32
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        clusterer = GraphLouvainClusterer(random_state=42)

        # k=None should work without error
        clusterer.fit_embedding(E, k=None)
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == n

    def test_fit_embedding_sets_knn_graph_attribute(self, rng):
        """fit_embedding should store the kNN graph for inspection."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        n, m, k = 30, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        clusterer = GraphLouvainClusterer(random_state=42)
        clusterer.fit_embedding(E, k=k)

        # knn_graph_ attribute should be set
        assert hasattr(clusterer, 'knn_graph_')
        assert isinstance(clusterer.knn_graph_, sparse.coo_matrix)
        assert clusterer.knn_graph_.shape == (n, n)

    def test_fit_embedding_deterministic_with_seed(self, rng):
        """Same E, same k, same random_state -> same labels."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        n, m, k = 50, 32, 10
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        c1 = GraphLouvainClusterer(random_state=42)
        c1.fit_embedding(E, k=k)

        c2 = GraphLouvainClusterer(random_state=42)
        c2.fit_embedding(E, k=k)

        np.testing.assert_array_equal(c1.labels_, c2.labels_)

    def test_fit_embedding_different_k_produces_different_graphs(self, rng):
        """Different k values should produce different kNN graphs."""
        from forest_clustering.graph_clustering import GraphLouvainClusterer

        n, m = 30, 32
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)

        c1 = GraphLouvainClusterer(random_state=42)
        c1.fit_embedding(E, k=3)

        c2 = GraphLouvainClusterer(random_state=42)
        c2.fit_embedding(E, k=10)

        assert c1.knn_graph_.nnz != c2.knn_graph_.nnz


# =============================================================================
# TESTS: Input dtype flexibility
# =============================================================================

class TestDtypeFlexibility:
    """Test various input dtypes that contain binary values."""

    @pytest.mark.parametrize("dtype", [
        np.uint8, np.int8, np.bool_,
    ])
    def test_accepted_binary_dtypes(self, dtype, rng):
        """uint8, int8, bool_ dtypes with binary values should work."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 16, 3
        E = rng.randint(0, 2, size=(n, m)).astype(dtype)

        G = batched_hamming_knn(E, k=k, batch_size=10)
        assert G.shape == (n, n)
        check_knn_correctness(G, E.astype(np.uint8), k)

    def test_float_with_binary_values(self, rng):
        """Float array with only 0.0 and 1.0 should be accepted."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        n, m, k = 20, 16, 3
        E = rng.randint(0, 2, size=(n, m)).astype(np.float32)

        G = batched_hamming_knn(E, k=k, batch_size=10)
        assert G.shape == (n, n)


# =============================================================================
# TESTS: Structured / Known-Correctness Cases
# =============================================================================

class TestStructuredCases:
    """Test with embeddings of known structure for verifiable correctness."""

    def test_small_five_points_k_equals_two(self):
        """5 points, k=2: each point has exactly 2 neighbors."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.array([
            [1, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 0],
        ], dtype=np.uint8)

        k = 2
        G = batched_hamming_knn(E, k=k, batch_size=10)
        k_eff = min(k, E.shape[0] - 1)

        # Check entry count
        assert G.nnz == E.shape[0] * k_eff

        # Each row should have exactly k_eff neighbors
        G_csr = G.tocsr()
        for i in range(E.shape[0]):
            assert len(G_csr[i].indices) == k_eff
            assert i not in G_csr[i].indices  # no self-loops

        # Verify against brute force
        check_knn_correctness(G, E, k)

    def test_orthogonal_vectors_max_distance(self):
        """Vectors at maximum Hamming distance should be nearest when k=1."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        m = 8
        # Point 0: all zeros, Point 1: all ones (max distance = m)
        E = np.array([
            [0] * m,
            [1] * m,
        ], dtype=np.uint8)

        k = 1
        G = batched_hamming_knn(E, k=k, batch_size=10)

        # The only edge should have distance = m
        assert G.nnz == 2  # 2 points * 1 neighbor each
        assert np.all(G.data == m), f"Expected distance {m}, got {G.data}"

    def test_nearest_neighbor_consistency(self):
        """For k=1, if A's nearest neighbor is B, B's distance to A should be stored."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.uint8)

        # d(0,1) = 1, d(0,2) = 8
        # d(1,0) = 1, d(1,2) = 7
        # d(2,0) = 8, d(2,1) = 7

        k = 1
        G = batched_hamming_knn(E, k=k, batch_size=10)

        # Point 0's nearest is Point 1 (dist 1)
        row0 = G.tocsr()[0]
        assert row0.indices[0] == 1
        assert row0.data[0] == 1

        # Point 1's nearest is Point 0 (dist 1)
        row1 = G.tocsr()[1]
        assert row1.indices[0] == 0
        assert row1.data[0] == 1

        # Point 2's nearest is Point 1 (dist 7)
        row2 = G.tocsr()[2]
        assert row2.indices[0] == 1
        assert row2.data[0] == 7


# =============================================================================
# TESTS: Property-Based Invariants
# =============================================================================

class TestPropertyInvariants:
    """Property-based invariants that should hold for all valid inputs."""

    @pytest.mark.parametrize("n,m,k", [
        (3, 4, 1),
        (5, 8, 2),
        (10, 16, 3),
        (20, 32, 5),
        (30, 64, 7),
    ])
    def test_all_properties_hold(self, n, m, k, rng):
        """Comprehensive property check for random embeddings."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=max(1, n // 3))

        check_knn_correctness(G, E, k)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_sparsity_pattern_properties(self, seed):
        """For various random seeds, verify structural invariants."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        rng = np.random.RandomState(seed)
        n, m, k = 25, 32, 5
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=10)

        k_eff = min(k, n - 1)

        # No self-loops
        for r, c in zip(G.row, G.col):
            assert r != c

        # Exactly k_eff entries per row
        G_csr = G.tocsr()
        for i in range(n):
            assert len(G_csr[i].data) == k_eff

        # All distances are valid Hamming distances
        D_full = compute_hamming_distance_matrix(E)
        for r, c, d in zip(G.row, G.col, G.data):
            assert D_full[r, c] == d, \
                f"Distance mismatch at ({r},{c}): stored {d}, actual {D_full[r, c]}"

    def test_triangle_inequality_not_required(self):
        """kNN graph doesn't need triangle inequality; just verify it works
        for non-metric-like embeddings too."""
        from forest_clustering.lsh_graph import batched_hamming_knn
        # Completely random binary data — no structure
        rng = np.random.RandomState(77)
        n, m, k = 15, 8, 3
        E = rng.randint(0, 2, size=(n, m), dtype=np.uint8)
        G = batched_hamming_knn(E, k=k, batch_size=5)

        check_knn_correctness(G, E, k)
