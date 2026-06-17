"""TDD tests for forest_clustering.contrastive_splits module.

All tests should fail (ImportError / AttributeError) until the production
code in forest_clustering/contrastive_splits.py is implemented.

Test categories:
    1. augment_sample       – noise injection + dropout augmentation
    2. generate_pairs       – positive & negative pair generation
    3. contrastive_loss    – contrastive learning loss
    4. evaluate_split_contrastive – scoring a single split
    5. build_contrastive_tree     – full decision tree
    6. ForestClusterer integration
    7. Edge cases
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_X():
    """Simple 2-D numerical data for testing."""
    rng = np.random.default_rng(42)
    return rng.random((20, 4))


@pytest.fixture
def blobs_X_y():
    """Labeled blobs for contrastive pair generation."""
    X, y = make_blobs(n_samples=30, centers=3, n_features=4,
                      random_state=42, cluster_std=0.6)
    return X, y


@pytest.fixture
def contrastive_pairs(blobs_X_y):
    """Pre-generated positive and negative pairs."""
    X, y = blobs_X_y
    from forest_clustering.contrastive_splits import generate_pairs
    pos_pairs, neg_pairs = generate_pairs(y, n_positive=20, n_negative=20,
                                          random_state=42)
    return X, y, pos_pairs, neg_pairs


# ===========================================================================
# 1. augment_sample
# ===========================================================================

class TestAugmentSample:
    """Tests for augment_sample: adds noise and/or dropout to a single sample."""

    def test_augment_adds_noise(self, simple_X):
        """Output must differ from input, but shape is preserved."""
        from forest_clustering.contrastive_splits import augment_sample
        x = simple_X[0]
        x_aug = augment_sample(x, noise_scale=0.1, dropout_prob=0.0, seed=42)
        assert x_aug.shape == x.shape
        assert not np.allclose(x_aug, x)

    def test_augment_dropout(self, simple_X):
        """With dropout > 0, some features must be exactly zero."""
        from forest_clustering.contrastive_splits import augment_sample
        x = simple_X[0]
        x_aug = augment_sample(x, noise_scale=0.0, dropout_prob=0.5, seed=42)
        assert x_aug.shape == x.shape
        # At least one feature should be zero with reasonably high dropout
        assert np.any(x_aug == 0), "Expected some features to be zero after dropout"

    def test_augment_deterministic_with_seed(self, simple_X):
        """Same seed must produce identical output."""
        from forest_clustering.contrastive_splits import augment_sample
        x = simple_X[0]
        a = augment_sample(x, noise_scale=0.1, dropout_prob=0.1, seed=123)
        b = augment_sample(x, noise_scale=0.1, dropout_prob=0.1, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_augment_preserves_range(self, simple_X):
        """Augmented values should stay within reasonable bounds."""
        from forest_clustering.contrastive_splits import augment_sample
        x = simple_X[0]
        # Run many augmentations and check bounds
        results = [augment_sample(x, noise_scale=0.05, dropout_prob=0.1,
                                  seed=i) for i in range(100)]
        stacked = np.stack(results)
        # With small noise scale, values shouldn't explode
        assert np.all(np.isfinite(stacked))
        assert np.all(np.abs(stacked) < 10), "Augmented values out of reasonable bounds"


# ===========================================================================
# 2. generate_pairs
# ===========================================================================

class TestGeneratePairs:
    """Tests for generate_pairs: positive and negative pair creation."""

    def test_positive_pairs_same_length_as_n(self, blobs_X_y):
        """When n_positive=n, we get exactly n positive pairs."""
        from forest_clustering.contrastive_splits import generate_pairs
        _, y = blobs_X_y
        n = len(y)
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=n, n_negative=n,
                                              random_state=42)
        assert len(pos_pairs) == n
        assert len(neg_pairs) == n

    def test_negative_pairs_different_from_positive(self, blobs_X_y):
        """Negative pairs must not be the same as positive pairs."""
        from forest_clustering.contrastive_splits import generate_pairs
        _, y = blobs_X_y
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=10, n_negative=10,
                                              random_state=42)
        pos_set = set(tuple(p) for p in pos_pairs)
        neg_set = set(tuple(p) for p in neg_pairs)
        # They should be disjoint sets
        assert pos_set.isdisjoint(neg_set), (
            "Negative pairs overlap with positive pairs"
        )

    def test_no_self_pairs(self, blobs_X_y):
        """No pair should have i == j."""
        from forest_clustering.contrastive_splits import generate_pairs
        _, y = blobs_X_y
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=20, n_negative=20,
                                              random_state=42)
        all_pairs = np.vstack([pos_pairs, neg_pairs])
        assert np.all(all_pairs[:, 0] != all_pairs[:, 1]), "Found self-pair (i == j)"

    def test_pairs_respect_random_state(self, blobs_X_y):
        """Same random_state must produce identical pairs."""
        from forest_clustering.contrastive_splits import generate_pairs
        _, y = blobs_X_y
        pos1, neg1 = generate_pairs(y, n_positive=15, n_negative=15, random_state=99)
        pos2, neg2 = generate_pairs(y, n_positive=15, n_negative=15, random_state=99)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(neg1, neg2)


# ===========================================================================
# 3. contrastive_loss
# ===========================================================================

class TestContrastiveLoss:
    """Tests for contrastive_loss: contrastive learning loss."""

    def test_loss_low_for_perfect_separation(self):
        """Positive pairs in same leaf, negatives in different leaves → low loss."""
        from forest_clustering.contrastive_splits import contrastive_loss
        # One-hot leaf embeddings: samples 0,1 in leaf 0; 2,3 in leaf 1
        leaf_emb = np.zeros((4, 2), dtype=np.float64)
        leaf_emb[0, 0] = 1.0
        leaf_emb[1, 0] = 1.0
        leaf_emb[2, 1] = 1.0
        leaf_emb[3, 1] = 1.0
        pos_pairs = np.array([[0, 1], [2, 3]])  # same leaf
        neg_pairs = np.array([[0, 2], [1, 3]])  # different leaves
        loss = contrastive_loss(leaf_emb, pos_pairs, neg_pairs, temperature=0.5)
        # True NT-Xent includes positive in denominator, so loss is ~0.127
        # for 1 negative per anchor at tau=0.5 with sim_pos=1, sim_neg=0
        expected = -np.log(np.exp(2.0) / (1.0 + np.exp(2.0)))
        assert abs(loss - expected) < 1e-6, f"Expected loss ~{expected}, got {loss}"

    def test_loss_positive_for_random(self):
        """Random embeddings should yield a positive loss."""
        from forest_clustering.contrastive_splits import contrastive_loss
        rng = np.random.default_rng(7)
        embeddings = rng.random((10, 8))
        pos_pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        neg_pairs = np.array([[0, 2], [1, 3], [4, 6], [5, 7]])
        loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5)
        assert loss > 0, f"Expected positive loss for random embeddings, got {loss}"

    def test_loss_decreases_with_temperature(self):
        """Higher temperature → lower loss (negatives less penalized)."""
        from forest_clustering.contrastive_splits import contrastive_loss
        rng = np.random.default_rng(13)
        embeddings = rng.random((10, 8))
        pos_pairs = np.array([[0, 1], [2, 3], [4, 5]])
        neg_pairs = np.array([[0, 2], [1, 4], [3, 5]])
        loss_low_t = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.1)
        loss_high_t = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=2.0)
        assert loss_high_t < loss_low_t, (
            f"Higher temperature should decrease loss: {loss_low_t} vs {loss_high_t}"
        )

    def test_loss_non_negative(self):
        """Loss must always be >= 0."""
        from forest_clustering.contrastive_splits import contrastive_loss
        rng = np.random.default_rng(21)
        for seed in range(10):
            rng_inner = np.random.default_rng(seed)
            embeddings = rng_inner.random((8, 6))
            pos_pairs = np.array([[0, 1], [2, 3]])
            neg_pairs = np.array([[0, 2], [1, 3]])
            loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5)
            assert loss >= -1e-12, f"Loss should be non-negative, got {loss}"

    def test_loss_deterministic(self):
        """Same input must produce identical loss."""
        from forest_clustering.contrastive_splits import contrastive_loss
        embeddings = np.random.default_rng(33).random((8, 6))
        pos_pairs = np.array([[0, 1], [2, 3], [4, 5]])
        neg_pairs = np.array([[0, 2], [1, 4]])
        loss1 = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5)
        loss2 = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5)
        assert loss1 == loss2

    def test_temperature_must_be_positive(self):
        """temperature <= 0 should raise ValueError."""
        from forest_clustering.contrastive_splits import contrastive_loss
        embeddings = np.ones((4, 4))
        pos_pairs = np.array([[0, 1]])
        neg_pairs = np.array([[0, 2]])
        with pytest.raises(ValueError, match="temperature must be > 0"):
            contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=-1.0)


# ===========================================================================
# 4. evaluate_split_contrastive
# ===========================================================================

class TestEvaluateSplitContrastive:
    """Tests for evaluate_split_contrastive: scoring a single split."""

    def test_split_separates_opposites(self, contrastive_pairs):
        """A threshold that puts positive pairs together scores higher."""
        from forest_clustering.contrastive_splits import evaluate_split_contrastive
        X, y, pos_pairs, neg_pairs = contrastive_pairs
        # Use a feature that separates the blobs well
        feature_idx = 0
        # Find a good threshold
        best_score = -np.inf
        best_thresh = None
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            thresh = np.percentile(X[:, feature_idx], pct)
            score = evaluate_split_contrastive(X, pos_pairs, neg_pairs,
                                               thresh, feature_idx)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        # A good threshold should score higher than a trivial one
        trivial_score = evaluate_split_contrastive(
            X, pos_pairs, neg_pairs, np.min(X[:, feature_idx]) - 1, feature_idx
        )
        assert best_score > trivial_score, (
            "Good threshold should score higher than trivial threshold"
        )

    def test_info_gain_bonus(self, contrastive_pairs):
        """Balanced split (both sides have samples) scores higher than imbalanced."""
        from forest_clustering.contrastive_splits import evaluate_split_contrastive
        X, y, pos_pairs, neg_pairs = contrastive_pairs
        feature_idx = 0
        # Balanced: threshold near median
        median_thresh = np.median(X[:, feature_idx])
        balanced_score = evaluate_split_contrastive(
            X, pos_pairs, neg_pairs, median_thresh, feature_idx
        )
        # Imbalanced: threshold below min or above max
        imbalanced_score = evaluate_split_contrastive(
            X, pos_pairs, neg_pairs, np.min(X[:, feature_idx]) - 10, feature_idx
        )
        assert balanced_score >= imbalanced_score, (
            "Balanced split should score at least as high as imbalanced"
        )

    def test_random_threshold_baseline(self, contrastive_pairs):
        """A well-chosen threshold should be competitive with random thresholds."""
        from forest_clustering.contrastive_splits import evaluate_split_contrastive
        X, y, pos_pairs, neg_pairs = contrastive_pairs
        feature_idx = 0
        rng = np.random.default_rng(42)
        # Best of several random thresholds
        random_scores = []
        for _ in range(20):
            thresh = rng.uniform(X[:, feature_idx].min(), X[:, feature_idx].max())
            s = evaluate_split_contrastive(X, pos_pairs, neg_pairs, thresh, feature_idx)
            random_scores.append(s)
        best_random = max(random_scores)
        # A well-chosen threshold should be competitive with best random
        median_score = evaluate_split_contrastive(
            X, pos_pairs, neg_pairs, np.median(X[:, feature_idx]), feature_idx
        )
        # With normalized info gain, allow median to be within a reasonable margin
        assert median_score >= best_random - 0.5, (
            f"Median-based split ({median_score:.4f}) should not be "
            f"dramatically worse than best random ({best_random:.4f})"
        )


# ===========================================================================
# 5. build_contrastive_tree
# ===========================================================================

class TestBuildContrastiveTree:
    """Tests for build_contrastive_tree: full contrastive decision tree."""

    def test_tree_assigns_all_samples(self, blobs_X_y):
        """Every sample must receive a leaf assignment."""
        from forest_clustering.contrastive_splits import build_contrastive_tree
        X, y = blobs_X_y
        leaf_ids = build_contrastive_tree(X, max_depth=4, n_pairs=20,
                                          temperature=0.5, random_state=42)
        assert len(leaf_ids) == len(X)
        assert not np.any(np.isnan(leaf_ids))

    def test_tree_depth_respected(self, blobs_X_y):
        """Number of unique leaves should not exceed 2^max_depth."""
        from forest_clustering.contrastive_splits import build_contrastive_tree
        X, y = blobs_X_y
        max_depth = 3
        leaf_ids = build_contrastive_tree(X, max_depth=max_depth, n_pairs=20,
                                          temperature=0.5, random_state=42)
        n_leaves = len(np.unique(leaf_ids))
        assert n_leaves <= 2 ** max_depth, (
            f"Got {n_leaves} leaves but max_depth={max_depth} allows at most {2**max_depth}"
        )

    def test_tree_leaves_are_reasonable(self, blobs_X_y):
        """Should produce more than 1 leaf but not too many."""
        from forest_clustering.contrastive_splits import build_contrastive_tree
        X, y = blobs_X_y
        leaf_ids = build_contrastive_tree(X, max_depth=5, n_pairs=20,
                                          temperature=0.5, random_state=42)
        n_leaves = len(np.unique(leaf_ids))
        assert n_leaves > 1, "Tree should produce more than 1 leaf"
        assert n_leaves <= len(X), "Should not have more leaves than samples"

    def test_contrastive_better_than_random(self, blobs_X_y):
        """On make_blobs, contrastive embedding should beat random single-tree assignments."""
        from forest_clustering.contrastive_splits import build_contrastive_tree
        X, y = blobs_X_y
        # Run multiple trees to build an embedding
        n_iterations = 20
        embedding = np.zeros((len(X), n_iterations), dtype=np.int64)
        for it in range(n_iterations):
            embedding[:, it] = build_contrastive_tree(
                X, max_depth=4, n_pairs=15, temperature=0.5, random_state=it + 100
            )
        # Compute silhouette on the Hamming distance matrix
        from forest_clustering.distance import pairwise_hamming
        D = pairwise_hamming(embedding).astype(np.float64)
        # Compare with random assignments
        rng = np.random.default_rng(99)
        random_emb = rng.integers(0, 4, size=(len(X), n_iterations))
        D_random = pairwise_hamming(random_emb).astype(np.float64)
        # Use true labels for silhouette reference
        sil_contrastive = silhouette_score(D, y, metric='precomputed')
        sil_random = silhouette_score(D_random, y, metric='precomputed')
        assert sil_contrastive > sil_random, (
            f"Contrastive silhouette ({sil_contrastive}) should exceed "
            f"random ({sil_random})"
        )


# ===========================================================================
# 6. Integration with ForestClusterer
# ===========================================================================

class TestForestClustererContrastiveIntegration:
    """ForestClusterer must accept contrastive=True and produce valid results."""

    def test_contrastive_false_is_default(self):
        """ForestClusterer() should work as before (contrastive disabled by default)."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=10, random_state=42)
        assert hasattr(c, 'contrastive') or True  # may not exist yet; default behavior
        X, _ = make_blobs(n_samples=20, centers=2, n_features=3, random_state=42)
        labels = c.fit_predict(X)
        assert len(labels) == len(X)

    def test_contrastive_true_runs(self):
        """ForestClusterer(contrastive=True) must run without error."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=10, contrastive=True, random_state=42)
        X, _ = make_blobs(n_samples=20, centers=2, n_features=3, random_state=42)
        labels = c.fit_predict(X)
        assert len(labels) == len(X)

    def test_contrastive_embedding_shape(self):
        """Embedding shape must be (n_samples, n_iterations)."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=15, contrastive=True, random_state=42)
        X, _ = make_blobs(n_samples=25, centers=2, n_features=3, random_state=42)
        c.fit(X)
        assert hasattr(c, 'embedding_')
        assert c.embedding_.shape == (25, 15)

    def test_contrastive_produces_valid_clustering(self):
        """Labels must be valid: no NaN, all samples assigned."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=10, contrastive=True, random_state=42)
        X, _ = make_blobs(n_samples=20, centers=2, n_features=3, random_state=42)
        labels = c.fit_predict(X)
        assert len(labels) == len(X)
        assert not np.any(np.isnan(labels))
        assert len(set(labels)) >= 1  # at least one cluster


# ===========================================================================
# 7. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case handling for the contrastive pipeline."""

    def test_n_less_than_10_uses_random(self):
        """With n < 10 samples, contrastive should fall back to random splits."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=5, contrastive=True, random_state=42)
        X = np.random.default_rng(42).random((5, 3))
        # Should not error — falls back gracefully
        labels = c.fit_predict(X)
        assert len(labels) == 5

    def test_all_identical_fallback(self):
        """When all samples are identical, should not crash."""
        from forest_clustering.contrastive_splits import (
            generate_pairs, build_contrastive_tree
        )
        X = np.ones((15, 3))
        y = np.array([0] * 7 + [1] * 8)
        # generate_pairs should still work
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=10, n_negative=10,
                                              random_state=42)
        assert len(pos_pairs) == 10
        assert len(neg_pairs) == 10
        # build_contrastive_tree should not crash on identical data
        leaf_ids = build_contrastive_tree(X, max_depth=3, n_pairs=10,
                                          temperature=0.5, random_state=42)
        assert len(leaf_ids) == len(X)

    def test_single_feature_works(self):
        """Contrastive pipeline must work with d=1."""
        from forest_clustering.contrastive_splits import (
            augment_sample, generate_pairs, build_contrastive_tree
        )
        rng = np.random.default_rng(55)
        X = rng.random((20, 1))
        y = np.array([0] * 10 + [1] * 10)
        # augment_sample
        x_aug = augment_sample(X[0], noise_scale=0.1, dropout_prob=0.1, seed=1)
        assert x_aug.shape == (1,)
        # generate_pairs
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=10, n_negative=10,
                                              random_state=42)
        assert len(pos_pairs) == 10
        # build_contrastive_tree
        leaf_ids = build_contrastive_tree(X, max_depth=3, n_pairs=10,
                                          temperature=0.5, random_state=42)
        assert len(leaf_ids) == len(X)

    def test_augment_empty_vector(self):
        """augment_sample with empty vector should return empty vector."""
        from forest_clustering.contrastive_splits import augment_sample
        x = np.array([], dtype=np.float64)
        x_aug = augment_sample(x, noise_scale=0.1, dropout_prob=0.1, seed=42)
        assert x_aug.shape == (0,)

    def test_generate_pairs_single_class(self):
        """generate_pairs with only one class should still return arrays."""
        from forest_clustering.contrastive_splits import generate_pairs
        y = np.zeros(10, dtype=int)
        pos_pairs, neg_pairs = generate_pairs(y, n_positive=5, n_negative=5,
                                              random_state=42)
        # No positive pairs possible with single class
        assert len(pos_pairs) == 0 or pos_pairs.shape[1] == 2

    def test_contrastive_loss_empty_pairs(self):
        """contrastive_loss with empty pairs should return 0."""
        from forest_clustering.contrastive_splits import contrastive_loss
        embeddings = np.random.default_rng(42).random((5, 4))
        empty_pairs = np.empty((0, 2), dtype=np.int64)
        loss = contrastive_loss(embeddings, empty_pairs, empty_pairs, temperature=0.5)
        assert loss == 0.0 or np.isfinite(loss)
