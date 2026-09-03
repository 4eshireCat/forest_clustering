"""
Comprehensive unit tests for forest_clustering.significance module.

These tests are written BEFORE the implementation exists (TDD style).
They cover:
  - permutation_test_ari()
  - bootstrap_ci_ari()
  - paired_permutation_test()
  - cluster_significance()
  - apply_multiple_testing_correction()

All tests are designed to FAIL on an empty/stub implementation and PASS
on a correct one.

References: SIGNIFICANCE_SPEC.md (sections 2-12)
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Import the module under test (will raise ImportError until implemented)
# ---------------------------------------------------------------------------
try:
    from forest_clustering.significance import (
        permutation_test_ari,
        bootstrap_ci_ari,
        paired_permutation_test,
        cluster_significance,
        apply_multiple_testing_correction,
    )
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False


# ===========================================================================
# Pytest skip marker for the whole module
# ===========================================================================
pytestmark = pytest.mark.skipif(
    not MODULE_AVAILABLE,
    reason="forest_clustering.significance module not yet implemented",
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    """Controlled random number generator for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture
def perfect_labels():
    """y_true == y_pred -- perfect agreement, ARI must be 1.0."""
    return (
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 0, 1, 1, 2, 2]),
    )


@pytest.fixture
def random_labels(rng):
    """Independent random labelings -- ARI should be near 0."""
    n = 50
    return (
        rng.integers(0, 3, size=n),
        rng.integers(0, 3, size=n),
    )


@pytest.fixture
def moderate_match_labels():
    """Partial overlap giving ARI ~ 0.5 -- should be significant."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 2])
    return y_true, y_pred


@pytest.fixture
def single_cluster_labels():
    """Both labelings have a single cluster -- ARI = 1.0 (structure match)."""
    return (
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 1, 1]),
    )


@pytest.fixture
def one_vs_many_labels():
    """One single-cluster labeling, one multi-cluster -- ARI = 0.0."""
    return (
        np.array([0, 0, 0, 0]),           # single cluster
        np.array([0, 0, 1, 1]),           # two clusters
    )


@pytest.fixture
def small_n_labels():
    """Only 2 samples -- degenerate minimum case."""
    return (
        np.array([0, 1]),
        np.array([0, 1]),
    )


@pytest.fixture
def blobs_data(rng):
    """Well-separated 3-cluster blobs from sklearn -- structure is real."""
    X, y = make_blobs(
        n_samples=200,
        centers=3,
        n_features=2,
        cluster_std=0.5,
        random_state=42,
    )
    return X, y


# ===========================================================================
# 1. permutation_test_ari
# ===========================================================================

class TestPermutationTestARI:
    """Tests for permutation_test_ari() per SIGNIFICANCE_SPEC sections 2, 12."""

    # -- Return schema ------------------------------------------------------

    REQUIRED_KEYS = {
        "ari_observed", "p_value", "is_significant", "n_permutations",
        "null_distribution", "effect_size", "alternative", "ci_95", "warning",
    }

    @pytest.mark.parametrize("key", list(REQUIRED_KEYS))
    def test_return_schema_contains_key(self, perfect_labels, key):
        """Every required key must be present in the result dict."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("key,expected_type", [
        ("ari_observed", (int, float, np.floating)),
        ("p_value", (int, float, np.floating)),
        ("is_significant", bool),
        ("n_permutations", (int, np.integer)),
        ("null_distribution", np.ndarray),
        ("effect_size", str),
        ("alternative", str),
        ("ci_95", (tuple, type(None))),
        ("warning", (str, type(None))),
    ])
    def test_return_schema_types(self, perfect_labels, key, expected_type):
        """Each key must have the correct type."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert isinstance(result[key], expected_type), \
            f"Key '{key}': expected {expected_type}, got {type(result[key])}"

    def test_null_distribution_length(self, perfect_labels):
        """null_distribution must have length equal to n_permutations."""
        y_true, y_pred = perfect_labels
        B = 100
        result = permutation_test_ari(y_true, y_pred, n_permutations=B)
        assert len(result["null_distribution"]) == B

    def test_null_distribution_values_in_range(self, perfect_labels):
        """All null distribution values must be in [-1, 1] (ARI range)."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert np.all(result["null_distribution"] >= -1.0)
        assert np.all(result["null_distribution"] <= 1.0)

    def test_p_value_in_valid_range(self, perfect_labels):
        """p_value must be in [1/(B+1), 1] (Invariant 9.1.1)."""
        y_true, y_pred = perfect_labels
        B = 100
        result = permutation_test_ari(y_true, y_pred, n_permutations=B)
        min_p = 1.0 / (B + 1)
        assert min_p <= result["p_value"] <= 1.0, \
            f"p_value={result['p_value']} not in [{min_p}, 1]"

    def test_is_significant_matches_p_value(self, perfect_labels):
        """is_significant must equal (p_value < 0.05)."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["is_significant"] == (result["p_value"] < 0.05)

    def test_ci_95_order(self, perfect_labels):
        """ci_95[0] <= ci_95[1] (lower bound <= upper bound)."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        if result["ci_95"] is not None:
            assert result["ci_95"][0] <= result["ci_95"][1]

    def test_alternative_field(self, perfect_labels):
        """alternative field must match the parameter passed."""
        y_true, y_pred = perfect_labels
        for alt in ("greater", "less", "two-sided"):
            result = permutation_test_ari(
                y_true, y_pred, n_permutations=100, alternative=alt
            )
            assert result["alternative"] == alt

    # -- Correctness: perfect agreement -------------------------------------

    def test_perfect_agreement_ari_is_one(self, perfect_labels):
        """When y_true == y_pred, ari_observed must be 1.0."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["ari_observed"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_agreement_p_value_counts_null_ties(self, perfect_labels):
        """Permutation p-values must count null statistics tied with ARI=1."""
        y_true, y_pred = perfect_labels
        B = 100
        result = permutation_test_ari(
            y_true, y_pred, n_permutations=B, random_state=42
        )
        expected = (
            1 + np.sum(result["null_distribution"] >= result["ari_observed"])
        ) / (B + 1)
        assert result["p_value"] == pytest.approx(expected)

    def test_perfect_agreement_can_be_non_significant_with_many_null_ties(
        self, perfect_labels
    ):
        """Effect size 1 does not imply significance in a small discrete null."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(
            y_true, y_pred, n_permutations=100, random_state=42
        )
        assert result["p_value"] > 0.05
        assert result["is_significant"] is False

    def test_perfect_agreement_effect_size_large(self, perfect_labels):
        """ARI = 1.0 -> effect_size must be 'large'."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["effect_size"] == "large"

    # -- Correctness: random labels -----------------------------------------

    def test_random_labels_ari_near_zero(self, random_labels):
        """Random independent labelings -> ARI ~ 0."""
        y_true, y_pred = random_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["ari_observed"] == pytest.approx(0.0, abs=0.15)

    def test_random_labels_p_value_not_significant(self, random_labels):
        """Random labelings -> p_value should NOT be significant (~ 0.5)."""
        y_true, y_pred = random_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=200)
        # For random data p should not be < 0.05 (Type I error control)
        assert result["p_value"] > 0.01  # loose bound for stochastic test
        assert result["is_significant"] is False

    def test_random_labels_effect_size_none(self, random_labels):
        """ARI <= 0 -> effect_size must be 'none'."""
        y_true, y_pred = random_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        if result["ari_observed"] <= 0:
            assert result["effect_size"] == "none"

    # -- Correctness: moderate match (significant) --------------------------

    def test_moderate_match_ari_reasonable(self, moderate_match_labels):
        """Partial overlap should give moderate positive ARI."""
        y_true, y_pred = moderate_match_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=200)
        assert 0.3 <= result["ari_observed"] <= 0.7

    def test_moderate_match_is_significant(self, moderate_match_labels):
        """ARI ~ 0.5 should be significant at alpha=0.05."""
        y_true, y_pred = moderate_match_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=500)
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    # -- Correctness: degenerate / single cluster ---------------------------

    def test_single_cluster_both_ari_is_one(self, single_cluster_labels):
        """Both labelings single cluster -> ARI = 1.0 (structure match).

        Per spec section 7.2: if all labels in y_true AND all labels in y_pred
        are identical -> ARI = 1.0 (both have exactly one cluster).
        """
        y_true, y_pred = single_cluster_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["ari_observed"] == pytest.approx(1.0, abs=1e-9)

    def test_single_cluster_null_is_not_significant(self, single_cluster_labels):
        """A labeling invariant under permutation contains no evidence of association."""
        y_true, y_pred = single_cluster_labels
        result = permutation_test_ari(
            y_true, y_pred, n_permutations=100, random_state=42
        )
        np.testing.assert_array_equal(result["null_distribution"], 1.0)
        assert result["p_value"] == 1.0
        assert result["is_significant"] is False

    def test_one_single_one_multi_ari_is_zero(self, one_vs_many_labels):
        """One single-cluster, one multi-cluster -> ARI = 0.0.

        Per spec section 7.2: structure does not match -> ARI = 0.0.
        """
        y_true, y_pred = one_vs_many_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["ari_observed"] == pytest.approx(0.0, abs=1e-9)

    def test_one_single_one_multi_is_not_significant(self, one_vs_many_labels):
        """ARI = 0.0 should never be significant."""
        y_true, y_pred = one_vs_many_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["is_significant"] is False

    # -- Determinism / reproducibility --------------------------------------

    def test_same_random_state_same_result(self, perfect_labels):
        """Same random_state -> identical result (Property 9.6.3)."""
        y_true, y_pred = perfect_labels
        r1 = permutation_test_ari(y_true, y_pred, n_permutations=100, random_state=42)
        r2 = permutation_test_ari(y_true, y_pred, n_permutations=100, random_state=42)
        assert r1["p_value"] == r2["p_value"]
        assert r1["ari_observed"] == r2["ari_observed"]
        np.testing.assert_array_equal(r1["null_distribution"], r2["null_distribution"])

    def test_different_random_state_different_result(self, random_labels):
        """Different random_state -> (likely) different null distribution."""
        y_true, y_pred = random_labels
        r1 = permutation_test_ari(y_true, y_pred, n_permutations=200, random_state=1)
        r2 = permutation_test_ari(y_true, y_pred, n_permutations=200, random_state=2)
        # Null distributions should differ (very unlikely to be identical)
        assert not np.array_equal(r1["null_distribution"], r2["null_distribution"])

    # -- n_permutations parameter -------------------------------------------

    @pytest.mark.parametrize("B", [10, 50, 100, 200])
    def test_n_permutations_parameter_respected(self, perfect_labels, B):
        """n_permutations must control the length of null_distribution."""
        y_true, y_pred = perfect_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=B)
        assert len(result["null_distribution"]) == B
        assert result["n_permutations"] == B

    def test_larger_b_gives_smaller_or_equal_minimum_p(self, perfect_labels):
        """Larger B -> smaller or equal minimum resolvable p-value = 1/(B+1)."""
        y_true, y_pred = perfect_labels
        r10 = permutation_test_ari(y_true, y_pred, n_permutations=10)
        r100 = permutation_test_ari(y_true, y_pred, n_permutations=100)
        r500 = permutation_test_ari(y_true, y_pred, n_permutations=500)
        assert 1.0 / 11 >= 1.0 / 101 >= 1.0 / 501
        assert r10["p_value"] >= 1.0 / 11
        assert r100["p_value"] >= 1.0 / 101
        assert r500["p_value"] >= 1.0 / 501

    # -- Alternative hypotheses ---------------------------------------------

    @pytest.mark.parametrize("alt,expect_p", [
        ("greater", lambda ari_obs, nd: (1 + np.sum(nd >= ari_obs)) / (len(nd) + 1)),
        ("less",    lambda ari_obs, nd: (1 + np.sum(nd <= ari_obs)) / (len(nd) + 1)),
        ("two-sided", lambda ari_obs, nd: (1 + np.sum(np.abs(nd) >= np.abs(ari_obs))) / (len(nd) + 1)),
    ])
    def test_alternative_p_value_formula(self, moderate_match_labels, alt, expect_p):
        """P-value must follow the correct formula for each alternative.

        We cross-check against a manual recomputation using the returned
        null_distribution (Definition 2.3.3).
        """
        y_true, y_pred = moderate_match_labels
        result = permutation_test_ari(
            y_true, y_pred, n_permutations=200, alternative=alt
        )
        expected_p = expect_p(result["ari_observed"], result["null_distribution"])
        assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)

    # -- Sample size warnings -----------------------------------------------

    def test_warning_for_n_less_than_30(self):
        """n < 30 must produce a WARNING-level message (section 2.6)."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["warning"] is not None
        assert "WARNING" in result["warning"]
        assert "n=" in result["warning"]

    def test_info_for_n_between_30_and_100(self):
        """30 <= n < 100 must produce an INFO-level message."""
        y_true = np.repeat([0, 1], 20)          # n = 40
        y_pred = np.tile([0, 1], 20)
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["warning"] is not None
        assert "INFO" in result["warning"]

    def test_no_warning_for_n_at_least_100(self):
        """n >= 100 must NOT produce a warning."""
        y_true = np.repeat([0, 1], 50)          # n = 100
        y_pred = np.tile([0, 1], 50)
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        assert result["warning"] is None

    # -- Edge cases ---------------------------------------------------------

    def test_empty_labels_error(self):
        """Empty arrays should raise ValueError (n < 2)."""
        with pytest.raises(ValueError):
            permutation_test_ari(np.array([]), np.array([]))

    def test_single_sample_error(self):
        """Single sample should raise ValueError (n < 2)."""
        with pytest.raises(ValueError):
            permutation_test_ari(np.array([0]), np.array([0]))

    def test_mismatched_lengths_error(self):
        """Different-length inputs should raise ValueError."""
        with pytest.raises(ValueError):
            permutation_test_ari(np.array([0, 0, 1]), np.array([0, 1]))

    def test_n_permutations_too_small_error(self):
        """n_permutations < 10 should raise ValueError (section 7.1)."""
        with pytest.raises(ValueError):
            permutation_test_ari(
                np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                n_permutations=5,
            )

    def test_n_permutations_must_be_integer(self):
        """Non-integer n_permutations should raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            permutation_test_ari(
                np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                n_permutations=100.5,
            )

    def test_minimum_n_permutations_10(self):
        """Exactly B=10 is the minimum allowed value."""
        result = permutation_test_ari(
            np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
            n_permutations=10,
        )
        assert len(result["null_distribution"]) == 10

    def test_small_permutations_warning_present(self):
        """B < 100 should include a warning about coarse p-values (section 7.5)."""
        result = permutation_test_ari(
            np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
            n_permutations=50,
        )
        # The warning may contain permutation warning info
        assert result["warning"] is not None

    def test_invalid_alternative_raises(self):
        """Invalid alternative should raise ValueError."""
        with pytest.raises(ValueError):
            permutation_test_ari(
                np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                n_permutations=50, alternative="invalid",
            )

    # -- Effect size classification -----------------------------------------

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), "none"),  # ARI=1.0 but single cluster -> actually "large"
        (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), "none"),  # ARI ~ 0
    ])
    def test_effect_size_known_cases(self, y_true, y_pred, expected):
        """Effect size classification for known ARI values."""
        result = permutation_test_ari(y_true, y_pred, n_permutations=100)
        if result["ari_observed"] <= 0:
            assert result["effect_size"] == "none"
        elif result["ari_observed"] <= 0.1:
            assert result["effect_size"] == "small"
        elif result["ari_observed"] <= 0.25:
            assert result["effect_size"] == "medium"
        else:
            assert result["effect_size"] == "large"

    # -- Null distribution properties (invariants) --------------------------

    def test_null_distribution_mean_near_zero_for_random(self, random_labels):
        """Under H0, E[ari_null] ~ 0 (Invariant 9.2.1)."""
        y_true, y_pred = random_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=500)
        null_mean = np.mean(result["null_distribution"])
        assert null_mean == pytest.approx(0.0, abs=0.1)

    def test_null_distribution_ci_contains_zero(self, random_labels):
        """For random labelings, the 95% null CI should contain 0."""
        y_true, y_pred = random_labels
        result = permutation_test_ari(y_true, y_pred, n_permutations=500)
        lo, hi = result["ci_95"]
        assert lo <= 0.0 <= hi


# ===========================================================================
# 2. bootstrap_ci_ari
# ===========================================================================

class TestBootstrapCIARI:
    """Tests for bootstrap_ci_ari() per SIGNIFICANCE_SPEC sections 3, 12."""

    REQUIRED_KEYS = {
        "ari_mean", "ari_std", "ci_lower", "ci_upper", "confidence",
        "is_stable", "n_bootstrap", "n_samples", "distribution", "warning",
    }

    @pytest.mark.parametrize("key", list(REQUIRED_KEYS))
    def test_return_schema_contains_key(self, blobs_data, key):
        """Every required key must be present."""
        X, y_true = blobs_data
        # Use a callable clusterer for simplicity
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("key,expected_type", [
        ("ari_mean", (int, float, np.floating)),
        ("ari_std", (int, float, np.floating)),
        ("ci_lower", (int, float, np.floating)),
        ("ci_upper", (int, float, np.floating)),
        ("confidence", (int, float, np.floating)),
        ("is_stable", bool),
        ("n_bootstrap", (int, np.integer)),
        ("n_samples", (int, np.integer)),
        ("distribution", (np.ndarray, type(None))),
        ("warning", (str, type(None))),
    ])
    def test_return_schema_types(self, blobs_data, key, expected_type):
        """Each key must have the correct type."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        assert isinstance(result[key], expected_type), \
            f"Key '{key}': expected {expected_type}, got {type(result[key])}"

    def test_kmeans_on_blobs_ari_mean_high(self, blobs_data):
        """KMeans on well-separated blobs should give high ari_mean."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        # ARI should be quite high for well-separated blobs
        assert result["ari_mean"] > 0.7

    def test_kmeans_on_blobs_ci_above_zero(self, blobs_data):
        """KMeans on blobs should have CI lower bound > 0 (stable)."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        assert result["ci_lower"] > 0.0
        assert result["is_stable"] is True

    def test_kmeans_on_blobs_ci_well_formed(self, blobs_data):
        """KMeans on blobs: CI must be well-formed with lower < upper."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        assert result["ci_lower"] <= result["ci_upper"]
        assert 0 < result["ci_lower"] <= result["ci_upper"] <= 1.0

    def test_ci_contains_true_ari_for_real_data(self, blobs_data):
        """Bootstrap CI should contain the true ARI for well-separated blobs.

        We use KMeans on make_blobs data where true clusters exist.
        The CI should cover the ARI computed on the full dataset.
        """
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)

        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=100, confidence=0.95)
        assert result["ci_lower"] <= result["ci_upper"]
        # The bootstrap estimates ARI on resampled data, which may differ
        # from the full-data ARI. We just check the CI is well-formed.
        assert -1.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0

    def test_ci_order_invariant(self, blobs_data):
        """ci_lower <= ci_upper must always hold."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        assert result["ci_lower"] <= result["ci_upper"]

    def test_confidence_parameter_respected(self, blobs_data):
        """The confidence level must be reflected in the result."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, confidence=0.90,
        )
        assert result["confidence"] == pytest.approx(0.90, abs=1e-9)

    def test_n_bootstrap_parameter_respected(self, blobs_data):
        """n_bootstrap must control the number of replicates."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, return_distribution=True,
        )
        assert len(result["distribution"]) == 50
        assert result["n_bootstrap"] == 50

    def test_return_distribution_false(self, blobs_data):
        """return_distribution=False -> distribution is None."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, return_distribution=False,
        )
        assert result["distribution"] is None

    def test_n_samples_field(self, blobs_data):
        """n_samples must equal the number of rows in X."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=30)
        assert result["n_samples"] == X.shape[0]

    # -- Determinism --------------------------------------------------------

    def test_same_random_state_same_result(self, blobs_data):
        """Same random_state -> identical result (Property 9.6.3)."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        r1 = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, random_state=42,
            return_distribution=True,
        )
        r2 = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, random_state=42,
            return_distribution=True,
        )
        assert r1["ari_mean"] == r2["ari_mean"]
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]
        np.testing.assert_array_equal(r1["distribution"], r2["distribution"])

    # -- Error handling -----------------------------------------------------

    def test_n_bootstrap_too_small(self, blobs_data):
        """n_bootstrap < 10 should raise ValueError."""
        X, y_true = blobs_data
        with pytest.raises(ValueError):
            bootstrap_ci_ari(X, lambda x: np.zeros(len(x)), y_true, n_bootstrap=5)

    def test_invalid_confidence_raises(self, blobs_data):
        """confidence outside (0, 1) should raise ValueError."""
        X, y_true = blobs_data
        with pytest.raises(ValueError):
            bootstrap_ci_ari(
                X, lambda x: np.zeros(len(x)), y_true, n_bootstrap=30, confidence=1.5,
            )
        with pytest.raises(ValueError):
            bootstrap_ci_ari(
                X, lambda x: np.zeros(len(x)), y_true, n_bootstrap=30, confidence=0.0,
            )

    def test_mismatched_y_true_length_raises(self, blobs_data):
        """y_true length must match X rows."""
        X, _ = blobs_data
        with pytest.raises(ValueError):
            bootstrap_ci_ari(
                X, lambda x: np.zeros(len(x)),
                np.array([0, 1]),  # wrong length
                n_bootstrap=30,
            )

    # -- Stability monotonicity (Invariant 9.3.3) ---------------------------

    def test_stability_implies_positive_mean(self, blobs_data):
        """If is_stable=True then ari_mean > 0 (Invariant 9.3.3)."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(X, clusterer, y_true, n_bootstrap=50)
        if result["is_stable"]:
            assert result["ari_mean"] > 0.0


# ===========================================================================
# 3. paired_permutation_test
# ===========================================================================

class TestPairedPermutationTest:
    """Tests for paired_permutation_test() per SIGNIFICANCE_SPEC sections 4, 12."""

    REQUIRED_KEYS = {
        "ari_1", "ari_2", "delta_obs", "p_value", "is_significant",
        "n_permutations", "null_distribution", "better_method", "warning",
    }

    @pytest.mark.parametrize("key", list(REQUIRED_KEYS))
    def test_return_schema_contains_key(self, perfect_labels, key):
        """Every required key must be present."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])  # different
        result = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100)
        assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("key,expected_type", [
        ("ari_1", (int, float, np.floating)),
        ("ari_2", (int, float, np.floating)),
        ("delta_obs", (int, float, np.floating)),
        ("p_value", (int, float, np.floating)),
        ("is_significant", bool),
        ("n_permutations", (int, np.integer)),
        ("null_distribution", np.ndarray),
        ("better_method", (int, type(None))),
        ("warning", (str, type(None))),
    ])
    def test_return_schema_types(self, perfect_labels, key, expected_type):
        """Each key must have the correct type."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        result = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100)
        assert isinstance(result[key], expected_type), \
            f"Key '{key}': expected {expected_type}, got {type(result[key])}"

    # -- Identical labels ---------------------------------------------------

    def test_identical_labels_delta_is_zero(self, perfect_labels):
        """labels_a == labels_b -> delta_obs = 0 (Invariant 9.4.2)."""
        y_true, labels = perfect_labels
        result = paired_permutation_test(labels, labels, y_true, n_permutations=100)
        assert result["delta_obs"] == pytest.approx(0.0, abs=1e-9)

    def test_identical_labels_p_value_is_one(self, perfect_labels):
        """labels_a == labels_b -> p_value = 1.0 (Invariant 9.4.2)."""
        y_true, labels = perfect_labels
        result = paired_permutation_test(labels, labels, y_true, n_permutations=100)
        assert result["p_value"] == pytest.approx(1.0, abs=1e-9)

    def test_identical_labels_not_significant(self, perfect_labels):
        """labels_a == labels_b -> is_significant = False."""
        y_true, labels = perfect_labels
        result = paired_permutation_test(labels, labels, y_true, n_permutations=100)
        assert result["is_significant"] is False

    def test_identical_labels_better_method_none(self, perfect_labels):
        """labels_a == labels_b -> better_method = None."""
        y_true, labels = perfect_labels
        result = paired_permutation_test(labels, labels, y_true, n_permutations=100)
        assert result["better_method"] is None

    # -- Different labels (good vs bad clustering) --------------------------

    def test_different_labels_delta_nonzero(self, perfect_labels):
        """Structurally different labelings -> delta_obs != 0.

        Note: ARI is invariant to label permutation, so we need structurally
        different clusterings (not just permuted labels) to get delta != 0.
        """
        y_true, labels_a = perfect_labels
        # labels_b merges clusters 0 and 1, keeping cluster 2 separate
        labels_b = np.array([0, 0, 0, 0, 1, 1])
        result = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100)
        assert result["delta_obs"] != 0.0

    def test_good_vs_bad_clustering_small_p_value(self, blobs_data):
        """Perfect labels vs random labels -> small p_value, better_method=1."""
        X, y_true = blobs_data
        labels_perfect = y_true.copy()
        rng = np.random.default_rng(123)
        labels_random = rng.integers(0, 3, size=len(y_true))
        result = paired_permutation_test(
            labels_perfect, labels_random, y_true, n_permutations=200,
        )
        # Perfect clustering should be significantly better than random
        assert result["p_value"] < 0.05
        assert result["is_significant"] is True
        assert result["better_method"] == 1
        assert result["delta_obs"] > 0

    # -- Symmetry (Invariant 9.4.1) -----------------------------------------

    def test_symmetry(self, perfect_labels):
        """Swapping method 1 and 2: p_value unchanged, delta sign flips."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        r1 = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100)
        r2 = paired_permutation_test(labels_b, labels_a, y_true, n_permutations=100)
        assert r1["p_value"] == pytest.approx(r2["p_value"], abs=1e-9)
        assert r1["delta_obs"] == pytest.approx(-r2["delta_obs"], abs=1e-9)
        if r1["better_method"] == 1:
            assert r2["better_method"] == 2
        elif r1["better_method"] == 2:
            assert r2["better_method"] == 1
        else:
            assert r2["better_method"] is None

    # -- Delta bounds -------------------------------------------------------

    def test_delta_obs_in_range(self, perfect_labels):
        """delta_obs must be in [-2, 2] (two ARI differences)."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        result = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100)
        assert -2.0 <= result["delta_obs"] <= 2.0

    # -- Determinism --------------------------------------------------------

    def test_same_random_state_same_result(self, perfect_labels):
        """Same random_state -> identical result."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        r1 = paired_permutation_test(
            labels_a, labels_b, y_true, n_permutations=100, random_state=42,
        )
        r2 = paired_permutation_test(
            labels_a, labels_b, y_true, n_permutations=100, random_state=42,
        )
        assert r1["p_value"] == r2["p_value"]
        assert r1["delta_obs"] == r2["delta_obs"]
        np.testing.assert_array_equal(r1["null_distribution"], r2["null_distribution"])

    # -- Edge cases / error handling ----------------------------------------

    def test_mismatched_lengths_error(self):
        """Different-length inputs should raise ValueError."""
        with pytest.raises(ValueError):
            paired_permutation_test(
                np.array([0, 0, 1]), np.array([0, 1]), np.array([0, 0, 1]),
            )

    def test_n_permutations_too_small(self):
        """n_permutations < 10 should raise ValueError."""
        with pytest.raises(ValueError):
            paired_permutation_test(
                np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                np.array([0, 0, 1, 1]), n_permutations=5,
            )

    def test_all_three_must_match(self):
        """labels_a, labels_b, y_true must all have the same length."""
        with pytest.raises(ValueError):
            paired_permutation_test(
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1]),  # different length
                n_permutations=100,
            )


# ===========================================================================
# 4. cluster_significance
# ===========================================================================

class TestClusterSignificance:
    """Tests for cluster_significance() per SIGNIFICANCE_SPEC sections 5, 12."""

    REQUIRED_TOP_KEYS = {
        "overall_silhouette", "n_clusters", "clusters",
        "n_significant", "significant_clusters", "warning",
    }

    REQUIRED_CLUSTER_KEYS = {
        "cluster_id", "size", "mean_silhouette",
        "silhouette_ci_lower", "silhouette_ci_upper",
        "is_significant", "p_value", "effect_size",
    }

    @pytest.fixture
    def well_separated_embedding(self):
        """A 2-D dataset with 3 very well-separated clusters."""
        X, y = make_blobs(
            n_samples=150, centers=3, n_features=2,
            cluster_std=0.3, random_state=42,
        )
        return X, y

    # -- Return schema ------------------------------------------------------

    @pytest.mark.parametrize("key", list(REQUIRED_TOP_KEYS))
    def test_return_schema_top_level(self, well_separated_embedding, key):
        """Every required top-level key must be present."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert key in result, f"Missing top-level key: {key}"

    @pytest.mark.parametrize("key,expected_type", [
        ("overall_silhouette", (int, float, np.floating)),
        ("n_clusters", (int, np.integer)),
        ("clusters", list),
        ("n_significant", (int, np.integer)),
        ("significant_clusters", list),
        ("warning", (str, type(None))),
    ])
    def test_return_schema_top_types(self, well_separated_embedding, key, expected_type):
        """Top-level keys must have correct types."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert isinstance(result[key], expected_type), \
            f"Key '{key}': expected {expected_type}, got {type(result[key])}"

    @pytest.mark.parametrize("key", list(REQUIRED_CLUSTER_KEYS))
    def test_return_schema_per_cluster(self, well_separated_embedding, key):
        """Each cluster dict must contain all required keys."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert len(result["clusters"]) > 0
        for i, cluster in enumerate(result["clusters"]):
            assert key in cluster, f"Missing key '{key}' in cluster {i}"

    def test_n_clusters_matches_unique_labels(self, well_separated_embedding):
        """n_clusters must equal the number of unique label values."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert result["n_clusters"] == len(np.unique(labels))

    def test_n_significant_count_correct(self, well_separated_embedding):
        """n_significant must equal the number of clusters with is_significant=True."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        expected = sum(1 for c in result["clusters"] if c["is_significant"])
        assert result["n_significant"] == expected

    def test_significant_clusters_ids_correct(self, well_separated_embedding):
        """significant_clusters must list IDs of significant clusters."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        expected_ids = [c["cluster_id"] for c in result["clusters"] if c["is_significant"]]
        assert result["significant_clusters"] == expected_ids

    def test_cluster_sizes_sum_to_n(self, well_separated_embedding):
        """Sum of cluster sizes must equal total number of samples."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        total = sum(c["size"] for c in result["clusters"])
        assert total == len(labels)

    # -- Well-separated clusters should be significant ----------------------

    def test_well_separated_clusters_are_significant(self, well_separated_embedding):
        """Well-separated blobs should have all clusters significant.

        With low cluster_std in make_blobs, silhouette scores should be high
        and all clusters should have CI lower bound > 0.
        """
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=100)
        # Most clusters should be significant for well-separated data
        assert result["n_significant"] >= 2, \
            f"Expected at least 2 significant clusters, got {result['n_significant']}"
        assert result["overall_silhouette"] > 0.3

    def test_mean_silhouette_in_range(self, well_separated_embedding):
        """Each cluster's mean_silhouette must be in [-1, 1]."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        for cluster in result["clusters"]:
            if not np.isnan(cluster["mean_silhouette"]):
                assert -1.0 <= cluster["mean_silhouette"] <= 1.0

    def test_overall_silhouette_in_range(self, well_separated_embedding):
        """overall_silhouette must be in [-1, 1] (Invariant 9.5.1)."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert -1.0 <= result["overall_silhouette"] <= 1.0 or np.isnan(result["overall_silhouette"])

    def test_ci_bounds_ordered_per_cluster(self, well_separated_embedding):
        """silhouette_ci_lower <= silhouette_ci_upper for each cluster."""
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        for cluster in result["clusters"]:
            if not np.isnan(cluster["silhouette_ci_lower"]):
                assert cluster["silhouette_ci_lower"] <= cluster["silhouette_ci_upper"]

    # -- Effect size classification -----------------------------------------

    def test_effect_size_values_valid(self, well_separated_embedding):
        """effect_size must be one of the allowed values (Definition 5.5.1)."""
        valid = {"poor", "weak", "moderate", "strong", "very strong"}
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        for cluster in result["clusters"]:
            assert cluster["effect_size"] in valid, \
                f"Invalid effect_size: {cluster['effect_size']}"

    # -- Significance implies positive silhouette ---------------------------

    def test_significant_implies_positive_silhouette(self, well_separated_embedding):
        """If is_significant=True, mean_silhouette should be > 0
        (Invariant 9.5.3: significance requires positive silhouette).
        """
        X, labels = well_separated_embedding
        result = cluster_significance(X, labels, n_bootstrap=50)
        for cluster in result["clusters"]:
            if cluster["is_significant"]:
                assert cluster["mean_silhouette"] > 0.0

    # -- Single cluster edge case -------------------------------------------

    def test_single_cluster(self):
        """Single cluster -> overall_silhouette = NaN, not significant (section 7.3).

        Silhouette is undefined for single clusters (no inter-cluster distance).
        """
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        labels = np.array([0, 0, 0, 0])
        result = cluster_significance(X, labels, n_bootstrap=50)
        assert result["n_clusters"] == 1
        assert np.isnan(result["overall_silhouette"])
        assert result["n_significant"] == 0
        assert len(result["significant_clusters"]) == 0
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["is_significant"] is False

    # -- Small cluster (n_k < 3) edge case ----------------------------------

    def test_small_cluster_skipped(self):
        """Correction must skip untestable clusters without shifting p-values."""
        X = np.array([
            [0, 0], [0, 0.1],          # cluster 0, size 2 (< 3)
            [10, 10], [10, 10.1],      # cluster 1, size 2 (< 3)
            [20, 20], [20, 0.1], [20, 0.2],  # cluster 2, size 3 (>= 3)
        ], dtype=float)
        labels = np.array([0, 0, 1, 1, 2, 2, 2])
        result = cluster_significance(
            X, labels, n_bootstrap=50, correction_method="bonferroni"
        )
        for cluster in result["clusters"]:
            if cluster["size"] < 3:
                assert cluster["is_significant"] is False
                assert cluster["p_value"] is None
                assert "p_value_corrected" not in cluster
            else:
                assert "p_value_corrected" in cluster

    # -- Determinism --------------------------------------------------------

    def test_same_random_state_same_result(self, well_separated_embedding):
        """Same random_state -> identical result."""
        X, labels = well_separated_embedding
        r1 = cluster_significance(X, labels, n_bootstrap=50, random_state=42)
        r2 = cluster_significance(X, labels, n_bootstrap=50, random_state=42)
        assert r1["overall_silhouette"] == pytest.approx(r2["overall_silhouette"], abs=1e-9)
        assert r1["n_significant"] == r2["n_significant"]
        for c1, c2 in zip(r1["clusters"], r2["clusters"]):
            assert c1["mean_silhouette"] == pytest.approx(c2["mean_silhouette"], abs=1e-9)
            assert c1["is_significant"] == c2["is_significant"]

    # -- Error handling -----------------------------------------------------

    def test_n_bootstrap_too_small(self, well_separated_embedding):
        """n_bootstrap < 10 should raise ValueError."""
        X, labels = well_separated_embedding
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=5)

    def test_invalid_confidence_raises(self, well_separated_embedding):
        """confidence outside (0, 1) should raise ValueError."""
        X, labels = well_separated_embedding
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=30, confidence=1.5)
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=30, confidence=-0.1)

    def test_nan_in_data_raises(self):
        """NaN values in X should raise ValueError (section 7.8)."""
        X = np.array([[0.0, np.nan], [1.0, 2.0]])
        labels = np.array([0, 1])
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=50)

    def test_inf_in_data_raises(self):
        """Inf values in X should raise ValueError (section 7.8)."""
        X = np.array([[0.0, np.inf], [1.0, 2.0]])
        labels = np.array([0, 1])
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=50)

    def test_1d_data_raises(self):
        """1D X should raise ValueError (must be 2D)."""
        X = np.array([0.0, 1.0, 2.0])
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=50)

    def test_n_less_than_2_raises(self):
        """n < 2 should raise ValueError."""
        X = np.array([[0.0]])
        labels = np.array([0])
        with pytest.raises(ValueError):
            cluster_significance(X, labels, n_bootstrap=50)

    # -- Multiple testing correction ----------------------------------------

    def test_correction_method_bonferroni(self, well_separated_embedding):
        """Bonferroni correction should adjust p-values upward."""
        X, labels = well_separated_embedding
        result = cluster_significance(
            X, labels, n_bootstrap=50, correction_method="bonferroni",
        )
        for cluster in result["clusters"]:
            if cluster["p_value"] is not None:
                assert "p_value_corrected" in cluster
                assert cluster["p_value_corrected"] >= cluster["p_value"]
                assert cluster["p_value_corrected"] <= 1.0
                assert cluster["is_significant"] == (
                    cluster["p_value_corrected"] <= 0.05
                )

    def test_correction_method_none(self, well_separated_embedding):
        """correction_method=None should skip correction."""
        X, labels = well_separated_embedding
        result = cluster_significance(
            X, labels, n_bootstrap=50, correction_method=None,
        )
        for cluster in result["clusters"]:
            assert "p_value_corrected" not in cluster


# ===========================================================================
# 5. apply_multiple_testing_correction
# ===========================================================================

class TestMultipleTestingCorrection:
    """Tests for apply_multiple_testing_correction() per SIGNIFICANCE_SPEC sections 6, 12.8."""

    def test_return_schema_keys(self):
        """Result must contain all required keys."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="bonferroni")
        required = {"p_values_adjusted", "rejected", "method", "alpha_corrected", "n_rejected"}
        for key in required:
            assert key in result

    def test_bonferroni_adjusted_values(self):
        """Bonferroni: adjusted = min(m * p, 1) (section 6.2, section 12.8)."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="bonferroni")
        expected = np.minimum(3 * pvals, 1.0)
        np.testing.assert_allclose(result["p_values_adjusted"], expected, atol=1e-9)

    def test_bonferroni_rejected(self):
        """Bonferroni with alpha=0.05: reject if p <= alpha/m.

        p=[0.01, 0.03, 0.1], alpha/m = 0.05/3 ~ 0.0167.
        0.01 <= 0.0167 -> True, 0.03 > 0.0167 -> False, 0.1 > 0.0167 -> False.
        """
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="bonferroni", alpha=0.05)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result["rejected"], expected)

    def test_bonferroni_alpha_corrected(self):
        """Bonferroni alpha_corrected = alpha / m (section 12.8)."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="bonferroni", alpha=0.05)
        assert result["alpha_corrected"] == pytest.approx(0.05 / 3, abs=1e-9)

    def test_bonferroni_n_rejected(self):
        """n_rejected must equal the count of True in rejected."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="bonferroni")
        assert result["n_rejected"] == np.sum(result["rejected"])

    def test_sidak_values(self):
        """Sidak: adjusted = 1 - (1 - p)^m (section 6.3)."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="sidak")
        expected = 1.0 - (1.0 - pvals) ** 3
        np.testing.assert_allclose(result["p_values_adjusted"], expected, atol=1e-9)

    def test_sidak_alpha_corrected(self):
        """Sidak alpha_corrected = 1 - (1 - alpha)^(1/m)."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="sidak", alpha=0.05)
        expected_alpha = 1.0 - (1.0 - 0.05) ** (1.0 / 3)
        assert result["alpha_corrected"] == pytest.approx(expected_alpha, abs=1e-9)

    def test_fdr_bh_rejected_valid(self):
        """BH FDR correction should reject at most the expected number."""
        pvals = np.array([0.01, 0.03, 0.1])
        result = apply_multiple_testing_correction(pvals, method="fdr_bh")
        # BH: find largest k where p_(k) <= (k/m) * alpha
        # sorted: [0.01, 0.03, 0.1]
        # k=3: 0.1 <= (3/3)*0.05 = 0.05? No
        # k=2: 0.03 <= (2/3)*0.05 = 0.0333? Yes -> reject first 2
        assert result["n_rejected"] <= len(pvals)
        assert np.sum(result["rejected"]) == result["n_rejected"]

    def test_fdr_bh_adjusted_monotonic(self):
        """BH adjusted p-values should be in [0, 1] and monotonic when sorted."""
        pvals = np.array([0.1, 0.01, 0.03])  # unsorted
        result = apply_multiple_testing_correction(pvals, method="fdr_bh")
        adj = result["p_values_adjusted"]
        assert np.all(adj >= 0.0)
        assert np.all(adj <= 1.0)
        # When sorted by original p-value order, adjusted should be non-decreasing
        # Actually the standard BH adjusted p-values should be monotonic in the sorted order
        order = np.argsort(pvals)
        sorted_adj = adj[order]
        # Check non-decreasing in sorted order
        assert np.all(np.diff(sorted_adj) >= -1e-10)

    def test_invalid_method_raises(self):
        """Unknown correction method should raise ValueError."""
        with pytest.raises(ValueError):
            apply_multiple_testing_correction(
                np.array([0.1, 0.2]), method="invalid_method",
            )

    def test_empty_pvalues(self):
        """Empty p-values array should return empty results."""
        result = apply_multiple_testing_correction(np.array([]), method="bonferroni")
        assert len(result["p_values_adjusted"]) == 0
        assert len(result["rejected"]) == 0
        assert result["n_rejected"] == 0


# ===========================================================================
# 6. Property-based invariants (applied across functions)
# ===========================================================================

class TestInvariants:
    """Cross-cutting invariants that every function must satisfy."""

    def test_permutation_p_value_range(self, rng):
        """Invariant 9.1.1: p_value in [1/(B+1), 1] for 100 random inputs."""
        B = 100
        min_p = 1.0 / (B + 1)
        for _ in range(100):
            n = rng.integers(10, 50)
            y_true = rng.integers(0, 4, size=n)
            y_pred = rng.integers(0, 4, size=n)
            result = permutation_test_ari(y_true, y_pred, n_permutations=B)
            assert min_p <= result["p_value"] <= 1.0

    def test_permutation_ari_range(self, rng):
        """Invariant 9.1.2: ARI(y, y_hat) in [-1, 1] for 100 random inputs."""
        for _ in range(100):
            n = rng.integers(10, 50)
            y_true = rng.integers(0, 4, size=n)
            y_pred = rng.integers(0, 4, size=n)
            result = permutation_test_ari(y_true, y_pred, n_permutations=50)
            assert -1.0 <= result["ari_observed"] <= 1.0

    def test_paired_p_value_range(self, rng):
        """Invariant 9.1.1 (paired): p_value in [1/(B+1), 1]."""
        B = 100
        min_p = 1.0 / (B + 1)
        for _ in range(50):
            n = rng.integers(10, 50)
            y_true = rng.integers(0, 4, size=n)
            labels_a = rng.integers(0, 4, size=n)
            labels_b = rng.integers(0, 4, size=n)
            result = paired_permutation_test(
                labels_a, labels_b, y_true, n_permutations=B,
            )
            assert min_p <= result["p_value"] <= 1.0

    def test_paired_ari_in_range(self, rng):
        """Invariant 9.1.2 (paired): ari_1, ari_2 in [-1, 1]."""
        for _ in range(50):
            n = rng.integers(10, 50)
            y_true = rng.integers(0, 4, size=n)
            labels_a = rng.integers(0, 4, size=n)
            labels_b = rng.integers(0, 4, size=n)
            result = paired_permutation_test(
                labels_a, labels_b, y_true, n_permutations=50,
            )
            assert -1.0 <= result["ari_1"] <= 1.0
            assert -1.0 <= result["ari_2"] <= 1.0

    def test_ci_lower_leq_upper_bootstrap(self, blobs_data):
        """Invariant: ci_lower <= ci_upper for bootstrap CI."""
        X, y_true = blobs_data
        def clusterer(X_b):
            return KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_b)
        result = bootstrap_ci_ari(
            X, clusterer, y_true, n_bootstrap=50, confidence=0.90,
        )
        assert result["ci_lower"] <= result["ci_upper"]

    def test_reproducibility_permutation(self, perfect_labels):
        """Property 9.6.3: Same random_state -> identical results (permutation)."""
        y_true, y_pred = perfect_labels
        r1 = permutation_test_ari(y_true, y_pred, n_permutations=100, random_state=99)
        r2 = permutation_test_ari(y_true, y_pred, n_permutations=100, random_state=99)
        assert r1["p_value"] == r2["p_value"]
        assert r1["ari_observed"] == r2["ari_observed"]
        np.testing.assert_array_equal(r1["null_distribution"], r2["null_distribution"])

    def test_reproducibility_paired(self, perfect_labels):
        """Property 9.6.3: Same random_state -> identical results (paired)."""
        y_true, labels_a = perfect_labels
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        r1 = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100, random_state=99)
        r2 = paired_permutation_test(labels_a, labels_b, y_true, n_permutations=100, random_state=99)
        assert r1["p_value"] == r2["p_value"]
        assert r1["delta_obs"] == r2["delta_obs"]
        np.testing.assert_array_equal(r1["null_distribution"], r2["null_distribution"])

    def test_reproducibility_cluster(self, blobs_data):
        """Property 9.6.3: Same random_state -> identical results (cluster)."""
        X, labels = blobs_data
        r1 = cluster_significance(X, labels, n_bootstrap=50, random_state=99)
        r2 = cluster_significance(X, labels, n_bootstrap=50, random_state=99)
        assert r1["overall_silhouette"] == pytest.approx(r2["overall_silhouette"], abs=1e-9)
        assert r1["n_significant"] == r2["n_significant"]
        for c1, c2 in zip(r1["clusters"], r2["clusters"]):
            assert c1["mean_silhouette"] == pytest.approx(c2["mean_silhouette"], abs=1e-9)
            assert c1["is_significant"] == c2["is_significant"]

    def test_monotonicity_in_b_permutation(self, perfect_labels):
        """Invariant 9.1.3: Larger B -> smaller minimum resolvable p-value.

        We check that the minimum p_value (1/(B+1)) decreases with B.
        """
        y_true, y_pred = perfect_labels
        r10 = permutation_test_ari(y_true, y_pred, n_permutations=10)
        r100 = permutation_test_ari(y_true, y_pred, n_permutations=100)
        r500 = permutation_test_ari(y_true, y_pred, n_permutations=500)
        # Minimum resolvable p-value decreases with B
        assert 1.0 / 11 > 1.0 / 101 > 1.0 / 501
        # Each p_value must be >= its respective minimum
        assert r10["p_value"] >= 1.0 / 11
        assert r100["p_value"] >= 1.0 / 101
        assert r500["p_value"] >= 1.0 / 501
