"""Integration tests for weighted-embedding support in ForestClusterer.

These tests verify that ForestClusterer accepts the new iteration_weighting
parameter, exposes iteration_weights_, and integrates correctly with the
weighted distance functions.

All tests should fail (ImportError / AttributeError) until production code exists.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mixed_df():
    """Small synthetic DataFrame with mixed feature types."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num_b": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "cat_a": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })


# ---------------------------------------------------------------------------
# Parameter acceptance
# ---------------------------------------------------------------------------

class TestIterationWeightingParameter:
    """ForestClusterer must accept iteration_weighting parameter."""

    def test_default_is_uniform(self):
        """Default iteration_weighting must be 'uniform'."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer()
        assert c.iteration_weighting == "uniform"

    def test_accepts_entropy(self):
        """ForestClusterer must accept 'entropy' strategy."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(iteration_weighting="entropy")
        assert c.iteration_weighting == "entropy"

    def test_accepts_inverse_gini(self):
        """ForestClusterer must accept 'inverse_gini' strategy."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(iteration_weighting="inverse_gini")
        assert c.iteration_weighting == "inverse_gini"

    def test_invalid_strategy_raises(self):
        """Invalid iteration_weighting must raise ValueError at init or fit time."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(iteration_weighting="bogus_strategy")
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            c.fit(X)

    def test_parameter_in_get_params(self):
        """iteration_weighting must appear in get_params()."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(iteration_weighting="entropy")
        params = c.get_params()
        assert "iteration_weighting" in params
        assert params["iteration_weighting"] == "entropy"


# ---------------------------------------------------------------------------
# Fit / predict with weighting strategies
# ---------------------------------------------------------------------------

class TestFitPredictWithWeighting:
    """fit_predict must succeed with all weighting strategies."""

    def test_entropy_runs_fit_predict(self, small_mixed_df):
        """ForestClusterer with 'entropy' must run fit_predict without error."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        labels = c.fit_predict(small_mixed_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(small_mixed_df)

    def test_inverse_gini_runs_fit_predict(self, small_mixed_df):
        """ForestClusterer with 'inverse_gini' must run fit_predict without error."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="inverse_gini",
        )
        labels = c.fit_predict(small_mixed_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(small_mixed_df)

    def test_uniform_runs_fit_predict(self, small_mixed_df):
        """ForestClusterer with 'uniform' (default) must run fit_predict."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="uniform",
        )
        labels = c.fit_predict(small_mixed_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(small_mixed_df)


# ---------------------------------------------------------------------------
# iteration_weights_ attribute
# ---------------------------------------------------------------------------

class TestIterationWeightsAttribute:
    """After fitting, iteration_weights_ must be available."""

    def test_attribute_exists_after_fit(self, small_mixed_df):
        """iteration_weights_ must exist after fit()."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        assert hasattr(c, "iteration_weights_")

    def test_attribute_shape(self, small_mixed_df):
        """iteration_weights_ must have shape (L,) where L = n_iterations."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        assert c.iteration_weights_.shape == (20,)

    def test_attribute_dtype(self, small_mixed_df):
        """iteration_weights_ must be float64."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        assert c.iteration_weights_.dtype == np.float64

    def test_attribute_non_negative(self, small_mixed_df):
        """All iteration weights must be non-negative."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        assert (c.iteration_weights_ >= 0).all()

    def test_attribute_mean_is_one(self, small_mixed_df):
        """Mean of iteration_weights_ must be 1.0 (normalization)."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        assert c.iteration_weights_.mean() == pytest.approx(1.0)

    def test_uniform_strategy_weights_are_ones(self, small_mixed_df):
        """With 'uniform' strategy, all weights must be exactly 1.0."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="uniform",
        )
        c.fit(small_mixed_df)
        np.testing.assert_array_equal(
            c.iteration_weights_, np.ones(20, dtype=np.float64)
        )


# ---------------------------------------------------------------------------
# get_iteration_weights() method
# ---------------------------------------------------------------------------

class TestGetIterationWeightsMethod:
    """ForestClusterer must expose get_iteration_weights()."""

    def test_method_exists(self, small_mixed_df):
        """get_iteration_weights must be a callable method."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=10, random_state=42)
        c.fit(small_mixed_df)
        assert hasattr(c, "get_iteration_weights")
        assert callable(c.get_iteration_weights)

    def test_method_returns_array(self, small_mixed_df):
        """get_iteration_weights() must return a numpy array."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        w = c.get_iteration_weights()
        assert isinstance(w, np.ndarray)

    def test_method_returns_same_as_attribute(self, small_mixed_df):
        """get_iteration_weights() must equal iteration_weights_ attribute."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        w_method = c.get_iteration_weights()
        w_attr = c.iteration_weights_
        np.testing.assert_array_equal(w_method, w_attr)

    def test_method_before_fit_raises(self):
        """get_iteration_weights() before fit() must raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        from forest_clustering import ForestClusterer
        c = ForestClusterer(n_iterations=10, random_state=42)
        with pytest.raises(NotFittedError):
            c.get_iteration_weights()


# ---------------------------------------------------------------------------
# pairwise_distance with weighted embedding
# ---------------------------------------------------------------------------

class TestPairwiseDistanceWithWeights:
    """pairwise_distance must use weighted Hamming when weights != uniform."""

    def test_pairwise_distance_returns_correct_shape(self, small_mixed_df):
        """pairwise_distance must return (n, n) matrix."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D = c.pairwise_distance()
        n = len(small_mixed_df)
        assert D.shape == (n, n)

    def test_pairwise_distance_dtype(self, small_mixed_df):
        """pairwise_distance must return float32."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D = c.pairwise_distance()
        assert D.dtype == np.float32

    def test_pairwise_distance_diagonal_zero(self, small_mixed_df):
        """Diagonal of pairwise_distance must be all zeros."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D = c.pairwise_distance()
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(len(small_mixed_df)))

    def test_pairwise_distance_symmetric(self, small_mixed_df):
        """pairwise_distance matrix must be symmetric."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D = c.pairwise_distance()
        np.testing.assert_array_almost_equal(D, D.T)

    def test_pairwise_distance_range(self, small_mixed_df):
        """pairwise_distance values must lie in [0, 1]."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D = c.pairwise_distance()
        assert D.min() >= 0.0
        assert D.max() <= 1.0

    def test_cross_distance_returns_correct_shape(self, small_mixed_df):
        """pairwise_distance(X, Y) must return (n_X, n_Y)."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        X_new = small_mixed_df.iloc[:3]
        Y_new = small_mixed_df.iloc[3:7]
        D_cross = c.pairwise_distance(X_new, Y_new)
        assert D_cross.shape == (3, 4)

    def test_cross_distance_dtype(self, small_mixed_df):
        """Cross-distance must be float32."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        D_cross = c.pairwise_distance(small_mixed_df[:3], small_mixed_df[3:7])
        assert D_cross.dtype == np.float32


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Default behavior must be identical to pre-weighting behavior."""

    def test_default_uniform_produces_identical_results(self, small_mixed_df):
        """Default iteration_weighting='uniform' must produce same labels as original."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="uniform",
        )
        c2 = ForestClusterer(
            n_iterations=20,
            random_state=42,
        )
        labels1 = c1.fit_predict(small_mixed_df)
        labels2 = c2.fit_predict(small_mixed_df)
        np.testing.assert_array_equal(labels1, labels2)

    def test_default_uniform_produces_same_embedding(self, small_mixed_df):
        """Default iteration_weighting='uniform' must produce same embedding."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="uniform",
        )
        c2 = ForestClusterer(
            n_iterations=20,
            random_state=42,
        )
        c1.fit(small_mixed_df)
        c2.fit(small_mixed_df)
        np.testing.assert_array_equal(c1.embedding_, c2.embedding_)

    def test_default_uniform_produces_same_pairwise_distance(self, small_mixed_df):
        """Default iteration_weighting='uniform' must produce same distance matrix."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="uniform",
        )
        c2 = ForestClusterer(
            n_iterations=20,
            random_state=42,
        )
        c1.fit(small_mixed_df)
        c2.fit(small_mixed_df)
        D1 = c1.pairwise_distance()
        D2 = c2.pairwise_distance()
        np.testing.assert_array_almost_equal(D1, D2)


# ---------------------------------------------------------------------------
# transform() on new data with weighted embedding
# ---------------------------------------------------------------------------

class TestTransformWithWeighting:
    """transform() on new data must work with weighted embedding."""

    def test_transform_returns_embedding(self, small_mixed_df):
        """transform() must return (n_new, L) int64 embedding."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        new_data = small_mixed_df.iloc[:3]
        E_new = c.transform(new_data)
        assert E_new.shape == (3, 20)
        assert E_new.dtype == np.int64

    def test_transform_consistent_with_embedding(self, small_mixed_df):
        """transform(train_data) should equal the fitted embedding."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        E_via_transform = c.transform(small_mixed_df)
        E_via_attr = c.get_embedding()
        np.testing.assert_array_equal(E_via_transform, E_via_attr)

    def test_transform_on_ndarray(self, small_mixed_df):
        """transform() must accept numpy ndarray."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
        )
        c.fit(small_mixed_df)
        X_arr = small_mixed_df.values[:3]
        E_new = c.transform(X_arr)
        assert E_new.shape == (3, 20)


# ---------------------------------------------------------------------------
# Multiple strategies comparison
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    """Different strategies should potentially produce different results."""

    def test_entropy_vs_uniform_can_differ(self, small_mixed_df):
        """With enough iterations, entropy weighting may produce different distances."""
        from forest_clustering import ForestClusterer

        c_uni = ForestClusterer(
            n_iterations=50,
            random_state=42,
            iteration_weighting="uniform",
        )
        c_ent = ForestClusterer(
            n_iterations=50,
            random_state=42,
            iteration_weighting="entropy",
        )
        c_uni.fit(small_mixed_df)
        c_ent.fit(small_mixed_df)

        # Embeddings should be identical (same random_state, same specs)
        np.testing.assert_array_equal(c_uni.embedding_, c_ent.embedding_)

        # But weights differ → distance matrices may differ
        D_uni = c_uni.pairwise_distance()
        D_ent = c_ent.pairwise_distance()

        # They are not necessarily different for all data, but the shapes match
        assert D_uni.shape == D_ent.shape

    def test_entropy_vs_inverse_gini_weights_can_differ(self, small_mixed_df):
        """entropy and inverse_gini may assign different weights."""
        from forest_clustering import ForestClusterer

        c_ent = ForestClusterer(
            n_iterations=50,
            random_state=42,
            iteration_weighting="entropy",
        )
        c_gini = ForestClusterer(
            n_iterations=50,
            random_state=42,
            iteration_weighting="inverse_gini",
        )
        c_ent.fit(small_mixed_df)
        c_gini.fit(small_mixed_df)

        w_ent = c_ent.iteration_weights_
        w_gini = c_gini.iteration_weights_

        # Both should have mean 1.0
        assert w_ent.mean() == pytest.approx(1.0)
        assert w_gini.mean() == pytest.approx(1.0)

        # Weights may differ between strategies
        # (not guaranteed for all data, but the vectors are both valid)
        assert w_ent.shape == w_gini.shape
