"""Tests for weight_temperature parameter in ForestClusterer.

These tests define the expected API and behavior BEFORE production code exists.
All tests should fail (TypeError / ValueError / AttributeError) until the
weight_temperature parameter is added to ForestClusterer.__init__ and wired
through to compute_iteration_weights during fit().
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
# 1. Parameter acceptance
# ---------------------------------------------------------------------------

class TestClustererAcceptsWeightTemperature:
    """ForestClusterer must accept weight_temperature parameter."""

    def test_clusterer_accepts_weight_temperature(self):
        """ForestClusterer(n_iterations=10, weight_temperature=0.5) must instantiate."""
        from forest_clustering import ForestClusterer

        c = ForestClusterer(n_iterations=10, weight_temperature=0.5)
        assert c.weight_temperature == 0.5

    def test_clusterer_default_weight_temperature_is_one(self):
        """Default weight_temperature must be 1.0."""
        from forest_clustering import ForestClusterer

        c = ForestClusterer()
        assert c.weight_temperature == 1.0

    def test_clusterer_temperature_in_get_params(self):
        """weight_temperature must appear in get_params()."""
        from forest_clustering import ForestClusterer

        c = ForestClusterer(n_iterations=10, weight_temperature=0.5)
        params = c.get_params()
        assert "weight_temperature" in params
        assert params["weight_temperature"] == 0.5


# ---------------------------------------------------------------------------
# 2. Temperature effects on weights
# ---------------------------------------------------------------------------

class TestClustererTemperatureEffects:
    """Different temperature values must produce different iteration weights."""

    def test_clusterer_temperature_sharpen_produces_different_weights(self, small_mixed_df):
        """temp=0.5 vs temp=1.0 must produce different iteration_weights_."""
        from forest_clustering import ForestClusterer

        c_sharpen = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
            weight_temperature=0.5,
        )
        c_default = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
            weight_temperature=1.0,
        )

        c_sharpen.fit(small_mixed_df)
        c_default.fit(small_mixed_df)

        w_sharpen = c_sharpen.iteration_weights_
        w_default = c_default.iteration_weights_

        # Should be different (not allclose)
        assert not np.allclose(w_sharpen, w_default), (
            "Sharpen (0.5) should produce different weights than default (1.0)"
        )

    def test_clusterer_temperature_different_values_produce_different_distances(self, small_mixed_df):
        """pairwise_distance must differ for temp=0.5 vs temp=2.0."""
        from forest_clustering import ForestClusterer

        c1 = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
            weight_temperature=0.5,
        )
        c2 = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
            weight_temperature=2.0,
        )

        c1.fit(small_mixed_df)
        c2.fit(small_mixed_df)

        D1 = c1.pairwise_distance()
        D2 = c2.pairwise_distance()

        assert D1.shape == D2.shape
        assert not np.allclose(D1, D2), (
            "Different temperatures should produce different distance matrices"
        )


# ---------------------------------------------------------------------------
# 3. Fit / predict with temperature and strategies
# ---------------------------------------------------------------------------

class TestClustererFitPredictWithTemperature:
    """fit_predict must succeed with temperature + weighting strategies."""

    def test_clusterer_temperature_with_entropy(self, small_mixed_df):
        """fit_predict with entropy + temperature=0.5 must succeed."""
        from forest_clustering import ForestClusterer

        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="entropy",
            weight_temperature=0.5,
        )
        labels = c.fit_predict(small_mixed_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(small_mixed_df)
        # iteration_weights_ should be set
        assert hasattr(c, "iteration_weights_")
        assert c.iteration_weights_.mean() == pytest.approx(1.0)

    def test_clusterer_temperature_with_inverse_gini(self, small_mixed_df):
        """fit_predict with inverse_gini + temperature=2.0 must succeed."""
        from forest_clustering import ForestClusterer

        c = ForestClusterer(
            n_iterations=20,
            random_state=42,
            iteration_weighting="inverse_gini",
            weight_temperature=2.0,
        )
        labels = c.fit_predict(small_mixed_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(small_mixed_df)
        assert hasattr(c, "iteration_weights_")
        assert c.iteration_weights_.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Invalid temperature values
# ---------------------------------------------------------------------------

class TestClustererInvalidTemperature:
    """Error handling for invalid weight_temperature values."""

    def test_invalid_temperature_raises(self, small_mixed_df):
        """temperature=0 or negative must raise ValueError at fit time."""
        from forest_clustering import ForestClusterer

        c_zero = ForestClusterer(
            n_iterations=10,
            iteration_weighting="entropy",
            weight_temperature=0.0,
        )
        with pytest.raises(ValueError):
            c_zero.fit(small_mixed_df)

        c_neg = ForestClusterer(
            n_iterations=10,
            iteration_weighting="entropy",
            weight_temperature=-0.5,
        )
        with pytest.raises(ValueError):
            c_neg.fit(small_mixed_df)
