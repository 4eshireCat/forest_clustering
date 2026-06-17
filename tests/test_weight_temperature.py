"""Tests for weight_temperature parameter in compute_iteration_weights.

These tests define the expected API and behavior BEFORE production code exists.
All tests should fail (TypeError / ValueError / unexpected result) until the
weight_temperature parameter is implemented in compute_iteration_weights.

Mathematical specification:
    raw_weights = [w_1, ..., w_L]  in [0, 1] from entropy / inverse_gini

    if temperature == 1.0:
        scaled = raw_weights
    elif temperature > 0:
        scaled = raw_weights ** (1.0 / temperature)
    else:
        raise ValueError

    if scaled.mean() > 0:
        weights = scaled / scaled.mean()   # normalize so mean = 1.0
    else:
        weights = ones(L)                  # fallback when all raw weights are 0

Effects:
    temperature = 1.0  (default): no-op, identical to current behavior
    temperature < 1.0  (e.g. 0.5): sharpen  → increases differences
    temperature > 1.0  (e.g. 2.0): soften   → decreases differences
    temperature -> 0+:  approaches one-hot (only max weight gets non-zero)
    temperature -> inf: approaches uniform  (all weights -> 1)
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def varying_raw_weights_embedding():
    """Embedding that produces clearly different raw weights across iterations.

    Column 0: perfect separation (5 unique cells) → raw weight ~1.0
    Column 1: moderate separation (3 unique)       → raw weight ~0.66
    Column 2: no separation (all same)             → raw weight 0.0
    """
    return np.array([
        [0, 0, 7],
        [1, 1, 7],
        [2, 1, 7],
        [3, 2, 7],
        [4, 2, 7],
    ], dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. Default / no-op behaviour
# ---------------------------------------------------------------------------

class TestTemperatureDefaultAndNoOp:
    """Default temperature=1.0 and explicit no-op behaviour."""

    def test_temperature_default_is_one(self, varying_raw_weights_embedding):
        """Not passing weight_temperature must behave identically to passing 1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        w_no_arg = compute_iteration_weights(E, strategy="entropy")
        w_explicit = compute_iteration_weights(E, strategy="entropy", weight_temperature=1.0)
        np.testing.assert_array_almost_equal(w_no_arg, w_explicit)

    def test_temperature_one_point_oh_no_op(self, varying_raw_weights_embedding):
        """Explicit temperature=1.0 must give exactly the same weights as before."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        w_old = compute_iteration_weights(E, strategy="entropy")
        w_new = compute_iteration_weights(E, strategy="entropy", weight_temperature=1.0)
        np.testing.assert_array_almost_equal(w_old, w_new)


# ---------------------------------------------------------------------------
# 2. Sharpen / soften effects on variance
# ---------------------------------------------------------------------------

class TestTemperatureVarianceEffects:
    """Temperature should increase or decrease variance of the weight vector."""

    def test_temperature_sharpen_increases_variance(self, varying_raw_weights_embedding):
        """temperature=0.5 must produce higher std than temperature=1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        w_default = compute_iteration_weights(E, strategy="entropy", weight_temperature=1.0)
        w_sharpen = compute_iteration_weights(E, strategy="entropy", weight_temperature=0.5)

        assert w_sharpen.std() > w_default.std(), (
            f"Expected sharpen (0.5) std {w_sharpen.std():.4f} > "
            f"default std {w_default.std():.4f}"
        )

    def test_temperature_soften_decreases_variance(self, varying_raw_weights_embedding):
        """temperature=2.0 must produce lower std than temperature=1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        w_default = compute_iteration_weights(E, strategy="entropy", weight_temperature=1.0)
        w_soften = compute_iteration_weights(E, strategy="entropy", weight_temperature=2.0)

        assert w_soften.std() < w_default.std(), (
            f"Expected soften (2.0) std {w_soften.std():.4f} < "
            f"default std {w_default.std():.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Limit behaviour
# ---------------------------------------------------------------------------

class TestTemperatureLimits:
    """Behaviour as temperature approaches extremes."""

    def test_temperature_limit_infinity_approaches_uniform(self, varying_raw_weights_embedding):
        """Large temperature (10.0, clamp max) should give weights all ~1.0."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=10.0)

        # All weights should be close to 1.0 (within 10%)
        assert np.abs(weights - 1.0).max() < 0.1, (
            f"Expected near-uniform weights, got {weights}"
        )

    def test_temperature_limit_zero_approaches_one_hot(self, varying_raw_weights_embedding):
        """Small temperature (0.1, clamp min) should give ~L to max weight, ~0 to others."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        L = E.shape[1]
        weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=0.1)

        # One weight should be larger than others (sharpen effect)
        max_w = weights.max()
        assert max_w > L * 0.3, f"Expected max weight > {L * 0.3}, got {max_w}"
        # Sum of remaining should be relatively small
        others_sum = weights.sum() - max_w
        assert others_sum < 3.0, f"Expected others_sum small, got {others_sum}"


# ---------------------------------------------------------------------------
# 4. Fallback and invariants
# ---------------------------------------------------------------------------

class TestTemperatureFallbackAndInvariants:
    """Fallback behaviour and output invariants for any valid temperature."""

    def test_temperature_all_zero_raw_weights_fallback(self):
        """When all raw weights are 0, any temperature should give uniform weights (fallback)."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = np.zeros((5, 4), dtype=np.int64)  # all same cell → all raw weights = 0
        for temp in [0.5, 1.0, 2.0, 10.0]:
            weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=temp)
            expected = np.ones(4, dtype=np.float64)
            np.testing.assert_array_almost_equal(
                weights, expected,
                err_msg=f"Fallback failed for temperature={temp}"
            )

    def test_temperature_preserves_non_negativity(self, varying_raw_weights_embedding):
        """All output weights must be >= 0 for any positive temperature."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=temp)
            assert (weights >= 0).all(), f"Negative weights found for temperature={temp}"

    def test_temperature_preserves_mean_one(self, varying_raw_weights_embedding):
        """Mean of output weights must be 1.0 for any valid temperature (unless all-zero fallback)."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=temp)
            assert weights.mean() == pytest.approx(1.0), (
                f"Mean != 1.0 for temperature={temp}: got {weights.mean()}"
            )


# ---------------------------------------------------------------------------
# 5. Strategy interaction
# ---------------------------------------------------------------------------

class TestTemperatureWithStrategies:
    """weight_temperature interaction with different strategies."""

    def test_temperature_with_uniform_strategy_no_op(self, varying_raw_weights_embedding):
        """Uniform strategy must ignore temperature (all weights = 1.0)."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        for temp in [0.5, 1.0, 2.0, 10.0]:
            weights = compute_iteration_weights(E, strategy="uniform", weight_temperature=temp)
            expected = np.ones(E.shape[1], dtype=np.float64)
            np.testing.assert_array_equal(
                weights, expected,
                err_msg=f"Uniform strategy should ignore temperature={temp}"
            )

    def test_temperature_with_entropy_strategy(self, varying_raw_weights_embedding):
        """Entropy strategy + temperature=0.5 must run without error."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=0.5)
        assert weights.shape == (E.shape[1],)
        assert weights.dtype == np.float64
        assert weights.mean() == pytest.approx(1.0)

    def test_temperature_with_inverse_gini_strategy(self, varying_raw_weights_embedding):
        """Inverse-gini strategy + temperature=2.0 must run without error."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        weights = compute_iteration_weights(E, strategy="inverse_gini", weight_temperature=2.0)
        assert weights.shape == (E.shape[1],)
        assert weights.dtype == np.float64
        assert weights.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Invalid temperature values
# ---------------------------------------------------------------------------

class TestTemperatureInvalidValues:
    """Error handling for invalid temperature inputs."""

    def test_negative_temperature_raises_value_error(self, varying_raw_weights_embedding):
        """temperature <= 0 must raise ValueError."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        with pytest.raises(ValueError):
            compute_iteration_weights(E, strategy="entropy", weight_temperature=-1.0)

    def test_temperature_zero_raises_value_error(self, varying_raw_weights_embedding):
        """temperature = 0 must raise ValueError."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        E = varying_raw_weights_embedding
        with pytest.raises(ValueError):
            compute_iteration_weights(E, strategy="entropy", weight_temperature=0.0)

    def test_temperature_extremely_small_positive_works(self):
        """temperature=1e-10 should give valid weights (no overflow/underflow)."""
        from forest_clustering.iteration_weights import compute_iteration_weights

        # Embedding with two different raw weights to see the extreme sharpening effect
        E = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
        ], dtype=np.int64)

        weights = compute_iteration_weights(E, strategy="entropy", weight_temperature=1e-10)
        assert weights.shape == (2,)
        assert (weights >= 0).all()
        assert weights.mean() == pytest.approx(1.0)
