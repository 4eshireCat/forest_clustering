"""Tests for kde_peaks_cut_points - density-aware cut generation.

These tests define the expected API and behavior BEFORE any production code exists.
All tests should fail (ImportError / AttributeError) until kde_cuts.py is implemented.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Shared random generator for reproducibility."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# T1: Bimodal distribution
# ---------------------------------------------------------------------------

class TestBimodal:
    """Two Gaussians N(-3, 0.5^2) and N(+3, 0.5^2), n=2000."""

    def test_finds_valley_between_two_peaks(self, rng):
        """Should find exactly 1 cut near 0, strategy='kde'."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert strategy == "kde"
        assert len(cuts) == 1
        assert -1.0 < cuts[0] < 1.0  # valley near 0

    def test_bimodal_with_2_cuts_requested_falls_back(self, rng):
        """Bimodal has only 1 valley; requesting 2 falls back to kde+uniform."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        # When only 1 valley exists but 2 cuts requested, strategy is kde+uniform
        assert strategy == "kde+uniform"
        assert len(cuts) == 2
        # The first cut should be the valley near 0
        valley = cuts[0] if -1.0 < cuts[0] < 1.0 else cuts[1]
        assert -1.0 < valley < 1.0

    def test_bimodal_returns_sorted_cuts(self, rng):
        """All returned cut points must be monotonically increasing."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        np.testing.assert_array_equal(cuts, np.sort(cuts))


# ---------------------------------------------------------------------------
# T2: Trimodal distribution
# ---------------------------------------------------------------------------

class TestTrimodal:
    """Three Gaussians at -5, 0, +5."""

    def test_finds_two_valleys(self, rng):
        """Should find 2 cuts near +/-2.5, strategy='kde'."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-5, 0.5, 1000),
            rng.normal(0, 0.5, 1000),
            rng.normal(5, 0.5, 1000),
        ])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        assert strategy == "kde"
        assert len(cuts) == 2
        assert -4.0 < cuts[0] < -1.0
        assert 1.0 < cuts[1] < 4.0

    def test_trimodal_cuts_are_sorted(self, rng):
        """Two valleys should be returned in ascending order."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-5, 0.5, 1000),
            rng.normal(0, 0.5, 1000),
            rng.normal(5, 0.5, 1000),
        ])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        assert cuts[0] < cuts[1]

    def test_trimodal_with_more_cuts_than_valleys(self, rng):
        """Requesting 3 cuts from trimodal (2 valleys) → kde+uniform."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-5, 0.5, 1000),
            rng.normal(0, 0.5, 1000),
            rng.normal(5, 0.5, 1000),
        ])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert strategy == "kde+uniform"
        assert len(cuts) == 3


# ---------------------------------------------------------------------------
# T3: Uniform distribution fallback
# ---------------------------------------------------------------------------

class TestUniformFallback:
    """U[0, 10] has no valleys in KDE → falls back to uniform."""

    def test_uniform_distribution_fallback(self, rng):
        """No valleys in uniform → strategy='uniform'."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.uniform(0, 10, 2000)
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        assert strategy == "uniform"

    def test_uniform_returns_requested_number_of_cuts(self, rng):
        """Uniform fallback should still return the requested number of cuts."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.uniform(0, 10, 2000)
        cuts, _ = kde_peaks_cut_points(data, n_cuts=4, rng=rng)
        assert len(cuts) == 4
        # All cuts should be within [0, 10]
        assert np.all((0 <= cuts) & (cuts <= 10))

    def test_uniform_cuts_sorted(self, rng):
        """Uniform fallback should return sorted cuts."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.uniform(0, 10, 2000)
        cuts, _ = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        np.testing.assert_array_equal(cuts, np.sort(cuts))


# ---------------------------------------------------------------------------
# T4: Constant data
# ---------------------------------------------------------------------------

class TestConstant:
    """All identical values → uniform fallback, cuts at the constant value."""

    def test_all_identical_values(self, rng):
        """Constant data → strategy='uniform', cuts at constant value."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.full(1000, 5.0)
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        assert strategy == "uniform"
        assert np.allclose(cuts, 5.0)

    def test_constant_returns_correct_number_of_cuts(self, rng):
        """Should return n_cuts copies of the constant value."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.full(500, 3.14)
        cuts, _ = kde_peaks_cut_points(data, n_cuts=5, rng=rng)
        assert len(cuts) == 5
        assert np.allclose(cuts, 3.14)


# ---------------------------------------------------------------------------
# T5: Skewed bimodal
# ---------------------------------------------------------------------------

class TestSkewedBimodal:
    """N(0, 0.5^2) + N(5, 1.5^2) → 1 cut between 1 and 4."""

    def test_skewed_finds_valley(self, rng):
        """Should find valley between modes, strategy='kde'."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(0, 0.5, 1000), rng.normal(5, 1.5, 1000)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert strategy == "kde"
        assert 1.0 < cuts[0] < 4.0

    def test_skewed_returns_single_cut(self, rng):
        """Requesting 1 cut from skewed bimodal should return exactly 1."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(0, 0.5, 1000), rng.normal(5, 1.5, 1000)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert len(cuts) == 1


# ---------------------------------------------------------------------------
# T6: Unimodal (single Gaussian)
# ---------------------------------------------------------------------------

class TestUnimodal:
    """N(0, 1) has no valleys → falls back to uniform."""

    def test_gaussian_no_valleys(self, rng):
        """Single Gaussian has no valleys → strategy='uniform'."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.normal(0, 1, 2000)
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=2, rng=rng)
        assert strategy == "uniform"

    def test_unimodal_returns_requested_cuts(self, rng):
        """Uniform fallback should return the requested number of cuts."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.normal(0, 1, 2000)
        cuts, _ = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert len(cuts) == 3
        # Cuts should be within data range
        assert np.all((data.min() <= cuts) | np.isclose(cuts, data.min()))
        assert np.all((cuts <= data.max()) | np.isclose(cuts, data.max()))


# ---------------------------------------------------------------------------
# T7: Two unique values
# ---------------------------------------------------------------------------

class TestTwoUniqueValues:
    """Only two distinct values → strategy='quantile', cut at midpoint."""

    def test_only_two_unique_values(self, rng):
        """Should use quantile strategy and cut at midpoint (5.0)."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([np.zeros(500), np.full(500, 10.0)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert strategy == "quantile"
        assert np.isclose(cuts[0], 5.0)

    def test_two_unique_values_unbalanced(self, rng):
        """Unequal counts should still cut at midpoint."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([np.zeros(100), np.full(900, 10.0)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert strategy == "quantile"
        assert np.isclose(cuts[0], 5.0)

    def test_two_unique_values_with_more_cuts(self, rng):
        """Requesting more than 1 cut with 2 unique values returns all at midpoint."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([np.zeros(500), np.full(500, 10.0)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert strategy == "quantile"
        assert len(cuts) == 3
        assert np.allclose(cuts, 5.0)


# ---------------------------------------------------------------------------
# T8: Five modes, need 3 cuts
# ---------------------------------------------------------------------------

class TestMultiModeSelectDeepestValleys:
    """5 peaks, 4 valleys → select 3 deepest valleys, strategy='kde'."""

    def test_selects_deepest_valleys_when_fewer_cuts_than_valleys(self, rng):
        """When requesting 3 cuts from 4 valleys, pick 3 deepest."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-8, 0.4, 500),
            rng.normal(-4, 0.4, 500),
            rng.normal(0, 0.4, 500),
            rng.normal(4, 0.4, 500),
            rng.normal(8, 0.4, 500),
        ])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert strategy == "kde"
        assert len(cuts) == 3
        # All cuts should be in the valley regions between modes
        assert -7.0 < cuts[0] < -5.5 or -7.0 < cuts[0] < 0
        assert cuts[0] < cuts[1] < cuts[2]

    def test_five_modes_all_four_valleys(self, rng):
        """Requesting 4 cuts from 4 valleys → use all valleys."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-8, 0.4, 500),
            rng.normal(-4, 0.4, 500),
            rng.normal(0, 0.4, 500),
            rng.normal(4, 0.4, 500),
            rng.normal(8, 0.4, 500),
        ])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=4, rng=rng)
        assert strategy == "kde"
        assert len(cuts) == 4
        assert cuts[0] < cuts[1] < cuts[2] < cuts[3]

    def test_five_modes_cuts_within_data_range(self, rng):
        """All cuts must lie within the data range."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([
            rng.normal(-8, 0.4, 500),
            rng.normal(-4, 0.4, 500),
            rng.normal(0, 0.4, 500),
            rng.normal(4, 0.4, 500),
            rng.normal(8, 0.4, 500),
        ])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert data.min() < cuts[0]
        assert cuts[-1] < data.max()


# ---------------------------------------------------------------------------
# T10: Deficit valleys
# ---------------------------------------------------------------------------

class TestDeficitValleys:
    """Bimodal with n_cuts=3 (only 1 valley) → strategy='kde+uniform'."""

    def test_kde_plus_uniform_strategy(self, rng):
        """When valleys < n_cuts, use kde+uniform strategy."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        assert strategy == "kde+uniform"
        assert len(cuts) == 3

    def test_kde_plus_uniform_includes_valley(self, rng):
        """kde+uniform should include the real valley among the cuts."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        # At least one cut should be near 0
        assert any(-1.0 < c < 1.0 for c in cuts)

    def test_kde_plus_uniform_cuts_sorted(self, rng):
        """kde+uniform should return sorted cuts."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=3, rng=rng)
        np.testing.assert_array_equal(cuts, np.sort(cuts))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same seed → same result."""

    def test_same_seed_same_result(self):
        """Same RNG seed should produce identical cuts and strategy."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        # Deterministic data generation
        data = np.concatenate([
            np.random.default_rng(123).normal(-3, 0.5, 500),
            np.random.default_rng(123).normal(3, 0.5, 500),
        ])
        rng1 = np.random.default_rng(42)
        cuts1, s1 = kde_peaks_cut_points(data, n_cuts=1, rng=rng1)
        rng2 = np.random.default_rng(42)
        cuts2, s2 = kde_peaks_cut_points(data, n_cuts=1, rng=rng2)
        np.testing.assert_array_equal(cuts1, cuts2)
        assert s1 == s2

    def test_different_seed_may_differ(self):
        """Different RNG seeds may produce different results (especially for uniform fallback)."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.random.default_rng(123).uniform(0, 10, 1000)
        rng1 = np.random.default_rng(1)
        cuts1, _ = kde_peaks_cut_points(data, n_cuts=2, rng=rng1)
        rng2 = np.random.default_rng(2)
        cuts2, _ = kde_peaks_cut_points(data, n_cuts=2, rng=rng2)
        # Cuts may differ with different seeds; at minimum they should both be valid
        assert len(cuts1) == 2
        assert len(cuts2) == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Edge cases and invalid inputs."""

    def test_n_cuts_zero_returns_empty(self, rng):
        """n_cuts=0 should return empty array and some valid strategy."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.normal(0, 1, 500)
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=0, rng=rng)
        assert len(cuts) == 0
        assert isinstance(strategy, str)

    def test_empty_data_raises(self, rng):
        """Empty array should raise ValueError."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        with pytest.raises(ValueError):
            kde_peaks_cut_points(np.array([]), n_cuts=1, rng=rng)

    def test_single_element(self, rng):
        """Single element should still work (returns uniform at that value)."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        cuts, strategy = kde_peaks_cut_points(np.array([5.0]), n_cuts=1, rng=rng)
        assert len(cuts) == 1
        assert np.isclose(cuts[0], 5.0)

    def test_all_nan_raises(self, rng):
        """All NaN data should raise ValueError."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        with pytest.raises(ValueError):
            kde_peaks_cut_points(np.full(100, np.nan), n_cuts=1, rng=rng)

    def test_data_with_nans_finite_only(self, rng):
        """NaN values should be ignored (filtered out) in KDE computation."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        clean = rng.normal(0, 1, 500)
        data = np.concatenate([clean, np.full(50, np.nan)])
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert isinstance(cuts, np.ndarray)
        assert len(cuts) >= 0

    def test_negative_n_cuts_raises(self, rng):
        """Negative n_cuts should raise ValueError."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        with pytest.raises(ValueError):
            kde_peaks_cut_points(rng.normal(0, 1, 100), n_cuts=-1, rng=rng)


# ---------------------------------------------------------------------------
# kde_params option
# ---------------------------------------------------------------------------

class TestKdeParams:
    """Passing kde_params should affect KDE behavior."""

    def test_kde_params_bandwidth(self, rng):
        """Should accept bandwidth parameter in kde_params."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
        cuts, strategy = kde_peaks_cut_points(
            data, n_cuts=1, rng=rng, kde_params={"bandwidth": 0.3}
        )
        assert strategy == "kde"
        assert len(cuts) == 1
        assert -1.0 < cuts[0] < 1.0

    def test_kde_params_grid_resolution(self, rng):
        """Should accept grid_resolution parameter."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
        cuts, strategy = kde_peaks_cut_points(
            data, n_cuts=1, rng=rng, kde_params={"grid_resolution": 256}
        )
        assert strategy == "kde"
        assert len(cuts) == 1

    def test_kde_params_none_default(self, rng):
        """kde_params=None should use defaults."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
        cuts, strategy = kde_peaks_cut_points(
            data, n_cuts=1, rng=rng, kde_params=None
        )
        assert strategy == "kde"
        assert len(cuts) == 1


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class TestReturnTypes:
    """Correct return types from kde_peaks_cut_points."""

    def test_returns_numpy_array(self, rng):
        """Cuts should be a numpy array."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert isinstance(cuts, np.ndarray)

    def test_returns_float64_cuts(self, rng):
        """Cuts should be float64 dtype."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
        cuts, _ = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert cuts.dtype == np.float64

    def test_strategy_is_string(self, rng):
        """Strategy should be a string."""
        from forest_clustering.kde_cuts import kde_peaks_cut_points
        data = rng.normal(0, 1, 500)
        cuts, strategy = kde_peaks_cut_points(data, n_cuts=1, rng=rng)
        assert isinstance(strategy, str)
        assert strategy in ("kde", "kde+uniform", "uniform", "quantile")
