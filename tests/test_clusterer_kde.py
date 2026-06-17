"""Integration tests for density-aware cuts (kde_peaks) in ForestClusterer.

These tests define the expected API and behavior BEFORE any production code exists.
All tests should fail (ImportError / AttributeError) until the feature is implemented.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Return a small synthetic mixed-type DataFrame for clustering."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "num_b": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "cat_a": ["x", "y", "x", "y", "x", "y", "x", "y"],
        "cat_b": ["a", "a", "b", "b", "a", "a", "b", "b"],
    })


@pytest.fixture
def bimodal_df():
    """DataFrame with clear bimodal numerical column for KDE cuts."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "bimodal": np.concatenate([rng.normal(-3, 0.5, 200), rng.normal(3, 0.5, 200)]),
        "uniform": rng.uniform(0, 10, 400),
    })


@pytest.fixture
def mixed_types_df():
    """Larger mixed DataFrame with bimodal numerical and categorical columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num_bimodal": np.concatenate([rng.normal(0, 1, 500), rng.normal(5, 1, 500)]),
        "num_normal": rng.normal(0, 1, 1000),
        "cat": rng.choice(["a", "b", "c"], size=1000),
    })


# ---------------------------------------------------------------------------
# Parameter acceptance
# ---------------------------------------------------------------------------

class TestClustererAcceptsCutStrategy:
    """ForestClusterer must accept the cut_strategy parameter."""

    def test_clusterer_accepts_cut_strategy(self):
        """ForestClusterer should accept cut_strategy='kde_peaks'."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="kde_peaks")
        assert c.cut_strategy == "kde_peaks"

    def test_clusterer_default_cut_strategy(self):
        """Default cut_strategy should be 'uniform'."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer()
        assert c.cut_strategy == "uniform"

    def test_accepts_uniform(self):
        """Should accept 'uniform' cut_strategy."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="uniform")
        assert c.cut_strategy == "uniform"

    def test_accepts_quantile(self):
        """Should accept 'quantile' cut_strategy."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="quantile")
        assert c.cut_strategy == "quantile"

    def test_accepts_kde_peaks(self):
        """Should accept 'kde_peaks' cut_strategy."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="kde_peaks")
        assert c.cut_strategy == "kde_peaks"

    def test_parameter_in_get_params(self):
        """cut_strategy must appear in get_params()."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="kde_peaks")
        params = c.get_params()
        assert "cut_strategy" in params
        assert params["cut_strategy"] == "kde_peaks"


# ---------------------------------------------------------------------------
# KDE params acceptance
# ---------------------------------------------------------------------------

class TestClustererAcceptsKdeParams:
    """ForestClusterer must accept and store kde_params."""

    def test_kde_params_passed_through(self):
        """kde_params should be stored and appear in get_params."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            cut_strategy="kde_peaks",
            kde_params={"grid_resolution": 256, "bandwidth": 0.5},
        )
        assert c.kde_params == {"grid_resolution": 256, "bandwidth": 0.5}

    def test_kde_params_in_get_params(self):
        """kde_params must appear in get_params()."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            cut_strategy="kde_peaks",
            kde_params={"grid_resolution": 256},
        )
        params = c.get_params()
        assert "kde_params" in params
        assert params["kde_params"] == {"grid_resolution": 256}

    def test_kde_params_default_none(self):
        """Default kde_params should be None."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer()
        assert c.kde_params is None

    def test_kde_params_with_uniform_strategy(self):
        """kde_params can be passed even with uniform strategy (ignored)."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="uniform", kde_params={"bandwidth": 0.5})
        assert c.kde_params == {"bandwidth": 0.5}


# ---------------------------------------------------------------------------
# Invalid strategy
# ---------------------------------------------------------------------------

class TestInvalidCutStrategy:
    """Invalid cut_strategy must raise ValueError."""

    def test_invalid_cut_strategy_raises(self):
        """Invalid cut_strategy at fit time should raise ValueError."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="bogus")
        with pytest.raises(ValueError):
            c.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))

    def test_empty_string_raises(self):
        """Empty string cut_strategy should raise ValueError."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(cut_strategy="")
        with pytest.raises(ValueError):
            c.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


# ---------------------------------------------------------------------------
# fit_predict with cut strategies
# ---------------------------------------------------------------------------

class TestFitPredictWithCutStrategies:
    """fit_predict must succeed with all cut strategies."""

    def test_kde_peaks_runs_fit_predict(self, sample_data):
        """ForestClusterer with kde_peaks should run fit_predict without error."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        labels = c.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_uniform_runs_fit_predict(self, sample_data):
        """ForestClusterer with uniform (default) should run fit_predict."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="uniform",
            random_state=42,
        )
        labels = c.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_quantile_runs_fit_predict(self, sample_data):
        """ForestClusterer with quantile should run fit_predict."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="quantile",
            random_state=42,
        )
        labels = c.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_kde_peaks_with_kde_params_runs(self, sample_data):
        """kde_peaks with custom kde_params should run fit_predict."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            kde_params={"bandwidth": 0.5, "grid_resolution": 128},
            random_state=42,
        )
        labels = c.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_kde_peaks_with_bimodal_data(self, bimodal_df):
        """kde_peaks on bimodal data should produce valid labels."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=20,
            n_bins=4,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        labels = c.fit_predict(bimodal_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(bimodal_df)

    def test_kde_peaks_with_mixed_data(self, mixed_types_df):
        """kde_peaks on mixed-type DataFrame should run successfully."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            n_bins=4,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        labels = c.fit_predict(mixed_types_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(mixed_types_df)


# ---------------------------------------------------------------------------
# Strategy comparison: cuts should differ
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    """Different cut strategies should potentially produce different cuts."""

    def test_kde_vs_uniform_produces_different_cuts(self, bimodal_df):
        """kde_peaks and uniform should produce different edges for bimodal data."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(
            n_iterations=10,
            n_bins=4,
            cut_strategy="uniform",
            random_state=42,
        )
        c2 = ForestClusterer(
            n_iterations=10,
            n_bins=4,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c1.fit(bimodal_df)
        c2.fit(bimodal_df)
        # specs should differ in their numerical edge placement
        assert not np.array_equal(
            c1.specs_[0].bin_specs[0].edges,
            c2.specs_[0].bin_specs[0].edges,
        )

    def test_quantile_vs_uniform_produces_different_cuts(self, bimodal_df):
        """quantile and uniform should potentially produce different edges."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(
            n_iterations=10,
            n_bins=4,
            cut_strategy="uniform",
            random_state=42,
        )
        c2 = ForestClusterer(
            n_iterations=10,
            n_bins=4,
            cut_strategy="quantile",
            random_state=42,
        )
        c1.fit(bimodal_df)
        c2.fit(bimodal_df)
        # At least one iteration's numerical edges should differ
        all_same = True
        for s1, s2 in zip(c1.specs_, c2.specs_):
            for bs1, bs2 in zip(s1.bin_specs, s2.bin_specs):
                if bs1.type == "numerical" and bs2.type == "numerical":
                    if not np.array_equal(bs1.edges, bs2.edges):
                        all_same = False
                        break
        assert not all_same

    def test_all_strategies_run_on_same_data(self, sample_data):
        """All strategies should produce valid embeddings of the same shape."""
        from forest_clustering import ForestClusterer
        for strategy in ["uniform", "quantile", "kde_peaks"]:
            c = ForestClusterer(
                n_iterations=10,
                cut_strategy=strategy,
                random_state=42,
            )
            c.fit(sample_data)
            assert c.embedding_.shape == (len(sample_data), 10)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Default behavior must match pre-density-aware behavior."""

    def test_default_strategy_is_uniform(self):
        """Without specifying cut_strategy, behavior should be uniform."""
        from forest_clustering import ForestClusterer
        c1 = ForestClusterer(n_iterations=10, random_state=42)
        c2 = ForestClusterer(
            n_iterations=10,
            cut_strategy="uniform",
            random_state=42,
        )
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        c1.fit(df)
        c2.fit(df)
        np.testing.assert_array_equal(c1.embedding_, c2.embedding_)

    def test_default_no_kde_params(self):
        """Default should have kde_params=None."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer()
        assert not hasattr(c, "kde_params") or c.kde_params is None

    def test_old_init_signature_still_works(self):
        """Creating clusterer without new kwargs should work."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=200,
            n_features="sqrt",
            n_bins=3,
            random_state=42,
        )
        assert c.n_iterations == 200
        assert c.n_bins == 3


# ---------------------------------------------------------------------------
# specs_ attribute inspection
# ---------------------------------------------------------------------------

class TestSpecsAttribute:
    """Inspect the specs_ attribute to verify cut_strategy effects."""

    def test_specs_exist_after_fit(self, sample_data):
        """specs_ should exist after fit with kde_peaks."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c.fit(sample_data)
        assert hasattr(c, "specs_")
        assert len(c.specs_) == 10

    def test_bin_specs_have_edges(self, sample_data):
        """Each numerical bin_spec should have edges array."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c.fit(sample_data)
        for spec in c.specs_:
            for bs in spec.bin_specs:
                if bs.type == "numerical":
                    assert bs.edges is not None
                    assert isinstance(bs.edges, np.ndarray)

    def test_bin_spec_edges_sorted(self, sample_data):
        """Edges in each numerical bin_spec should be sorted."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c.fit(sample_data)
        for spec in c.specs_:
            for bs in spec.bin_specs:
                if bs.type == "numerical":
                    np.testing.assert_array_equal(bs.edges, np.sort(bs.edges))


# ---------------------------------------------------------------------------
# transform() works with kde_peaks
# ---------------------------------------------------------------------------

class TestTransformWithKdePeaks:
    """transform() on new data must work with kde_peaks strategy."""

    def test_transform_returns_embedding(self, sample_data):
        """transform() should return (n_new, L) int64 embedding."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c.fit(sample_data)
        new_data = sample_data.iloc[:3]
        E_new = c.transform(new_data)
        assert E_new.shape == (3, 10)
        assert E_new.dtype == np.int64

    def test_transform_consistent_with_embedding(self, sample_data):
        """transform(train_data) should equal the fitted embedding."""
        from forest_clustering import ForestClusterer
        c = ForestClusterer(
            n_iterations=10,
            cut_strategy="kde_peaks",
            random_state=42,
        )
        c.fit(sample_data)
        E_via_transform = c.transform(sample_data)
        E_via_attr = c.get_embedding()
        np.testing.assert_array_equal(E_via_transform, E_via_attr)
