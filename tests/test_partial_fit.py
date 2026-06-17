"""Tests for partial_fit / online mode."""

import numpy as np
import pandas as pd
import pytest


class TestPartialFitBasic:
    """Basic partial_fit functionality."""

    def test_first_call_like_fit(self, sample_data):
        """First partial_fit call must produce same result as fit."""
        from forest_clustering import ForestClusterer

        fc1 = ForestClusterer(n_iterations=10, random_state=42)
        fc1.fit(sample_data)

        fc2 = ForestClusterer(n_iterations=10, random_state=42)
        fc2.partial_fit(sample_data)

        np.testing.assert_array_equal(fc1.embedding_, fc2.embedding_)

    def test_partial_fit_returns_self(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        result = fc.partial_fit(sample_data)
        assert result is fc

    def test_second_call_does_not_crash(self, sample_data):
        """Two partial_fit calls in a row must work."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.partial_fit(sample_data)
        fc.partial_fit(sample_data)  # should not raise
        assert hasattr(fc, "embedding_")

    def test_embedding_accumulated(self, sample_data):
        """After two partial_fit calls, total samples should be sum."""
        from forest_clustering import ForestClusterer

        n = len(sample_data)
        half = n // 2
        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.partial_fit(sample_data.iloc[:half])
        fc.partial_fit(sample_data.iloc[half:])
        # Total embedding rows should equal n
        assert fc.embedding_.shape[0] == n

    def test_accepts_numpy_array(self):
        from forest_clustering import ForestClusterer

        X = np.random.default_rng(42).random((20, 4))
        fc = ForestClusterer(n_iterations=5, random_state=42)
        fc.partial_fit(X)
        assert fc.embedding_.shape[0] == 20


class TestPartialFitConsistency:
    """Consistency across calls."""

    def test_same_distribution_no_rebuild(self, sample_data):
        """Two calls with same distribution → consistent embeddings."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, random_state=42, partial_fit_strategy="never"
        )
        fc.partial_fit(sample_data)
        emb1 = fc.embedding_.copy()
        fc.partial_fit(sample_data)
        emb2 = fc.embedding_[len(sample_data) :]  # second batch
        np.testing.assert_array_equal(emb1, emb2)

    def test_transform_after_partial_fit(self, sample_data):
        """transform() must work after partial_fit."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.partial_fit(sample_data)
        E = fc.transform(sample_data)
        np.testing.assert_array_equal(fc.embedding_, E)


class TestDriftDetection:
    """Drift detection behavior."""

    def test_drift_report_exists(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.partial_fit(sample_data)
        report = fc.get_drift_report()
        assert isinstance(report, dict)

    def test_drift_detected_on_shifted_data(self, sample_data):
        """Strongly shifted data should trigger drift."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=10, random_state=42, partial_fit_strategy="drift"
        )
        fc.partial_fit(sample_data)

        # Shift data by +100
        shifted = sample_data.copy()
        num_cols = shifted.select_dtypes(include=[np.number]).columns
        shifted[num_cols] = shifted[num_cols] + 100

        fc.partial_fit(shifted)
        report = fc.get_drift_report()
        assert isinstance(report, dict)

    def test_periodic_strategy_rebuilds(self, sample_data):
        """periodic strategy should eventually rebuild."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=5,
            random_state=42,
            partial_fit_strategy="periodic",
            partial_fit_max_samples=5,
        )
        fc.partial_fit(sample_data.iloc[:3])
        fc.partial_fit(sample_data.iloc[:3])  # exceeds max_samples → rebuild
        assert hasattr(fc, "embedding_")

    def test_never_strategy_no_rebuild(self, sample_data):
        """never strategy should not rebuild specs."""
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(
            n_iterations=5, random_state=42, partial_fit_strategy="never"
        )
        fc.partial_fit(sample_data)
        specs_before = fc.specs_
        fc.partial_fit(sample_data)
        # specs should be the same object
        assert fc.specs_ is specs_before


class TestPartialFitBackwardCompat:
    """Backward compatibility."""

    def test_fit_still_works(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        labels = fc.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)

    def test_fit_after_partial_fit_works(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=10, random_state=42)
        fc.partial_fit(sample_data)
        fc.fit(sample_data)  # reset and full fit
        assert hasattr(fc, "embedding_")

    def test_default_strategy_is_drift(self):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer()
        assert fc.partial_fit_strategy == "drift"


class TestPartialFitInvalid:
    """Invalid inputs."""

    def test_invalid_strategy_raises(self, sample_data):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(partial_fit_strategy="bogus")
        with pytest.raises(ValueError):
            fc.partial_fit(sample_data)

    def test_empty_data_raises(self):
        from forest_clustering import ForestClusterer

        fc = ForestClusterer(n_iterations=5, random_state=42)
        with pytest.raises(ValueError):
            fc.partial_fit(pd.DataFrame())
