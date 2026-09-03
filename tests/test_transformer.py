"""Tests for forest_clustering.transformer.ForestTransformer.

Defines the expected API and behavior BEFORE production code exists.
ForestTransformer is a sklearn-compatible transformer that produces the (n, L)
embedding matrix.

All tests should fail (ImportError / AttributeError) until production code exists.
"""

import numpy as np
import pandas as pd
import pytest


class TestImport:
    """Module and class importability."""

    def test_module_importable(self):
        """The transformer module can be imported."""
        from forest_clustering import transformer  # noqa: F401

    def test_class_importable(self):
        """ForestTransformer can be imported directly."""
        from forest_clustering.transformer import ForestTransformer  # noqa: F401

    def test_class_available_from_package(self):
        """ForestTransformer is re-exported from the package root."""
        from forest_clustering import ForestTransformer  # noqa: F401


class TestSklearnCompatibility:
    """sklearn BaseEstimator / TransformerMixin interface."""

    def test_inherits_base_estimator(self):
        """ForestTransformer must inherit from sklearn.base.BaseEstimator."""
        from sklearn.base import BaseEstimator
        from forest_clustering.transformer import ForestTransformer

        assert issubclass(ForestTransformer, BaseEstimator)

    def test_inherits_transformer_mixin(self):
        """ForestTransformer must inherit from sklearn.base.TransformerMixin."""
        from sklearn.base import TransformerMixin
        from forest_clustering.transformer import ForestTransformer

        assert issubclass(ForestTransformer, TransformerMixin)

    def test_has_get_params(self):
        """ForestTransformer must expose get_params() (from BaseEstimator)."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer()
        params = t.get_params()
        assert isinstance(params, dict)
        assert "n_iterations" in params

    def test_has_set_params(self):
        """ForestTransformer must expose set_params() (from BaseEstimator)."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer()
        t.set_params(n_iterations=100)
        assert t.get_params()["n_iterations"] == 100

    def test_set_params_returns_self(self):
        """set_params must return self (sklearn convention)."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer()
        result = t.set_params(n_iterations=100)
        assert result is t


class TestInitialization:
    """Constructor parameter defaults and validation."""

    def test_default_parameters(self):
        """Default parameters must match ForestClusterer defaults."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer()
        assert t.n_iterations == 200
        assert t.n_features == "sqrt"
        assert t.n_bins == 3
        assert t.corr_threshold == 0.7
        assert t.corr_sample_size == 10_000
        assert t.feature_types is None
        assert t.cat_threshold == 10
        assert t.quantile_cuts is False
        assert t.n_jobs == -1
        assert t.random_state is None

    def test_custom_parameters(self):
        """Custom parameters must be stored correctly."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(
            n_iterations=100,
            n_features=5,
            n_bins=5,
            corr_threshold=None,
            corr_sample_size=5_000,
            feature_types={"a": "numerical"},
            cat_threshold=5,
            quantile_cuts=True,
            n_jobs=2,
            random_state=42,
        )
        assert t.n_iterations == 100
        assert t.n_features == 5
        assert t.n_bins == 5
        assert t.corr_threshold is None
        assert t.corr_sample_size == 5_000
        assert t.feature_types == {"a": "numerical"}
        assert t.cat_threshold == 5
        assert t.quantile_cuts is True
        assert t.n_jobs == 2
        assert t.random_state == 42


class TestFit:
    """fit() behavior."""

    def test_fit_returns_self(self, sample_data):
        """fit() must return self (sklearn convention)."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        result = t.fit(sample_data)
        assert result is t

    def test_fit_sets_embedding_attribute(self, sample_data):
        """fit() must set the embedding_ attribute."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        assert hasattr(t, "embedding_")

    def test_fit_sets_specs_attribute(self, sample_data):
        """fit() must set the specs_ attribute (needed for transform)."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        assert hasattr(t, "specs_")

    def test_fit_sets_encoder_attribute(self, sample_data):
        """fit() must set the encoder_ attribute."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        assert hasattr(t, "encoder_")

    def test_fit_with_ndarray(self):
        """fit() must work with a plain numpy ndarray."""
        from forest_clustering.transformer import ForestTransformer

        X = np.random.default_rng(42).random((20, 4))
        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(X)
        assert hasattr(t, "embedding_")

    def test_fit_with_pandas_df(self, sample_data):
        """fit() must work with a pandas DataFrame."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        assert hasattr(t, "embedding_")


class TestTransform:
    """transform() behavior."""

    def test_transform_returns_int64_embedding(self, sample_data):
        """transform() must return (n, L) int64 array."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        E = t.transform(sample_data)
        assert isinstance(E, np.ndarray)
        assert E.dtype == np.int64

    def test_transform_shape_matches_n_samples(self, sample_data):
        """transform() output must have n rows (same as input) and L columns."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        E = t.transform(sample_data)
        assert E.shape[0] == len(sample_data)
        assert E.shape[1] == 10  # n_iterations

    def test_transform_on_new_data(self, sample_data):
        """transform() must work on new, unseen data."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)

        new_data = pd.DataFrame({
            "num_a": [1.5, 2.5],
            "num_b": [0.5, 1.5],
            "cat_a": ["x", "y"],
            "cat_b": ["a", "b"],
        })
        E_new = t.transform(new_data)
        assert E_new.shape == (2, 10)

    def test_transform_without_fit_raises(self, sample_data):
        """transform() before fit() must raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        with pytest.raises(NotFittedError):
            t.transform(sample_data)


class TestFitTransform:
    """fit_transform() behavior."""

    def test_fit_transform_equals_fit_then_transform(self, sample_data):
        """fit_transform(X) must equal fit(X).transform(X)."""
        from forest_clustering.transformer import ForestTransformer

        t1 = ForestTransformer(n_iterations=10, random_state=42)
        t2 = ForestTransformer(n_iterations=10, random_state=42)

        E_fit_transform = t1.fit_transform(sample_data)
        E_fit_then_transform = t2.fit(sample_data).transform(sample_data)

        np.testing.assert_array_equal(E_fit_transform, E_fit_then_transform)

    def test_fit_transform_returns_int64(self, sample_data):
        """fit_transform() must return int64 array."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        E = t.fit_transform(sample_data)
        assert E.dtype == np.int64

    def test_fit_transform_shape(self, sample_data):
        """fit_transform() must return (n, L) shape."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        E = t.fit_transform(sample_data)
        assert E.shape == (len(sample_data), 10)


class TestGetEmbedding:
    """get_embedding() convenience method."""

    def test_get_embedding_returns_fitted_embedding(self, sample_data):
        """get_embedding() must return the same array as the embedding_ attribute."""
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        t.fit(sample_data)
        E_via_attr = t.embedding_
        E_via_method = t.get_embedding()
        np.testing.assert_array_equal(E_via_attr, E_via_method)

    def test_get_embedding_before_fit_raises(self):
        """get_embedding() before fit() must raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        from forest_clustering.transformer import ForestTransformer

        t = ForestTransformer(n_iterations=10, random_state=42)
        with pytest.raises(NotFittedError):
            t.get_embedding()


class TestPipelineCompatibility:
    """Compatibility with sklearn.pipeline."""

    def test_compatible_with_make_pipeline(self, sample_data):
        """ForestTransformer must work inside make_pipeline."""
        from sklearn.pipeline import make_pipeline
        from sklearn.cluster import DBSCAN
        from forest_clustering.transformer import ForestTransformer

        pipe = make_pipeline(
            ForestTransformer(n_iterations=10, random_state=42),
            DBSCAN(metric="hamming"),
        )
        labels = pipe.fit_predict(sample_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_data)

    def test_pipeline_transform_step(self, sample_data):
        """The first step in a pipeline must produce an embedding."""
        from sklearn.pipeline import make_pipeline
        from forest_clustering.transformer import ForestTransformer

        pipe = make_pipeline(
            ForestTransformer(n_iterations=10, random_state=42),
        )
        E = pipe.fit_transform(sample_data)
        assert E.dtype == np.int64
        assert E.shape == (len(sample_data), 10)


class TestReproducibility:
    """Random seed reproducibility."""

    def test_same_random_state_same_embedding(self, sample_data):
        """Two transformers with the same random_state must produce identical embeddings."""
        from forest_clustering.transformer import ForestTransformer

        t1 = ForestTransformer(n_iterations=20, random_state=123)
        t2 = ForestTransformer(n_iterations=20, random_state=123)

        E1 = t1.fit_transform(sample_data)
        E2 = t2.fit_transform(sample_data)

        np.testing.assert_array_equal(E1, E2)

    def test_different_random_state_different_embeddings(self, sample_data):
        """Different random_state values should generally produce different embeddings."""
        from forest_clustering.transformer import ForestTransformer

        t1 = ForestTransformer(n_iterations=20, random_state=1)
        t2 = ForestTransformer(n_iterations=20, random_state=2)

        E1 = t1.fit_transform(sample_data)
        E2 = t2.fit_transform(sample_data)

        # They should differ with high probability for 20 iterations
        assert not np.array_equal(E1, E2)


def test_categorical_renaming_does_not_change_correlation_weights():
    """Nominal category codes must never be interpreted as ordered ranks."""
    from forest_clustering.transformer import ForestTransformer

    values = ["a", "b", "c"] * 10
    original = pd.DataFrame({"left": values, "right": values})
    renamed = original.copy()
    renamed["right"] = renamed["right"].map({"a": "z", "b": "x", "c": "y"})

    first = ForestTransformer(n_iterations=4, random_state=0).fit(original)
    second = ForestTransformer(n_iterations=4, random_state=0).fit(renamed)

    np.testing.assert_array_equal(first.feature_weights_, np.ones(2))
    np.testing.assert_array_equal(second.feature_weights_, np.ones(2))


class TestParameterMirroring:
    """ForestTransformer parameters must mirror ForestClusterer parameters."""

    def test_has_n_iterations(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "n_iterations")

    def test_has_n_features(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "n_features")

    def test_has_n_bins(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "n_bins")

    def test_has_corr_threshold(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "corr_threshold")

    def test_has_corr_sample_size(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "corr_sample_size")

    def test_has_feature_types(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "feature_types")

    def test_has_cat_threshold(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "cat_threshold")

    def test_has_quantile_cuts(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "quantile_cuts")

    def test_has_n_jobs(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "n_jobs")

    def test_has_random_state(self):
        from forest_clustering.transformer import ForestTransformer
        t = ForestTransformer()
        assert hasattr(t, "random_state")
