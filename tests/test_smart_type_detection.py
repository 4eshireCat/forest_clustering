"""Tests for smart categorical type detection."""

import numpy as np
import pandas as pd
import pytest


class TestSmartDetection:
    """Test detect_feature_types with strategy='smart'."""

    def test_integer_encoded_categorical(self):
        """country_code [1,2,3,4,5]*100 → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'country_code': [1, 2, 3, 4, 5] * 100})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['country_code'] == 'categorical'

    def test_high_cardinality_id(self):
        """user_id [1001..1100] → numerical (ID, not categorical)."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'user_id': list(range(1001, 1101))})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['user_id'] == 'numerical'

    def test_binary_feature(self):
        """is_active [0,1]*500 → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'is_active': [0, 1] * 500})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['is_active'] == 'categorical'

    def test_few_unique_floats(self):
        """rating [1.0,2.0,3.0,4.0,5.0]*100 → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'rating': [1.0, 2.0, 3.0, 4.0, 5.0] * 100})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['rating'] == 'categorical'

    def test_continuous_float(self):
        """temperature [20.1, 20.2, ...] → numerical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'temperature': np.linspace(20.0, 30.0, 1000)})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['temperature'] == 'numerical'

    def test_object_column(self):
        """name ['Alice','Bob',...] → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Diana'] * 250})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['name'] == 'categorical'

    def test_known_override(self):
        """known_types should override auto-detection."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'a': [1, 2, 3] * 100})
        types = DataEncoder.detect_feature_types(df, strategy='smart',
                                                  known_types={'a': 'numerical'})
        assert types['a'] == 'numerical'

    def test_naive_vs_smart(self):
        """naive and smart should agree on obvious cases but differ on edge cases."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({
            'name': ['x', 'y'] * 50,        # object → both categorical
            'code': [1, 2, 3] * 33 + [1],    # int → naive:numerical, smart:categorical
        })
        naive = DataEncoder.detect_feature_types(df, strategy='naive')
        smart = DataEncoder.detect_feature_types(df, strategy='smart')

        assert naive['name'] == 'categorical'
        assert smart['name'] == 'categorical'
        # code: naive should be numerical, smart should be categorical
        assert naive['code'] == 'numerical'
        assert smart['code'] == 'categorical'

    def test_all_nan_column(self):
        """All-NaN column → categorical (default)."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'empty': [np.nan] * 100})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['empty'] == 'categorical'

    def test_constant_column(self):
        """Constant column → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'const': [5] * 100})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['const'] == 'categorical'

    def test_boolean_column(self):
        """Boolean column → categorical."""
        from forest_clustering.feature_encoder import DataEncoder
        df = pd.DataFrame({'flag': [True, False] * 50})
        types = DataEncoder.detect_feature_types(df, strategy='smart')
        assert types['flag'] == 'categorical'


class TestClustererIntegration:
    """Test ForestClusterer with smart detection."""

    def test_clusterer_auto_detect_smart(self):
        """Clusterer with auto_feature_types='smart'."""
        from forest_clustering import ForestClusterer
        df = pd.DataFrame({
            'code': [1, 2, 3] * 50,           # int → should be categorical
            'value': np.random.default_rng(42).random(150),  # float → numerical
        })
        fc = ForestClusterer(n_iterations=10, random_state=42, auto_feature_types='smart')
        labels = fc.fit_predict(df)
        assert isinstance(labels, np.ndarray)


class TestInputValidation:
    def test_invalid_strategy_raises(self):
        from forest_clustering.feature_encoder import DataEncoder
        with pytest.raises(ValueError):
            DataEncoder.detect_feature_types(pd.DataFrame({'a': [1]}), strategy='bogus')
