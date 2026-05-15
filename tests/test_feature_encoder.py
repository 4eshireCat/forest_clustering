import numpy as np
import pandas as pd
import pytest
from forest_clustering.feature_encoder import DataEncoder


def test_numerical_array():
    X = np.random.randn(50, 3)
    enc = DataEncoder()
    X_enc = enc.fit_transform(X)
    assert X_enc.shape == (50, 3)
    assert all(t == "numerical" for t in enc.feature_types_)


def test_categorical_detection_low_cardinality():
    X = np.column_stack([np.random.randn(50), np.random.randint(0, 5, 50)])
    enc = DataEncoder(cat_threshold=10)
    enc.fit_transform(X)
    assert enc.feature_types_ == ["numerical", "categorical"]


def test_dataframe_mixed_types():
    df = pd.DataFrame(
        {
            "num": np.random.randn(40),
            "cat": np.random.choice(["a", "b", "c"], 40),
            "int_cat": np.random.randint(0, 4, 40),
        }
    )
    enc = DataEncoder()
    X_enc = enc.fit_transform(df)
    assert X_enc.shape == (40, 3)
    assert enc.feature_types_[0] == "numerical"
    assert enc.feature_types_[1] == "categorical"
    assert enc.feature_types_[2] == "categorical"


def test_transform_unknown_categories():
    df_train = pd.DataFrame({"cat": ["a", "b", "c", "a"]})
    df_test = pd.DataFrame({"cat": ["a", "z", "b"]})  # 'z' is unknown
    enc = DataEncoder()
    enc.fit_transform(df_train)
    X_test = enc.transform(df_test)
    assert X_test[1, 0] == -1.0  # unknown → -1


def test_roundtrip_numerical():
    # Use 50 rows so each column has 50 unique values > cat_threshold=10
    X = np.random.randn(50, 5)
    enc = DataEncoder()
    X_enc = enc.fit_transform(X)
    np.testing.assert_allclose(X_enc, X)


def test_feature_types_override():
    X = np.column_stack([np.random.randn(50), np.random.randn(50)])
    enc = DataEncoder(feature_types_override={1: "categorical"})
    enc.fit_transform(X)
    assert enc.feature_types_[0] == "numerical"
    assert enc.feature_types_[1] == "categorical"
