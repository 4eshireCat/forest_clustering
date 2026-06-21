import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN

from forest_clustering import (
    ForestClusterer,
    UnsupervisedRandomForestClusterer,
    ExtraTreesProximityClusterer,
    UnsupervisedBinaryTreeClusterer,
)
from forest_clustering.feature_encoder import DataEncoder


class RecordingClusterer:
    def __init__(self):
        self.X_seen = None
    def fit_predict(self, X):
        self.X_seen = X
        return np.zeros(X.shape[0], dtype=int)


def test_dbscan_no_longer_auto_retries_unless_opted_in():
    X = pd.DataFrame({"x": [0, 0, 1, 1, 10, 10], "c": list("aaabbb")})
    db = DBSCAN(metric="hamming", eps=1.0)
    model = ForestClusterer(n_iterations=20, clusterer=db, cluster_input="embedding", random_state=0, n_jobs=1)
    labels = model.fit_predict(X)
    # eps=1.0 connects every row in Hamming space; without the historical retry
    # this must remain one cluster instead of silently replacing eps.
    assert len(set(labels)) == 1


def test_cluster_input_distance_and_similarity_are_explicit():
    X = pd.DataFrame({"x": [0.0, 0.1, 4.0, 4.1], "c": ["a", "a", "b", "b"]})

    rec_d = RecordingClusterer()
    ForestClusterer(n_iterations=10, clusterer=rec_d, cluster_input="distance", random_state=0, n_jobs=1).fit(X)
    assert rec_d.X_seen.shape == (4, 4)
    assert np.allclose(np.diag(rec_d.X_seen), 0.0)
    assert np.all(rec_d.X_seen >= 0.0)

    rec_s = RecordingClusterer()
    ForestClusterer(n_iterations=10, clusterer=rec_s, cluster_input="similarity", random_state=0, n_jobs=1).fit(X)
    assert rec_s.X_seen.shape == (4, 4)
    assert np.allclose(np.diag(rec_s.X_seen), 1.0)
    assert np.all((rec_s.X_seen >= 0.0) & (rec_s.X_seen <= 1.0))


def test_cluster_input_onehot_does_not_pass_raw_nominal_ids():
    X = pd.DataFrame({"x": [0.0, 0.1, 4.0, 4.1], "c": ["a", "a", "b", "b"]})
    rec = RecordingClusterer()
    ForestClusterer(n_iterations=12, clusterer=rec, cluster_input="onehot", random_state=0, n_jobs=1).fit(X)
    assert rec.X_seen.shape[0] == len(X)
    assert rec.X_seen.shape[1] >= 12
    assert rec.X_seen.shape[1] != 12 or sparse.issparse(rec.X_seen)


def test_missing_indicators_add_stable_binary_features():
    X = pd.DataFrame({"x": [1.0, np.nan, 2.0], "cat": ["a", "b", None]})
    enc = DataEncoder(add_missing_indicators=True, coerce_numeric_strings=True)
    Xt = enc.fit_transform(X)
    names = [m.name for m in enc.columns_]
    assert "x__missing" in names
    assert "cat__missing" in names
    assert Xt.shape[1] == 4
    Xt2 = enc.transform(X)
    assert np.array_equal(Xt, Xt2)


def test_rare_categories_map_rare_and_unseen_to_same_bucket():
    X = pd.DataFrame({"city": ["Austin"] * 5 + ["Paris", "Rome"]})
    enc = DataEncoder(rare_category_min_count=2)
    Xt = enc.fit_transform(X)
    meta = enc.columns_[0]
    assert meta.rare_value is not None
    rare_code = meta.cat_encoder[meta.rare_value]
    assert Xt[-1, 0] == rare_code
    assert Xt[-2, 0] == rare_code
    X_new = pd.DataFrame({"city": ["Berlin", "Austin"]})
    Xt_new = enc.transform(X_new)
    assert Xt_new[0, 0] == rare_code
    assert Xt_new[1, 0] != rare_code


def test_numeric_string_coercion_treats_numeric_objects_as_numerical():
    X = pd.DataFrame({"amount": ["1.1", "2.2", "3.3", "4.4"], "label": ["a", "a", "b", "b"]})
    enc = DataEncoder(coerce_numeric_strings=True)
    enc.fit_transform(X)
    assert enc.columns_[0].type == "numerical"
    model = ForestClusterer(n_iterations=6, n_clusters=2, coerce_numeric_strings=True, random_state=0, n_jobs=1)
    model.fit(X)
    assert model.encoder_.columns_[0].type == "numerical"


def test_n_bins_auto_resolves_to_valid_integer():
    X = pd.DataFrame({"x": np.linspace(0, 1, 20), "c": ["a", "b"] * 10})
    model = ForestClusterer(n_bins="auto", n_iterations=8, n_clusters=2, random_state=0, n_jobs=1).fit(X)
    assert isinstance(model.n_bins_, int)
    assert 2 <= model.n_bins_ <= model.max_bins
    assert model.embedding_.shape == (20, 8)


def test_tree_estimators_accept_quality_preprocessing_options():
    X = pd.DataFrame({
        "fare": ["7.25", "8.05", None, "71.28", "8.05", "7.25"],
        "cabin": [None, "C85", None, "C123", "rare1", "rare2"],
        "sex": ["male", "female", "female", "female", "male", "male"],
    })
    kwargs = dict(
        n_clusters=2,
        add_missing_indicators=True,
        rare_category_min_count=2,
        coerce_numeric_strings=True,
        random_state=0,
    )
    for cls in [UnsupervisedRandomForestClusterer, ExtraTreesProximityClusterer, UnsupervisedBinaryTreeClusterer]:
        est = cls(n_estimators=10, **kwargs) if cls is not UnsupervisedBinaryTreeClusterer else cls(**kwargs)
        labels = est.fit_predict(X)
        assert labels.shape == (len(X),)
        assert est.proximity_matrix().shape == (len(X), len(X))
