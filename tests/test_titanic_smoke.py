from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forest_clustering import UnsupervisedRandomForestClusterer


TITANIC_CANDIDATES = [
    Path('/opt/pyvenv/lib/python3.13/site-packages/gradio/media_assets/data/titanic.csv'),
]


def _load_local_titanic():
    for path in TITANIC_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    pytest.skip('Local Titanic dataset is not available in this environment')


def test_urf_titanic_smoke():
    df = _load_local_titanic()
    wanted = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cols = [c for c in wanted if c in df.columns]
    X = df[cols].head(300).copy()
    est = UnsupervisedRandomForestClusterer(
        n_estimators=40,
        n_clusters=3,
        random_state=123,
        min_samples_leaf=2,
        n_jobs=1,
    )
    labels = est.fit_predict(X)
    P = est.proximity_matrix()
    assert labels.shape == (len(X),)
    assert est.leaf_embedding_.shape == (len(X), 40)
    assert P.shape == (len(X), len(X))
    assert np.allclose(np.diag(P), 1.0)
    assert len(set(labels)) >= 2

from forest_clustering import ExtraTreesProximityClusterer


def test_extra_trees_titanic_smoke():
    df = _load_local_titanic()
    wanted = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cols = [c for c in wanted if c in df.columns]
    X = df[cols].head(300).copy()
    est = ExtraTreesProximityClusterer(
        n_estimators=40,
        n_clusters=3,
        random_state=123,
        min_samples_leaf=2,
        n_jobs=1,
    )
    labels = est.fit_predict(X)
    P = est.proximity_matrix()
    assert labels.shape == (len(X),)
    assert est.leaf_embedding_.shape == (len(X), 40)
    assert P.shape == (len(X), len(X))
    assert np.allclose(np.diag(P), 1.0)
    assert len(set(labels)) >= 2

from forest_clustering import UnsupervisedBinaryTreeClusterer


def test_binary_tree_titanic_smoke():
    df = _load_local_titanic()
    wanted = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cols = [c for c in wanted if c in df.columns]
    X = df[cols].head(300).copy()
    est = UnsupervisedBinaryTreeClusterer(
        n_clusters=3,
        random_state=123,
        min_samples_leaf=10,
        min_samples_split=20,
        max_depth=5,
        n_thresholds=16,
    )
    labels = est.fit_predict(X)
    P = est.proximity_matrix()
    assert labels.shape == (len(X),)
    assert est.leaf_embedding_.shape == (len(X), 1)
    assert P.shape == (len(X), len(X))
    assert np.allclose(np.diag(P), 1.0)
    assert len(set(labels)) >= 2
    assert len(est.rules()) == est.n_leaves_
