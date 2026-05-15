import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from forest_clustering import ForestClusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blobs(n=300, centers=3, seed=0):
    X, y = make_blobs(n_samples=n, centers=centers, random_state=seed, cluster_std=0.8)
    return X, y


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X, _ = blobs()
    clf = ForestClusterer(n_iterations=20, random_state=0)
    assert clf.fit(X) is clf


def test_fit_predict_shape():
    X, _ = blobs()
    clf = ForestClusterer(n_iterations=30, random_state=0)
    labels = clf.fit_predict(X)
    assert labels.shape == (len(X),)


def test_embedding_shape():
    X, _ = blobs(n=100)
    clf = ForestClusterer(n_iterations=50, random_state=0)
    clf.fit(X)
    E = clf.get_embedding()
    assert E.shape == (100, 50)
    assert E.dtype == np.int64


def test_transform_new_data():
    X, _ = blobs(n=200)
    clf = ForestClusterer(n_iterations=40, random_state=1)
    clf.fit(X)
    X_new = np.random.randn(30, X.shape[1])
    E_new = clf.transform(X_new)
    assert E_new.shape == (30, 40)


def test_pairwise_distance_shape():
    X, _ = blobs(n=50)
    clf = ForestClusterer(n_iterations=20, random_state=2)
    clf.fit(X)
    D = clf.pairwise_distance()
    assert D.shape == (50, 50)
    assert D.dtype == np.float32
    # Diagonal must be 0
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)
    # Must be symmetric
    np.testing.assert_allclose(D, D.T, atol=1e-6)
    # Values in [0, 1]
    assert D.min() >= 0.0
    assert D.max() <= 1.0 + 1e-6


def test_pairwise_distance_cross():
    X, _ = blobs(n=60)
    clf = ForestClusterer(n_iterations=20, random_state=3)
    clf.fit(X)
    D = clf.pairwise_distance(X=X[:20], Y=X[20:40])
    assert D.shape == (20, 20)


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def test_separates_well_separated_blobs():
    from sklearn.cluster import AgglomerativeClustering
    X, y = blobs(n=300, centers=3)
    inner = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
    clf = ForestClusterer(n_iterations=150, n_bins=3, clusterer=inner, random_state=42)
    labels = clf.fit_predict(X)
    ari = adjusted_rand_score(y, labels)
    assert ari > 0.7, f"ARI too low: {ari:.3f}"


def test_agglomerative_precomputed():
    X, y = blobs(n=100, centers=3)
    inner = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
    clf = ForestClusterer(n_iterations=80, clusterer=inner, random_state=5)
    labels = clf.fit_predict(X)
    assert len(np.unique(labels)) == 3


def test_kmeans_on_embedding():
    X, y = blobs(n=150, centers=3)
    inner = KMeans(n_clusters=3, random_state=0, n_init=5)
    clf = ForestClusterer(n_iterations=80, clusterer=inner, random_state=6)
    labels = clf.fit_predict(X)
    assert len(np.unique(labels)) == 3


# ---------------------------------------------------------------------------
# Mixed / categorical data
# ---------------------------------------------------------------------------

def test_dataframe_with_categorical():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {
            "num1": rng.standard_normal(120),
            "num2": rng.standard_normal(120),
            "cat": rng.choice(["A", "B", "C", "D"], 120),
        }
    )
    clf = ForestClusterer(n_iterations=30, random_state=0)
    labels = clf.fit_predict(df)
    assert labels.shape == (120,)


def test_all_categorical():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "c1": rng.choice(["x", "y", "z"], 80),
            "c2": rng.choice([1, 2, 3], 80),
        }
    )
    clf = ForestClusterer(n_iterations=20, random_state=0)
    labels = clf.fit_predict(df)
    assert labels.shape == (80,)


# ---------------------------------------------------------------------------
# Correlation weighting
# ---------------------------------------------------------------------------

def test_corr_weighting_does_not_crash():
    rng = np.random.default_rng(20)
    base = rng.standard_normal(200)
    X = np.column_stack([base, base + rng.standard_normal(200) * 0.01, rng.standard_normal(200)])
    clf = ForestClusterer(n_iterations=20, corr_threshold=0.9, corr_sample_size=200, random_state=0)
    clf.fit(X)
    w = clf.feature_weights_
    assert w[0] == pytest.approx(w[1], abs=1e-6)
    assert w[0] < 1.0


def test_corr_disabled():
    X, _ = blobs(n=100)
    clf = ForestClusterer(n_iterations=20, corr_threshold=None, random_state=0)
    clf.fit(X)
    np.testing.assert_allclose(clf.feature_weights_, 1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_with_seed():
    X, _ = blobs(n=100)
    clf1 = ForestClusterer(n_iterations=30, random_state=42)
    clf2 = ForestClusterer(n_iterations=30, random_state=42)
    E1 = clf1.fit(X).get_embedding()
    E2 = clf2.fit(X).get_embedding()
    np.testing.assert_array_equal(E1, E2)


def test_different_seeds_differ():
    X, _ = blobs(n=100)
    clf1 = ForestClusterer(n_iterations=30, random_state=0)
    clf2 = ForestClusterer(n_iterations=30, random_state=1)
    E1 = clf1.fit(X).get_embedding()
    E2 = clf2.fit(X).get_embedding()
    assert not np.array_equal(E1, E2)


# ---------------------------------------------------------------------------
# n_features variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_features", ["sqrt", "log2", 1, 3, 0.5])
def test_n_features_variants(n_features):
    X, _ = blobs(n=80)
    clf = ForestClusterer(n_iterations=10, n_features=n_features, random_state=0)
    clf.fit(X)
    assert clf.embedding_.shape[0] == 80


# ---------------------------------------------------------------------------
# quantile_cuts
# ---------------------------------------------------------------------------

def test_quantile_cuts():
    X = np.random.exponential(2, (200, 5))
    clf = ForestClusterer(n_iterations=20, quantile_cuts=True, random_state=0)
    labels = clf.fit_predict(X)
    assert labels.shape == (200,)
