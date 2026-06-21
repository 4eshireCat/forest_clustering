import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from forest_clustering import AutoTreeClusterer


def _toy_data(n=36):
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-2.0, scale=0.25, size=(n // 2, 2))
    b = rng.normal(loc=2.0, scale=0.25, size=(n - n // 2, 2))
    X = np.vstack([a, b])
    return pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "group_hint": ["left"] * (n // 2) + ["right"] * (n - n // 2),
    })


def test_auto_tree_clusterer_selects_best_candidate_and_exposes_results():
    X = _toy_data()
    model = AutoTreeClusterer(
        algorithms=("forest", "binary_tree"),
        k_range=(2, 3),
        scoring="silhouette",
        n_restarts=1,
        estimator_params={
            "forest": {"n_iterations": 24, "n_bins": "auto", "n_jobs": 1},
            "binary_tree": {"n_thresholds": 8},
        },
        random_state=7,
    )
    labels = model.fit_predict(X)

    assert labels.shape == (len(X),)
    assert model.best_algorithm_ in {"forest", "binary_tree"}
    assert model.best_n_clusters_ in {2, 3}
    assert np.isfinite(model.best_score_)
    assert len(model.cv_results_) == 4
    for col in ["algorithm", "n_clusters", "mean_score", "rank"]:
        assert col in model.cv_results_.columns
    assert model.labels_.shape == (len(X),)


def test_auto_tree_clusterer_delegates_matrices_and_transform():
    X = _toy_data(24)
    model = AutoTreeClusterer(
        algorithms=("forest",),
        k_range=(2,),
        scoring="silhouette",
        n_restarts=1,
        estimator_params={"forest": {"n_iterations": 12, "n_jobs": 1}},
        random_state=0,
    ).fit(X)

    Z = model.transform(X)
    S = model.similarity_matrix()
    D = model.pairwise_distance()
    assert Z.shape[0] == len(X)
    assert S.shape == (len(X), len(X))
    assert D.shape == (len(X), len(X))
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(np.diag(D), 0.0)


def test_auto_tree_clusterer_combined_scoring_uses_stability():
    X = _toy_data(30)
    model = AutoTreeClusterer(
        algorithms=("urf",),
        k_range=(2,),
        scoring="combined",
        n_restarts=2,
        stability_weight=0.2,
        estimator_params={"urf": {"n_estimators": 8, "n_jobs": 1}},
        random_state=123,
    ).fit(X)
    assert "stability" in model.cv_results_.columns
    assert np.isfinite(model.cv_results_["stability"].iloc[0])
    assert np.isfinite(model.best_score_)


def test_auto_tree_clusterer_is_sklearn_cloneable_and_reproducible():
    X = _toy_data(30)
    model = AutoTreeClusterer(
        algorithms=("extratrees",),
        k_range=(2, 3),
        scoring="silhouette",
        n_restarts=1,
        estimator_params={"extratrees": {"n_estimators": 10, "n_jobs": 1}},
        random_state=42,
    )
    cloned = clone(model)
    labels1 = model.fit_predict(X)
    labels2 = cloned.fit_predict(X)
    assert np.array_equal(labels1, labels2)
    assert cloned.best_params_ == model.best_params_


def test_auto_tree_clusterer_rejects_bad_scoring_and_algorithm():
    X = _toy_data(12)
    with pytest.raises(ValueError, match="scoring"):
        AutoTreeClusterer(scoring="bad", algorithms=("forest",), k_range=(2,), estimator_params={"forest": {"n_iterations": 4, "n_jobs": 1}}).fit(X)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        AutoTreeClusterer(algorithms=("unknown",), k_range=(2,)).fit(X)


def test_auto_tree_binary_tree_scoring_is_not_label_leaky_on_three_blobs():
    """Regression for 0.6.0: binary-tree self-distance made k=2 and k=3 both score 1.0."""
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X_arr, y_true = make_blobs(
        n_samples=90,
        centers=[(-6, 0), (0, 6), (6, 0)],
        cluster_std=0.45,
        random_state=0,
    )
    X = pd.DataFrame(X_arr, columns=["x", "y"])
    model = AutoTreeClusterer(
        algorithms=("binary_tree",),
        k_range=(2, 3),
        scoring="silhouette",
        n_restarts=1,
        estimator_params={"binary_tree": {"n_thresholds": 64}},
        random_state=0,
    ).fit(X)

    assert model.best_n_clusters_ == 3
    assert adjusted_rand_score(y_true, model.labels_) > 0.95
    scores = model.cv_results_.set_index("n_clusters")["mean_silhouette"].to_dict()
    assert scores[3] > scores[2]
    assert scores[3] < 1.0


def test_auto_tree_proximity_scoring_mode_is_explicit_compatibility_mode():
    from sklearn.datasets import make_blobs

    X_arr, _ = make_blobs(
        n_samples=60,
        centers=[(-6, 0), (0, 6), (6, 0)],
        cluster_std=0.45,
        random_state=1,
    )
    X = pd.DataFrame(X_arr, columns=["x", "y"])
    model = AutoTreeClusterer(
        algorithms=("binary_tree",),
        k_range=(2, 3),
        scoring="silhouette",
        scoring_space="proximity",
        n_restarts=1,
        estimator_params={"binary_tree": {"n_thresholds": 64}},
        random_state=0,
    ).fit(X)

    assert "mean_silhouette" in model.cv_results_.columns
    assert np.isclose(model.cv_results_["mean_silhouette"].max(), 1.0)


def test_auto_tree_validates_scoring_space_and_sample_size():
    X = _toy_data(12)
    with pytest.raises(ValueError, match="scoring_space"):
        AutoTreeClusterer(algorithms=("forest",), k_range=(2,), scoring_space="bad").fit(X)
    with pytest.raises(ValueError, match="scoring_sample_size"):
        AutoTreeClusterer(algorithms=("forest",), k_range=(2,), scoring_sample_size=1).fit(X)
