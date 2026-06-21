import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from forest_clustering import (
    ClusterDiagnosticsReport,
    StabilityAnalyzer,
    compare_clusterings,
    ForestClusterer,
    AutoTreeClusterer,
)


def _data():
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=0.35, random_state=0)
    return pd.DataFrame(X, columns=["x", "y"]), y


def test_report_summary_health_cards_and_profiles():
    X, _ = _data()
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    report = ClusterDiagnosticsReport(model, X, labels=model.labels_, random_state=0)
    summary = report.summary()
    assert summary.loc[0, "n_samples"] == 90
    assert summary.loc[0, "n_clusters"] == 3
    assert "silhouette" in summary.columns
    checks = report.health_checks()
    assert isinstance(checks, list) and checks
    cards = report.cluster_cards()
    assert len(cards) == 3
    assert "Cluster" in cards[0]
    profiles = report.cluster_profiles()
    assert set(["cluster", "size", "share"]).issubset(profiles.columns)


def test_uncertain_samples_and_proximity_summary():
    X, _ = _data()
    model = ForestClusterer(n_iterations=20, n_bins="auto", n_clusters=3, random_state=0).fit(X)
    report = ClusterDiagnosticsReport(model, X, random_state=0)
    uncertain = report.uncertain_samples(top_n=5)
    assert len(uncertain) == 5
    assert {"index", "cluster", "silhouette"}.issubset(uncertain.columns)
    block = report.proximity_block_summary()
    assert len(block) == 3
    assert "within_similarity_mean" in block.columns


def test_plots_return_matplotlib_objects():
    X, _ = _data()
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    report = ClusterDiagnosticsReport(model, X, labels=model.labels_, random_state=0)
    assert report.plot_cluster_sizes() is not None
    assert report.plot_embedding() is not None
    assert report.plot_silhouette() is not None
    assert report.plot_cluster_profiles() is not None
    assert report.plot_proximity_heatmap() is not None
    assert report.plot_uncertainty() is not None
    fig = report.plot_overview()
    assert fig is not None
    plt.close("all")


def test_stability_analyzer_and_comparison():
    X, _ = _data()
    estimator = KMeans(n_clusters=3, n_init=5, random_state=0)
    stability = StabilityAnalyzer(estimator, n_runs=3, random_state=0).fit(X)
    assert not stability.pairwise_scores_.empty
    assert stability.summary().loc[0, "mean_ari"] > 0.9
    ax = stability.plot_score_distribution()
    assert ax is not None
    comparison = compare_clusterings(X, {"kmeans": estimator, "forest": ForestClusterer(n_iterations=10, n_clusters=3, random_state=0)}, random_state=0)
    assert set(comparison.results_["model"]) == {"kmeans", "forest"}
    assert comparison.agreement_matrix_.shape == (2, 2)
    assert comparison.plot_scores() is not None
    assert comparison.plot_pairwise_agreement() is not None
    plt.close("all")


def test_autotree_search_plots():
    X, _ = _data()
    auto = AutoTreeClusterer(
        algorithms=("binary_tree",),
        k_range=(2, 3),
        n_restarts=1,
        scoring="silhouette",
        random_state=0,
    ).fit(X)
    assert auto.plot_search_results() is not None
    assert auto.plot_k_selection() is not None
    assert auto.plot_parameter_sensitivity("missing") is not None
    plt.close("all")

def test_compare_clusterings_encodes_mixed_data_for_sklearn_baseline():
    X, _ = _data()
    X = X.copy()
    X["category"] = ["a" if i < 45 else "b" for i in range(len(X))]
    comparison = compare_clusterings(X, {"kmeans": KMeans(n_clusters=3, n_init=5, random_state=0)}, random_state=0)
    assert comparison.results_.loc[0, "input_space"] == "encoded_fallback"
    assert comparison.results_.loc[0, "n_clusters"] == 3
