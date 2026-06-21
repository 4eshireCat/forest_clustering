import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from forest_clustering import (
    AutoTreeClusterer,
    ClusterLabelClassifier,
    ClusterSurrogateTree,
    ForestClusterer,
)


def _toy_frame():
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=0.45, random_state=42)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["segment_hint"] = np.where(df["x0"] > df["x0"].median(), "right", "left")
    return df, y


def test_cluster_label_classifier_fits_predicts_and_reports_fidelity():
    X, _ = _toy_frame()
    clusterer = AutoTreeClusterer(
        algorithms=("binary_tree",),
        k_range=(3,),
        n_restarts=1,
        random_state=42,
    )
    clf = ClusterLabelClassifier(clusterer=clusterer, cv=3, random_state=42).fit(X)
    pred = clf.predict(X.iloc[:5])
    proba = clf.predict_proba(X.iloc[:5])

    assert pred.shape == (5,)
    assert proba.shape[0] == 5
    assert set(["accuracy", "balanced_accuracy", "f1_macro"]).issubset(clf.fidelity_report_)
    assert clf.fidelity_summary().shape[0] == 2
    assert "Cluster" in clf.explain_clusters()


def test_cluster_label_classifier_can_reject_low_confidence_predictions():
    X, _ = _toy_frame()
    clf = ClusterLabelClassifier(
        clusterer=ForestClusterer(n_clusters=3, n_iterations=20, random_state=42),
        cv=2,
        classifier=RandomForestClassifier(n_estimators=5, max_depth=1, random_state=42),
        confidence_threshold=0.99,
        unknown_policy="reject",
        random_state=42,
    ).fit(X)
    pred = clf.predict(X.iloc[:10])
    assert any(x == -1 for x in pred)


def test_cluster_surrogate_tree_exports_rules_and_dataframe():
    X, _ = _toy_frame()
    clusterer = ForestClusterer(n_clusters=3, n_iterations=25, random_state=42)
    explainer = ClusterSurrogateTree(
        clusterer=clusterer,
        max_depth=3,
        min_samples_leaf=5,
        cv=3,
        random_state=42,
    ).fit(X)

    rules = explainer.export_text()
    rules_df = explainer.rules_dataframe(min_purity=0.0)
    assert "class:" in rules
    assert {"cluster", "rule", "samples", "purity"}.issubset(rules_df.columns)
    assert explainer.predict(X.iloc[:7]).shape == (7,)


def test_surrogate_can_accept_existing_cluster_labels():
    X, _ = _toy_frame()
    y = np.repeat([0, 1, 2], 30)
    explainer = ClusterSurrogateTree(max_depth=2, min_samples_leaf=3, random_state=42).fit(X, y=y)
    assert explainer.clusterer_ is None
    assert explainer.labels_.shape == (90,)
    assert len(explainer.extract_leaf_rules()) > 0
    assert "Cluster" in explainer.explain_rules()


def test_visualization_methods_return_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, _ = _toy_frame()
    labels = np.repeat([0, 1, 2], 30)
    clf = ClusterLabelClassifier(
        classifier=DecisionTreeClassifier(max_depth=3, random_state=42),
        cv=2,
        random_state=42,
    ).fit(X, y=labels)

    ax1 = clf.plot_cluster_sizes()
    ax2 = clf.plot_embedding()
    ax3 = clf.plot_feature_importances(top_n=5)
    ax4 = clf.plot_fidelity_confusion_matrix()
    assert ax1 is not None and ax2 is not None and ax3 is not None and ax4 is not None
    plt.close("all")
