"""Inductive cluster-label assignment and human-readable explanations.

This module adds a supervised layer on top of unsupervised clusterers.  The
clusterer still defines the segmentation; the classifier learns to reproduce
those cluster labels so that users can assign new samples, inspect fidelity and
extract explanations.  Metrics reported by these tools are *fidelity to cluster
labels*, not external clustering quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils.validation import check_is_fitted

from ._tree_common import build_tree_preprocessor, to_frame
from .auto import AutoTreeClusterer


@dataclass
class ClusterRule:
    """A compact rule object used by ``ClusterSurrogateTree.extract_leaf_rules``."""

    cluster: Any
    rule: str
    samples: int
    purity: float


def _safe_array(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _as_labels(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("cluster labels must be a 1D array")
    return labels


def _label_counts(labels) -> dict:
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v) if isinstance(v, np.integer) else v: int(c) for v, c in zip(vals, counts)}


def _effective_cv(labels, cv) -> int:
    if cv is None or int(cv) <= 1:
        return 0
    labels = _as_labels(labels)
    _, counts = np.unique(labels, return_counts=True)
    if counts.size == 0:
        return 0
    return max(0, min(int(cv), int(counts.min())))


def _fidelity_metrics(y_true, y_pred, labels=None) -> dict[str, Any]:
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def _make_default_classifier(random_state=None):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=None,
    )


def _feature_names_from_preprocessor(preprocessor, n_features: int) -> list[str]:
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return [f"feature_{i}" for i in range(n_features)]


def _short_feature_name(name: str) -> str:
    for prefix in ("preprocess__num__", "preprocess__cat__", "num__", "cat__", "normalize__"):
        name = name.replace(prefix, "")
    name = name.replace("onehot__", "")
    return name




def _format_condition(feature: str, op: str, threshold: float) -> str:
    if abs(float(threshold) - 0.5) <= 1e-9 and op in {"<=", ">"}:
        # Most one-hot encoded categorical rules split at 0.5.  Phrase those as
        # presence/absence instead of exposing the numeric dummy variable.
        return f"not {feature}" if op == "<=" else f"{feature}"
    return f"{feature} {op} {threshold:.3g}"

def _top_feature_importances(estimator, feature_names: list[str], top_n: int) -> pd.DataFrame:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None and hasattr(estimator, "estimator"):
        importances = getattr(estimator.estimator, "feature_importances_", None)
    if importances is None and hasattr(estimator, "base_estimator"):
        importances = getattr(estimator.base_estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    importances = np.asarray(importances, dtype=float)
    n = min(len(importances), len(feature_names))
    df = pd.DataFrame({
        "feature": [_short_feature_name(f) for f in feature_names[:n]],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False)
    return df.head(int(top_n)).reset_index(drop=True)


class _OriginalFeatureMixin:
    """Shared preprocessing, profiling and plotting helpers."""

    def _fit_original_features(self, X):
        X_df = to_frame(X)
        self.input_columns_ = list(X_df.columns)
        self.preprocessor_ = build_tree_preprocessor(
            X_df,
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        )
        X_feat = self.preprocessor_.fit_transform(X_df)
        self.feature_names_ = _feature_names_from_preprocessor(self.preprocessor_, X_feat.shape[1])
        return X_df, X_feat

    def _transform_original_features(self, X):
        check_is_fitted(self, "preprocessor_")
        X_df = to_frame(X)
        if list(X_df.columns) != list(self.input_columns_):
            raise ValueError(f"Column names do not match fit() columns. Expected {self.input_columns_}, got {list(X_df.columns)}")
        return self.preprocessor_.transform(X_df)

    def cluster_profile(self, X=None, labels=None, max_categorical_levels: int = 3) -> pd.DataFrame:
        """Summarise each cluster in a compact human-readable table.

        Numeric columns show cluster means and global contrast.  Categorical
        columns show the most frequent values inside the cluster.  This is a
        diagnostic/explanation view, not a statistical test.
        """
        check_is_fitted(self, "labels_")
        if X is None:
            X_df = self.X_original_
        else:
            X_df = to_frame(X)
        labels = self.labels_ if labels is None else _as_labels(labels)
        if len(X_df) != len(labels):
            raise ValueError("X and labels must have the same number of rows")

        rows = []
        n = len(labels)
        numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
        global_means = {c: pd.to_numeric(X_df[c], errors="coerce").mean() for c in numeric_cols}
        for cluster in np.unique(labels):
            mask = labels == cluster
            part = X_df.loc[mask]
            row: dict[str, Any] = {
                "cluster": int(cluster) if isinstance(cluster, np.integer) else cluster,
                "size": int(mask.sum()),
                "share": float(mask.mean()),
            }
            contrasts = []
            for c in numeric_cols:
                mean = pd.to_numeric(part[c], errors="coerce").mean()
                row[f"{c}__mean"] = float(mean) if pd.notna(mean) else np.nan
                glob = global_means[c]
                if pd.notna(mean) and pd.notna(glob):
                    diff = mean - glob
                    contrasts.append((abs(diff), f"{c}: mean {mean:.3g} ({diff:+.3g} vs global)"))
            for c in categorical_cols:
                vals = part[c].astype("object").where(part[c].notna(), "<missing>")
                top = vals.value_counts(dropna=False).head(max_categorical_levels)
                row[f"{c}__top"] = "; ".join(f"{idx}={cnt}" for idx, cnt in top.items())
            contrasts = sorted(contrasts, reverse=True)[:5]
            row["numeric_highlights"] = " | ".join(text for _, text in contrasts)
            rows.append(row)
        return pd.DataFrame(rows)

    def explain_clusters(self, X=None, labels=None, top_n_numeric: int = 3, max_categorical_levels: int = 2) -> str:
        """Return plain-language descriptions of clusters."""
        profile = self.cluster_profile(X=X, labels=labels, max_categorical_levels=max_categorical_levels)
        lines = []
        for _, row in profile.iterrows():
            lines.append(f"Cluster {row['cluster']} ({int(row['size'])} samples, {100 * float(row['share']):.1f}%):")
            if row.get("numeric_highlights"):
                highlights = [h.strip() for h in str(row["numeric_highlights"]).split("|") if h.strip()]
                for h in highlights[:top_n_numeric]:
                    lines.append(f"  - {h}")
            cat_cols = [c for c in profile.columns if c.endswith("__top")]
            for c in cat_cols[:max_categorical_levels]:
                val = row.get(c, "")
                if isinstance(val, str) and val:
                    lines.append(f"  - frequent {c[:-5]} values: {val}")
        return "\n".join(lines)

    def plot_cluster_sizes(self, ax=None):
        """Plot cluster counts as a bar chart."""
        check_is_fitted(self, "labels_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        counts = pd.Series(self.labels_).value_counts().sort_index()
        ax.bar([str(x) for x in counts.index], counts.values)
        ax.set_title("Cluster sizes")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Samples")
        return ax

    def plot_embedding(self, X=None, labels=None, ax=None, title: str = "Cluster assignment projection"):
        """Plot a 2D PCA/SVD projection of the classifier feature space."""
        check_is_fitted(self, "labels_")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA, TruncatedSVD

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        X_feat = self._transform_original_features(self.X_original_ if X is None else X)
        labels = self.labels_ if labels is None else _as_labels(labels)
        if X_feat.shape[1] < 2:
            coords = np.column_stack([_safe_array(X_feat).ravel(), np.zeros(X_feat.shape[0])])
        elif sparse.issparse(X_feat):
            coords = TruncatedSVD(n_components=2, random_state=getattr(self, "random_state", None)).fit_transform(X_feat)
        else:
            coords = PCA(n_components=2, random_state=getattr(self, "random_state", None)).fit_transform(np.asarray(X_feat))
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.80, label=f"cluster {lab}")
        ax.set_title(title)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.legend(loc="best")
        return ax

    def plot_feature_importances(self, top_n: int = 20, ax=None):
        """Plot surrogate classifier feature importances when available."""
        check_is_fitted(self, "classifier_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        df = self.feature_importances_dataframe(top_n=top_n)
        if df.empty:
            ax.text(0.5, 0.5, "Classifier has no feature_importances_", ha="center", va="center")
            ax.set_axis_off()
            return ax
        y = np.arange(len(df))[::-1]
        ax.barh(y, df["importance"].values[::-1])
        ax.set_yticks(y)
        ax.set_yticklabels(df["feature"].values[::-1])
        ax.set_title("Top surrogate feature importances")
        ax.set_xlabel("Importance")
        return ax

    def feature_importances_dataframe(self, top_n: int = 20) -> pd.DataFrame:
        check_is_fitted(self, "classifier_")
        return _top_feature_importances(self.classifier_, self.feature_names_, top_n=top_n)


class ClusterLabelClassifier(_OriginalFeatureMixin, BaseEstimator, ClassifierMixin):
    """Learn a classifier that reproduces cluster labels.

    The class turns a transductive clustering result into an inductive model with
    ``predict`` / ``predict_proba`` for new rows.  Its metrics measure fidelity
    to the clusterer labels, not ground-truth clustering quality.
    """

    def __init__(
        self,
        clusterer=None,
        classifier=None,
        cv: int = 5,
        calibrate: bool = False,
        confidence_threshold: float = 0.0,
        unknown_policy: str = "force",
        random_state: int | None = None,
        add_missing_indicators: bool = True,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = True,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.clusterer = clusterer
        self.classifier = classifier
        self.cv = cv
        self.calibrate = calibrate
        self.confidence_threshold = confidence_threshold
        self.unknown_policy = unknown_policy
        self.random_state = random_state
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction

    def fit(self, X, y=None):
        if self.unknown_policy not in {"force", "reject"}:
            raise ValueError("unknown_policy must be 'force' or 'reject'")
        if float(self.confidence_threshold) < 0 or float(self.confidence_threshold) > 1:
            raise ValueError("confidence_threshold must be in [0, 1]")

        self.X_original_, X_feat = self._fit_original_features(X)
        self.n_features_in_ = self.X_original_.shape[1]
        self.feature_names_in_ = np.asarray(self.X_original_.columns, dtype=object)

        if y is None:
            base_clusterer = self.clusterer if self.clusterer is not None else AutoTreeClusterer(random_state=self.random_state)
            self.clusterer_ = clone(base_clusterer)
            labels = self.clusterer_.fit_predict(X)
        else:
            self.clusterer_ = None
            labels = y
        self.labels_ = _as_labels(labels)
        if self.labels_.shape[0] != self.X_original_.shape[0]:
            raise ValueError("cluster labels must have the same length as X")
        self.classes_ = np.unique(self.labels_)
        self.cluster_counts_ = _label_counts(self.labels_)

        base_classifier = clone(self.classifier) if self.classifier is not None else _make_default_classifier(self.random_state)
        self.fidelity_report_ = self._cross_validated_fidelity(base_classifier, X_feat, self.labels_)
        self.classifier_ = self._fit_final_classifier(base_classifier, X_feat, self.labels_)
        train_pred = self.classifier_.predict(X_feat)
        self.train_fidelity_report_ = _fidelity_metrics(self.labels_, train_pred, labels=self.classes_)
        if hasattr(self.classifier_, "predict_proba"):
            proba = self.classifier_.predict_proba(X_feat)
            self.mean_confidence_ = float(np.max(proba, axis=1).mean())
            self.low_confidence_rate_ = float((np.max(proba, axis=1) < float(self.confidence_threshold)).mean())
        else:
            self.mean_confidence_ = np.nan
            self.low_confidence_rate_ = np.nan
        return self

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        X_feat = self._transform_original_features(X)
        labels = np.asarray(self.classifier_.predict(X_feat))
        if self.unknown_policy == "reject" and hasattr(self.classifier_, "predict_proba"):
            proba = self.classifier_.predict_proba(X_feat)
            conf = np.max(proba, axis=1)
            labels = labels.astype(object)
            labels[conf < float(self.confidence_threshold)] = -1
        return labels

    def predict_proba(self, X):
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(f"{type(self.classifier_).__name__} does not implement predict_proba")
        X_feat = self._transform_original_features(X)
        return self.classifier_.predict_proba(X_feat)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def score(self, X, y=None):
        check_is_fitted(self, "fidelity_report_")
        if y is None:
            return float(self.train_fidelity_report_["accuracy"])
        return float(accuracy_score(y, self.predict(X)))

    def fidelity_summary(self) -> pd.DataFrame:
        """Return train and out-of-fold fidelity metrics as a small table."""
        check_is_fitted(self, "fidelity_report_")
        rows = []
        for name, report in (("out_of_fold", self.fidelity_report_), ("train", self.train_fidelity_report_)):
            rows.append({
                "split": name,
                "accuracy": report.get("accuracy", np.nan),
                "balanced_accuracy": report.get("balanced_accuracy", np.nan),
                "f1_macro": report.get("f1_macro", np.nan),
            })
        return pd.DataFrame(rows)

    def plot_fidelity_confusion_matrix(self, ax=None, normalize: bool = False):
        """Plot out-of-fold confusion matrix when CV was possible."""
        check_is_fitted(self, "fidelity_report_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        cm = np.asarray(self.fidelity_report_.get("confusion_matrix"))
        if cm.size == 0:
            ax.text(0.5, 0.5, "No out-of-fold confusion matrix", ha="center", va="center")
            ax.set_axis_off()
            return ax
        cm_plot = cm.astype(float)
        if normalize:
            denom = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm_plot, np.maximum(denom, 1), out=np.zeros_like(cm_plot), where=denom > 0)
        im = ax.imshow(cm_plot, aspect="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        labels = [str(x) for x in self.fidelity_report_.get("labels", self.classes_)]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted cluster")
        ax.set_ylabel("Cluster label")
        ax.set_title("Out-of-fold cluster-label fidelity")
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                val = cm_plot[i, j]
                text = f"{val:.2f}" if normalize else str(int(val))
                ax.text(j, i, text, ha="center", va="center")
        return ax

    def _fit_final_classifier(self, base_classifier, X_feat, labels):
        if not self.calibrate:
            clf = clone(base_classifier).fit(X_feat, labels)
            return clf
        cv_eff = _effective_cv(labels, min(3, int(self.cv) if self.cv else 3))
        if cv_eff < 2:
            return clone(base_classifier).fit(X_feat, labels)
        try:
            calibrated = CalibratedClassifierCV(estimator=clone(base_classifier), cv=cv_eff)
        except TypeError:  # sklearn < 1.2
            calibrated = CalibratedClassifierCV(base_estimator=clone(base_classifier), cv=cv_eff)
        return calibrated.fit(X_feat, labels)

    def _cross_validated_fidelity(self, base_classifier, X_feat, labels) -> dict[str, Any]:
        cv_eff = _effective_cv(labels, self.cv)
        if cv_eff < 2:
            pred = clone(base_classifier).fit(X_feat, labels).predict(X_feat)
            out = _fidelity_metrics(labels, pred, labels=np.unique(labels))
            out["cv"] = 0
            out["note"] = "CV skipped because at least one cluster is too small. Metrics are in-sample."
            return out
        preds = np.empty(labels.shape, dtype=labels.dtype)
        skf = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            clf = clone(base_classifier)
            clf.fit(X_feat[train_idx], labels[train_idx])
            preds[test_idx] = clf.predict(X_feat[test_idx])
        out = _fidelity_metrics(labels, preds, labels=np.unique(labels))
        out["cv"] = cv_eff
        out["note"] = "Out-of-fold fidelity to cluster labels."
        return out


class ClusterSurrogateTree(_OriginalFeatureMixin, BaseEstimator, ClassifierMixin):
    """Fit an interpretable decision tree that explains cluster labels."""

    def __init__(
        self,
        clusterer=None,
        max_depth: int | None = 4,
        min_samples_leaf: int = 10,
        criterion: str = "gini",
        cv: int = 5,
        random_state: int | None = None,
        add_missing_indicators: bool = True,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = True,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.clusterer = clusterer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.cv = cv
        self.random_state = random_state
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction

    def fit(self, X, y=None):
        self.X_original_, X_feat = self._fit_original_features(X)
        self.n_features_in_ = self.X_original_.shape[1]
        self.feature_names_in_ = np.asarray(self.X_original_.columns, dtype=object)
        if y is None:
            base_clusterer = self.clusterer if self.clusterer is not None else AutoTreeClusterer(random_state=self.random_state)
            self.clusterer_ = clone(base_clusterer)
            labels = self.clusterer_.fit_predict(X)
        else:
            self.clusterer_ = None
            labels = y
        self.labels_ = _as_labels(labels)
        self.classes_ = np.unique(self.labels_)
        self.cluster_counts_ = _label_counts(self.labels_)
        self.classifier_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
            class_weight="balanced",
        )
        base = clone(self.classifier_)
        self.fidelity_report_ = self._cross_validated_fidelity(base, X_feat, self.labels_)
        self.classifier_.fit(X_feat, self.labels_)
        train_pred = self.classifier_.predict(X_feat)
        self.train_fidelity_report_ = _fidelity_metrics(self.labels_, train_pred, labels=self.classes_)
        self.rules_ = self.export_text()
        return self

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        X_feat = self._transform_original_features(X)
        return self.classifier_.predict(X_feat)

    def predict_proba(self, X):
        check_is_fitted(self, "classifier_")
        X_feat = self._transform_original_features(X)
        return self.classifier_.predict_proba(X_feat)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def score(self, X, y=None):
        check_is_fitted(self, "train_fidelity_report_")
        if y is None:
            return float(self.train_fidelity_report_["accuracy"])
        return float(accuracy_score(y, self.predict(X)))

    def export_text(self, max_depth: int | None = None) -> str:  # type: ignore[override]
        """Return sklearn-style rules with readable feature names."""
        check_is_fitted(self, "classifier_")
        names = [_short_feature_name(f) for f in self.feature_names_]
        return export_text(self.classifier_, feature_names=names, max_depth=self.classifier_.get_depth() if max_depth is None else max_depth)

    def extract_leaf_rules(self, min_purity: float = 0.0) -> list[ClusterRule]:
        """Extract approximate root-to-leaf rules and assigned clusters.

        The result is intentionally compact; use ``export_text`` for the full
        decision-tree dump.
        """
        check_is_fitted(self, "classifier_")
        tree = self.classifier_.tree_
        names = [_short_feature_name(f) for f in self.feature_names_]
        rules: list[ClusterRule] = []

        def visit(node: int, conditions: list[str]):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                counts = tree.value[node][0]
                total = float(counts.sum())
                if total <= 0:
                    return
                best = int(np.argmax(counts))
                purity = float(counts[best] / total)
                if purity < min_purity:
                    return
                cluster = self.classifier_.classes_[best]
                rule = " and ".join(conditions) if conditions else "all samples"
                rules.append(ClusterRule(cluster=cluster, rule=rule, samples=int(tree.n_node_samples[node]), purity=purity))
                return
            feature = names[tree.feature[node]]
            thr = tree.threshold[node]
            visit(left, conditions + [_format_condition(feature, "<=", thr)])
            visit(right, conditions + [_format_condition(feature, ">", thr)])

        visit(0, [])
        return sorted(rules, key=lambda r: (r.cluster, -r.purity, -r.samples))


    def explain_rules(self, min_purity: float = 0.0, max_rules: int | None = None) -> str:
        """Return compact, human-readable cluster assignment rules."""
        rules = self.extract_leaf_rules(min_purity=min_purity)
        if max_rules is not None:
            rules = rules[:int(max_rules)]
        lines = []
        for rule in rules:
            lines.append(
                f"Cluster {rule.cluster}: if {rule.rule} "
                f"(samples={rule.samples}, purity={rule.purity:.2f})"
            )
        return "\n".join(lines)

    def rules_dataframe(self, min_purity: float = 0.0) -> pd.DataFrame:
        """Return extracted leaf rules as a DataFrame."""
        rules = self.extract_leaf_rules(min_purity=min_purity)
        return pd.DataFrame([r.__dict__ for r in rules])

    def fidelity_summary(self) -> pd.DataFrame:
        check_is_fitted(self, "fidelity_report_")
        rows = []
        for name, report in (("out_of_fold", self.fidelity_report_), ("train", self.train_fidelity_report_)):
            rows.append({
                "split": name,
                "accuracy": report.get("accuracy", np.nan),
                "balanced_accuracy": report.get("balanced_accuracy", np.nan),
                "f1_macro": report.get("f1_macro", np.nan),
            })
        return pd.DataFrame(rows)

    def plot_tree(self, ax=None, max_depth: int | None = None, filled: bool = True, rounded: bool = True, fontsize: int = 8):
        """Plot the surrogate decision tree."""
        check_is_fitted(self, "classifier_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(14, 7))
        names = [_short_feature_name(f) for f in self.feature_names_]
        plot_tree(
            self.classifier_,
            feature_names=names,
            class_names=[str(c) for c in self.classes_],
            max_depth=max_depth,
            filled=filled,
            rounded=rounded,
            fontsize=fontsize,
            ax=ax,
        )
        ax.set_title("Surrogate decision tree for cluster labels")
        return ax

    def _cross_validated_fidelity(self, base_classifier, X_feat, labels) -> dict[str, Any]:
        cv_eff = _effective_cv(labels, self.cv)
        if cv_eff < 2:
            pred = clone(base_classifier).fit(X_feat, labels).predict(X_feat)
            out = _fidelity_metrics(labels, pred, labels=np.unique(labels))
            out["cv"] = 0
            out["note"] = "CV skipped because at least one cluster is too small. Metrics are in-sample."
            return out
        preds = np.empty(labels.shape, dtype=labels.dtype)
        skf = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            clf = clone(base_classifier)
            clf.fit(X_feat[train_idx], labels[train_idx])
            preds[test_idx] = clf.predict(X_feat[test_idx])
        out = _fidelity_metrics(labels, preds, labels=np.unique(labels))
        out["cv"] = cv_eff
        out["note"] = "Out-of-fold fidelity to cluster labels."
        return out


__all__ = [
    "ClusterLabelClassifier",
    "ClusterSurrogateTree",
    "ClusterRule",
]
