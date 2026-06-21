"""Diagnostics, visualisation and reporting helpers for clustering results.

The tools in this module are deliberately pragmatic.  They do not try to prove
that a clustering is "true"; instead they expose common failure modes: unstable
solutions, dominant clusters, tiny clusters, weak separation, uncertain samples,
and model choices that are hard to explain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
)
from sklearn.utils.validation import check_is_fitted

from ._tree_common import build_tree_preprocessor, to_frame


_NOISE_LABEL = -1


def _as_1d_labels(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    return labels


def _label_key(x):
    try:
        return int(x)
    except Exception:
        return x


def _safe_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _safe_feature_names(X_df: pd.DataFrame, feature_names=None) -> list[str]:
    if feature_names is not None:
        return [str(x) for x in feature_names]
    return [str(c) for c in X_df.columns]


def _n_effective_clusters(labels: np.ndarray, include_noise: bool = False) -> int:
    labs = np.unique(labels)
    if not include_noise:
        labs = labs[labs != _NOISE_LABEL]
    return int(len(labs))


def _can_score(labels: np.ndarray) -> bool:
    n = len(labels)
    k = len(np.unique(labels))
    return n >= 3 and 2 <= k <= n - 1


def _short_name(name: str) -> str:
    for prefix in ("preprocess__num__", "preprocess__cat__", "normalize__", "num__", "cat__", "onehot__"):
        name = name.replace(prefix, "")
    return name


@dataclass
class HealthCheck:
    """One human-readable diagnostic warning or note."""

    level: str
    code: str
    message: str
    value: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "code": self.code, "message": self.message, "value": self.value}


class ClusterDiagnosticsReport:
    """Diagnostic report for an already fitted clustering result.

    Parameters
    ----------
    clusterer : estimator or None
        Fitted sklearn-style clusterer.  If ``labels`` is omitted, the report
        reads ``clusterer.labels_`` or calls ``clusterer.fit_predict(X)``.
    X : array-like
        Original input data.  Mixed-type tabular data are supported.
    labels : array-like or None
        Optional cluster labels.  Supplying labels is useful when diagnostics are
        run on an external clusterer such as sklearn KMeans.
    distance_matrix, similarity_matrix : array-like or None
        Optional precomputed matrices.  If omitted, the report uses the
        clusterer's ``pairwise_distance`` / ``similarity_matrix`` when available,
        or falls back to a robust preprocessed feature space.
    """

    def __init__(
        self,
        clusterer=None,
        X=None,
        labels=None,
        feature_names=None,
        distance_matrix=None,
        similarity_matrix=None,
        random_state: int | None = None,
        add_missing_indicators: bool = True,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = True,
        numeric_string_min_fraction: float = 0.90,
    ):
        if X is None:
            raise ValueError("X must be supplied")
        self.clusterer = clusterer
        self.X = X
        self.X_df_ = to_frame(X)
        self.feature_names_ = _safe_feature_names(self.X_df_, feature_names)
        self.random_state = random_state
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction
        self.distance_matrix = distance_matrix
        self.similarity_matrix_input = similarity_matrix
        if labels is None:
            labels = self._labels_from_clusterer(clusterer, X)
        self.labels_ = _as_1d_labels(labels)
        if len(self.labels_) != len(self.X_df_):
            raise ValueError("labels length must match X rows")
        self._feature_matrix_cache = None
        self._distance_cache = None
        self._similarity_cache = None
        self._silhouette_samples_cache = None

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------
    def _labels_from_clusterer(self, clusterer, X):
        if clusterer is None:
            raise ValueError("Either clusterer with labels_ or explicit labels must be supplied")
        if hasattr(clusterer, "labels_"):
            return getattr(clusterer, "labels_")
        if hasattr(clusterer, "fit_predict"):
            return clusterer.fit_predict(X)
        raise ValueError("clusterer does not expose labels_ or fit_predict")

    def feature_matrix(self):
        """Return leak-safe encoded features used for most diagnostics."""
        if self._feature_matrix_cache is None:
            pre = build_tree_preprocessor(
                self.X_df_,
                add_missing_indicators=self.add_missing_indicators,
                rare_category_min_count=self.rare_category_min_count,
                rare_category_min_freq=self.rare_category_min_freq,
                coerce_numeric_strings=self.coerce_numeric_strings,
                numeric_string_min_fraction=self.numeric_string_min_fraction,
            )
            self.preprocessor_ = pre
            Z = pre.fit_transform(self.X_df_)
            self._feature_matrix_cache = Z
            try:
                self.encoded_feature_names_ = [_short_name(str(x)) for x in pre.get_feature_names_out()]
            except Exception:
                self.encoded_feature_names_ = [f"feature_{i}" for i in range(Z.shape[1])]
        return self._feature_matrix_cache

    def distance_matrix_(self, prefer_clusterer: bool = False):
        """Return an n x n distance matrix.

        By default this uses a leak-safe preprocessed feature space.  Passing
        ``prefer_clusterer=True`` uses the clusterer's own distance when it is
        available, which is appropriate for proximity heatmaps but less safe for
        model-selection metrics.
        """
        if self._distance_cache is not None and not prefer_clusterer:
            return self._distance_cache
        D = None
        if self.distance_matrix is not None:
            D = np.asarray(self.distance_matrix, dtype=float)
        elif prefer_clusterer and self.clusterer is not None and hasattr(self.clusterer, "pairwise_distance"):
            try:
                D = np.asarray(self.clusterer.pairwise_distance(), dtype=float)
            except Exception:
                D = None
        if D is None:
            Z = self.feature_matrix()
            D = pairwise_distances(Z, metric="euclidean")
        if D.shape != (len(self.labels_), len(self.labels_)):
            raise ValueError("distance matrix must be square with shape (n_samples, n_samples)")
        np.fill_diagonal(D, 0.0)
        if not prefer_clusterer:
            self._distance_cache = D
        return D

    def similarity_matrix_(self, prefer_clusterer: bool = True):
        """Return an n x n similarity matrix for proximity diagnostics."""
        if self._similarity_cache is not None and prefer_clusterer:
            return self._similarity_cache
        S = None
        if self.similarity_matrix_input is not None:
            S = np.asarray(self.similarity_matrix_input, dtype=float)
        elif prefer_clusterer and self.clusterer is not None and hasattr(self.clusterer, "similarity_matrix"):
            try:
                S = np.asarray(self.clusterer.similarity_matrix(), dtype=float)
            except Exception:
                S = None
        elif prefer_clusterer and self.clusterer is not None and hasattr(self.clusterer, "proximity_matrix"):
            try:
                S = np.asarray(self.clusterer.proximity_matrix(), dtype=float)
            except Exception:
                S = None
        if S is None:
            D = self.distance_matrix_(prefer_clusterer=False)
            mx = float(np.nanmax(D)) if D.size else 0.0
            S = 1.0 - D / mx if mx > 0 else np.ones_like(D)
        if S.shape != (len(self.labels_), len(self.labels_)):
            raise ValueError("similarity matrix must be square with shape (n_samples, n_samples)")
        S = np.asarray(S, dtype=float)
        np.fill_diagonal(S, 1.0)
        if prefer_clusterer:
            self._similarity_cache = S
        return S

    # ------------------------------------------------------------------
    # Metrics and diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return a one-row table with common clustering diagnostics."""
        labels = self.labels_
        Z = self.feature_matrix()
        row: dict[str, Any] = {
            "n_samples": int(len(labels)),
            "n_clusters": _n_effective_clusters(labels),
            "noise_rate": float(np.mean(labels == _NOISE_LABEL)),
            "min_cluster_size": int(pd.Series(labels[labels != _NOISE_LABEL]).value_counts().min()) if _n_effective_clusters(labels) else 0,
            "max_cluster_share": float(pd.Series(labels).value_counts().max() / max(len(labels), 1)),
        }
        if _can_score(labels):
            try:
                row["silhouette"] = float(silhouette_score(Z, labels))
                row["calinski_harabasz"] = float(calinski_harabasz_score(_safe_dense(Z), labels))
                row["davies_bouldin"] = float(davies_bouldin_score(_safe_dense(Z), labels))
                sil = self.silhouette_samples()
                row["negative_silhouette_rate"] = float(np.mean(sil < 0))
            except Exception:
                row["silhouette"] = np.nan
                row["calinski_harabasz"] = np.nan
                row["davies_bouldin"] = np.nan
                row["negative_silhouette_rate"] = np.nan
        else:
            row["silhouette"] = np.nan
            row["calinski_harabasz"] = np.nan
            row["davies_bouldin"] = np.nan
            row["negative_silhouette_rate"] = np.nan
        try:
            block = self.proximity_block_summary()
            row["within_similarity_mean"] = float(block["within_similarity_mean"].mean())
            row["between_similarity_mean"] = float(block["between_similarity_mean"].mean())
            row["separation_ratio_mean"] = float(block["separation_ratio"].mean())
        except Exception:
            row["within_similarity_mean"] = np.nan
            row["between_similarity_mean"] = np.nan
            row["separation_ratio_mean"] = np.nan
        return pd.DataFrame([row])

    def silhouette_samples(self) -> np.ndarray:
        if self._silhouette_samples_cache is None:
            labels = self.labels_
            if not _can_score(labels):
                self._silhouette_samples_cache = np.full(len(labels), np.nan)
            else:
                try:
                    self._silhouette_samples_cache = silhouette_samples(self.feature_matrix(), labels)
                except Exception:
                    self._silhouette_samples_cache = np.full(len(labels), np.nan)
        return self._silhouette_samples_cache

    def cluster_sizes(self) -> pd.DataFrame:
        s = pd.Series(self.labels_).value_counts().sort_index()
        return pd.DataFrame({"cluster": [_label_key(x) for x in s.index], "size": s.values, "share": s.values / max(len(self.labels_), 1)})

    def cluster_profiles(self, max_categorical_levels: int = 3) -> pd.DataFrame:
        """Return cluster-level feature summaries and contrasts."""
        X = self.X_df_
        labels = self.labels_
        rows = []
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        global_num = {c: pd.to_numeric(X[c], errors="coerce").mean() for c in numeric_cols}
        global_cat = {c: X[c].astype("object").where(X[c].notna(), "<missing>").value_counts(normalize=True) for c in categorical_cols}
        sil = self.silhouette_samples()
        for lab in np.unique(labels):
            mask = labels == lab
            part = X.loc[mask]
            row: dict[str, Any] = {"cluster": _label_key(lab), "size": int(mask.sum()), "share": float(mask.mean())}
            if np.isfinite(sil).any():
                row["silhouette_mean"] = float(np.nanmean(sil[mask]))
            contrasts = []
            for c in numeric_cols:
                vals = pd.to_numeric(part[c], errors="coerce")
                mean = vals.mean()
                glob = global_num[c]
                row[f"{c}__mean"] = float(mean) if pd.notna(mean) else np.nan
                if pd.notna(mean) and pd.notna(glob):
                    diff = float(mean - glob)
                    scale = float(pd.to_numeric(X[c], errors="coerce").std()) or 1.0
                    effect = diff / scale
                    row[f"{c}__effect_size"] = effect
                    contrasts.append((abs(effect), f"{c}: mean {mean:.3g} ({diff:+.3g} vs global)"))
            for c in categorical_cols:
                vals = part[c].astype("object").where(part[c].notna(), "<missing>")
                top = vals.value_counts(normalize=True, dropna=False).head(max_categorical_levels)
                pieces = []
                for val, frac in top.items():
                    glob_frac = float(global_cat[c].get(val, 0.0))
                    lift = float(frac / glob_frac) if glob_frac > 0 else np.inf
                    pieces.append(f"{val}: {100*frac:.1f}% ({lift:.2g}x)")
                row[f"{c}__top"] = "; ".join(pieces)
            row["highlights"] = " | ".join(t for _, t in sorted(contrasts, reverse=True)[:5])
            rows.append(row)
        return pd.DataFrame(rows)

    def proximity_block_summary(self) -> pd.DataFrame:
        """Summarise within-cluster and between-cluster similarity blocks."""
        S = self.similarity_matrix_(prefer_clusterer=True)
        labels = self.labels_
        rows = []
        labs = np.unique(labels)
        for lab in labs:
            mask = labels == lab
            other = ~mask
            within = S[np.ix_(mask, mask)]
            if within.size > mask.sum():
                # Exclude diagonal when possible.
                within_vals = within[~np.eye(within.shape[0], dtype=bool)]
            else:
                within_vals = within.ravel()
            between_vals = S[np.ix_(mask, other)].ravel() if other.any() else np.array([])
            w = float(np.nanmean(within_vals)) if within_vals.size else 1.0
            b = float(np.nanmean(between_vals)) if between_vals.size else np.nan
            rows.append({
                "cluster": _label_key(lab),
                "within_similarity_mean": w,
                "between_similarity_mean": b,
                "separation_ratio": float(w / b) if b and np.isfinite(b) and b > 0 else np.inf,
            })
        return pd.DataFrame(rows)

    def uncertain_samples(self, top_n: int | None = 20) -> pd.DataFrame:
        """Return samples with low silhouette or weak proximity margin."""
        labels = self.labels_
        sil = self.silhouette_samples()
        rows = []
        try:
            S = self.similarity_matrix_(prefer_clusterer=True)
            labs = np.unique(labels)
            affinities = np.zeros((len(labels), len(labs)), dtype=float)
            for j, lab in enumerate(labs):
                mask = labels == lab
                affinities[:, j] = np.nanmean(S[:, mask], axis=1)
            for i in range(len(labels)):
                own_idx = int(np.where(labs == labels[i])[0][0])
                own = float(affinities[i, own_idx])
                others = np.delete(affinities[i], own_idx)
                second = float(np.max(others)) if others.size else np.nan
                margin = own - second if np.isfinite(second) else np.nan
                rows.append({"index": int(i), "cluster": _label_key(labels[i]), "silhouette": float(sil[i]) if np.isfinite(sil[i]) else np.nan, "own_affinity": own, "second_affinity": second, "margin": margin})
        except Exception:
            for i in range(len(labels)):
                rows.append({"index": int(i), "cluster": _label_key(labels[i]), "silhouette": float(sil[i]) if np.isfinite(sil[i]) else np.nan, "own_affinity": np.nan, "second_affinity": np.nan, "margin": np.nan})
        df = pd.DataFrame(rows)
        sort_cols = [c for c in ["silhouette", "margin"] if c in df]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True, na_position="last")
        return df.head(int(top_n)).reset_index(drop=True) if top_n is not None else df.reset_index(drop=True)

    def health_checks(
        self,
        max_cluster_share_warn: float = 0.80,
        min_cluster_size_warn: int = 5,
        negative_silhouette_rate_warn: float = 0.20,
        low_separation_ratio_warn: float = 1.20,
    ) -> list[dict[str, Any]]:
        """Return diagnostic warnings and notes as dictionaries."""
        checks: list[HealthCheck] = []
        summary = self.summary().iloc[0]
        n_clusters = int(summary["n_clusters"])
        if n_clusters < 2:
            checks.append(HealthCheck("error", "single_cluster", "Only one non-noise cluster was found; most cluster diagnostics are not meaningful.", n_clusters))
        if float(summary["max_cluster_share"]) >= max_cluster_share_warn:
            checks.append(HealthCheck("warning", "dominant_cluster", f"One cluster contains {100*float(summary['max_cluster_share']):.1f}% of samples.", float(summary["max_cluster_share"])))
        if int(summary["min_cluster_size"]) < int(min_cluster_size_warn) and n_clusters > 1:
            checks.append(HealthCheck("warning", "tiny_cluster", f"At least one cluster has fewer than {min_cluster_size_warn} samples.", int(summary["min_cluster_size"])))
        neg = summary.get("negative_silhouette_rate", np.nan)
        if pd.notna(neg) and float(neg) >= negative_silhouette_rate_warn:
            checks.append(HealthCheck("warning", "many_negative_silhouettes", f"{100*float(neg):.1f}% of samples have negative silhouette.", float(neg)))
        sep = summary.get("separation_ratio_mean", np.nan)
        if pd.notna(sep) and np.isfinite(sep) and float(sep) < low_separation_ratio_warn:
            checks.append(HealthCheck("warning", "weak_proximity_separation", "Within-cluster similarity is only slightly higher than between-cluster similarity.", float(sep)))
        if not checks:
            checks.append(HealthCheck("ok", "no_major_warnings", "No major diagnostic warning triggered.", None))
        return [c.as_dict() for c in checks]

    def cluster_cards(self, top_n: int = 4) -> list[str]:
        """Return human-readable cards for each cluster."""
        profiles = self.cluster_profiles()
        block = None
        try:
            block = self.proximity_block_summary().set_index("cluster")
        except Exception:
            pass
        cards = []
        for _, row in profiles.iterrows():
            cluster = row["cluster"]
            lines = [f"Cluster {cluster}", f"Size: {int(row['size'])} samples ({100*float(row['share']):.1f}%)."]
            if pd.notna(row.get("silhouette_mean", np.nan)):
                lines.append(f"Mean silhouette: {float(row['silhouette_mean']):.3f}.")
            if block is not None and cluster in block.index:
                b = block.loc[cluster]
                lines.append(f"Proximity separation ratio: {float(b['separation_ratio']):.3g}.")
            highlights = str(row.get("highlights", ""))
            parts = [p.strip() for p in highlights.split("|") if p.strip()]
            if parts:
                lines.append("Most distinctive numeric traits:")
                lines.extend([f"- {p}" for p in parts[:top_n]])
            cat_cols = [c for c in profiles.columns if c.endswith("__top")]
            cats = []
            for c in cat_cols:
                val = row.get(c, "")
                if isinstance(val, str) and val:
                    cats.append(f"- {c[:-5]}: {val}")
            if cats:
                lines.append("Most frequent categorical traits:")
                lines.extend(cats[:top_n])
            cards.append("\n".join(lines))
        return cards

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_cluster_sizes(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        df = self.cluster_sizes()
        ax.bar(df["cluster"].astype(str), df["size"])
        ax.set_title("Cluster sizes")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Samples")
        return ax

    def embedding(self, method: str = "pca") -> np.ndarray:
        """Return a 2D embedding for visualisation."""
        method = method.lower()
        Z = self.feature_matrix()
        if Z.shape[1] < 2:
            arr = _safe_dense(Z).reshape(Z.shape[0], -1)
            return np.column_stack([arr[:, 0], np.zeros(Z.shape[0])])
        if method == "pca":
            if sparse.issparse(Z):
                return TruncatedSVD(n_components=2, random_state=self.random_state).fit_transform(Z)
            return PCA(n_components=2, random_state=self.random_state).fit_transform(np.asarray(Z))
        if method == "svd":
            return TruncatedSVD(n_components=2, random_state=self.random_state).fit_transform(Z)
        raise ValueError("method must be 'pca' or 'svd'; optional UMAP is intentionally not a hard dependency")

    def plot_embedding(self, method: str = "pca", color_by: str = "cluster", ax=None, annotate_centroids: bool = True):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        coords = self.embedding(method=method)
        if color_by == "cluster":
            vals = self.labels_
            for lab in np.unique(vals):
                mask = vals == lab
                ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.80, label=f"cluster {lab}")
                if annotate_centroids and mask.any():
                    cx, cy = coords[mask].mean(axis=0)
                    ax.text(cx, cy, str(lab), ha="center", va="center", fontsize=10, fontweight="bold")
            ax.legend(loc="best")
        elif color_by == "silhouette":
            vals = self.silhouette_samples()
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=30, alpha=0.85)
            ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            raise ValueError("color_by must be 'cluster' or 'silhouette'")
        ax.set_title(f"Cluster embedding ({method.upper()})")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        return ax

    def plot_silhouette(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        sil = self.silhouette_samples()
        labels = self.labels_
        if not np.isfinite(sil).any():
            ax.text(0.5, 0.5, "Silhouette is not defined", ha="center", va="center")
            ax.set_axis_off()
            return ax
        y_lower = 0
        yticks = []
        for lab in np.unique(labels):
            vals = np.sort(sil[labels == lab])
            y_upper = y_lower + len(vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.6)
            yticks.append((y_lower + y_upper) / 2)
            y_lower = y_upper + 5
        ax.axvline(float(np.nanmean(sil)), linestyle="--")
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(x) for x in np.unique(labels)])
        ax.set_xlabel("Silhouette")
        ax.set_ylabel("Cluster")
        ax.set_title("Per-sample silhouette by cluster")
        return ax

    def plot_cluster_profiles(self, top_n: int = 12, ax=None):
        """Plot strongest numeric effect sizes by cluster."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
        prof = self.cluster_profiles()
        rows = []
        for _, row in prof.iterrows():
            for col in prof.columns:
                if col.endswith("__effect_size"):
                    rows.append({"cluster": row["cluster"], "feature": col[:-13], "effect": row[col]})
        df = pd.DataFrame(rows).dropna()
        if df.empty:
            ax.text(0.5, 0.5, "No numeric profile effects available", ha="center", va="center")
            ax.set_axis_off()
            return ax
        df["abs_effect"] = df["effect"].abs()
        df = df.sort_values("abs_effect", ascending=False).head(int(top_n))
        labels = [f"C{r.cluster}: {r.feature}" for r in df.itertuples()]
        y = np.arange(len(df))[::-1]
        ax.barh(y, df["effect"].values[::-1])
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Effect size vs global mean")
        ax.set_title("Most distinctive numeric cluster traits")
        return ax

    def plot_proximity_heatmap(self, ax=None, order_by: str = "cluster", max_samples: int = 500):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        S = self.similarity_matrix_(prefer_clusterer=True)
        labels = self.labels_
        n = len(labels)
        idx = np.arange(n)
        if n > int(max_samples):
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(idx, size=int(max_samples), replace=False))
        if order_by == "cluster":
            idx = idx[np.argsort(labels[idx], kind="mergesort")]
        elif order_by != "input":
            raise ValueError("order_by must be 'cluster' or 'input'")
        im = ax.imshow(S[np.ix_(idx, idx)], aspect="auto", vmin=np.nanmin(S), vmax=np.nanmax(S))
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Similarity / proximity heatmap")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Samples")
        return ax

    def plot_uncertainty(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        df = self.uncertain_samples(top_n=None)
        if "margin" in df and df["margin"].notna().any():
            ax.hist(df["margin"].dropna().values, bins=30)
            ax.set_xlabel("Affinity margin")
            ax.set_title("Assignment uncertainty: lower margin means more uncertain")
        else:
            ax.hist(df["silhouette"].dropna().values, bins=30)
            ax.set_xlabel("Silhouette")
            ax.set_title("Assignment uncertainty")
        ax.set_ylabel("Samples")
        return ax

    def plot_overview(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        self.plot_cluster_sizes(ax=axes[0, 0])
        self.plot_embedding(ax=axes[0, 1])
        self.plot_silhouette(ax=axes[1, 0])
        self.plot_uncertainty(ax=axes[1, 1])
        fig.tight_layout()
        return fig


class StabilityAnalyzer(BaseEstimator):
    """Resampling/seed stability diagnostics for a clusterer."""

    def __init__(self, estimator, n_runs: int = 10, sample_fraction: float = 1.0, random_state: int | None = None):
        self.estimator = estimator
        self.n_runs = n_runs
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def fit(self, X, y=None):
        X_df = to_frame(X)
        rng = np.random.default_rng(self.random_state)
        labels_list = []
        rows = []
        indices_list = []
        n = len(X_df)
        for r in range(int(self.n_runs)):
            est = clone(self.estimator)
            seed = int(rng.integers(0, 2**31 - 1))
            if hasattr(est, "set_params") and "random_state" in est.get_params(deep=False):
                est.set_params(random_state=seed)
            if float(self.sample_fraction) < 1.0:
                m = max(2, int(np.ceil(float(self.sample_fraction) * n)))
                idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
                X_run = X_df.iloc[idx]
            else:
                idx = np.arange(n)
                X_run = X_df
            labels = est.fit_predict(X_run)
            labels_list.append(np.asarray(labels))
            indices_list.append(idx)
            rows.append({"run": r, "random_state": seed, "n_samples": len(idx), "n_clusters": len(np.unique(labels))})
        pairs = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                common, ii, jj = np.intersect1d(indices_list[i], indices_list[j], return_indices=True)
                if len(common) < 2:
                    continue
                pairs.append({
                    "run_i": i,
                    "run_j": j,
                    "n_common": int(len(common)),
                    "ari": float(adjusted_rand_score(labels_list[i][ii], labels_list[j][jj])),
                    "nmi": float(normalized_mutual_info_score(labels_list[i][ii], labels_list[j][jj])),
                })
        self.run_summary_ = pd.DataFrame(rows)
        self.pairwise_scores_ = pd.DataFrame(pairs)
        self.labels_per_run_ = labels_list
        self.indices_per_run_ = indices_list
        return self

    def summary(self) -> pd.DataFrame:
        check_is_fitted(self, "pairwise_scores_")
        if self.pairwise_scores_.empty:
            return pd.DataFrame([{"mean_ari": np.nan, "std_ari": np.nan, "mean_nmi": np.nan, "std_nmi": np.nan}])
        return pd.DataFrame([{
            "mean_ari": float(self.pairwise_scores_["ari"].mean()),
            "std_ari": float(self.pairwise_scores_["ari"].std(ddof=0)),
            "mean_nmi": float(self.pairwise_scores_["nmi"].mean()),
            "std_nmi": float(self.pairwise_scores_["nmi"].std(ddof=0)),
        }])

    def plot_score_distribution(self, metric: str = "ari", ax=None):
        import matplotlib.pyplot as plt
        check_is_fitted(self, "pairwise_scores_")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        if self.pairwise_scores_.empty or metric not in self.pairwise_scores_:
            ax.text(0.5, 0.5, "No pairwise stability scores", ha="center", va="center")
            ax.set_axis_off()
            return ax
        ax.hist(self.pairwise_scores_[metric].dropna().values, bins=20)
        ax.set_title(f"Stability distribution ({metric.upper()})")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Run pairs")
        return ax


class ClusterComparison(BaseEstimator):
    """Comparison table and plots for several clustering models.

    The comparison tries each model on the original data first.  If a standard
    sklearn estimator rejects mixed/string columns, it retries on the same
    leak-safe encoded feature matrix used by ``ClusterDiagnosticsReport``.  This
    keeps comparisons convenient for mixed tabular datasets without forcing all
    forest-clustering estimators to give up their native preprocessing.
    """

    def __init__(self, X, models: Mapping[str, Any], labels: Mapping[str, Any] | None = None, random_state: int | None = None, encode_fallback: bool = True):
        self.X = X
        self.models = dict(models)
        self.labels = dict(labels or {})
        self.random_state = random_state
        self.encode_fallback = encode_fallback

    def _encoded_X(self):
        if not hasattr(self, "encoded_X_"):
            X_df = to_frame(self.X)
            self.comparison_preprocessor_ = build_tree_preprocessor(X_df, add_missing_indicators=True, coerce_numeric_strings=True)
            self.encoded_X_ = self.comparison_preprocessor_.fit_transform(X_df)
        return self.encoded_X_

    def _fit_labels(self, est):
        try:
            labs = est.fit_predict(self.X) if hasattr(est, "fit_predict") else est.fit(self.X).labels_
            return labs, "original"
        except Exception as exc:
            if not self.encode_fallback:
                raise
            Z = self._encoded_X()
            try:
                labs = est.fit_predict(Z) if hasattr(est, "fit_predict") else est.fit(Z).labels_
                return labs, "encoded_fallback"
            except Exception:
                raise exc

    def fit(self):
        rows = []
        labels_by_name = {}
        fitted = {}
        for name, model in self.models.items():
            input_space = "provided_labels"
            if name in self.labels:
                labs = _as_1d_labels(self.labels[name])
                fitted[name] = model
            else:
                est = clone(model)
                labs, input_space = self._fit_labels(est)
                fitted[name] = est
            labels_by_name[name] = labs
            report = ClusterDiagnosticsReport(fitted[name] if fitted[name] is not None else None, self.X, labels=labs, random_state=self.random_state)
            s = report.summary().iloc[0].to_dict()
            rows.append({"model": name, "input_space": input_space, **s})
        self.fitted_models_ = fitted
        self.labels_by_name_ = labels_by_name
        self.results_ = pd.DataFrame(rows)
        names = list(labels_by_name)
        A = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j <= i:
                    continue
                score = adjusted_rand_score(labels_by_name[a], labels_by_name[b])
                A.loc[a, b] = A.loc[b, a] = float(score)
        self.agreement_matrix_ = A
        return self

    def rank(self, by: str = "silhouette") -> pd.DataFrame:
        check_is_fitted(self, "results_")
        if by not in self.results_.columns:
            raise ValueError(f"Unknown ranking column {by!r}")
        ascending = by in {"davies_bouldin", "negative_silhouette_rate"}
        return self.results_.sort_values(by, ascending=ascending).reset_index(drop=True)

    def plot_scores(self, metric: str = "silhouette", ax=None):
        import matplotlib.pyplot as plt
        check_is_fitted(self, "results_")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        df = self.results_.sort_values(metric, ascending=True)
        ax.barh(df["model"], df[metric])
        ax.set_title(f"Model comparison: {metric}")
        ax.set_xlabel(metric)
        return ax

    def plot_pairwise_agreement(self, ax=None):
        import matplotlib.pyplot as plt
        check_is_fitted(self, "agreement_matrix_")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(self.agreement_matrix_.values, vmin=0, vmax=1, aspect="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(self.agreement_matrix_.columns)))
        ax.set_yticks(np.arange(len(self.agreement_matrix_.index)))
        ax.set_xticklabels(self.agreement_matrix_.columns, rotation=45, ha="right")
        ax.set_yticklabels(self.agreement_matrix_.index)
        ax.set_title("Pairwise clustering agreement (ARI)")
        return ax


def compare_clusterings(X, models: Mapping[str, Any], labels: Mapping[str, Any] | None = None, random_state: int | None = None, encode_fallback: bool = True) -> ClusterComparison:
    """Fit and compare multiple clustering models.

    If ``encode_fallback=True`` (default), sklearn estimators that cannot fit
    raw mixed-type data are retried on a robust encoded feature matrix.
    """
    return ClusterComparison(X, models=models, labels=labels, random_state=random_state, encode_fallback=encode_fallback).fit()
