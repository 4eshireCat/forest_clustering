"""Automatic model and parameter selection for tree-based clustering.

``AutoTreeClusterer`` is intentionally small and sklearn-like: it tries a grid of
supported tree-clustering estimators, cluster counts and random restarts, scores
valid clusterings with internal unsupervised criteria and keeps the fitted best
estimator for downstream use.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.utils.validation import check_is_fitted

from .binary_tree import UnsupervisedBinaryTreeClusterer
from .clusterer import ForestClusterer
from .extra_trees import ExtraTreesProximityClusterer
from .urf import UnsupervisedRandomForestClusterer


_ALLOWED_ALGORITHMS = {"forest", "urf", "extratrees", "binary_tree"}
_ALLOWED_SCORING = {"silhouette", "calinski_harabasz", "davies_bouldin", "stability", "combined"}


class AutoTreeClusterer(BaseEstimator, ClusterMixin):
    """Automatically select a tree-clustering algorithm and cluster count.

    Parameters
    ----------
    algorithms : sequence of str, default=("forest", "urf", "extratrees", "binary_tree")
        Algorithms to try. Supported names are ``"forest"``
        (:class:`ForestClusterer`), ``"urf"``
        (:class:`UnsupervisedRandomForestClusterer`), ``"extratrees"``
        (:class:`ExtraTreesProximityClusterer`) and ``"binary_tree"``
        (:class:`UnsupervisedBinaryTreeClusterer`).
    k_range : iterable of int or int, default=(2, 3, 4, 5, 6)
        Cluster counts to try. If an integer ``m`` is supplied, the range
        ``2..m`` is used.
    scoring : {"silhouette", "calinski_harabasz", "davies_bouldin", "stability", "combined"}, default="combined"
        Internal objective. ``combined`` uses mean silhouette plus
        ``stability_weight * stability`` across restarts.
    n_restarts : int, default=3
        Number of random seeds per algorithm / parameter / cluster-count
        candidate. More restarts improve robustness at additional cost.
    stability_weight : float, default=0.10
        Weight of restart stability in ``combined`` scoring.
    scoring_space : {"auto", "features", "proximity"}, default="auto"
        Representation used for internal quality metrics. ``"auto"`` and
        ``"features"`` use leak-safe feature representations that are not
        constructed from the candidate final labels. ``"proximity"`` keeps the
        older proximity-distance silhouette behaviour for advanced users, but
        it is unsafe for estimators whose distance is defined directly from
        final cluster labels.
    scoring_sample_size : int or None, default=None
        Optional sample size for silhouette scoring on large datasets. ``None``
        uses all samples.
    estimator_params : dict or None, default=None
        Algorithm-specific parameter grids. Example::

            {
                "forest": {"n_iterations": [100, 200], "n_bins": ["auto", 4]},
                "urf": {"n_estimators": [100]},
            }

        Scalar values are treated as fixed parameters; lists/tuples/sets are
        expanded as a grid.
    n_iterations : int, default=100
        Default ``ForestClusterer.n_iterations`` when not overridden.
    n_estimators : int, default=100
        Default ``n_estimators`` for forest-based estimators when not overridden.
    n_bins : int or "auto", default="auto"
        Default ``ForestClusterer.n_bins`` when not overridden.
    n_jobs : int or None, default=None
        Parallelism forwarded where supported.
    random_state : int or None, default=None
        Base seed for all restarts.
    add_missing_indicators, rare_category_min_count, rare_category_min_freq,
    coerce_numeric_strings, numeric_string_min_fraction : preprocessing options
        Forwarded to all supported estimators.

    Attributes
    ----------
    best_estimator_ : estimator
        Fitted winning estimator.
    best_algorithm_ : str
        Winning algorithm name.
    best_n_clusters_ : int
        Winning cluster count.
    best_params_ : dict
        Parameters used for the winning estimator, including algorithm and seed.
    best_score_ : float
        Winning internal score.
    cv_results_ : pandas.DataFrame
        One row per algorithm / cluster-count / parameter-grid candidate,
        aggregated across restarts.
    search_results_ : pandas.DataFrame
        One row per individual fitted restart.
    labels_ : ndarray of shape (n_samples,)
        Labels from ``best_estimator_``.
    """

    def __init__(
        self,
        algorithms=("forest", "urf", "extratrees", "binary_tree"),
        k_range=(2, 3, 4, 5, 6),
        scoring: str = "combined",
        n_restarts: int = 3,
        stability_weight: float = 0.10,
        scoring_space: str = "auto",
        scoring_sample_size: int | None = None,
        estimator_params: dict | None = None,
        n_iterations: int = 100,
        n_estimators: int = 100,
        n_bins="auto",
        n_jobs=None,
        random_state: int | None = None,
        add_missing_indicators: bool = True,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = True,
        numeric_string_min_fraction: float = 0.90,
        refit: bool = False,
    ):
        self.algorithms = algorithms
        self.k_range = k_range
        self.scoring = scoring
        self.n_restarts = n_restarts
        self.stability_weight = stability_weight
        self.scoring_space = scoring_space
        self.scoring_sample_size = scoring_sample_size
        self.estimator_params = estimator_params
        self.n_iterations = n_iterations
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction
        self.refit = refit

    def fit(self, X, y=None):
        """Run the search and keep the best fitted estimator."""
        algorithms = self._validate_algorithms(self.algorithms)
        k_values = self._validate_k_range(self.k_range)
        scoring = self._validate_scoring(self.scoring)
        scoring_space = self._validate_scoring_space(self.scoring_space)
        scoring_sample_size = self._validate_scoring_sample_size(self.scoring_sample_size)
        n_restarts = int(self.n_restarts)
        self._fit_X_ = X
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if float(self.stability_weight) < 0:
            raise ValueError("stability_weight must be >= 0")

        run_rows: list[dict[str, Any]] = []
        fitted_runs: list[tuple[int, BaseEstimator]] = []
        group_id = 0
        for algorithm in algorithms:
            for params in self._parameter_grid(algorithm):
                params_key = self._params_key(params)
                for k in k_values:
                    labels_for_group = []
                    run_indices = []
                    for restart in range(n_restarts):
                        seed = self._seed_for(group_id, restart)
                        est = self._make_estimator(algorithm, k, seed, params)
                        labels = est.fit_predict(X)
                        metrics = self._score_estimator(est, labels, scoring_space=scoring_space, scoring_sample_size=scoring_sample_size)
                        row = {
                            "group_id": group_id,
                            "algorithm": algorithm,
                            "n_clusters": int(k),
                            "restart": restart,
                            "random_state": seed,
                            "params": dict(params),
                            "params_key": params_key,
                            **metrics,
                        }
                        run_rows.append(row)
                        fitted_runs.append((len(run_rows) - 1, est))
                        labels_for_group.append(np.asarray(labels))
                        run_indices.append(len(run_rows) - 1)
                    stability = self._stability(labels_for_group)
                    for idx in run_indices:
                        run_rows[idx]["stability"] = stability
                    group_id += 1

        self.search_results_ = pd.DataFrame(run_rows)
        self.cv_results_ = self._aggregate_results(self.search_results_, scoring)
        if self.cv_results_.empty:
            raise RuntimeError("No AutoTreeClusterer candidates were evaluated")
        if not np.isfinite(self.cv_results_["mean_score"]).any():
            raise RuntimeError("No candidate produced a valid clustering")

        best_row = self.cv_results_.iloc[0]
        best_group_id = int(best_row["group_id"])
        group_runs = self.search_results_[self.search_results_["group_id"] == best_group_id]
        # Keep the best individual restart for the winning candidate.  This
        # avoids changing the selected model after search unless refit=True.
        run_score_col = self._run_score_column(scoring)
        best_run_idx = int(group_runs.sort_values(
            [run_score_col, "primary_score", "calinski_harabasz", "neg_davies_bouldin", "restart"],
            ascending=[False, False, False, False, True],
        ).iloc[0].name)
        best_est = next(est for idx, est in fitted_runs if idx == best_run_idx)

        self.best_algorithm_ = str(best_row["algorithm"])
        self.best_n_clusters_ = int(best_row["n_clusters"])
        self.best_score_ = float(best_row["mean_score"])
        self.best_index_ = int(best_group_id)
        self.best_run_index_ = int(best_run_idx)
        best_seed = self.search_results_.loc[best_run_idx, "random_state"]
        if pd.isna(best_seed):
            best_seed = None
        elif best_seed is not None:
            best_seed = int(best_seed)
        self.best_params_ = {
            "algorithm": self.best_algorithm_,
            "n_clusters": self.best_n_clusters_,
            "random_state": best_seed,
            **dict(best_row["params"]),
        }

        if self.refit:
            best_est = self._make_estimator(
                self.best_algorithm_,
                self.best_n_clusters_,
                self.best_params_["random_state"],
                dict(best_row["params"]),
            ).fit(X)

        self.best_estimator_ = best_est
        self.labels_ = np.asarray(self.best_estimator_.labels_)
        self.n_features_in_ = getattr(self.best_estimator_, "n_features_in_", None)
        self.feature_names_in_ = getattr(self.best_estimator_, "feature_names_in_", None)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def transform(self, X):
        check_is_fitted(self, "best_estimator_")
        if not hasattr(self.best_estimator_, "transform"):
            raise AttributeError(f"Best estimator {type(self.best_estimator_).__name__} does not implement transform")
        return self.best_estimator_.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        if not hasattr(self.best_estimator_, "predict"):
            raise AttributeError(f"Best estimator {type(self.best_estimator_).__name__} does not implement predict")
        return self.best_estimator_.predict(X)

    def proximity_matrix(self, X=None, Y=None):
        check_is_fitted(self, "best_estimator_")
        if hasattr(self.best_estimator_, "proximity_matrix"):
            return self.best_estimator_.proximity_matrix(X=X, Y=Y)
        return self.best_estimator_.similarity_matrix(X=X, Y=Y)

    def similarity_matrix(self, X=None, Y=None):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.similarity_matrix(X=X, Y=Y)

    def pairwise_distance(self, X=None, Y=None):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.pairwise_distance(X=X, Y=Y)

    def score(self, X=None, y=None):
        """Return the selected internal score after ``fit``.

        ``X`` is accepted for sklearn API compatibility and ignored: the score is
        the search criterion computed during fitting.
        """
        check_is_fitted(self, "best_score_")
        return self.best_score_

    # ------------------------------------------------------------------
    # Search internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_algorithms(algorithms):
        if isinstance(algorithms, str):
            algorithms = (algorithms,)
        algorithms = tuple(algorithms)
        if not algorithms:
            raise ValueError("algorithms must contain at least one algorithm")
        for algorithm in algorithms:
            if algorithm not in _ALLOWED_ALGORITHMS:
                raise ValueError(f"Unknown algorithm {algorithm!r}; supported: {sorted(_ALLOWED_ALGORITHMS)}")
        return algorithms

    @staticmethod
    def _validate_k_range(k_range):
        if isinstance(k_range, int):
            if k_range < 2:
                raise ValueError("integer k_range must be >= 2")
            values = tuple(range(2, int(k_range) + 1))
        else:
            values = tuple(int(k) for k in k_range)
        if not values:
            raise ValueError("k_range must contain at least one cluster count")
        if any(k < 1 for k in values):
            raise ValueError("all k_range values must be >= 1")
        return values

    @staticmethod
    def _validate_scoring(scoring):
        if scoring not in _ALLOWED_SCORING:
            raise ValueError(f"scoring must be one of {sorted(_ALLOWED_SCORING)}, got {scoring!r}")
        return scoring

    @staticmethod
    def _validate_scoring_space(scoring_space):
        allowed = {"auto", "features", "proximity"}
        if scoring_space not in allowed:
            raise ValueError(f"scoring_space must be one of {sorted(allowed)}, got {scoring_space!r}")
        return scoring_space

    @staticmethod
    def _validate_scoring_sample_size(scoring_sample_size):
        if scoring_sample_size is None:
            return None
        scoring_sample_size = int(scoring_sample_size)
        if scoring_sample_size < 2:
            raise ValueError("scoring_sample_size must be >= 2 or None")
        return scoring_sample_size

    def _seed_for(self, group_id: int, restart: int) -> int | None:
        if self.random_state is None:
            return None
        return int(self.random_state) + 1009 * int(group_id) + int(restart)

    def _parameter_grid(self, algorithm: str):
        by_algo = self.estimator_params or {}
        params = dict(by_algo.get(algorithm, {}))
        if not params:
            return [dict()]
        keys = list(params.keys())
        values = [self._as_grid_values(params[k]) for k in keys]
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @staticmethod
    def _as_grid_values(value):
        if isinstance(value, (str, bytes)) or value is None:
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _params_key(params: dict) -> str:
        if not params:
            return "{}"
        return repr(tuple(sorted(params.items())))

    def _quality_kwargs(self):
        return dict(
            add_missing_indicators=self.add_missing_indicators,
            rare_category_min_count=self.rare_category_min_count,
            rare_category_min_freq=self.rare_category_min_freq,
            coerce_numeric_strings=self.coerce_numeric_strings,
            numeric_string_min_fraction=self.numeric_string_min_fraction,
        )

    def _make_estimator(self, algorithm: str, k: int, seed: int | None, params: dict):
        q = self._quality_kwargs()
        common = dict(n_clusters=int(k), random_state=seed, **q)
        if algorithm == "forest":
            base = dict(n_iterations=self.n_iterations, n_bins=self.n_bins, n_jobs=self.n_jobs, **common)
            base.update(params)
            return ForestClusterer(**base)
        if algorithm == "urf":
            base = dict(n_estimators=self.n_estimators, n_jobs=self.n_jobs, **common)
            base.update(params)
            return UnsupervisedRandomForestClusterer(**base)
        if algorithm == "extratrees":
            base = dict(n_estimators=self.n_estimators, n_jobs=self.n_jobs, **common)
            base.update(params)
            return ExtraTreesProximityClusterer(**base)
        if algorithm == "binary_tree":
            base = dict(**common)
            base.update(params)
            return UnsupervisedBinaryTreeClusterer(**base)
        raise ValueError(f"Unknown algorithm {algorithm!r}")

    def _score_estimator(self, est, labels, scoring_space: str, scoring_sample_size: int | None):
        labels = np.asarray(labels)
        primary = self._silhouette_score(est, labels, scoring_space=scoring_space, sample_size=scoring_sample_size)
        out = {"silhouette": primary, "primary_score": primary}
        out["calinski_harabasz"] = self._feature_score(est, labels, calinski_harabasz_score)
        db = self._feature_score(est, labels, davies_bouldin_score)
        out["davies_bouldin"] = db
        out["neg_davies_bouldin"] = -db if np.isfinite(db) else -np.inf
        return out

    @staticmethod
    def _valid_labeling(labels) -> bool:
        labels = np.asarray(labels)
        n_labels = np.unique(labels).size
        return 2 <= n_labels <= labels.size - 1

    def _silhouette_score(self, est, labels, scoring_space: str, sample_size: int | None):
        if not self._valid_labeling(labels):
            return -np.inf

        # Default to leak-safe scoring.  The previous AutoTreeClusterer used
        # est.pairwise_distance() for every estimator.  That is invalid for
        # estimators such as UnsupervisedBinaryTreeClusterer whose distance is
        # defined from the final leaf/cluster labels themselves: it makes every
        # non-trivial labeling look perfectly separated and can select k=2 over
        # a correct k=3.  Feature-space scoring below uses representations built
        # before the final labels are assigned.
        if scoring_space in {"auto", "features"}:
            try:
                X = self._scoring_features(est)
                return float(silhouette_score(
                    X,
                    labels,
                    metric="euclidean",
                    sample_size=self._effective_silhouette_sample_size(labels, sample_size),
                    random_state=self.random_state,
                ))
            except Exception:
                return -np.inf

        # Explicit opt-in compatibility mode.  Kept for users who intentionally
        # want proximity-distance scoring and understand the leakage risk.
        try:
            D = est.pairwise_distance()
            D = np.asarray(D, dtype=float)
            np.fill_diagonal(D, 0.0)
            return float(silhouette_score(
                D,
                labels,
                metric="precomputed",
                sample_size=self._effective_silhouette_sample_size(labels, sample_size),
                random_state=self.random_state,
            ))
        except Exception:
            return -np.inf

    @staticmethod
    def _effective_silhouette_sample_size(labels, sample_size):
        if sample_size is None:
            return None
        labels = np.asarray(labels)
        n = labels.size
        n_labels = np.unique(labels).size
        sample_size = min(int(sample_size), n)
        # silhouette requires n_labels <= n_samples - 1.  If the requested
        # sample is too small for the chosen k, fall back to the smallest safe
        # sample, capped by n.  If that means all rows, sklearn accepts None.
        sample_size = max(sample_size, min(n, n_labels + 1))
        return None if sample_size >= n else sample_size

    def _feature_score(self, est, labels, scorer):
        if not self._valid_labeling(labels):
            return -np.inf
        try:
            X = self._scoring_features(est)
            if hasattr(X, "toarray"):
                X = X.toarray()
            return float(scorer(np.asarray(X), labels))
        except Exception:
            return -np.inf

    def _scoring_features(self, est):
        """Return leak-safe features for AutoTree internal model selection.

        The returned matrix must not be constructed from final cluster labels.
        Raw leaf ids are also avoided for Euclidean metrics because leaf ids are
        nominal.  The method intentionally prefers one-hot/weighted embeddings
        or the preprocessed feature matrix used by an interpretable binary tree.
        """
        if hasattr(est, "get_scoring_features"):
            X = est.get_scoring_features()
        elif isinstance(est, UnsupervisedBinaryTreeClusterer) and hasattr(est, "X_transformed_"):
            X = est.X_transformed_
        elif hasattr(est, "leaf_onehot_embedding_"):
            X = est.leaf_onehot_embedding_
        elif hasattr(est, "transform_onehot"):
            X = est.transform_onehot(self._fit_X_)
        elif hasattr(est, "_embedding_as_weighted_features") and hasattr(est, "embedding_"):
            X = est._embedding_as_weighted_features(est.embedding_, sparse_output=True)
        else:
            X = est.transform(self._fit_X_)
        return X

    @staticmethod
    def _stability(labelings) -> float:
        if any(not AutoTreeClusterer._valid_labeling(labels) for labels in labelings):
            return -np.inf
        if len(labelings) < 2:
            return 0.0
        scores = [adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]
        return float(np.mean(scores)) if scores else 0.0

    def _aggregate_results(self, runs: pd.DataFrame, scoring: str) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        rows = []
        for group_id, part in runs.groupby("group_id", sort=False):
            mean_sil = self._finite_mean(part["silhouette"].to_numpy())
            mean_ch = self._finite_mean(part["calinski_harabasz"].to_numpy())
            mean_db = self._finite_mean(part["davies_bouldin"].to_numpy())
            stability = float(part["stability"].iloc[0]) if "stability" in part else 0.0
            if scoring == "silhouette":
                score = mean_sil
            elif scoring == "calinski_harabasz":
                score = mean_ch
            elif scoring == "davies_bouldin":
                score = -mean_db if np.isfinite(mean_db) else -np.inf
            elif scoring == "stability":
                score = stability
            else:  # combined
                score = mean_sil + float(self.stability_weight) * stability
            first = part.iloc[0]
            rows.append({
                "group_id": int(group_id),
                "algorithm": first["algorithm"],
                "n_clusters": int(first["n_clusters"]),
                "params": first["params"],
                "params_key": first["params_key"],
                "mean_score": float(score),
                "mean_silhouette": mean_sil,
                "mean_calinski_harabasz": mean_ch,
                "mean_davies_bouldin": mean_db,
                "stability": stability,
                "n_restarts": int(len(part)),
            })
        out = pd.DataFrame(rows)
        out["neg_mean_davies_bouldin"] = -out["mean_davies_bouldin"].replace(-np.inf, np.inf)
        out = out.sort_values(
            [
                "mean_score",
                "mean_silhouette",
                "stability",
                "mean_calinski_harabasz",
                "neg_mean_davies_bouldin",
                "n_clusters",
            ],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    @staticmethod
    def _run_score_column(scoring: str) -> str:
        if scoring == "calinski_harabasz":
            return "calinski_harabasz"
        if scoring == "davies_bouldin":
            return "neg_davies_bouldin"
        return "primary_score"

    @staticmethod
    def _finite_mean(values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return -np.inf
        return float(np.mean(arr))

    # Fit data is kept only during scoring to compute feature-space metrics.
    def _more_tags(self):
        return {"requires_y": False}


# ---------------------------------------------------------------------------
# Lightweight plotting helpers attached after class definition to keep the
# search logic above focused on fitting and scoring.
# ---------------------------------------------------------------------------
def _auto_plot_search_results(self, x="n_clusters", y="mean_score", hue="algorithm", ax=None):
    """Plot AutoTree search results from ``cv_results_``.

    Parameters are column names in ``cv_results_``.  The method intentionally
    returns a matplotlib Axes so users can keep customising the chart.
    """
    check_is_fitted(self, "cv_results_")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    df = self.cv_results_.copy()
    if x not in df.columns or y not in df.columns or hue not in df.columns:
        raise ValueError("x, y and hue must be columns in cv_results_")
    for name, part in df.groupby(hue):
        part = part.sort_values(x)
        ax.plot(part[x], part[y], marker="o", label=str(name))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("AutoTree search results")
    ax.legend(loc="best")
    return ax


def _auto_plot_k_selection(self, ax=None):
    """Plot the best score for each candidate number of clusters."""
    check_is_fitted(self, "cv_results_")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    df = self.cv_results_.groupby("n_clusters", as_index=False)["mean_score"].max().sort_values("n_clusters")
    ax.plot(df["n_clusters"], df["mean_score"], marker="o")
    ax.axvline(self.best_n_clusters_, linestyle="--")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("best mean_score")
    ax.set_title("AutoTree k selection")
    return ax


def _auto_plot_parameter_sensitivity(self, parameter, ax=None):
    """Plot average search score by a parameter stored in ``params``."""
    check_is_fitted(self, "cv_results_")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for _, row in self.cv_results_.iterrows():
        params = row.get("params", {}) or {}
        if parameter in params:
            rows.append({"value": str(params[parameter]), "score": row["mean_score"]})
    df = pd.DataFrame(rows)
    if df.empty:
        ax.text(0.5, 0.5, f"Parameter {parameter!r} was not varied", ha="center", va="center")
        ax.set_axis_off()
        return ax
    agg = df.groupby("value", as_index=False)["score"].mean().sort_values("score")
    ax.barh(agg["value"], agg["score"])
    ax.set_xlabel("mean score")
    ax.set_ylabel(parameter)
    ax.set_title(f"AutoTree sensitivity: {parameter}")
    return ax


AutoTreeClusterer.plot_search_results = _auto_plot_search_results
AutoTreeClusterer.plot_k_selection = _auto_plot_k_selection
AutoTreeClusterer.plot_parameter_sensitivity = _auto_plot_parameter_sensitivity
