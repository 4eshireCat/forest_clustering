# Changelog

## 0.8.0 - prototype sampling and subsampled clustering

- Added `PrototypeSampler` with `leaf_signature` and `birch` modes.
- Added weighted prototypes, `inverse_assignment_`, `expand_labels()` and compression diagnostics.
- Added `SubsampledClusterer` wrapper to fit expensive clustering on prototypes and return full-data labels.
- Added visualization helpers for compression, prototype weights and reconstruction error.
- Added tests for prototype weights, label expansion, rare-bucket preservation, BIRCH mode and full-data label expansion.


## 0.6.1 - AutoTree scoring leakage fix

### Fixed

- Fixed a model-selection bug in `AutoTreeClusterer`: silhouette scoring no longer uses estimator distances that may be derived directly from final cluster labels.
- `UnsupervisedBinaryTreeClusterer` is now scored in its fitted preprocessed feature space during AutoTree search, preventing self-confirming distance scores where every non-trivial leaf partition looked perfect.
- Added robust secondary tie-breakers for AutoTree candidate ranking: silhouette, stability, Calinski-Harabasz, negative Davies-Bouldin, then `n_clusters`.
- Added regression tests showing that AutoTree now selects `k=3` on a three-blob dataset where version 0.6.0 selected `k=2`.

### Added

- `scoring_space` parameter for `AutoTreeClusterer`: `"auto"` / `"features"` use leak-safe feature scoring; `"proximity"` is an explicit compatibility mode.
- `scoring_sample_size` parameter for optional silhouette subsampling on larger datasets.

## 0.6.0 - Automatic parameter selection

- Added `AutoTreeClusterer`, a sklearn-style meta-estimator for automatic selection over algorithm families, cluster counts, parameter grids and random restarts.
- Added internal scoring modes: `silhouette`, `calinski_harabasz`, `davies_bouldin`, `stability`, and `combined`.
- Added `best_estimator_`, `best_algorithm_`, `best_n_clusters_`, `best_score_`, `best_params_`, `cv_results_`, and `search_results_`.
- Delegated `transform`, `similarity_matrix`, `proximity_matrix`, and `pairwise_distance` from `AutoTreeClusterer` to the selected estimator.
- Added TDD tests for model selection, matrix delegation, stability scoring, sklearn cloneability, reproducibility and validation errors.

## 0.5.1 - Quality fixes

- Made DBSCAN eps retry opt-in via `auto_tune_dbscan=False` default.
- Added explicit `cluster_input={"auto", "embedding", "onehot", "distance", "similarity"}` to avoid ambiguous downstream-clusterer inputs.
- Added missing-value indicator features through `add_missing_indicators`.
- Added rare-category grouping via `rare_category_min_count` and `rare_category_min_freq`.
- Added numeric-string coercion via `coerce_numeric_strings` and `numeric_string_min_fraction`.
- Added `n_bins="auto"` for simple Sturges-style bin selection.
- Propagated quality preprocessing options to URF, ExtraTrees proximity and binary tree estimators.
- Added regression tests for these fixes.


## 0.5.0

Release-preparation version consolidating the patched fork into a deployable package.

### Added

- `UnsupervisedRandomForestClusterer`: Breiman-style unsupervised random forest clustering.
- `ExtraTreesProximityClusterer`: ExtraTrees-based proximity clustering.
- `UnsupervisedBinaryTreeClusterer`: interpretable greedy binary tree clustering.
- `transform_onehot(X)` for URF and ExtraTrees.
- Shared tree utilities in `_tree_common.py`.
- Expanded tests for proximity semantics, sklearn API compatibility, missing values, Titanic smoke checks, and downstream clustering behavior.
- `README.md`, `ALGORITHM.md`, `RELEASE.md`, `MANIFEST.in`, and deployment metadata.

### Fixed

- Corrected quantile cut-point sampling in `ForestClusterer`.
- Avoided Spearman correlation weighting for arbitrary label-encoded categoricals.
- Added explicit `similarity_matrix()` API.
- Moved heavy estimator validation out of `__init__` and into `fit()`.
- Fixed downstream clustering for tree estimators: non-precomputed clusterers now receive sparse one-hot leaf features rather than raw leaf ids.
- Improved robustness on all-missing mixed columns.
- Added compatibility helpers for sklearn API differences around `OneHotEncoder` and `SimpleImputer`.

### Notes

This version keeps the public API sklearn-like and intentionally does not upload credentials or tokens. Build and upload commands are documented in `RELEASE.md`.

## 0.7.0 - cluster-label assignment and explanations

Added an inductive explanation layer on top of tree-based clustering:

- `ClusterLabelClassifier` learns a supervised classifier that reproduces cluster labels.
  It provides `predict`, `predict_proba`, out-of-fold fidelity metrics, confidence-based rejection,
  cluster profiles and diagnostic plots.
- `ClusterSurrogateTree` fits an interpretable decision tree to cluster labels and exports
  human-readable rules, rule tables and tree visualizations.
- New plotting helpers:
  - `plot_cluster_sizes()`
  - `plot_embedding()`
  - `plot_feature_importances()`
  - `plot_fidelity_confusion_matrix()`
  - `ClusterSurrogateTree.plot_tree()`
- New explanation helpers:
  - `cluster_profile()`
  - `explain_clusters()`
  - `explain_rules()`
  - `rules_dataframe()`
- Added regression tests for inductive assignment, rule extraction, confidence rejection and plotting.

Important terminology: fidelity metrics measure how well the supervised surrogate reproduces
cluster labels. They do not prove that the original clustering is externally correct.

## 0.9.0 - Diagnostics and visualisation

Added:

- `ClusterDiagnosticsReport` for clustering diagnostics, health checks, cluster cards and plots.
- `StabilityAnalyzer` for seed/resampling stability analysis using ARI/NMI.
- `compare_clusterings()` and `ClusterComparison` for model comparison tables and agreement heatmaps.
- AutoTree visualisation helpers: `plot_search_results()`, `plot_k_selection()` and `plot_parameter_sensitivity()`.
- Diagnostic visualisations: cluster sizes, PCA/SVD embeddings, silhouette plots, profile plots, proximity heatmaps and uncertainty histograms.
- Mixed-data fallback in `compare_clusterings()` for ordinary sklearn baselines such as `KMeans`.

Changed:

- Bumped package version to `0.9.0`.
- Documentation now presents diagnostics as a first-class workflow: fit -> diagnose -> explain -> compare -> deploy.

Notes:

- Diagnostic metrics are internal validity/stability signals. They help catch failure modes, but they do not prove that a clustering is the only correct segmentation.
