# forest-clustering

Tree-based and random-partition clustering for mixed-type tabular data.

`forest-clustering` provides sklearn-style estimators that convert tabular rows into tree or partition embeddings, compute sample-to-sample similarities, and then run a downstream clustering algorithm. It is designed for practical unsupervised learning on mixed numerical/categorical data where Euclidean distance is often a poor default.

## What is included

| Estimator | Core idea | Best use case |
|---|---|---|
| `ForestClusterer` | Random feature partitions produce an integer embedding; Hamming similarity measures co-partitioning. | Fast mixed-type clustering, large tabular data, robust random-partition baseline. |
| `UnsupervisedRandomForestClusterer` | Breiman-style unsupervised random forest: real rows vs synthetic null rows, then same-leaf proximity. | Canonical random-forest proximity clustering. |
| `ExtraTreesProximityClusterer` | ExtraTrees version of the synthetic-null forest; more randomized splits and fast leaf proximity. | Faster tree-proximity baseline with strong randomization. |
| `UnsupervisedBinaryTreeClusterer` | Greedy interpretable binary tree that recursively splits rows to reduce within-leaf variance. | Explainable cluster rules and small/medium tabular datasets. |
| `AutoTreeClusterer` | Tries several tree-based estimators, cluster counts, parameter grids and random restarts; selects the best by internal score/stability. | Practical autoparameter selection when you do not know the best algorithm or `k`. |
| `ForestTransformer` | Transformer-only wrapper around `ForestClusterer`. | sklearn pipelines and custom downstream models. |

The package also includes weighted Hamming distances, graph helpers, adaptive bins, KDE-based cut points, permutation importance, clusterability checks, and significance utilities.

## Autotuning in 0.6.x

The 0.6 release line adds `AutoTreeClusterer`, a sklearn-style meta-estimator for simple, practical autoparameter selection. It can search across algorithms, cluster counts, small algorithm-specific grids and random restarts. Version 0.6.1 fixes an important model-selection issue: internal silhouette scoring now uses leak-safe feature representations by default instead of distances that may be derived from the candidate final labels.

```python
from forest_clustering import AutoTreeClusterer

model = AutoTreeClusterer(
    algorithms=("forest", "urf", "extratrees", "binary_tree"),
    k_range=range(2, 8),
    scoring="combined",          # leak-safe silhouette + stability
    scoring_space="auto",         # default; avoids label-derived distance leakage
    n_restarts=3,
    estimator_params={
        "forest": {"n_iterations": [100, 200], "n_bins": ["auto", 4]},
        "urf": {"n_estimators": [100]},
        "extratrees": {"n_estimators": [100]},
        "binary_tree": {"max_depth": [None, 6]},
    },
    add_missing_indicators=True,
    rare_category_min_count=5,
    coerce_numeric_strings=True,
    random_state=42,
)

labels = model.fit_predict(X)
print(model.best_algorithm_, model.best_n_clusters_, model.best_score_)
print(model.cv_results_.head())
```

Supported scoring modes are `"silhouette"`, `"calinski_harabasz"`, `"davies_bouldin"`, `"stability"`, and `"combined"`. By default, `scoring_space="auto"` uses leak-safe feature-space scoring: weighted partition features for `ForestClusterer`, one-hot leaf features for URF/ExtraTrees, and the preprocessed feature matrix for `UnsupervisedBinaryTreeClusterer`. `scoring_space="proximity"` is available only as an explicit compatibility mode. The selected estimator is available as `model.best_estimator_`, and matrix APIs delegate to it: `similarity_matrix()`, `pairwise_distance()`, and `transform(X)`.

## Quality fixes in 0.5.1

The 0.5.1 release focuses on practical clustering quality and safer defaults:

- `auto_tune_dbscan=False` by default: DBSCAN parameters are no longer silently changed.
- `cluster_input="auto" | "embedding" | "onehot" | "distance" | "similarity"`: explicit downstream input control.
- `add_missing_indicators=True`: append binary missingness indicators.
- `rare_category_min_count` / `rare_category_min_freq`: group rare and unseen categories into a stable rare bucket.
- `coerce_numeric_strings=True`: treat numeric object/string columns as numeric features.
- `n_bins="auto"`: simple sample-size-aware bin selection.

```python
from forest_clustering import ForestClusterer

model = ForestClusterer(
    n_bins="auto",
    n_clusters=3,
    add_missing_indicators=True,
    rare_category_min_count=5,
    coerce_numeric_strings=True,
    cluster_input="auto",
    random_state=42,
)
labels = model.fit_predict(X)
```

## Installation

```bash
pip install forest-clustering
```

Python `>=3.10` is required.

## Quick start

```python
import pandas as pd
from forest_clustering import ForestClusterer

X = pd.DataFrame({
    "age": [22, 38, 26, 35, 54, 2],
    "fare": [7.25, 71.28, 7.92, 53.10, 51.86, 21.08],
    "sex": ["male", "female", "female", "female", "male", "male"],
    "pclass": [3, 1, 3, 1, 1, 3],
})

model = ForestClusterer(
    n_iterations=200,
    n_bins=3,
    cut_strategy="quantile",
    random_state=42,
)

labels = model.fit_predict(X)
embedding = model.get_embedding()
distance = model.pairwise_distance()
similarity = model.similarity_matrix()
```

## Tree-proximity clustering

### Breiman-style unsupervised random forest

```python
from forest_clustering import UnsupervisedRandomForestClusterer

urf = UnsupervisedRandomForestClusterer(
    n_estimators=300,
    n_clusters=4,
    synthetic="permute_marginals",
    random_state=42,
)

labels = urf.fit_predict(X)
proximity = urf.proximity_matrix()
leaf_ids = urf.transform(X)
leaf_onehot = urf.transform_onehot(X)
```

The estimator builds a synthetic null dataset, trains a random forest to separate observed rows from synthetic rows, and uses same-leaf co-occurrence as a proximity score.

### ExtraTrees proximity clustering

```python
from forest_clustering import ExtraTreesProximityClusterer

xt = ExtraTreesProximityClusterer(
    n_estimators=300,
    n_clusters=4,
    synthetic="uniform_box",
    random_state=42,
)

labels = xt.fit_predict(X)
```

This estimator uses the same synthetic-null idea as URF, but with `ExtraTreesClassifier`, producing more randomized split geometry.

### Interpretable binary tree clustering

```python
from forest_clustering import UnsupervisedBinaryTreeClusterer

bt = UnsupervisedBinaryTreeClusterer(
    n_clusters=4,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
)

labels = bt.fit_predict(X)
rules = bt.rules()
```

Use this when you need human-readable cluster rules instead of only a proximity matrix.

### Automatic parameter selection

```python
from forest_clustering import AutoTreeClusterer

auto = AutoTreeClusterer(
    algorithms=("forest", "urf", "extratrees"),
    k_range=(2, 3, 4, 5),
    scoring="combined",
    n_restarts=3,
    random_state=42,
)

labels = auto.fit_predict(X)
best_model = auto.best_estimator_
results = auto.cv_results_
```

`AutoTreeClusterer` is deliberately conservative: it does not claim to find a universally true clustering. It automates the practical choices users normally tune by hand: algorithm family, cluster count, selected hyperparameters and seed robustness.

## Using custom downstream clusterers

All estimators keep a sklearn-like interface. For proximity-based tree estimators, downstream clusterers with `metric="precomputed"` receive a distance matrix. Other clusterers receive a sparse one-hot leaf embedding, not raw leaf ids.

```python
from sklearn.cluster import AgglomerativeClustering, KMeans
from forest_clustering import UnsupervisedRandomForestClusterer

# Distance-matrix downstream clustering
agg = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
model = UnsupervisedRandomForestClusterer(clusterer=agg, random_state=0)
labels = model.fit_predict(X)

# Feature-matrix downstream clustering
km = KMeans(n_clusters=3, n_init="auto", random_state=0)
model = UnsupervisedRandomForestClusterer(clusterer=km, random_state=0)
labels = model.fit_predict(X)
```

## sklearn pipeline usage

```python
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans
from forest_clustering import ForestTransformer

pipe = make_pipeline(
    ForestTransformer(n_iterations=300, n_bins=3, random_state=42),
    MiniBatchKMeans(n_clusters=5, random_state=42),
)

labels = pipe.fit_predict(X)
```

## Practical recommendations

| Goal | Recommendation |
|---|---|
| Fast baseline | `ForestClusterer(n_iterations=100, n_bins=3)` |
| Robust mixed-type clustering | `ForestClusterer(cut_strategy="quantile", corr_threshold=0.8)` |
| Canonical tree proximity | `UnsupervisedRandomForestClusterer(n_estimators=300)` |
| Faster randomized tree proximity | `ExtraTreesProximityClusterer(n_estimators=300)` |
| Explainable clusters | `UnsupervisedBinaryTreeClusterer(n_clusters=k)` |
| Unknown algorithm / cluster count | `AutoTreeClusterer(k_range=range(2, 8), scoring="combined")` |
| Unknown number of clusters but fixed algorithm | Pair proximity/distance with `DBSCAN`, HDBSCAN, or graph clustering. |
| Large data | Use `MiniBatchKMeans` on embeddings or graph/LSH helpers. |

## API summary

Most estimators implement:

```python
fit(X, y=None)
fit_predict(X, y=None)
fit_transform(X, y=None)
transform(X)
pairwise_distance(...)
similarity_matrix(...)
```

`AutoTreeClusterer` additionally exposes `best_estimator_`, `best_algorithm_`, `best_n_clusters_`, `best_score_`, `best_params_`, `cv_results_`, and `search_results_`.

Tree-proximity estimators also implement:

```python
proximity_matrix(X=None, Y=None)
transform_onehot(X)   # URF and ExtraTrees
```

`UnsupervisedBinaryTreeClusterer` also implements:

```python
predict(X)
rules()
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Build and deploy

Build distribution archives:

```bash
python -m pip install --upgrade build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI after verifying the TestPyPI package:

```bash
python -m twine upload dist/*
```

Use API tokens rather than account passwords. For token-based uploads, username is usually `__token__` and the password is the token value.

## Documentation

See [`ALGORITHM.md`](ALGORITHM.md) for the mathematical and implementation details.

## License

MIT


## Explaining clusters and assigning new samples

Version 0.7.0 adds a supervised explanation layer.  The clusterer still defines the segmentation; the classifier learns to reproduce those labels for deployment and interpretation.

```python
from forest_clustering import AutoTreeClusterer, ClusterLabelClassifier, ClusterSurrogateTree

clusterer = AutoTreeClusterer(
    algorithms=("forest", "urf", "extratrees", "binary_tree"),
    k_range=range(2, 8),
    scoring="combined",
    random_state=42,
)

assigner = ClusterLabelClassifier(
    clusterer=clusterer,
    cv=5,
    confidence_threshold=0.60,
    unknown_policy="reject",
    random_state=42,
)
assigner.fit(X)

labels = assigner.labels_              # labels from the clusterer
new_labels = assigner.predict(X_new)    # -1 for low-confidence rows when reject mode is enabled
proba = assigner.predict_proba(X_new)
print(assigner.fidelity_summary())
print(assigner.explain_clusters())
```

For compact rules, fit a shallow surrogate decision tree:

```python
explainer = ClusterSurrogateTree(
    clusterer=clusterer,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
).fit(X)

print(explainer.explain_rules(min_purity=0.70))
print(explainer.rules_dataframe())
```

Visualization helpers return matplotlib axes and can be used in notebooks:

```python
assigner.plot_cluster_sizes()
assigner.plot_embedding()
assigner.plot_feature_importances(top_n=15)
assigner.plot_fidelity_confusion_matrix(normalize=True)
explainer.plot_tree()
```

The reported accuracy, balanced accuracy and F1 are **fidelity metrics**: they measure how well the supervised surrogate reproduces cluster labels.  They are not external clustering-quality scores.

## Prototype sampling and subsampled clustering

Version 0.8.0 added a conservative compression layer for large datasets.  The goal is not to throw rows away blindly; the sampler builds **weighted prototypes** and stores the map from every original row back to its prototype.

```python
from forest_clustering import PrototypeSampler, SubsampledClusterer, AutoTreeClusterer

sampler = PrototypeSampler(
    method="leaf_signature",
    compression=0.20,          # keep roughly up to 20% prototypes
    n_partitions=128,
    n_bins="auto",
    preserve_rare=True,        # protect tiny buckets / rare groups
    rare_bucket_min_size=3,
    random_state=42,
)

X_proto, weights = sampler.fit_resample(X)
print(sampler.compression_summary())

clusterer = AutoTreeClusterer(k_range=range(2, 8), random_state=42)
model = SubsampledClusterer(sampler=sampler, clusterer=clusterer)
labels = model.fit_predict(X)          # labels for all original rows
labels_new = model.predict(X_new)      # assigned through nearest prototype
```

Two sampler modes are provided:

| Method | Idea | Use when |
|---|---|---|
| `leaf_signature` | Use the library's random partition signatures, group rows with the same coarse signature, keep representative rows plus weights. | Mixed-type tabular data, many duplicates or near-duplicates, tree-clustering workflows. |
| `birch` | Use sklearn BIRCH on a numeric encoded feature space and keep medoid prototypes for subclusters. | Dense numeric data or already well-encoded features. |

The sampler exposes diagnostics and plots:

```python
sampler.plot_compression()
sampler.plot_prototype_weights()
sampler.plot_reconstruction_error()
```

Important: prototype sampling can speed up expensive clustering, especially when pairwise proximity matrices would otherwise be large.  It can also damage rare microclusters if configured aggressively. Keep `preserve_rare=True` unless you are deliberately compressing noise.


## Diagnostics and visualisation

Version 0.9.0 adds a diagnostic workflow for checking whether a clustering result is usable.

```python
from forest_clustering import AutoTreeClusterer, ClusterDiagnosticsReport

model = AutoTreeClusterer(k_range=range(2, 8), random_state=42).fit(X)
report = ClusterDiagnosticsReport(model, X)

print(report.summary())
print(report.health_checks())
print("\n\n".join(report.cluster_cards()))

report.plot_overview()
report.plot_proximity_heatmap()
report.plot_cluster_profiles()
```

For stochastic algorithms, inspect stability:

```python
from forest_clustering import StabilityAnalyzer

stability = StabilityAnalyzer(model, n_runs=10, random_state=42).fit(X)
print(stability.summary())
stability.plot_score_distribution()
```

For model comparison:

```python
from forest_clustering import compare_clusterings, ForestClusterer, UnsupervisedRandomForestClusterer
from sklearn.cluster import KMeans

comparison = compare_clusterings(X, {
    "forest": ForestClusterer(n_clusters=3, random_state=42),
    "urf": UnsupervisedRandomForestClusterer(n_clusters=3, random_state=42),
    "kmeans": KMeans(n_clusters=3, n_init=10, random_state=42),
})

print(comparison.rank())
comparison.plot_scores()
comparison.plot_pairwise_agreement()
```

The diagnostics are practical warning signals, not mathematical proof that a segmentation is uniquely correct. Treat health checks, stability and cluster cards as a review process before using clusters in production.

`compare_clusterings()` automatically retries ordinary sklearn estimators on a robust encoded feature matrix when they cannot consume mixed string/categorical columns. This makes baseline comparisons convenient without changing the native preprocessing of forest-clustering estimators.
