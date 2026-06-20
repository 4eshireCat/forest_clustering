# forest-clustering

**Random-partition similarity clustering for mixed-type tabular data with outlier robustness, correlation-aware feature selection and scalable large‑n paths.**

`forest-clustering` builds a compact integer embedding from random feature partitions and then applies any sklearn-compatible clustering algorithm to that embedding. It handles **numerical**, **categorical**, and **mixed** data natively, automatically detects feature types, down-weights correlated features, supports density-aware cut-points, contrastive tree splits, and scales to 100 K+ rows without materialising a dense `O(n²)` distance matrix.

---

## What makes it different

| Capability | forest-clustering | KMeans | DBSCAN | HDBSCAN | Agglomerative |
|---|---|---|---|---|---|
| Mixed-type (num + cat) | **native** | no | no | no | no |
| Outlier-robust cuts | **quantile / KDE peaks** | no | no | no | no |
| Correlated-feature down-weighting | **built-in** | no | no | no | no |
| Adaptive bins per feature | **yes** | no | no | no | no |
| Contrastive tree splits | **yes** | no | no | no | no |
| Graph clustering (Louvain / Leiden) | **yes** | no | no | no | no |
| `O(n·L)` memory for large `n` | **yes** | no | no | no | no |
| Incremental / online `partial_fit` | **yes** | no | no | no | no |
| Permutation feature importance | **yes** | no | no | no | no |
| Preflight clusterability tests | **yes** | no | no | no | no |
| Statistical significance of clusters | **yes** | no | no | no | no |

*ForestClusterer does not replace the above algorithms — it wraps them.* You still use KMeans, DBSCAN, AgglomerativeClustering, etc., but you run them on a robust, mixed-type-aware embedding instead of raw data.

---

## Installation

### PyPI (recommended)

```bash
# Core library — everything except optional graph backends
pip install forest-clustering

# With NetworkX Louvain support
pip install "forest-clustering[graph]"

# With Leiden (leidenalg + igraph) support — faster, better communities
pip install "forest-clustering[leiden]"

# With numba — accelerates weighted Hamming distance computation (2–5× faster)
pip install "forest-clustering[numba]"

# All optional backends
pip install "forest-clustering[graph,leiden,numba]"
```

Python ≥ 3.10 is required.

### Editable install from source

```bash
git clone https://github.com/<your-org>/forest-clustering.git
cd forest-clustering
pip install -e ".[dev,graph,leiden]"
pytest -q          # 979 tests expected
```

### Dependencies

| Package | Minimum | Notes |
|---|---|---|
| numpy | — | — |
| pandas | — | — |
| scipy | — | KDE, correlation, sparse matrices |
| scikit-learn | — | Base estimators, metrics |
| joblib | — | Parallel embedding computation |
| networkx | optional | `community_method='louvain'` |
| leidenalg + igraph | optional | `community_method='leiden'` |
| numba | optional | Accelerated weighted Hamming distances |

---

## How it works (in 30 seconds)

1. **Encode** — `DataEncoder` auto-detects numerical vs categorical columns and label-encodes categories.
2. **Partition** — for `L` iterations randomly select `m` features, draw `K‑1` cut-points (uniform, quantile, or KDE peaks), and assign each sample a **cell ID**.
3. **Embed** — the result is an `n × L` integer matrix `E`. Two points that land in the same cell are similar.
4. **Cluster** — Hamming distance on `E` approximates true similarity. Any sklearn clusterer can consume `E` directly, a precomputed distance matrix, or a sparse weighted one-hot expansion of `E`.

The full algorithm description with diagrams is in [ALGORITHM.md](ALGORITHM.md).

---

## Quick start

### Basic clustering

```python
from forest_clustering import ForestClusterer
from sklearn.cluster import KMeans

fc = ForestClusterer(
    n_iterations=200,
    n_bins=3,
    quantile_cuts=True,          # robust to outliers
    corr_threshold=0.9,          # down-weight correlated duplicates
    clusterer=KMeans(n_clusters=5, n_init="auto", random_state=0),
    random_state=42,
)

labels = fc.fit_predict(df)      # DataFrame or ndarray
```

### Graph community detection (Louvain & Leiden)

Graph-based community detection is recommended when the number of clusters is unknown and the dataset is large (`n > 12 000`), because it uses a sparse kNN graph instead of a dense `O(n²)` distance matrix.

#### String shortcuts (simplest)

```python
# Louvain — works with the [graph] extra
fc = ForestClusterer(
    n_iterations=300,
    clusterer='louvain:k=20,resolution=1.2',
    random_state=42,
)
labels = fc.fit_predict(df)

# Leiden — needs [leiden] extra; usually faster and avoids badly-connected communities
fc = ForestClusterer(
    n_iterations=300,
    clusterer='leiden:k=20,resolution=1.5',
    random_state=42,
)
labels = fc.fit_predict(df)
```

#### Direct GraphLouvainClusterer (full control)

```python
from forest_clustering import GraphLouvainClusterer

# Louvain with custom parameters
graph_clf = GraphLouvainClusterer(
    n_neighbors=15,
    resolution=1.0,
    weight_transform='exp',      # 'exp', 'linear', 'inverse'
    noise_strategy='mark',       # 'mark' (-1), 'merge', 'singleton'
    mutual_knn=False,            # True for stricter connectivity
    community_method='louvain',
    random_state=42,
)

fc = ForestClusterer(
    n_iterations=300,
    clusterer=graph_clf,
    random_state=42,
)
labels = fc.fit_predict(df)
```

#### Leiden with aggressive resolution (more, smaller clusters)

```python
from forest_clustering import GraphLouvainClusterer

graph_clf = GraphLouvainClusterer(
    n_neighbors=20,
    resolution=2.0,              # higher → more communities
    community_method='leiden',
    random_state=42,
)

fc = ForestClusterer(
    n_iterations=300,
    clusterer=graph_clf,
    random_state=42,
)
labels = fc.fit_predict(df)
```

#### Matrix-free graph clustering on embedding (no dense distance matrix)

```python
from forest_clustering import ForestClusterer, GraphLouvainClusterer

# Step 1: build embedding only
fc = ForestClusterer(
    n_iterations=300,
    n_bins=3,
    random_state=42,
)
fc.fit(df)
E = fc.get_embedding()           # (n, L) int64

# Step 2: cluster on embedding directly (matrix-free)
graph_clf = GraphLouvainClusterer(
    n_neighbors=15,
    resolution=1.0,
    community_method='leiden',
    random_state=42,
)
graph_clf.fit_embedding(E, method='auto')   # 'auto' | 'knn' | 'banding'
labels = graph_clf.labels_
```

#### Manual kNN graph construction and symmetrization

```python
from forest_clustering import batched_hamming_knn, symmetrize_knn
from forest_clustering import GraphLouvainClusterer

# Build embedding
fc = ForestClusterer(n_iterations=200, random_state=42)
E = fc.fit_transform(df)

# Exact batched kNN graph (Hamming distances)
G = batched_hamming_knn(E, k=15)

# Symmetrize: A_bar[i,j] = max(A[i,j], A[j,i])
G_sym = symmetrize_knn(G)

# Run Louvain on the symmetrized graph
graph_clf = GraphLouvainClusterer(
    n_neighbors=15,
    resolution=1.0,
    community_method='louvain',
    random_state=42,
)
# Convert to distance matrix or use fit_embedding
graph_clf.fit_embedding(E, method='knn')
labels = graph_clf.labels_
```

#### LSH banding for very large n (sub-quadratic, bounded memory)

```python
from forest_clustering import lsh_banding_knn, GraphLouvainClusterer

# For n > 20 000, banding avoids O(n²) memory
G = lsh_banding_knn(E, k=15, band_size='auto', max_bucket=150)

graph_clf = GraphLouvainClusterer(
    n_neighbors=15,
    resolution=1.0,
    community_method='leiden',
    random_state=42,
)
graph_clf.fit_embedding(E, method='banding')  # or pass G manually
labels = graph_clf.labels_
```

#### Noise handling strategies

```python
# 'mark' — singleton communities become -1 (noise)
fc = ForestClusterer(
    clusterer='louvain:k=15,resolution=1.0',
    random_state=42,
)
labels = fc.fit_predict(df)
noise_mask = labels == -1

# 'merge' — singletons merged into nearest non-singleton community
graph_clf = GraphLouvainClusterer(
    noise_strategy='merge',
    community_method='louvain',
)

# 'singleton' — keep singletons as separate 1-point clusters
graph_clf = GraphLouvainClusterer(
    noise_strategy='singleton',
    community_method='louvain',
)
```

### Advanced: adaptive bins + correlation-aware selection + KDE peaks

```python
fc = ForestClusterer(
    n_iterations=300,
    adaptive_bins=True,
    min_bins=2,
    max_bins=10,
    correlation_aware=True,
    corr_group_threshold=0.7,
    cut_strategy='kde_peaks',   # density-aware cut-points
    clusterer=KMeans(n_clusters=5, n_init="auto"),
    random_state=42,
)
labels = fc.fit_predict(df)
```

### Contrastive splits (learned trees instead of random cuts)

```python
fc = ForestClusterer(
    n_iterations=100,
    contrastive=True,           # contrastive tree building per iteration
    n_bins=3,
    clusterer=KMeans(n_clusters=3, n_init="auto"),
    random_state=42,
)
labels = fc.fit_predict(df)
```

### Iteration weighting (entropy / inverse-Gini)

```python
fc = ForestClusterer(
    n_iterations=200,
    iteration_weighting='entropy',      # or 'inverse_gini'
    weight_temperature=0.5,             # < 1 sharpens, > 1 softens
    clusterer=KMeans(n_clusters=3, n_init="auto"),
    random_state=42,
)
labels = fc.fit_predict(df)
```

### Online / incremental learning

```python
fc = ForestClusterer(
    n_iterations=200,
    partial_fit_strategy='drift',     # 'drift', 'periodic', 'never'
    partial_fit_drift_threshold=0.3,
    partial_fit_rebuild_threshold=0.3,
    random_state=42,
)
fc.fit(X_first)
fc.partial_fit(X_new)   # detects drift, rebuilds specs if needed
```

### Permutation feature importance

```python
fc = ForestClusterer(
    n_iterations=200,
    compute_importance=True,
    importance_repeats=5,
    importance_metric='silhouette',
    random_state=42,
)
fc.fit(df)
print(fc.get_feature_importances(detailed=True))
```

### Preflight clusterability check

```python
from forest_clustering import hopkins_statistic, gap_statistic, clusterability_test

report = clusterability_test(X, method='both', random_state=42)
print(report['recommendation'])
# "proceed with clustering"  or  "no significant cluster structure detected"
```

### Statistical significance of clusters

```python
from forest_clustering import cluster_significance, permutation_test_ari

# Per-cluster silhouette significance
sig = cluster_significance(X, labels, n_bootstrap=100, correction_method='bonferroni')
print(f"Significant clusters: {sig['significant_clusters']}")

# ARI significance vs ground truth
res = permutation_test_ari(y_true, labels, n_permutations=1000)
print(f"ARI={res['ari_observed']:.3f}, p={res['p_value']:.4f}")
```

### Transform new data

```python
E_new = fc.transform(X_new)              # (n_new, L) embedding
D_new = fc.pairwise_distance(Y=X_new)  # (n_train, n_new) Hamming distance train→new
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_iterations` | `200` | Number of random partitioning iterations `L`. More → more stable embeddings. |
| `n_features` | `"sqrt"` | Features selected per iteration: `int`, `float` fraction, `"sqrt"`, `"log2"`. |
| `n_bins` | `3` | Default number of bins per feature per iteration `K`. |
| `clusterer` | `None` | Any sklearn-compatible estimator, or strings `'louvain'`, `'leiden'` with optional params (e.g. `'leiden:k=20,resolution=1.5'`). |
| `corr_threshold` | `0.7` | Spearman \|r\| threshold for grouping correlated features. `None` disables. |
| `corr_sample_size` | `10_000` | Rows sampled for correlation estimation. |
| `feature_types` | `None` | Override auto-detection: `{col: "numerical"\|"categorical"}`. |
| `cat_threshold` | `10` | Integer columns with ≤ this many unique values are treated as categorical. |
| `quantile_cuts` | `False` | Sample cut-points from empirical quantiles (outlier robust). |
| `cut_strategy` | `"uniform"` | `"uniform"`, `"quantile"`, or `"kde_peaks"` (density-aware). |
| `kde_params` | `None` | Dict of overrides for KDE peak detection (bandwidth, grid_resolution, …). |
| `adaptive_bins` | `False` | Compute optimal `K` per feature from spread and cardinality. |
| `min_bins` / `max_bins` | `2` / `10` | Bounds when `adaptive_bins=True`. |
| `correlation_aware` | `False` | Ensure at most one feature per correlated group is selected per iteration. |
| `corr_group_threshold` | `0.7` | Pearson threshold for grouping in correlation-aware mode. |
| `iteration_weighting` | `"uniform"` | `"uniform"`, `"entropy"`, `"inverse_gini"`. |
| `weight_temperature` | `1.0` | Temperature for sharpening / softening iteration weights. |
| `contrastive` | `False` | Use contrastive trees instead of random cuts. |
| `compute_importance` | `False` | Compute permutation feature importance after fit. |
| `auto_feature_types` | `"naive"` | `"naive"` (dtype-only) or `"smart"` (heuristic ID / cardinality detection). |
| `partial_fit_strategy` | `"drift"` | `"drift"`, `"periodic"`, `"never"`. |
| `n_jobs` | `-1` | Parallelism for embedding computation. |
| `random_state` | `None` | Seed for reproducibility. |

See docstrings for the full parameter list (e.g. `partial_fit_max_samples`, `importance_repeats`, etc.).

---

## Hyperparameter guidelines

| Goal | Recommendation |
|---|---|
| Fast prototype | `n_iterations=50`, `n_bins=3` |
| Balanced quality / speed | `n_iterations=200`, `n_bins=3` (default) |
| High stability | `n_iterations=500`, `n_bins=4` |
| Outlier-heavy data | `quantile_cuts=True` or `cut_strategy='kde_peaks'` |
| Many correlated features | `corr_threshold=0.8–0.9`, `correlation_aware=True` |
| Unknown K, noise present | `clusterer='louvain'` or `DBSCAN(metric='hamming')` |
| Very large `n` (> 50 K) | `clusterer='louvain'` (LSH-banding kNN, no dense matrix) |
| Mixed-type data | leave `feature_types=None`, use `auto_feature_types='smart'` |

---

## Utilities

```python
# Raw embedding
E = fc.get_embedding()                 # (n, L) int64

# Pairwise Hamming distance (chunked automatically for large n)
D = fc.pairwise_distance()              # (n, n) float32
D_cross = fc.pairwise_distance(Y=X_new) # (n, n_new) train→new Hamming

# Iteration weights
w = fc.get_iteration_weights()         # (L,) float64, mean = 1.0

# Drift report (after partial_fit)
report = fc.get_drift_report()
```

---

## License

MIT
