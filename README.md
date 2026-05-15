# forest-clustering

**Random-partition similarity clustering for mixed-type tabular data.**

`forest-clustering` builds a compact binary embedding from random feature partitions and then applies any sklearn-compatible clustering algorithm to that embedding. It handles numerical, categorical, and mixed data natively, with built-in correlation-based feature weighting and outlier-robust cut-point generation.

## How it works

Each of the *L* iterations:
1. Samples *m* features (weighted by inverse correlation-group size).
2. Draws *K−1* random cut-points per numerical feature (uniform or quantile-based).
3. Assigns each sample a mixed-radix *cell ID* based on which bin it falls in for each selected feature.

The result is an *n × L* integer embedding. Two samples that consistently land in the same cell are similar; Hamming distance on this embedding approximates the true similarity.

```
Hamming(i, j) = (1/L) · Σ_l [ E[i,l] ≠ E[j,l] ]  ∈ [0, 1]
```

See [ALGORITHM.md](ALGORITHM.md) for a detailed description with diagrams.

## Installation

```bash
pip install forest-clustering
```

Python ≥ 3.10 required.

## Quick start

```python
from forest_clustering import ForestClusterer
from sklearn.cluster import KMeans

fc = ForestClusterer(
    n_iterations=200,
    n_bins=3,
    quantile_cuts=True,       # robust to outliers
    corr_threshold=0.9,       # down-weight correlated features
    clusterer=KMeans(n_clusters=5, n_init="auto", random_state=0),
    random_state=42,
)

labels = fc.fit_predict(df)   # works with DataFrame or ndarray
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_iterations` | 200 | Number of random partitioning iterations *L*. More → more stable. |
| `n_bins` | 3 | Number of bins per feature per iteration *K*. |
| `n_features` | `"sqrt"` | Features selected per iteration: int, float fraction, `"sqrt"`, `"log2"`. |
| `quantile_cuts` | `False` | Sample cut-points from empirical quantiles (robust to outliers). |
| `corr_threshold` | 0.7 | Spearman \|r\| threshold for grouping correlated features. `None` disables. |
| `corr_sample_size` | 10 000 | Rows used to estimate feature correlations. |
| `clusterer` | `DBSCAN(metric="hamming")` | Any sklearn-compatible `fit_predict` estimator. |
| `feature_types` | `None` | Override type detection: `{col: "numerical"\|"categorical"}`. |
| `cat_threshold` | 10 | Numerical columns with ≤ this many unique values → treated as categorical. |
| `n_jobs` | −1 | Parallelism for embedding computation (joblib). |
| `random_state` | `None` | Seed for reproducibility. |

## Downstream clustering algorithms

| Algorithm | When to use |
|---|---|
| `KMeans(n_clusters=K)` on embedding | Known K, any n — fast and stable |
| `AgglomerativeClustering(metric="precomputed")` | n ≤ 50 K, non-spherical clusters |
| `DBSCAN(metric="hamming")` on embedding | Unknown K, need outlier detection |
| `HDBSCAN(metric="precomputed")` | Variable-density clusters (pass `.astype(float64)`) |
| `MiniBatchKMeans` on embedding | n > 100 K |

```python
from sklearn.cluster import AgglomerativeClustering

fc = ForestClusterer(
    n_iterations=300,
    clusterer=AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average"),
)
labels = fc.fit_predict(X)
```

## Utilities

```python
# Get the raw n×L embedding
E = fc.get_embedding()

# Pairwise Hamming distance matrix (chunked for large n)
D = fc.pairwise_distance()

# Transform new data using fitted partition specs
E_new = fc.transform(X_new)
```

## Hyperparameter guidelines

| Goal | Recommendation |
|---|---|
| Fast prototype | `n_iterations=50`, `n_bins=3` |
| Balanced quality/speed | `n_iterations=200`, `n_bins=3` (default) |
| High stability | `n_iterations=500`, `n_bins=4` |
| Outlier-heavy data | `quantile_cuts=True` |
| Many correlated features | `corr_threshold=0.8–0.9` |

## License

MIT
