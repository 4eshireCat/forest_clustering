# Algorithms

This document explains the algorithms implemented by `forest-clustering`, their mathematical interpretation, and the main implementation choices.

## 1. Random-partition clustering: `ForestClusterer`

`ForestClusterer` is a random-partition similarity method. It does not train supervised decision trees. Instead, it repeatedly samples a small subset of features, partitions each selected feature into bins, and encodes every row by the cell it falls into.

For iteration `l`:

1. Select `m` features from `p` input columns.
2. For every selected numerical feature, draw `K - 1` cut points.
3. For categorical features, group categories into bins.
4. Assign each row to a mixed-radix cell id.

After `L` iterations, the embedding is an integer matrix:

```text
E in N^(n x L)
```

where `E[i, l]` is the random cell id of sample `i` at iteration `l`.

The default dissimilarity is Hamming distance over random partition ids:

```text
D(i, j) = (1 / L) * sum_l 1[E[i, l] != E[j, l]]
S(i, j) = 1 - D(i, j)
```

`S(i, j)` estimates how often two samples co-occur in the same random partition cell under the package's partition generator. It should be interpreted as a random-partition similarity, not as a universal ground-truth similarity.

### Numerical cut strategies

- `cut_strategy="uniform"`: cut points are sampled uniformly inside the observed feature range.
- `cut_strategy="quantile"` or `quantile_cuts=True`: cut points are sampled by drawing quantile probabilities uniformly and applying empirical quantiles. This is robust to heavy-tailed features and outliers.
- KDE helpers can be used to choose density-aware cut points.

### Correlation handling

Highly correlated features can make random partitions over-count one latent direction. `ForestClusterer` can down-weight correlated numerical feature groups. Label-encoded categorical features are intentionally excluded from Spearman-based weighting, because arbitrary category codes do not define an ordinal scale.

### Downstream clustering

The embedding can be consumed by any sklearn-compatible clusterer. Common choices:

- `KMeans` / `MiniBatchKMeans` on the embedding or weighted one-hot features.
- `AgglomerativeClustering(metric="precomputed")` on pairwise Hamming distance.
- `DBSCAN(metric="precomputed")` or HDBSCAN-style algorithms on distance matrices.
- Graph clustering after building a sparse nearest-neighbor graph from Hamming distances.

## 2. Breiman-style unsupervised random forest: `UnsupervisedRandomForestClusterer`

This estimator implements the classic unsupervised random-forest trick:

1. Build a synthetic null dataset `X_synth` with the same number of rows as `X`.
2. Label real rows as `1` and synthetic rows as `0`.
3. Train a `RandomForestClassifier` to discriminate real data from null data.
4. Transform real rows into forest leaf ids.
5. Define proximity as the fraction of trees where two rows land in the same leaf.

For trees `t = 1, ..., T` and leaf ids `L_t(i)`:

```text
P(i, j) = (1 / T) * sum_t 1[L_t(i) = L_t(j)]
D(i, j) = 1 - P(i, j)
```

The proximity is high when the forest repeatedly considers two rows similar under splits that separate real joint structure from synthetic noise.

### Synthetic null modes

- `synthetic="permute_marginals"`: independently permutes each column. This preserves one-dimensional marginal distributions but breaks cross-feature dependence.
- `synthetic="uniform_box"`: samples numerical columns uniformly from their observed range and categorical columns from observed categories. This is a stronger null and can be useful when marginal preservation is not desired.

### Downstream clustering safety

Raw leaf ids are nominal labels. Their numeric values are arbitrary. Therefore:

- clusterers with `metric="precomputed"` receive `1 - proximity`;
- other clusterers receive a sparse one-hot leaf embedding from `transform_onehot(X)`.

This avoids invalid Euclidean geometry over raw leaf numbers.

## 3. ExtraTrees proximity clustering: `ExtraTreesProximityClusterer`

`ExtraTreesProximityClusterer` uses the same synthetic-null procedure as URF, but trains an `ExtraTreesClassifier` instead of a standard random forest.

The proximity definition is the same:

```text
P(i, j) = fraction of ExtraTrees where i and j share a leaf
```

Compared with URF, ExtraTrees usually produces more randomized split boundaries. This can be faster and can reduce variance in some high-dimensional settings, but it can also be noisier when the synthetic discrimination task is weak.

Use it as a companion baseline to URF rather than a strict replacement.

## 4. Greedy unsupervised binary tree: `UnsupervisedBinaryTreeClusterer`

This estimator builds a single interpretable clustering tree.

At each step, it considers candidate binary splits and chooses the split with the largest reduction in within-node squared error. It recursively splits until it reaches the requested number of leaves/clusters or the stopping constraints.

For a node containing rows `A`, define within-node SSE:

```text
SSE(A) = sum_{i in A} ||x_i - mean(A)||^2
```

For candidate split `A -> A_left, A_right`, the gain is:

```text
gain = SSE(A) - SSE(A_left) - SSE(A_right)
```

The best split maximizes `gain`, subject to `min_samples_split`, `min_samples_leaf`, `max_depth`, and feature/threshold sampling constraints.

After fitting, each leaf is a cluster. `rules()` returns human-readable decision rules for each cluster.

## 5. Automatic selection: `AutoTreeClusterer`

`AutoTreeClusterer` is a meta-estimator. It does not introduce a new clustering geometry; instead, it automates a small model-selection loop over existing estimators.

For every candidate algorithm `a`, cluster count `k`, parameter-grid point `theta`, and restart `r`, it fits an estimator:

```text
M[a, k, theta, r].fit(X) -> labels[a, k, theta, r]
```

It then computes internal unsupervised scores. Since version 0.6.1, the default silhouette score is computed on a leak-safe scoring representation rather than blindly using each estimator's public `pairwise_distance()` output. This matters because some estimators, especially the interpretable binary tree, define public proximity from final leaf assignments. Scoring such a model on a distance matrix derived from the same final labels would be self-confirming.

Default scoring representations are:

```text
ForestClusterer                 -> weighted one-hot random-partition features
UnsupervisedRandomForestClusterer -> one-hot forest leaf features
ExtraTreesProximityClusterer      -> one-hot ExtraTrees leaf features
UnsupervisedBinaryTreeClusterer   -> fitted preprocessed feature matrix
```

The default silhouette objective is therefore:

```text
silhouette(labels, Phi), where Phi is independent of the final labels
```

For multiple restarts of the same `(a, k, theta)` candidate, stability is computed with the adjusted Rand index between restart labelings:

```text
stability = mean_{r1 < r2} ARI(labels_r1, labels_r2)
```

The default `scoring="combined"` objective is:

```text
score = mean_silhouette + stability_weight * stability
```

Other supported objectives are `silhouette`, `calinski_harabasz`, `davies_bouldin`, and `stability`. `scoring_space="proximity"` is kept as an explicit compatibility mode, but it should not be used with estimators whose distance is constructed directly from final cluster labels. After search, the best fitted estimator is stored in `best_estimator_`; `labels_`, `transform`, `similarity_matrix`, and `pairwise_distance` delegate to it.

This procedure is intentionally simple. It is useful because many failures in practical clustering come from choosing a poor cluster count, weak default seed or mismatched algorithm family. It is not a proof that the selected clustering is globally correct.

## 6. Distance, similarity, and proximity APIs

The package uses these conventions:

```text
proximity_matrix: high means similar, diagonal is 1
similarity_matrix: alias or equivalent high-similarity matrix
distance_matrix: low means similar, diagonal is 0
```

For tree-proximity estimators:

```text
distance = 1 - proximity
```

For `ForestClusterer`:

```text
distance = Hamming(random_partition_embedding)
similarity = 1 - distance
```

## 7. Mixed data preprocessing

Tree-proximity estimators use a sklearn preprocessing pipeline:

- numeric columns: imputation, then numeric values are passed to the forest;
- categorical columns: imputation, then one-hot encoding;
- all-missing columns are handled robustly where supported by the installed sklearn version.

This makes the new tree estimators usable on pandas DataFrames with missing values and mixed dtypes.

## 8. Choosing an algorithm

| Situation | Recommended estimator |
|---|---|
| You want a fast robust baseline | `ForestClusterer` |
| You want canonical random-forest proximity | `UnsupervisedRandomForestClusterer` |
| You want a faster/more randomized proximity baseline | `ExtraTreesProximityClusterer` |
| You need interpretable if/then cluster rules | `UnsupervisedBinaryTreeClusterer` |
| You do not know algorithm or `k` | `AutoTreeClusterer(k_range=range(2, 8), scoring="combined")` |
| You need sklearn pipeline integration | `ForestTransformer` + downstream estimator |
| You have large data | Random-partition embedding + MiniBatchKMeans or graph/LSH helpers |

## 9. Limitations

- Clustering quality is data-dependent; there is no universally correct unsupervised objective.
- URF/ExtraTrees proximities depend on the synthetic null distribution.
- Proximity matrices are `O(n^2)` memory; use embeddings or graph approximations for large datasets.
- Binary tree clustering is interpretable but less expressive than an ensemble.
- Categorical split semantics depend on preprocessing and category frequency.
- `AutoTreeClusterer` uses internal validation metrics; these are useful diagnostics, not ground-truth labels.

## 10. Reproducibility

Set `random_state` on the estimator and downstream clusterer when reproducible output is required.

```python
model = UnsupervisedRandomForestClusterer(
    n_estimators=300,
    n_clusters=3,
    random_state=42,
)
```

## Cluster-label classifiers and surrogate explanations

Version 0.7.0 adds a supervised layer that can be fitted after clustering.  The workflow is:

1. Fit a clustering estimator and obtain labels `z`.
2. Train a supervised classifier `g(x) -> z` using those labels as pseudo-targets.
3. Use the classifier for inductive assignment, confidence estimates and explanations.

This does not change the original clustering objective.  The classifier is a *surrogate* for the
cluster assignment function.  Its metrics are therefore fidelity metrics: they describe how well
`g` reproduces the existing cluster labels, not whether the clusters are semantically correct.

### ClusterLabelClassifier

`ClusterLabelClassifier` fits a default balanced random forest, or any user-supplied sklearn
classifier, to reproduce cluster labels.  It first converts mixed-type tabular data to numeric
features using the same robust preprocessing used by the tree-clustering estimators:

- missing indicators;
- rare-category grouping;
- numeric-string coercion;
- median imputation and scaling for numeric columns;
- one-hot encoding for categorical columns.

It exposes:

- `labels_`: labels produced by the underlying clusterer;
- `predict(X_new)`: assign new rows to clusters;
- `predict_proba(X_new)`: cluster-label probabilities when the classifier supports them;
- `fidelity_report_`: out-of-fold accuracy, balanced accuracy, macro F1 and confusion matrix;
- `train_fidelity_report_`: in-sample reproduction metrics;
- `cluster_profile()` and `explain_clusters()` for readable cluster summaries;
- plotting methods for cluster sizes, projection plots, feature importances and fidelity matrices.

The optional rejection mode is useful in production.  With `unknown_policy="reject"` and a
`confidence_threshold`, low-confidence predictions are returned as `-1` instead of being forced
into a known cluster.

### ClusterSurrogateTree

`ClusterSurrogateTree` fits a shallow `DecisionTreeClassifier` to cluster labels.  It is meant for
interpretability rather than maximum fidelity.  It provides:

- `export_text()` for a full sklearn-style tree dump;
- `extract_leaf_rules()` for root-to-leaf rules with sample counts and purity;
- `explain_rules()` for compact human-readable rules;
- `rules_dataframe()` for reporting;
- `plot_tree()` for visual inspection.

For one-hot categorical splits, rules near threshold `0.5` are phrased as presence/absence rather
than raw numeric dummy thresholds where possible.

### Recommended use

Use `ClusterLabelClassifier` when the goal is deployment or assigning future observations.  Use
`ClusterSurrogateTree` when the goal is to explain a segmentation to humans.  If the surrogate has
low out-of-fold fidelity, treat the clustering as difficult to express in the original feature space
and avoid over-interpreting simple rules.

## 0.8.0 Prototype sampling layer

The prototype layer reduces the number of rows before clustering while preserving a reversible mapping back to the original dataset.

Let `X = {x_i}_{i=1}^n`. A sampler produces:

- prototype rows `P = {p_j}_{j=1}^m`, `m <= n`;
- weights `w_j`, where `w_j` is the number of original rows represented by prototype `j`;
- inverse assignment `a_i in {0, ..., m-1}` so that labels can be expanded by `label_i = label^P_{a_i}`.

### Leaf-signature prototypes

For mixed tabular data, `PrototypeSampler(method="leaf_signature")` fits a `ForestTransformer` and obtains a random-partition signature matrix `E in Z^{n x L}`.  A prefix or full signature is used as a bucket key. Rows in the same bucket are considered redundant under the selected random-partition view.

For each bucket the sampler chooses a representative row:

- `first`: deterministic first row;
- `medoid`: row nearest to the bucket center in signature/feature space;
- `centroid`: only meaningful for numeric/BIRCH workflows; mixed leaf-signature data keeps valid original rows instead.

Small buckets can be preserved as individual prototypes with `preserve_rare=True`. This protects small clusters and outliers from being removed by compression.

### BIRCH prototypes

`PrototypeSampler(method="birch")` builds a robust numeric/categorical preprocessing pipeline and fits sklearn BIRCH.  Samples assigned to the same BIRCH subcluster are collapsed into a weighted representative. If the target prototype budget is lower than the natural number of subclusters, centers are deterministically merged by farthest-first representative selection and nearest-center assignment.

### SubsampledClusterer

`SubsampledClusterer` composes a sampler and a clusterer:

1. fit sampler on full data;
2. fit the clusterer on prototypes;
3. expand prototype labels to all original rows;
4. predict new rows by assigning them to their nearest fitted prototype.

This is a practical acceleration layer. It is not a new clustering objective, and it should be validated with compression diagnostics.

## Diagnostics and visualisation layer in 0.9.0

`ClusterDiagnosticsReport` is a post-fit diagnostic layer. It intentionally separates two spaces:

1. **Leak-safe feature space**: a robust encoded representation of the original data. This is used for internal metrics such as silhouette, Calinski-Harabasz and Davies-Bouldin.
2. **Model proximity space**: a clusterer's own similarity/proximity matrix when available. This is used for proximity heatmaps and block-separation diagnostics.

This separation avoids the scoring leakage problem fixed in 0.6.1: a distance matrix derived directly from final labels must not be used as evidence that those labels are good.

The diagnostics module includes:

- cluster size balance;
- negative silhouette rate;
- per-cluster profile contrasts;
- proximity block summaries;
- uncertain sample ranking using silhouette and affinity margin;
- health checks and cluster cards;
- stability analysis across repeated fits;
- model comparison with pairwise adjusted Rand agreement.

`StabilityAnalyzer` repeatedly clones an estimator, changes its random seed where possible, and compares labelings with ARI/NMI. This gives a practical robustness signal for stochastic algorithms.

`compare_clusterings()` fits several models, computes a shared diagnostics table and builds a pairwise ARI agreement matrix. Agreement across different algorithms is not proof of truth, but it is useful evidence that the discovered structure is not merely a random artifact.

When an ordinary sklearn baseline cannot consume mixed-type data, `compare_clusterings()` can retry the model on the same robust encoded feature matrix used for diagnostics. The resulting `input_space` column records whether the model used original data or encoded fallback.
