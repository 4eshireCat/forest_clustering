# forest-clustering patched fork

Local patched fork based on the PyPI `forest-clustering==0.4.0` package.

## Primary fixes

- `quantile_cuts=True` / `cut_strategy="quantile"` now draws cut points from uniformly sampled empirical quantile probabilities instead of re-sampling observed values as cut points.
- Correlation-based feature weights now ignore label-encoded categorical columns, because Spearman correlation on arbitrary category codes is not mathematically meaningful.
- `ForestClusterer.fit()` now rejects empty sample sets and zero-feature inputs with clear `ValueError`s.
- Added `ForestClusterer.similarity_matrix()` as the explicit `1 - Hamming` random-partition similarity API.
- Version marker changed to `0.4.1-patched` in `forest_clustering.__version__`.

## Test command

```bash
python -m pytest -q
```

## Notes

This is a conservative patch set. It does not rename the public estimator, rewrite `partial_fit`, or replace categorical association with Cramér's V / mutual information. Those are recommended next steps, but they are API-impacting or larger design changes.

## 0.4.2-patched: Breiman-style unsupervised random forest

Added `UnsupervisedRandomForestClusterer`, a sklearn-style estimator implementing Breiman-style unsupervised random forest clustering:

1. Build a synthetic null dataset from the observed table (`permute_marginals` or `uniform_box`).
2. Train a `RandomForestClassifier` to distinguish real rows from synthetic rows.
3. Use same-leaf co-occurrence across trees as a proximity matrix.
4. Cluster `1 - proximity` with average-linkage `AgglomerativeClustering` by default, or pass a custom sklearn-compatible downstream clusterer.

Public API:

```python
from forest_clustering import UnsupervisedRandomForestClusterer

est = UnsupervisedRandomForestClusterer(
    n_estimators=200,
    n_clusters=3,
    synthetic="permute_marginals",
    random_state=42,
)
labels = est.fit_predict(X)
proximity = est.proximity_matrix()
distance = est.pairwise_distance()
leaf_embedding = est.transform(X_new)
```

TDD coverage added:

- sklearn clone compatibility and `fit_predict` shape
- proximity symmetry, `[0, 1]` bounds, and unit diagonal
- distance/proximity complement
- reproducibility with fixed `random_state`
- transform/cross-proximity on new samples
- input validation for synthetic modes
- local Titanic smoke test

Smoke-test command used:

```bash
cd /mnt/data/forest_clustering_patched
PYTHONPATH=. pytest -q
# 10 passed
```

## 0.4.3-patched

Added the second tree-based clustering estimator:

- `ExtraTreesProximityClusterer`
  - sklearn-style API: `fit`, `fit_predict`, `transform`, `fit_transform`
  - proximity API: `proximity_matrix`, `similarity_matrix`, `pairwise_distance`
  - synthetic-null modes: `permute_marginals`, `uniform_box`
  - default downstream clustering: average-linkage agglomerative clustering on `1 - proximity`
  - supports mixed numeric/categorical pandas data through preprocessing pipeline

TDD additions:

- sklearn clone + fit/predict shape
- symmetric/unit-diagonal proximity
- distance complement check
- reproducibility with fixed `random_state`
- transform/cross-proximity on new samples
- invalid synthetic mode validation
- Titanic smoke test

Validation command:

```bash
PYTHONPATH=. pytest -q
```

Observed result in this environment:

```text
16 passed
```

## 0.4.4-patched

Added `UnsupervisedBinaryTreeClusterer`, a greedy CART-like unsupervised binary tree clustering estimator.

Key properties:
- sklearn-style API: `fit`, `fit_predict`, `predict`, `transform`, `fit_transform`.
- Leaf-equality proximity API: `proximity_matrix`, `similarity_matrix`, `pairwise_distance`.
- Mixed numeric/categorical preprocessing via sklearn `ColumnTransformer`.
- Human-readable `rules()` for leaf clusters.
- TDD coverage for API shape, proximity semantics, reproducibility, validation, cross-proximity, and Titanic smoke test.

Current full test suite: `22 passed`.

## 0.4.5-patched: self-review hardening pass

Maintainer review found and fixed several issues in the patched tree estimators and the original estimator API surface:

- Added shared `_tree_common.py` utilities to remove duplicated preprocessing/proximity/downstream-clustering code from URF and ExtraTrees.
- Fixed a mathematical downstream-clustering bug: non-precomputed clusterers now receive a one-hot leaf embedding instead of raw numeric leaf ids. Raw leaf ids are nominal labels; Euclidean distances between them are arbitrary and invalid.
- Added `transform_onehot(X)` to `UnsupervisedRandomForestClusterer` and `ExtraTreesProximityClusterer`.
- Made tree preprocessing robust to all-missing numeric/categorical columns by preserving empty features where sklearn supports it and by treating `None` / `pd.NA` consistently as missing values.
- Added sklearn-version compatibility helpers for `OneHotEncoder(sparse_output=...)` and `SimpleImputer(keep_empty_features=...)`.
- Reused the shared robust preprocessing path in `UnsupervisedBinaryTreeClusterer`.
- Moved `ForestClusterer` constructor validation into `fit()` for a more sklearn-like lightweight `__init__`.
- Synced `pyproject.toml` version and `forest_clustering.__version__`.

Additional regression coverage:

- Custom Euclidean downstream clusterer receives sparse one-hot leaf features for URF/ExtraTrees.
- All-missing mixed columns do not crash URF, ExtraTrees, or BinaryTree.
- `ForestClusterer(n_bins=0)` can be constructed and fails at `fit()`, matching sklearn estimator conventions.
- Package version metadata stays consistent.

Current full test suite: `27 passed`.
