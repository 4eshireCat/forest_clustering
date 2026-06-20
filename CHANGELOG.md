# Changelog

## 0.4.0 — Leiden backend, API/docstring audit, small-`n` parity

### Added
- **Leiden community detection** (`leidenalg` + `igraph`) as an alternative to
  Louvain: `GraphLouvainClusterer(community_method='leiden')` and the
  `ForestClusterer` string shortcuts `clusterer='leiden'` /
  `'leiden:k=20,resolution=1.5'`. Installs via the optional `leiden` extra
  (`pip install forest-clustering[leiden]`).
- `auto_band_size` is now exported from the package root.

### Changed
- **Graph clusterers are size-aware.** For `n <= 12000` the Louvain/Leiden route
  uses the exact dense Hamming distance matrix (cheap at this scale and
  identical to the classic behaviour, including tie-breaking); above it the
  matrix-free LSH-banding kNN graph is used. This guarantees small-dataset
  results are unchanged from earlier versions while keeping large-`n` scalable.
- `networkx` is now imported lazily (only when `community_method='louvain'`), so
  a Leiden-only install does not require it. Declared as the optional `graph`
  extra.
- Docstring/API audit: rewrote `GraphLouvainClusterer`'s docstring, documented
  the `clusterer` string shortcuts on `ForestClusterer`, and filled previously
  missing public docstrings (`fit`, `fit_predict`, `get_embedding`).

- For `n <= 12000` the centroid clusterers also use the dense weighted one-hot
  (sparse CSR only above the threshold), so small-data KMeans results are exact
  too.

### Validation
- Regression-checked against the pre-refactor version on small samples (1500
  rows) of Adult, Nursery, Car, Mushroom and Tic-tac-toe: **bit-for-bit identical
  ARI/NMI on all five for both the KMeans and Louvain paths.**

## 0.3.0 — Scalable large-`n` paths (no dense distance matrix)

The dense `n x n` distance matrix is no longer on the critical path for the
common large-`n` algorithms; it remains available as a small-`n` convenience and
for genuinely `precomputed`-only estimators.

### Added
- **Sparse weighted one-hot features** (`forest_clustering.sparse_features`,
  `weighted_onehot_features`). The cell-id embedding is encoded as a sparse CSR
  matrix with exactly `L` non-zeros per row (`O(n*L)` memory, independent of
  per-column cardinality), where squared Euclidean distance equals the weighted
  Hamming distance. Centroid estimators (KMeans / MiniBatchKMeans / Birch) now
  cluster directly on this representation. This also removes the dense one-hot
  out-of-memory failure in the default KMeans path.
- **LSH banding kNN graph** (`forest_clustering.lsh_graph.lsh_banding_knn`) — a
  fully vectorised, sub-quadratic, drop-in replacement for
  `batched_hamming_knn`. Candidate neighbours come from shared per-band cell-id
  tuples; exact Hamming is computed only on candidates. Memory is `O(n*c)`,
  never `O(n^2)`, and independent of per-column cardinality.
- **`auto_band_size`** and `band_size='auto'`: picks the smallest band size that
  keeps the bucket-collision load bounded, adapting to data entropy.
- `GraphLouvainClusterer.fit_embedding(..., method='auto'|'knn'|'banding')`
  builds its graph straight from the embedding; `'auto'` switches to banding
  above `banding_threshold` rows.
- `networkx` declared as the optional `graph` extra
  (`pip install forest-clustering[graph]`).

### Changed
- `ForestClusterer` Louvain paths build a sparse kNN graph from the embedding
  instead of a dense distance matrix.
- Banding internals are fully vectorised (compact-code factorisation,
  triangular-inverse pair enumeration, single packed-key top-k) — roughly 3x
  faster graph construction with bounded memory.

### Fixed
- `GraphLouvainClusterer._run_louvain_on_knn` cast kNN distances to float before
  the similarity transform, fixing an unsigned-integer overflow that turned
  `exp(-d^2/...)` into `inf` and crashed Louvain.
- Updated a stale partitioner test that assumed the obsolete positional
  `K**M` cell-id bound (cell ids are 52-bit hashes).

### Validation (UCI Adult, 48,842 rows)
- Centroid: old dense one-hot needs ~7 GB and fails; sparse CSR clusters all
  rows in ~2 s at ~430 MB (60x smaller features).
- Graph: old exact-kNN needs ~9.8 GB and fails; LSH banding builds in ~7 s at
  ~1.1 GB with kNN recall@15 ≈ 0.94 and exact-vs-banding label agreement
  ARI ≈ 0.89.
