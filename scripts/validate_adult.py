"""Validate the two new scalable paths on the UCI Adult dataset.

Compares the OLD dense paths against the NEW sparse/banding paths on:
  (1) clustering quality   -- ARI / NMI vs the income label, silhouette
  (2) peak memory          -- tracemalloc around the cluster step
  (3) wall-clock time

Path 1 (centroid): dense one-hot KMeans  vs  sparse CSR KMeans
Path 2 (graph):    dense-matrix Louvain  vs  LSH-banding Louvain
"""
import time, tracemalloc, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
warnings.filterwarnings("ignore")

from forest_clustering import ForestClusterer
from forest_clustering.sparse_features import weighted_onehot_features
from forest_clustering.graph_clustering import GraphLouvainClusterer

# ── Load & encode Adult ────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/adult.csv")
y = (df["income"].astype(str).str.contains(">50K")).astype(int).values
X = df.drop(columns=["income"]).copy()
num = X.select_dtypes("number").columns
cat = X.select_dtypes("object").columns
X[cat] = X[cat].fillna("NA")
Xnum = StandardScaler().fit_transform(X[num])
Xcat = OrdinalEncoder().fit_transform(X[cat])
Xall = np.hstack([Xnum, Xcat]).astype(np.float64)
print(f"Adult: {Xall.shape[0]} rows x {Xall.shape[1]} features | "
      f">50K base rate = {y.mean():.3f}")

def quality(name, labels, t, peak_mb, ref=Xall):
    m = labels >= 0
    nlab = len(np.unique(labels[m]))
    ari = adjusted_rand_score(y[m], labels[m]) if m.sum() > 1 else float("nan")
    nmi = normalized_mutual_info_score(y[m], labels[m]) if m.sum() > 1 else float("nan")
    try:
        sil = silhouette_score(ref[m], labels[m]) if 1 < nlab < m.sum() else float("nan")
    except Exception:
        sil = float("nan")
    print(f"  {name:<26} clusters={nlab:<3} ARI={ari:+.3f} NMI={nmi:.3f} "
          f"silhouette={sil:+.3f} | {t:6.1f}s peak={peak_mb:7.1f}MB "
          f"noise={100*(~m).mean():.1f}%")
    return ari

# Shared embedding so both paths cluster identical features.
N = Xall.shape[0]
print(f"\nComputing forest embedding on all {N} rows ...")
fc = ForestClusterer(n_iterations=200, n_bins=4, n_clusters=2,
                     corr_threshold=None, random_state=0)
fc.fit_embedding_only = None  # marker; we call internals below
fc._encode_and_build_specs = None
# Build embedding via the public fit path but stop at embedding:
t0 = time.time()
fc.fit(Xall)                      # full default pipeline (now sparse KMeans)
E = fc.embedding_
print(f"  embedding shape={E.shape}  built+clustered in {time.time()-t0:.1f}s")
w = getattr(fc, "iteration_weights_", None)

# ── PATH 1: centroid — dense one-hot vs sparse CSR ─────────────────────────
from sklearn.cluster import KMeans
print("\n[Path 1] Centroid KMeans on weighted one-hot features")
# old: dense  (expected to OOM on Adult -- that is the result)
tracemalloc.start(); t = time.time()
lab_dense = None
try:
    Pd = weighted_onehot_features(E, weights=w, sparse_output=False)
    lab_dense = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(Pd)
    peak_d = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    quality("dense one-hot (OLD)", lab_dense, time.time() - t, peak_d)
    dense_width = Pd.shape[1]; dense_mb = Pd.nbytes / 1e6
except MemoryError:
    tracemalloc.stop()
    width = sum(np.unique(E[:, l]).size for l in range(E.shape[1]))
    dense_width = width; dense_mb = E.shape[0] * width * 8 / 1e6
    print(f"  {'dense one-hot (OLD)':<26} *** MemoryError: needs "
          f"{dense_mb/1e3:.1f} GB for ({E.shape[0]}, {width}) float64 -- INFEASIBLE ***")
# new: sparse
tracemalloc.start(); t = time.time()
Ps = weighted_onehot_features(E, weights=w, sparse_output=True)
lab_sparse = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(Ps)
peak_s = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
quality("sparse CSR (NEW)", lab_sparse, time.time() - t, peak_s)
csr_mb = (Ps.data.nbytes + Ps.indices.nbytes + Ps.indptr.nbytes) / 1e6
if lab_dense is not None:
    agree = (lab_dense == lab_sparse).mean()
    agree = max(agree, (lab_dense != lab_sparse).mean())
    print(f"  -> identical assignment (label-perm invariant, k=2): {agree:.4f}")
    print(f"  -> peak memory: {peak_d:.0f}MB -> {peak_s:.0f}MB  ({peak_d/max(peak_s,1e-9):.1f}x lower)")
else:
    print(f"  -> OLD path infeasible; NEW path clustered all {N} rows fine.")
print(f"  -> features: dense ({E.shape[0]}, {dense_width}) float64 = {dense_mb/1e3:.1f} GB "
      f"vs CSR nnz={Ps.nnz} = {csr_mb:.0f}MB  ({dense_mb/max(csr_mb,1e-9):.0f}x smaller)")

# ── PATH 2: graph — dense Hamming matrix Louvain vs LSH banding ────────────
print("\n[Path 2] Louvain: dense distance matrix vs LSH banding")
k = 15
# old: exact kNN built by brute-force batched Hamming (OOMs at large n)
tracemalloc.start(); t = time.time()
ari_knn = None
try:
    gc_knn = GraphLouvainClusterer(n_neighbors=k, random_state=0)
    gc_knn.fit_embedding(E, method="knn")
    peak_knn = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    ari_knn = quality("exact kNN Louvain (OLD)", gc_knn.labels_, time.time() - t, peak_knn)
except MemoryError:
    tracemalloc.stop()
    print(f"  {'exact kNN Louvain (OLD)':<26} *** MemoryError: brute-force "
          f"batch needs ~{1000*N*E.shape[1]/1e9:.1f} GB -- INFEASIBLE ***")
# new: banding
tracemalloc.start(); t = time.time()
gc_band = GraphLouvainClusterer(n_neighbors=k, random_state=0)
gc_band.fit_embedding(E, method="banding", band_size="auto")
peak_band = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
ari_band = quality("LSH banding Louvain (NEW)", gc_band.labels_, time.time() - t, peak_band)

# recall of banding kNN vs exact (on a random subset for tractability)
rng = np.random.default_rng(0)
sub = rng.choice(N, size=2000, replace=False)
Es = E[sub]
D = (Es[:, None, :] != Es[None, :, :]).sum(2); np.fill_diagonal(D, E.shape[1] + 1)
true = np.argsort(D, axis=1)[:, :k]
from forest_clustering.lsh_graph import lsh_banding_knn
Gb = lsh_banding_knn(Es, k=k, band_size="auto", random_state=0).tocsr()
recall = np.mean([len(set(true[i]) & set(Gb[i].indices)) / k for i in range(len(sub))])
print(f"  -> banding kNN recall@{k} (2k subset) = {recall:.3f}")
if ari_knn is not None:
    print(f"  -> peak memory: {peak_knn:.0f}MB -> {peak_band:.0f}MB")
else:
    print(f"  -> OLD exact path infeasible at n={N}; NEW banding peak={peak_band:.0f}MB")
print(f"  -> dense full distance matrix would need "
      f"{N*N*4/1e9:.1f} GB (float32) -- neither path builds it")

# Head-to-head on a subset where BOTH the exact and banding paths are feasible,
# to show banding preserves Louvain quality (not just memory).
print("\n[Path 2b] Exact vs banding on an 8000-row subset (both feasible)")
sub2 = rng.choice(N, size=8000, replace=False)
Esub = E[sub2]; ysub = y[sub2]
ge = GraphLouvainClusterer(n_neighbors=k, random_state=0); ge.fit_embedding(Esub, method="knn")
gb = GraphLouvainClusterer(n_neighbors=k, random_state=0); gb.fit_embedding(Esub, method="banding", band_size="auto")
from sklearn.metrics import adjusted_rand_score as _ari
print(f"  exact kNN  : clusters={len(np.unique(ge.labels_[ge.labels_>=0]))} "
      f"ARI_income={_ari(ysub, ge.labels_):+.3f}")
print(f"  banding    : clusters={len(np.unique(gb.labels_[gb.labels_>=0]))} "
      f"ARI_income={_ari(ysub, gb.labels_):+.3f}")
print(f"  agreement exact-vs-banding (ARI of labelings) = {_ari(ge.labels_, gb.labels_):+.3f}")
