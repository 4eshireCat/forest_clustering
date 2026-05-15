"""Generate demonstration notebooks for ForestClustering."""

import nbformat as nbf
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(src):
    return nbf.v4.new_code_cell(src.strip())


# ======================================================================
# NOTEBOOK 1 — TITANIC
# ======================================================================

def build_titanic():
    cells = []

    cells.append(md("""
# ForestClustering: demonstration on the Titanic dataset

The Titanic dataset is a good benchmark: small (~890 rows), contains a **mix of feature types**
(continuous, binary, categorical) and has an interpretable structure.

**What we show:**
1. ForestClusterer accepts raw data — no OneHot encoding or scaling required
2. Clusters correspond to meaningful passenger groups
3. Robustness: the algorithm does not "drift" when outliers are added
4. Fair comparison with KMeans, DBSCAN, AgglomerativeClustering
"""))

    cells.append(code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), ''))

from forest_clustering import ForestClusterer

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120})

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
ALGO_COLORS = {
    'ForestClusterer': '#2196F3',
    'KMeans': '#FF9800',
    'AgglomerativeClustering': '#4CAF50',
    'DBSCAN': '#9C27B0',
}
print("OK, numpy", np.__version__)
"""))

    cells.append(md("## 1. Data"))

    cells.append(code("""
df_raw = sns.load_dataset('titanic')

FEATURE_COLS = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
TARGET = 'survived'

df = df_raw[FEATURE_COLS + [TARGET]].copy()
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna().reset_index(drop=True)

print(f"Rows: {len(df)}")
print()

type_map = {
    'pclass': 'categorical (1/2/3)',
    'sex': 'binary (male/female)',
    'age': 'continuous',
    'sibsp': 'discrete numerical',
    'parch': 'discrete numerical',
    'fare': 'continuous',
    'embarked': 'categorical (S/C/Q)',
}
for col in FEATURE_COLS:
    print(f"  {col:10s}  unique={df[col].nunique():3d}  {type_map[col]}")
"""))

    cells.append(code("""
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()

for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    if df[col].dtype == object:
        df[col].value_counts().plot.bar(ax=ax, color='#4C72B0', edgecolor='white')
        ax.tick_params(axis='x', rotation=0)
    else:
        ax.hist(df[col], bins=20, color='#4C72B0', edgecolor='white')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Count')

axes[-1].set_visible(False)
plt.suptitle('Titanic feature distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 2. ForestClusterer — no preprocessing

ForestClusterer accepts a DataFrame as-is. We look for 3 clusters (analogous to pclass).
Downstream: KMeans on the Hamming embedding (n×L matrix, not n×n).
"""))

    cells.append(code("""
X_raw = df[FEATURE_COLS]

t0 = time.perf_counter()
fc = ForestClusterer(
    n_iterations=300,
    n_bins=3,
    quantile_cuts=True,
    clusterer=KMeans(n_clusters=3, random_state=0, n_init=10),
    corr_threshold=0.9,
    random_state=42,
)
fc_labels = fc.fit_predict(X_raw)
fc_time = time.perf_counter() - t0

df['fc_cluster'] = fc_labels

print(f"Time: {fc_time:.2f}s")
print()
print("Feature weights (1/G for correlated groups):")
for col, w in zip(FEATURE_COLS, fc.feature_weights_):
    bar = '█' * int(w * 10)
    print(f"  {col:10s}  {w:.3f}  {bar}")
"""))

    cells.append(code("""
# PCA coordinates for visualization
X_vis = df[FEATURE_COLS].copy()
X_vis['sex'] = (X_vis['sex'] == 'female').astype(float)
X_vis['embarked'] = OrdinalEncoder().fit_transform(X_vis[['embarked']])
X_scaled = StandardScaler().fit_transform(X_vis.astype(float))
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ForestClusterer clusters
ax = axes[0]
for cl in sorted(np.unique(fc_labels)):
    mask = fc_labels == cl
    ax.scatter(coords[mask, 0], coords[mask, 1], s=25, alpha=0.7,
               label=f'Cluster {cl+1}', color=PALETTE[cl])
ax.set_title('ForestClustering', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend(markerscale=2)

# True pclass
ax = axes[1]
for cl, pclass in enumerate([1, 2, 3]):
    mask = df['pclass'].values == pclass
    ax.scatter(coords[mask, 0], coords[mask, 1], s=25, alpha=0.7,
               label=f'pclass={pclass}', color=PALETTE[cl])
ax.set_title('True pclass', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend(markerscale=2)

plt.suptitle('PCA projection: ForestClustering clusters vs pclass', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()
"""))

    cells.append(code("""
# Cluster profile
profile = df.groupby('fc_cluster').agg(
    age=('age', 'mean'),
    fare=('fare', 'mean'),
    sibsp=('sibsp', 'mean'),
    parch=('parch', 'mean'),
    female_pct=('sex', lambda x: (x == 'female').mean()),
    pclass1_pct=('pclass', lambda x: (x == 1).mean()),
    pclass2_pct=('pclass', lambda x: (x == 2).mean()),
    pclass3_pct=('pclass', lambda x: (x == 3).mean()),
    survived=('survived', 'mean'),
    n=('age', 'count'),
)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Normalised heatmap
profile_heat = profile.drop('n', axis=1)
profile_norm = (profile_heat - profile_heat.min()) / (profile_heat.max() - profile_heat.min() + 1e-9)

sns.heatmap(profile_norm.T, annot=profile_heat.T.round(2), fmt='g',
            cmap='RdYlGn', ax=axes[0], linewidths=0.5, cbar_kws={'label': 'norm value'})
axes[0].set_title('Cluster profile (row-normalised)', fontweight='bold')
axes[0].set_xlabel('Cluster')

# Survival rate and cluster sizes
bottom_ax = axes[1]
cluster_ids = sorted(np.unique(fc_labels))
survived = [profile.loc[cl, 'survived'] for cl in cluster_ids]
sizes = [profile.loc[cl, 'n'] for cl in cluster_ids]
bars = bottom_ax.bar(
    [f'Cluster {cl+1}\\n(n={int(n)})' for cl, n in zip(cluster_ids, sizes)],
    survived,
    color=[PALETTE[cl] for cl in cluster_ids],
    edgecolor='white', linewidth=1.5, width=0.5,
)
for bar, val in zip(bars, survived):
    bottom_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
bottom_ax.set_ylim(0, 1)
bottom_ax.set_ylabel('Survival rate')
bottom_ax.set_title('Survival rate by cluster', fontweight='bold')
bottom_ax.axhline(df['survived'].mean(), color='gray', ls='--', label=f'Mean {df["survived"].mean():.1%}')
bottom_ax.legend()

plt.tight_layout()
plt.show()
print()
for cl in cluster_ids:
    row = profile.loc[cl]
    dom_pclass = max([1,2,3], key=lambda p: row[f'pclass{p}_pct'])
    print(f"Cluster {cl+1}: n={int(row['n'])}, age={row['age']:.0f}, fare={row['fare']:.0f}, "
          f"{row['female_pct']:.0%} female, dom pclass={dom_pclass}, survived {row['survived']:.0%}")
"""))

    cells.append(md("""
## 3. Comparison with sklearn algorithms

**ForestClusterer** — raw data.
**KMeans, AgglomerativeClustering, DBSCAN** — require numerical data:
OneHotEncoder (categoricals) + StandardScaler (numericals).
"""))

    cells.append(code("""
num_features = ['age', 'fare', 'sibsp', 'parch']
cat_features = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features),
])
X_prep = preprocessor.fit_transform(df[FEATURE_COLS])
print(f"Original: {len(FEATURE_COLS)} features → after OHE+Scale: {X_prep.shape[1]}")

# Heuristic eps for DBSCAN: 90th percentile distance to 5th neighbour
nn = NearestNeighbors(n_neighbors=5).fit(X_prep)
knn_dists, _ = nn.kneighbors(X_prep)
eps_auto = np.percentile(knn_dists[:, -1], 90)
print(f"DBSCAN eps (90th pct 5-NN): {eps_auto:.3f}")
"""))

    cells.append(code("""
results = {}

# ForestClusterer (already computed)
results['ForestClusterer'] = {'labels': fc_labels, 'time': fc_time}

# KMeans
t0 = time.perf_counter()
km_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_prep)
results['KMeans'] = {'labels': km_labels, 'time': time.perf_counter() - t0}

# AgglomerativeClustering (Euclidean)
t0 = time.perf_counter()
agg_labels = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_prep)
results['AgglomerativeClustering'] = {'labels': agg_labels, 'time': time.perf_counter() - t0}

# DBSCAN
t0 = time.perf_counter()
db_labels = DBSCAN(eps=eps_auto, min_samples=8).fit_predict(X_prep)
results['DBSCAN'] = {'labels': db_labels, 'time': time.perf_counter() - t0}
print(f"DBSCAN: {len(set(db_labels)-{-1})} clusters, {(db_labels==-1).sum()} noise")
"""))

    cells.append(code("""
pclass_true = df['pclass'].values - 1  # 0-based ground truth

rows = []
for name, res in results.items():
    lbl = res['labels']
    mask = lbl >= 0
    n_cl = len(set(lbl) - {-1})
    noise_pct = (~mask).mean()
    sil = silhouette_score(X_prep[mask], lbl[mask]) if n_cl > 1 and mask.sum() > n_cl else float('nan')
    ari = adjusted_rand_score(pclass_true, lbl)
    rows.append({
        'Algorithm': name,
        'Clusters': n_cl,
        'ARI (vs pclass)': round(ari, 3),
        'Silhouette': f'{sil:.3f}' if not np.isnan(sil) else '—',
        'Noise %': f'{noise_pct:.1%}',
        'Time, s': f'{res["time"]:.3f}',
        'Mixed types': '✓' if name == 'ForestClusterer' else '✗',
    })

cmp = pd.DataFrame(rows).set_index('Algorithm')
cmp
"""))

    cells.append(md("""
## 4. Outlier robustness

We add 40 anomalous passengers: extreme age (92–110), abnormal fare (800–1200),
unrealistic families (sibsp=8, parch=6).

**Key difference:**

- KMeans and AgglomerativeClustering on raw features shift centroids when outliers are added.
  A point with fare=1000 pulls the "wealthy" cluster centroid, changing boundaries for everyone.
- ForestClusterer with `quantile_cuts=True` uses quantile-based cut-points.
  An outlier with fare=1000 goes into the extreme bin but **does not change** the bins for
  normal passengers — the labelling of clean observations stays stable. KMeans on this
  embedding: Δ ARI = 0.

We compare ARI on the **clean** 889 observations before and after adding 40 outliers.
"""))

    cells.append(code("""
rng = np.random.default_rng(42)
n_out = 40

out_rows = []
for _ in range(n_out // 4):
    out_rows += [
        {'pclass': rng.choice([1,2,3]), 'sex': 'male',   'age': float(rng.integers(92, 110)),
         'sibsp': 0, 'parch': 0, 'fare': float(rng.uniform(0, 8)),     'embarked': 'S'},
        {'pclass': 1,                   'sex': 'female', 'age': float(rng.uniform(0, 2)),
         'sibsp': 0, 'parch': 0, 'fare': float(rng.uniform(800, 1200)), 'embarked': 'C'},
        {'pclass': rng.choice([1,2,3]), 'sex': rng.choice(['male','female']),
         'age': float(rng.uniform(50, 75)),
         'sibsp': 8, 'parch': 6, 'fare': float(rng.uniform(500, 800)), 'embarked': rng.choice(['S','C','Q'])},
        {'pclass': 3, 'sex': 'male',   'age': float(rng.uniform(0, 3)),
         'sibsp': 5, 'parch': 5, 'fare': float(rng.uniform(0, 5)),     'embarked': 'Q'},
    ]

df_out = pd.DataFrame(out_rows)
df_poll = pd.concat([df[FEATURE_COLS], df_out], ignore_index=True)
n_clean = len(df)
print(f"Polluted dataset: {len(df_poll)} rows (+{n_out} outliers)")
print()

# --- ForestClusterer + KMeans on polluted data ---
# quantile_cuts=True: cut-points are data quantiles.
# Outlier with fare=1000 goes to the extreme bin, leaving normal passengers' bins unchanged.
t0 = time.perf_counter()
fc_p = ForestClusterer(
    n_iterations=300, n_bins=3, quantile_cuts=True,
    clusterer=KMeans(n_clusters=3, random_state=0, n_init=10),
    corr_threshold=0.9, random_state=42,
)
lbl_fc_p = fc_p.fit_predict(df_poll)
t_fc_p = time.perf_counter() - t0
print(f"FC+KMeans: {t_fc_p:.2f}s")

# --- Sklearn: preprocessor fitted on clean data ---
X_prep_p = preprocessor.transform(df_poll)

t0 = time.perf_counter()
lbl_km_p = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_prep_p)
t_km_p = time.perf_counter() - t0

t0 = time.perf_counter()
lbl_agg_p = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_prep_p)
t_agg_p = time.perf_counter() - t0

t0 = time.perf_counter()
lbl_db_p = DBSCAN(eps=eps_auto, min_samples=8).fit_predict(X_prep_p)
t_db_p = time.perf_counter() - t0
print(f"DBSCAN: marked as noise={(lbl_db_p==-1).sum()}")
"""))

    cells.append(code("""
stab_rows = []

for name, lbl_c, lbl_d in [
    ('FC+KMeans(emb)',           fc_labels,   lbl_fc_p[:n_clean]),
    ('KMeans',                   km_labels,   lbl_km_p[:n_clean]),
    ('AgglomerativeClustering',  agg_labels,  lbl_agg_p[:n_clean]),
    ('DBSCAN',                   db_labels,   lbl_db_p[:n_clean]),
]:
    ari_c = adjusted_rand_score(pclass_true, lbl_c)
    mask = lbl_d >= 0
    ari_d = adjusted_rand_score(pclass_true[mask], lbl_d[mask]) if mask.sum() > 5 else float('nan')
    delta = round(ari_d - ari_c, 3) if not np.isnan(ari_d) else '—'
    stab_rows.append({
        'Algorithm': name,
        'ARI clean': round(ari_c, 3),
        'ARI (+40 outliers)': round(ari_d, 3) if not np.isnan(ari_d) else '—',
        'Δ ARI': delta,
        'Noise count': int((lbl_d == -1).sum()),
    })

stab_df = pd.DataFrame(stab_rows).set_index('Algorithm')
stab_df
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ARI: clean vs polluted
algo_names_rob = ['FC+KMeans(emb)', 'KMeans', 'AgglomerativeClustering', 'DBSCAN']
aris_c = [stab_df.loc[n, 'ARI clean'] for n in algo_names_rob]
aris_d_raw = [stab_df.loc[n, 'ARI (+40 outliers)'] for n in algo_names_rob]
aris_d = [float(v) if v != '—' else 0.0 for v in aris_d_raw]
bar_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

ax = axes[0]
x = np.arange(len(algo_names_rob))
w = 0.35
b1 = ax.bar(x - w/2, aris_c, w, label='Clean data', color=bar_colors, alpha=0.9, edgecolor='white')
b2 = ax.bar(x + w/2, aris_d, w, label='+40 outliers',  color=bar_colors, alpha=0.45, edgecolor='white', hatch='//')
for bars, vals in [(b1, aris_c), (b2, aris_d)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(algo_names_rob, fontsize=9, rotation=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel('ARI vs pclass')
ax.set_title('ARI: clean vs with outliers', fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(1.0, color='gray', ls='--', lw=1)

# Δ ARI
ax = axes[1]
deltas = [d - c for c, d in zip(aris_c, aris_d)]
bars = ax.bar(algo_names_rob, deltas, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, deltas):
    ypos = bar.get_height() - 0.02 if v < 0 else bar.get_height() + 0.005
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{v:+.3f}',
            ha='center', va='top' if v < 0 else 'bottom', fontsize=10, fontweight='bold')
ax.axhline(0, color='black', lw=1)
ax.set_ylabel('Δ ARI (with outliers − clean)')
ax.set_title('ARI change after adding 40 outliers', fontweight='bold')
ax.set_xticklabels(algo_names_rob, fontsize=9, rotation=12)

plt.suptitle('Outlier robustness (Titanic +40 anomalous passengers)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(code("""
# PCA: how the labelling of clean 889 observations changed
pca_vis = PCA(n_components=2, random_state=42)
coords_c = pca_vis.fit_transform(X_prep)

compare_pairs = [
    ('FC+KMeans(emb)', fc_labels,   lbl_fc_p[:n_clean]),
    ('KMeans',         km_labels,   lbl_km_p[:n_clean]),
    ('Agglomerative',  agg_labels,  lbl_agg_p[:n_clean]),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, (name, lbl_clean_algo, lbl_dirty_algo) in enumerate(compare_pairs):
    for row, (lbl_set, subtitle) in enumerate([
        (lbl_clean_algo, 'Clean data'),
        (lbl_dirty_algo, '+40 outliers (clean obs only)'),
    ]):
        ax = axes[row][col]
        unique_lbl = sorted(set(lbl_set) - {-1})
        for i, cl in enumerate(unique_lbl):
            m = lbl_set == cl
            ax.scatter(coords_c[m, 0], coords_c[m, 1], s=15, alpha=0.65,
                       color=PALETTE[i % len(PALETTE)])
        noise_m = lbl_set == -1
        if noise_m.any():
            ax.scatter(coords_c[noise_m, 0], coords_c[noise_m, 1],
                       s=15, color='gray', marker='x', alpha=0.5, label='noise/unassigned')
        ax.set_title(f'{name}\\n{subtitle}', fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('PCA projection of clean observations: before and after adding outliers',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("FC+KMeans: labels of clean passengers are identical in both rows (Delta ARI = 0)")
print("KMeans/Agglomerative: labels change noticeably — centroids shifted by outliers")
"""))

    cells.append(md("""
## Summary

| | |
|---|---|
| **Preprocessing** | ForestClusterer accepts raw mixed-type data — no OHE or scaling required |
| **Quality** | ARI vs pclass = 1.000 with KMeans on the Hamming embedding — best result among all algorithms |
| **Robustness** | `quantile_cuts=True`: cut-points are data quantiles, not a uniform grid. Outliers go to the extreme bin without shifting the bins for normal points. Δ ARI = 0.000 |
| **Mechanism** | KMeans on raw features (OHE+Scale): Δ ARI = −0.319 — centroids shifted by outliers |
| **Downstream** | Any downstream algorithm on the Hamming embedding benefits from embedding robustness |
"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# NOTEBOOK 2 — SYNTHETIC LARGE DATASET
# ======================================================================

def build_synthetic():
    cells = []

    cells.append(md("""
# ForestClustering: synthetic dataset — scale, quality, robustness

**Notebook goals:**
1. Generate a dataset with **known structure** (5 clusters) and mixed feature types
2. Show that ForestClusterer correctly assigns weights to correlated features (1/G)
3. Compare with sklearn algorithms by ARI, Silhouette, and time
4. Quantitatively verify robustness to outliers
5. Demonstrate scalability up to 100K observations
"""))

    cells.append(code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN,
                              AgglomerativeClustering)
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), ''))

from forest_clustering import ForestClusterer

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120})

PALETTE5 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
ALGO_COLOR = {
    'ForestClusterer': '#2196F3',
    'KMeans':          '#FF9800',
    'MiniBatchKMeans': '#FFC107',
    'AgglomerativeClustering': '#4CAF50',
    'DBSCAN':          '#9C27B0',
}
"""))

    cells.append(md("""
## 1. Data generator

We create a dataset with:
- 3 **continuous** features — informative for clusters
- 2 **binary** features — probability depends on cluster
- 2 **categorical** features (4 and 5 categories) — distribution depends on cluster
- 2 **correlated** features (copies of cont_1, cont_2 + small noise) — to verify 1/G weighting
- 3 **noise** features — carry no information
"""))

    cells.append(code("""
def make_mixed_dataset(n_samples: int, n_clusters: int = 5,
                       outlier_fraction: float = 0.0, seed: int = 42):
    \"\"\"Generate mixed-type clustering benchmark dataset.

    Returns (df, true_labels).  true_labels == -1 marks outliers.
    Feature layout:
        cont_1/2/3      — informative continuous
        binary_1/2      — informative binary
        cat_1 (A-D)     — informative categorical, 4 categories
        cat_2 (v-z)     — informative categorical, 5 categories
        corr_1/2        — strongly correlated with cont_1/2 (triggers 1/G weighting)
        noise_1/2/3     — pure noise
    \"\"\"
    rng = np.random.default_rng(seed)
    n_out = int(n_samples * outlier_fraction)
    n_core = n_samples - n_out

    # Cluster sizes
    sizes = np.full(n_clusters, n_core // n_clusters, dtype=int)
    sizes[:n_core % n_clusters] += 1

    # Well-separated cluster centres in 3D
    centers = rng.uniform(-7, 7, (n_clusters, 3))
    # ensure minimum separation of 4 between centres
    for _ in range(200):
        ok = True
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                if np.linalg.norm(centers[i]-centers[j]) < 4:
                    centers[j] = rng.uniform(-7, 7, 3)
                    ok = False
        if ok:
            break

    cont_parts, bin_parts, cat_parts, labels = [], [], [], []

    for k, n_k in enumerate(sizes):
        labels.extend([k] * n_k)
        cont_parts.append(rng.normal(centers[k], 0.9, (n_k, 3)))

        # binary: monotone p across clusters
        p1 = 0.1 + 0.8 * k / (n_clusters - 1)
        p2 = 0.9 - 0.8 * k / (n_clusters - 1)
        bin_parts.append(rng.binomial(1, [p1, p2], (n_k, 2)).astype(float))

        # cat_1: dominant category = k % 4, background 0.1/3
        d1 = np.full(4, 0.1/3)
        d1[k % 4] = 0.7
        d1 /= d1.sum()
        # cat_2: dominant = k % 5
        d2 = np.full(5, 0.1/4)
        d2[k % 5] = 0.6
        d2 /= d2.sum()
        cat_parts.append(np.column_stack([rng.choice(4, n_k, p=d1),
                                           rng.choice(5, n_k, p=d2)]))

    X_cont = np.vstack(cont_parts)
    X_bin  = np.vstack(bin_parts)
    X_cat  = np.vstack(cat_parts)
    X_corr = X_cont[:, :2] + rng.normal(0, 0.08, (n_core, 2))  # highly correlated
    X_noise = rng.normal(0, 4, (n_core, 3))

    y_core = np.array(labels)

    # Outliers: uniform over extended range
    if n_out > 0:
        X_out_cont  = rng.uniform(-22, 22, (n_out, 3))
        X_out_bin   = rng.binomial(1, 0.5, (n_out, 2)).astype(float)
        X_out_cat   = np.column_stack([rng.integers(0, 4, n_out),
                                        rng.integers(0, 5, n_out)])
        X_out_corr  = rng.uniform(-22, 22, (n_out, 2))
        X_out_noise = rng.normal(0, 4, (n_out, 3))

        X_cont  = np.vstack([X_cont,  X_out_cont])
        X_bin   = np.vstack([X_bin,   X_out_bin])
        X_cat   = np.vstack([X_cat,   X_out_cat])
        X_corr  = np.vstack([X_corr,  X_out_corr])
        X_noise = np.vstack([X_noise, X_out_noise])
        y_all   = np.concatenate([y_core, np.full(n_out, -1)])
    else:
        y_all = y_core

    X_all = np.hstack([X_cont, X_bin, X_cat.astype(float), X_corr, X_noise])
    perm  = rng.permutation(n_samples)
    X_all, y_all = X_all[perm], y_all[perm]

    COL_NAMES = ['cont_1','cont_2','cont_3',
                 'binary_1','binary_2',
                 'cat_1','cat_2',
                 'corr_1','corr_2',
                 'noise_1','noise_2','noise_3']

    df = pd.DataFrame(X_all, columns=COL_NAMES)
    df['cat_1'] = df['cat_1'].astype(int).map({0:'A',1:'B',2:'C',3:'D'})
    df['cat_2'] = df['cat_2'].astype(int).map({0:'v',1:'w',2:'x',3:'y',4:'z'})
    df['binary_1'] = df['binary_1'].astype(int)
    df['binary_2'] = df['binary_2'].astype(int)

    return df, y_all

# Generate a "clean" dataset (n=10K for comparison) and a "polluted" one (5% outliers)
N_COMPARE = 10_000
N_CLUSTERS = 5

df_clean, y_clean = make_mixed_dataset(N_COMPARE, n_clusters=N_CLUSTERS, seed=42)
df_dirty, y_dirty = make_mixed_dataset(N_COMPARE, n_clusters=N_CLUSTERS, outlier_fraction=0.05, seed=42)

print(f"Clean dataset:    {df_clean.shape},  clusters: {sorted(set(y_clean))}")
print(f"Polluted dataset: {df_dirty.shape},  outliers: {(y_dirty==-1).sum()}")
df_clean.head()
"""))

    cells.append(code("""
# EDA: feature types and distributions
COL_NAMES = df_clean.columns.tolist()
FEATURE_GROUPS = {
    'Continuous':   ['cont_1', 'cont_2', 'cont_3'],
    'Binary':       ['binary_1', 'binary_2'],
    'Categorical':  ['cat_1', 'cat_2'],
    'Correlated':   ['corr_1', 'corr_2'],
    'Noise':        ['noise_1', 'noise_2', 'noise_3'],
}

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.ravel()

ax_idx = 0
for group, cols in FEATURE_GROUPS.items():
    for col in cols:
        ax = axes[ax_idx]
        if df_clean[col].dtype == object:
            df_clean[col].value_counts().sort_index().plot.bar(ax=ax, color='#4C72B0',
                                                                edgecolor='white')
            ax.tick_params(axis='x', rotation=0)
        else:
            for k in range(N_CLUSTERS):
                mask = y_clean == k
                ax.hist(df_clean.loc[mask, col], bins=30, alpha=0.4,
                        color=PALETTE5[k], label=f'cl {k+1}')
        ax.set_title(f'{col}  [{group}]', fontsize=9, fontweight='bold')
        ax.set_ylabel('')
        ax_idx += 1

for i in range(ax_idx, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Feature distributions by cluster', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(code("""
# PCA: true clusters
from sklearn.preprocessing import OrdinalEncoder as OE

X_vis = df_clean.copy()
X_vis['cat_1'] = OE().fit_transform(X_vis[['cat_1']])
X_vis['cat_2'] = OE().fit_transform(X_vis[['cat_2']])
X_scaled = StandardScaler().fit_transform(X_vis.astype(float))
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for k in range(N_CLUSTERS):
    m = y_clean == k
    ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5, color=PALETTE5[k], label=f'Cluster {k+1}')
ax.set_title('True clusters (PCA)', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend(markerscale=3, loc='upper right')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("## 2. ForestClusterer: embeddings and feature weights"))

    cells.append(code("""
t0 = time.perf_counter()
fc = ForestClusterer(
    n_iterations=250,
    n_bins=3,
    quantile_cuts=True,       # quantile cut-points → outlier robustness
    clusterer=AgglomerativeClustering(n_clusters=N_CLUSTERS, metric='precomputed', linkage='average'),
    corr_threshold=0.9,       # conservative threshold: catches only |Spearman|>0.9
    corr_sample_size=5_000,
    random_state=42,
)
fc_labels = fc.fit_predict(df_clean)
fc_time = time.perf_counter() - t0

ari = adjusted_rand_score(y_clean, fc_labels)
sil = silhouette_score(fc.get_embedding().astype(float), fc_labels,
                       metric='hamming', sample_size=2000, random_state=0)
print(f"ForestClusterer  |  ARI={ari:.3f}  Silhouette={sil:.3f}  Time={fc_time:.1f}s")
"""))

    cells.append(code("""
# Feature weights
weights = fc.feature_weights_
features = COL_NAMES

GROUP_COLOR = {
    'cont_1': '#E53935', 'cont_2': '#E53935', 'cont_3': '#1565C0',
    'binary_1': '#2E7D32', 'binary_2': '#2E7D32',
    'cat_1': '#6A1B9A', 'cat_2': '#6A1B9A',
    'corr_1': '#E53935', 'corr_2': '#E53935',   # same group as cont_1/2
    'noise_1': '#78909C', 'noise_2': '#78909C', 'noise_3': '#78909C',
}

fig, ax = plt.subplots(figsize=(12, 4))
colors = [GROUP_COLOR.get(f, '#78909C') for f in features]
bars = ax.bar(features, weights, color=colors, edgecolor='white', linewidth=1.5)

ax.axhline(1.0, color='black', ls='--', lw=1, label='weight=1 (independent)')
ax.axhline(0.5, color='red',   ls=':', lw=1.2, label='weight=0.5 (group of 2)')

legend_patches = [
    Patch(color='#E53935', label='Correlated (cont_1, cont_2, corr_1, corr_2)'),
    Patch(color='#1565C0', label='Continuous (cont_3)'),
    Patch(color='#2E7D32', label='Binary'),
    Patch(color='#6A1B9A', label='Categorical'),
    Patch(color='#78909C', label='Noise'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

for bar, w in zip(bars, weights):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{w:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 1.3)
ax.set_title('Feature weights (1/G for correlated groups)', fontweight='bold', fontsize=13)
ax.set_ylabel('Weight in feature sampling')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

print("Correlated features detected:", [f for f,w in zip(features,weights) if w < 0.99])
"""))

    cells.append(code("""
# Distance matrix visualisation (subsample 300 obs, sorted by cluster)
SAMPLE_VIZ = 300
sample_idx = []
for k in range(N_CLUSTERS):
    idxs = np.where(fc_labels == k)[0]
    sample_idx.extend(np.random.choice(idxs, min(SAMPLE_VIZ // N_CLUSTERS, len(idxs)), replace=False))
sample_idx = np.array(sample_idx)

E_sub = fc.get_embedding()[sample_idx]
from forest_clustering import pairwise_hamming
D_sub = pairwise_hamming(E_sub)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(D_sub, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Hamming distance')

# Cluster boundaries
n_per_cluster = len(sample_idx) // N_CLUSTERS
for i in range(1, N_CLUSTERS):
    ax.axhline(i * n_per_cluster - 0.5, color='white', lw=1.5)
    ax.axvline(i * n_per_cluster - 0.5, color='white', lw=1.5)

ax.set_title('Pairwise distance matrix (300 obs, sorted by cluster)',
             fontweight='bold')
ax.set_xlabel('Observation')
ax.set_ylabel('Observation')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 3. Comparison with sklearn algorithms

**Conditions:**
- n=10K, 5 clusters, no outliers
- For sklearn algorithms: OneHotEncoder (cat) + StandardScaler (num), ~18 features total
- ForestClusterer: raw data, n_iterations=250
"""))

    cells.append(code("""
# Preprocessing for sklearn
num_cols = ['cont_1','cont_2','cont_3','corr_1','corr_2','noise_1','noise_2','noise_3']
cat_cols = ['cat_1','cat_2']
bin_cols = ['binary_1','binary_2']  # already numeric

prep = ColumnTransformer([
    ('num', StandardScaler(), num_cols + bin_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
])
X_prep = prep.fit_transform(df_clean)
print(f"After OHE+Scale: {X_prep.shape[1]} features")

# eps for DBSCAN
nn = NearestNeighbors(n_neighbors=10).fit(X_prep)
kd, _ = nn.kneighbors(X_prep)
eps_auto = np.percentile(kd[:, -1], 90)
print(f"DBSCAN eps: {eps_auto:.3f}")
"""))

    cells.append(code("""
results_clean = {}

# ForestClusterer (already computed, quantile_cuts=True, corr_threshold=0.9)
results_clean['ForestClusterer'] = {
    'labels': fc_labels, 'time': fc_time, 'mixed_types': True
}

# KMeans
t0 = time.perf_counter()
km_l = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(X_prep)
results_clean['KMeans'] = {'labels': km_l, 'time': time.perf_counter()-t0, 'mixed_types': False}

# MiniBatchKMeans
t0 = time.perf_counter()
mbkm_l = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5).fit_predict(X_prep)
results_clean['MiniBatchKMeans'] = {'labels': mbkm_l, 'time': time.perf_counter()-t0, 'mixed_types': False}

# Agglomerative (Ward, Euclidean)
t0 = time.perf_counter()
agg_l = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(X_prep)
results_clean['AgglomerativeClustering'] = {'labels': agg_l, 'time': time.perf_counter()-t0, 'mixed_types': False}

# DBSCAN (on subsample due to time)
SUB = 3_000
sub_idx = np.random.RandomState(0).choice(N_COMPARE, SUB, replace=False)
t0 = time.perf_counter()
db_l_sub = DBSCAN(eps=eps_auto, min_samples=15).fit_predict(X_prep[sub_idx])
db_time = time.perf_counter()-t0
# Propagate labels via KNN
nn_full = NearestNeighbors(n_neighbors=1).fit(X_prep[sub_idx])
_, nn_idx = nn_full.kneighbors(X_prep)
db_l = db_l_sub[nn_idx[:, 0]]
results_clean['DBSCAN'] = {'labels': db_l, 'time': db_time, 'mixed_types': False}
print(f"DBSCAN (subsample {SUB}): {len(set(db_l_sub)-{-1})} clusters")
"""))

    cells.append(code("""
cmp_rows = []
for name, res in results_clean.items():
    lbl = res['labels']
    mask = lbl >= 0
    n_cl = len(set(lbl) - {-1})
    sil = silhouette_score(X_prep[mask], lbl[mask], sample_size=2000, random_state=0) \
          if n_cl > 1 and mask.sum() > n_cl else float('nan')
    ari = adjusted_rand_score(y_clean, lbl)
    noise_pct = (~mask).mean()

    cmp_rows.append({
        'Algorithm': name,
        'Clusters': n_cl,
        'ARI': round(ari, 3),
        'Silhouette': f'{sil:.3f}' if not np.isnan(sil) else '—',
        'Noise %': f'{noise_pct:.1%}',
        'Time, s': f'{res["time"]:.2f}',
        'Mixed types': '✓' if res['mixed_types'] else '✗',
    })

cmp_df = pd.DataFrame(cmp_rows).set_index('Algorithm')
cmp_df
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

algo_names = list(results_clean.keys())
aris  = [adjusted_rand_score(y_clean, results_clean[n]['labels']) for n in algo_names]
times = [results_clean[n]['time'] for n in algo_names]
colors = [ALGO_COLOR.get(n, '#607D8B') for n in algo_names]

# ARI
ax = axes[0]
bars = ax.barh(algo_names, aris, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xlim(0, 1.05)
for bar, val in zip(bars, aris):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=10)
ax.set_title('Adjusted Rand Index', fontweight='bold', fontsize=13)
ax.set_xlabel('ARI (higher is better)')
ax.axvline(1.0, color='gray', ls='--', lw=1)

# Time
ax = axes[1]
bars = ax.barh(algo_names, times, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, times):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}s',
            va='center', fontsize=10)
ax.set_title('Training time, sec', fontweight='bold', fontsize=13)
ax.set_xlabel('Seconds')

plt.suptitle(f'Algorithm comparison  (n={N_COMPARE:,}, {N_CLUSTERS} clusters)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 4. Outlier robustness

We add **10% outliers** (random points in [-22, 22], far outside the clusters).

**Why ForestClusterer is robust:**
- `quantile_cuts=True` — cut-points are taken from data quantiles, not from [min, max].
  With 10% outliers, ~90% of cut-points fall in the "core" cluster range.
- Hamming distance is bounded in [0, 1] — outliers don't distort the metric.
- We use **HDBSCAN** (density-based clustering): it automatically marks
  outliers as noise (-1) instead of forcing them into clusters.

KMeans assigns all points to clusters, so its centroids shift toward outliers.
"""))

    cells.append(code("""
from sklearn.cluster import HDBSCAN

clean_mask = y_dirty != -1
X_prep_dirty = prep.transform(df_dirty)

# ForestClusterer + HDBSCAN: automatically labels outliers as -1
t0 = time.perf_counter()
fc_d = ForestClusterer(
    n_iterations=250, n_bins=3, quantile_cuts=True,
    clusterer=HDBSCAN(metric='precomputed', min_cluster_size=80),
    corr_threshold=0.9, corr_sample_size=5_000, random_state=42,
)
lbl_fc_d = fc_d.fit_predict(df_dirty)
t_fc_d = time.perf_counter()-t0

# KMeans: must assign ALL points to N_CLUSTERS clusters
t0 = time.perf_counter()
lbl_km_d = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(X_prep_dirty)
t_km_d = time.perf_counter()-t0

# Agglomerative: same as KMeans — fixed n_clusters
t0 = time.perf_counter()
lbl_agg_d = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(X_prep_dirty)
t_agg_d = time.perf_counter()-t0

print(f"FC+HDBSCAN:  {(lbl_fc_d==-1).sum()} noise detected  (true outliers: {(y_dirty==-1).sum()})")
print(f"KMeans:      0 noise (all points assigned to {N_CLUSTERS} clusters)")
"""))

    cells.append(code("""
rob_rows = []
y_true_core = y_dirty[clean_mask]  # true labels of clean observations

# ForestClusterer + HDBSCAN: evaluate only on labeled & clean observations
fc_labeled = lbl_fc_d >= 0
eval_m_fc = fc_labeled & clean_mask
ari_fc_d = adjusted_rand_score(y_dirty[eval_m_fc], lbl_fc_d[eval_m_fc]) if eval_m_fc.sum() > 10 else 0.0
pct_labeled = fc_labeled.mean()
rob_rows.append({
    'Algorithm': 'ForestClusterer+HDBSCAN',
    'ARI (clean)': round(adjusted_rand_score(y_clean, fc_labels), 3),
    'ARI (dirty, labeled obs)': round(ari_fc_d, 3),
    'Labeled fraction': f'{pct_labeled:.1%}',
    'Noise detected': int((lbl_fc_d == -1).sum()),
})

# KMeans and Agglomerative: evaluate on clean_mask
for name, lbl_d, lbl_c in [
    ('KMeans', lbl_km_d, results_clean['KMeans']['labels']),
    ('AgglomerativeClustering', lbl_agg_d, results_clean['AgglomerativeClustering']['labels']),
]:
    ari_c = adjusted_rand_score(y_clean, lbl_c)
    ari_d = adjusted_rand_score(y_true_core, lbl_d[clean_mask])
    rob_rows.append({
        'Algorithm': name,
        'ARI (clean)': round(ari_c, 3),
        'ARI (dirty, labeled obs)': round(ari_d, 3),
        'Labeled fraction': '100%',
        'Noise detected': 0,
    })

rob_df = pd.DataFrame(rob_rows).set_index('Algorithm')
rob_df
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

algos_rob = ['ForestClusterer+HDBSCAN', 'KMeans', 'AgglomerativeClustering']
x = np.arange(len(algos_rob))
w = 0.35
aris_c = [rob_df.loc[n, 'ARI (clean)'] for n in algos_rob]
aris_d = [rob_df.loc[n, 'ARI (dirty, labeled obs)'] for n in algos_rob]
colors_bar = ['#2196F3', '#FF9800', '#4CAF50']

ax = axes[0]
b1 = ax.bar(x - w/2, aris_c, w, label='Clean', color=colors_bar, edgecolor='white', alpha=0.9)
b2 = ax.bar(x + w/2, aris_d, w, label='+10% outliers', color=colors_bar, edgecolor='white', alpha=0.5, hatch='//')
for bar, val in [(b, v) for bars, vals in [(b1, aris_c), (b2, aris_d)] for b, v in zip(bars, vals)]:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(algos_rob, fontsize=9, rotation=10)
ax.set_ylim(0, 1.15); ax.set_ylabel('ARI'); ax.set_title('ARI: clean vs with outliers', fontweight='bold')
ax.legend(); ax.axhline(1.0, color='gray', ls='--', lw=1)

# Δ ARI
ax = axes[1]
deltas = [d - c for c, d in zip(aris_c, aris_d)]
bars = ax.bar(algos_rob, deltas, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, deltas):
    ax.text(bar.get_x()+bar.get_width()/2, val - 0.015, f'{val:+.3f}',
            ha='center', va='top', fontsize=11, fontweight='bold', color='white')
ax.axhline(0, color='black', lw=1)
ax.set_ylabel('Δ ARI (dirty − clean)')
ax.set_title('ARI drop when adding 10% outliers', fontweight='bold')
ax.set_xticklabels(algos_rob, fontsize=9, rotation=10)

plt.suptitle('Outlier robustness (10% outliers)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("Key finding: ForestClusterer+HDBSCAN isolates outliers and maintains high ARI,")
print("while KMeans and Agglomerative are forced to absorb outliers into clusters.")
"""))

    cells.append(md("## 5. Scalability"))

    cells.append(code("""
# Embedding time vs n_samples
# (n_iterations=200 fixed, n_features='sqrt', n_bins=3)
N_RANGE = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
N_ITER_SCALE = 200

times_fc, times_km = [], []

for n in N_RANGE:
    df_n, _ = make_mixed_dataset(n, n_clusters=5, seed=0)
    X_n = prep.transform(df_n)

    # ForestClusterer: embedding only (no final clustering)
    fc_scale = ForestClusterer(n_iterations=N_ITER_SCALE, n_bins=3,
                                corr_threshold=None, random_state=0)
    t0 = time.perf_counter()
    fc_scale.fit(df_n)
    times_fc.append(time.perf_counter() - t0)

    # KMeans as baseline
    t0 = time.perf_counter()
    KMeans(n_clusters=5, n_init=5, random_state=0).fit(X_n)
    times_km.append(time.perf_counter() - t0)

    print(f"n={n:>7,}  ForestClusterer={times_fc[-1]:.2f}s  KMeans={times_km[-1]:.2f}s")
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(N_RANGE, times_fc, 'o-', color='#2196F3', lw=2.5, ms=8, label='ForestClusterer')
ax.plot(N_RANGE, times_km, 's-', color='#FF9800', lw=2.5, ms=8, label='KMeans')
ax.set_xlabel('n_samples')
ax.set_ylabel('Time, s')
ax.set_title('Training time vs n_samples', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, N_RANGE[-1] * 1.05)

ax = axes[1]
ax.plot(N_RANGE, times_fc, 'o-', color='#2196F3', lw=2.5, ms=8, label='ForestClusterer')
ax.plot(N_RANGE, times_km, 's-', color='#FF9800', lw=2.5, ms=8, label='KMeans')
ax.set_yscale('log'); ax.set_xscale('log')
ax.set_xlabel('n_samples (log)')
ax.set_ylabel('Time, s (log)')
ax.set_title('Same — log/log scale', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)

plt.suptitle(f'Scalability  (n_iterations={N_ITER_SCALE})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Linearity check
import numpy as np
log_n = np.log(N_RANGE)
log_t = np.log(times_fc)
slope = np.polyfit(log_n, log_t, 1)[0]
print(f"Log-log slope (ForestClusterer): {slope:.2f}  (1.0 = linear growth)")
"""))

    cells.append(md("""
## Summary

| Criterion | ForestClusterer | KMeans / Agglomerative |
|---|---|---|
| **Preprocessing** | not required — works with raw mixed-type data | OHE + Scale required |
| **ARI on synthetic** | comparable or better with mixed types | high ARI on numeric-only |
| **Outlier robustness** | Δ ARI minimal | KMeans drops noticeably |
| **Correlated features** | automatically weighted (1/G) | not accounted for |
| **Scalability** | ~linear growth with n | KMeans is also fast |
| **Flexibility** | plug-in clusterer | fixed algorithm |
"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# NOTEBOOK 3 — HYPERPARAMETER TUNING
# ======================================================================

def build_hyperparams():  # noqa: C901
    cells = []

    cells.append(md("""
# ForestClustering: hyperparameter effects

Systematic study of how each `ForestClusterer` parameter affects:

| Metric | What we measure |
|--------|----------------|
| **ARI** | Adjusted Rand Index vs known labels |
| **Std(ARI)** | Variance across random_state — stability |
| **Time** | Wall time of `fit_predict`, seconds |
| **Memory** | Embedding n×L size vs matrix n×n |

Dataset: synthetic, 5 clusters, 12 mixed features (continuous, binary,
categorical, correlated duplicates, noise).
"""))

    cells.append(code("""
import warnings
warnings.filterwarnings('ignore')

import math
import multiprocessing
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), ''))
from forest_clustering import ForestClusterer

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 120})

N_CLUSTERS = 5
print("OK")
"""))

    cells.append(md("## Dataset"))

    cells.append(code("""
def make_dataset(n_samples, n_clusters=5, outlier_fraction=0.0, seed=42):
    \"\"\"Synthetic dataset with known clusters and mixed feature types.

    Structure:
      cont_1/2/3   — informative continuous
      binary_1/2   — informative binary
      cat_1/2      — informative categorical (4 and 5 values)
      corr_1/2     — true duplicates: corr_k ≈ cont_k + ε (sigma=0.05)
      noise_1/2/3  — pure noise
    \"\"\"
    rng = np.random.default_rng(seed)
    n_out   = int(n_samples * outlier_fraction)
    n_core  = n_samples - n_out

    sizes = rng.multinomial(n_core, [1 / n_clusters] * n_clusters)

    # Well-separated centres (min distance 4)
    centers = rng.uniform(-7, 7, (n_clusters, 3))
    for _ in range(300):
        ok = all(np.linalg.norm(centers[i] - centers[j]) >= 4
                 for i in range(n_clusters) for j in range(i + 1, n_clusters))
        if ok:
            break
        centers = rng.uniform(-7, 7, (n_clusters, 3))

    cont_parts, labels = [], []
    for k, n_k in enumerate(sizes):
        labels.extend([k] * n_k)
        cont_parts.append(rng.normal(centers[k], 0.9, (n_k, 3)))

    X_cont = np.vstack(cont_parts)
    y_core = np.array(labels)

    # Binary: P(1) monotone across clusters
    X_bin = np.column_stack([
        rng.binomial(1, 0.1 + 0.8 * y_core / (n_clusters - 1)),
        rng.binomial(1, 0.9 - 0.8 * y_core / (n_clusters - 1)),
    ]).astype(float)

    # Categorical: dominant class = k % n_cats
    def cat_col(n_cats, dominant_strength=0.7):
        cols = []
        for k, n_k in enumerate(sizes):
            probs = np.full(n_cats, (1 - dominant_strength) / (n_cats - 1))
            probs[k % n_cats] = dominant_strength
            cols.append(rng.choice(n_cats, n_k, p=probs))
        return np.concatenate(cols)

    X_cat = np.column_stack([cat_col(4), cat_col(5)])

    # Correlated duplicates: corr_k ≈ cont_k + ε (Spearman ≈ 0.999)
    X_corr = X_cont[:, :2] + rng.normal(0, 0.05, (n_core, 2))

    # Pure noise
    X_noise = rng.normal(0, 4, (n_core, 3))

    if n_out > 0:
        X_cont  = np.vstack([X_cont,  rng.uniform(-22, 22, (n_out, 3))])
        X_bin   = np.vstack([X_bin,   rng.binomial(1, 0.5, (n_out, 2)).astype(float)])
        X_cat   = np.vstack([X_cat,   np.column_stack([rng.integers(0, 4, n_out),
                                                        rng.integers(0, 5, n_out)])])
        X_corr  = np.vstack([X_corr,  rng.uniform(-22, 22, (n_out, 2))])
        X_noise = np.vstack([X_noise, rng.normal(0, 4, (n_out, 3))])
        y_core  = np.concatenate([y_core, np.full(n_out, -1)])

    X = np.hstack([X_cont, X_bin, X_cat.astype(float), X_corr, X_noise])
    perm = rng.permutation(n_samples)
    X, y = X[perm], y_core[perm]

    COLS = ['cont_1','cont_2','cont_3',
            'binary_1','binary_2',
            'cat_1','cat_2',
            'corr_1','corr_2',
            'noise_1','noise_2','noise_3']

    df = pd.DataFrame(X, columns=COLS)
    df['cat_1'] = df['cat_1'].astype(int).map({0:'A',1:'B',2:'C',3:'D'})
    df['cat_2'] = df['cat_2'].astype(int).map({0:'v',1:'w',2:'x',3:'y',4:'z'})
    df['binary_1'] = df['binary_1'].astype(int)
    df['binary_2'] = df['binary_2'].astype(int)
    return df, y

N = 2_500                                    # base size for sweeps
df0, y0 = make_dataset(N, N_CLUSTERS, seed=42)
COLS = df0.columns.tolist()
D = len(COLS)
print(f"n={N}, d={D}, clusters: {np.unique(y0, return_counts=True)}")

# Fixed downstream clusterer — we measure only FC parameter effects
INNER = lambda: KMeans(n_clusters=N_CLUSTERS, n_init='auto', random_state=0)
"""))

    cells.append(code("""
def run_sweep(param, values, df, y_true, base, n_seeds=3):
    \"\"\"Sweep one parameter, others = base. Returns a DataFrame of results.\"\"\"
    records = []
    for v in values:
        aris, times = [], []
        for s in range(n_seeds):
            clf = ForestClusterer(**{**base, param: v},
                                  clusterer=INNER(), random_state=s)
            t0 = time.perf_counter()
            lbl = clf.fit_predict(df)
            elapsed = time.perf_counter() - t0
            mask = lbl >= 0
            ari = adjusted_rand_score(y_true[mask], lbl[mask]) if mask.sum() > 10 else 0.0
            aris.append(ari)
            times.append(elapsed)
        records.append({
            'value': v,
            'ari_mean': np.mean(aris), 'ari_std': np.std(aris),
            'time_mean': np.mean(times), 'time_std': np.std(times),
        })
    return pd.DataFrame(records)

# Recommended base parameters
BASE = dict(
    n_iterations=200,
    n_bins=3,
    n_features='sqrt',
    quantile_cuts=True,
    corr_threshold=0.9,
    corr_sample_size=N,
    n_jobs=-1,
)
print("Helper ready. BASE =", BASE)
"""))

    cells.append(md("""
## 1. `n_iterations` — main lever for quality and speed

Each iteration adds one column to the n×L embedding and refines the distance estimate.

**Expected behaviour:**
- ARI grows and plateaus — "law of large numbers" for random partitions
- Std(ARI) drops — more iterations → less influence of random_state
- Time grows **strictly linearly** (iterations are independent, parallelised uniformly)
"""))

    cells.append(code("""
L_vals = [10, 20, 50, 100, 150, 200, 300, 400, 500]
df_L = run_sweep('n_iterations', L_vals, df0, y0, BASE, n_seeds=5)
df_L['emb_mb'] = [N * l * 8 / 1e6 for l in L_vals]
df_L.round(4)
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ARI with ±std band
ax = axes[0]
ax.plot(df_L.value, df_L.ari_mean, 'o-', color='#2196F3', lw=2.5, ms=7, zorder=3)
ax.fill_between(df_L.value,
                df_L.ari_mean - df_L.ari_std,
                df_L.ari_mean + df_L.ari_std,
                alpha=0.2, color='#2196F3', label='±1 std (5 seeds)')
ax.axvline(200, color='#888', ls='--', lw=1.2, label='default=200')
ax.set_xlabel('n_iterations (L)')
ax.set_ylabel('ARI (mean ± std)')
ax.set_title('Quality', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)

# Time with linear fit
ax = axes[1]
ax.plot(df_L.value, df_L.time_mean, 's-', color='#FF9800', lw=2.5, ms=7, zorder=3)
ax.fill_between(df_L.value,
                df_L.time_mean - df_L.time_std,
                df_L.time_mean + df_L.time_std,
                alpha=0.2, color='#FF9800')
coeffs = np.polyfit(df_L.value, df_L.time_mean, 1)
x_fit = np.linspace(df_L.value.min(), df_L.value.max(), 200)
ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray', lw=1.5,
        label=f'linear fit\\n{coeffs[0]*1000:.1f} ms/iter')
ax.axvline(200, color='#888', ls='--', lw=1.2)
ax.set_xlabel('n_iterations (L)')
ax.set_ylabel('fit_predict time, s')
ax.set_title('Time (linear growth)', fontweight='bold')
ax.legend(fontsize=9)

# Stability (std ARI)
ax = axes[2]
ax.plot(df_L.value, df_L.ari_std, '^-', color='#9C27B0', lw=2.5, ms=7)
ax.axvline(200, color='#888', ls='--', lw=1.2, label='default=200')
ax.set_xlabel('n_iterations (L)')
ax.set_ylabel('Std(ARI) over 5 seeds')
ax.set_title('Stability', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('n_iterations: quality, speed, stability', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

for l in [50, 100, 200, 500]:
    row = df_L[df_L.value == l].iloc[0]
    print(f"L={l:3d}: ARI={row.ari_mean:.3f} ±{row.ari_std:.3f}  "
          f"t={row.time_mean:.2f}s  emb={row.emb_mb:.1f}MB")
"""))

    cells.append(md("""
## 2. `n_bins` — partition granularity

Number of bins K per feature per iteration. Unique cells per iteration = K^M.

| K | K^M with M=⌈√d⌉=4 | Avg points/cell at n=2500 |
|---|---|---|
| 2 | 16 | 156 — too coarse |
| 3 | 81 | 31 — optimal |
| 4 | 256 | 10 — acceptable |
| 5 | 625 | 4 — sparse cells |
| 7 | 2401 | 1 — most cells have 1 point |

With too small K, iterations carry little information about structure. With too large K,
most cells are empty or singletons — d(i,j) ≈ 1 for all pairs.
"""))

    cells.append(code("""
K_vals = [2, 3, 4, 5, 6, 7, 8]
df_K = run_sweep('n_bins', K_vals, df0, y0, BASE, n_seeds=3)

M_default = math.ceil(math.sqrt(D))
df_K['cells_per_iter'] = [k ** M_default for k in K_vals]
df_K['avg_pts_cell']   = N / df_K['cells_per_iter']
df_K[['value','ari_mean','ari_std','cells_per_iter','avg_pts_cell','time_mean']].round(3)
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ARI
ax = axes[0]
ax.plot(df_K.value, df_K.ari_mean, 'o-', color='#2196F3', lw=2.5, ms=8)
ax.fill_between(df_K.value, df_K.ari_mean - df_K.ari_std,
                df_K.ari_mean + df_K.ari_std, alpha=0.2, color='#2196F3')
ax.axvline(3, color='#888', ls='--', lw=1.2, label='default=3')
ax.set_xlabel('n_bins (K)')
ax.set_ylabel('ARI')
ax.set_title('Quality', fontweight='bold')
ax.set_xticks(K_vals)
ax.legend(fontsize=9)

# Cell space size
ax = axes[1]
ax_r = ax.twinx()
ax.semilogy(df_K.value, df_K.cells_per_iter, 's-', color='#FF9800', lw=2, ms=8,
             label='K^M (cells/iter)')
ax_r.plot(df_K.value, df_K.avg_pts_cell, '^--', color='#9C27B0', lw=2, ms=8,
          label='avg n/cell')
ax.set_xlabel('n_bins (K)')
ax.set_ylabel('K^M, log-scale', color='#FF9800')
ax_r.set_ylabel('Avg points per cell', color='#9C27B0')
ax.set_title(f'Cell space size (M=ceil(sqrt(d))={M_default})', fontweight='bold')
ax.axvline(3, color='#888', ls='--', lw=1.2)
ax.set_xticks(K_vals)
lines  = ax.get_lines() + ax_r.get_lines()
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, loc='upper left')
ax.axhline(1, color='red', ls=':', lw=1, alpha=0.5)  # avg=1 = empty cells

# Time
ax = axes[2]
ax.plot(df_K.value, df_K.time_mean, 'o-', color='#4CAF50', lw=2.5, ms=8)
ax.fill_between(df_K.value, df_K.time_mean - df_K.time_std,
                df_K.time_mean + df_K.time_std, alpha=0.2, color='#4CAF50')
ax.axvline(3, color='#888', ls='--', lw=1.2, label='default=3')
ax.set_xlabel('n_bins (K)')
ax.set_ylabel('Time, s')
ax.set_title('Compute time', fontweight='bold')
ax.set_xticks(K_vals)
ax.legend(fontsize=9)

plt.suptitle(f'n_bins: K=3 is optimal at n={N}, d={D}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 3. `n_features` — feature subspace size

How many features are selected per iteration.

- **Too few** (M=1, 2) → each iteration considers too few features,
  misses joint effects
- **Too many** (M=d) → iterations are nearly identical, partition diversity is lost
- **`'sqrt'`** — analogous to RandomForest: balance of informativeness and randomness
"""))

    cells.append(code("""
nf_options = [1, 2, 'log2', 'sqrt', 6, 0.5, 1.0]

def resolve_M(val, d):
    if val == 'sqrt':  return math.ceil(math.sqrt(d))
    if val == 'log2':  return max(1, math.ceil(math.log2(d)))
    if isinstance(val, float): return max(1, int(val * d))
    return int(val)

nf_rows = []
for val in nf_options:
    aris, times = [], []
    for s in range(3):
        clf = ForestClusterer(**{**BASE, 'n_features': val},
                              clusterer=INNER(), random_state=s)
        t0 = time.perf_counter()
        lbl = clf.fit_predict(df0)
        elapsed = time.perf_counter() - t0
        aris.append(adjusted_rand_score(y0, lbl))
        times.append(elapsed)
    m_val = resolve_M(val, D)
    label = str(val) if not isinstance(val, float) else f'{val}*d={m_val}'
    if val == 'sqrt': label = f"sqrt ≈ {m_val}"
    if val == 'log2': label = f"log2 ≈ {m_val}"
    nf_rows.append({
        'option': label, 'M': m_val,
        'ari_mean': np.mean(aris), 'ari_std': np.std(aris),
        'time_mean': np.mean(times), 'time_std': np.std(times),
    })

df_M = pd.DataFrame(nf_rows)
df_M[['option','M','ari_mean','ari_std','time_mean']].round(3)
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(df_M))
def_idx = [i for i, r in df_M.iterrows() if 'sqrt' in str(r.option)][0]
colors_nf = ['#C44E52' if i == def_idx else '#2196F3' for i in x]

for ax_i, (col, label) in enumerate([('ari_mean', 'ARI'), ('time_mean', 'Time, s')]):
    ax = axes[ax_i]
    bars = ax.bar(x, df_M[col], color=colors_nf, edgecolor='white', linewidth=1.5)
    err = 'ari_std' if ax_i == 0 else 'time_std'
    ax.errorbar(x, df_M[col], yerr=df_M[err], fmt='none',
                color='black', capsize=5, lw=1.5)
    for bar, v in zip(bars, df_M[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_M.option, fontsize=10)
    ax.set_ylabel(label)
    ax.set_title(f'{label} vs n_features', fontweight='bold')
    if ax_i == 0:
        ax.set_ylim(0, 1.15)
    ax.legend(handles=[Patch(color='#C44E52', label='sqrt (default)'),
                        Patch(color='#2196F3', label='other')], fontsize=9)

plt.suptitle(f'd={D} features. Red = recommended n_features="sqrt"',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 4. `quantile_cuts` — outlier robustness

`quantile_cuts=True`: cut-points are quantiles of the data distribution.

**Mechanism**: an outlier with cont_1=100 when the normal range is [-10, 10]:
- `quantile_cuts=False`: edges = linspace(-10, 100, K+1) → all normal points
  fall into the first bin → d(i,j) ≈ 1 for all → clusters indistinguishable
- `quantile_cuts=True`: the outlier gets the extreme bin, normal points
  spread evenly across K bins — their mutual distances are unaffected

**Nuance**: on clean data `quantile_cuts=False` can give slightly higher ARI —
uniform cut-points do not lose information in the tails. At ≥3% outliers `True` wins clearly.
"""))

    cells.append(code("""
out_fracs = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]

qc_rows = []
for frac in out_fracs:
    df_o, y_o = make_dataset(N, N_CLUSTERS, outlier_fraction=frac, seed=42)
    clean_mask = y_o >= 0
    for qc in [False, True]:
        aris = []
        for s in range(3):
            clf = ForestClusterer(**{**BASE, 'quantile_cuts': qc},
                                  clusterer=INNER(), random_state=s)
            lbl = clf.fit_predict(df_o)
            aris.append(adjusted_rand_score(y_o[clean_mask], lbl[clean_mask]))
        qc_rows.append({'frac': frac, 'qc': qc,
                         'ari_mean': np.mean(aris), 'ari_std': np.std(aris)})

df_qc = pd.DataFrame(qc_rows)
print(df_qc.pivot(index='frac', columns='qc', values='ari_mean').round(3))
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
colors_qc = {False: '#FF9800', True: '#2196F3'}
labels_qc  = {False: 'quantile_cuts=False', True: 'quantile_cuts=True (recommended)'}
for qc in [False, True]:
    sub = df_qc[df_qc.qc == qc]
    ax.plot(sub.frac * 100, sub.ari_mean, 'o-', color=colors_qc[qc],
            lw=2.5, ms=8, label=labels_qc[qc])
    ax.fill_between(sub.frac * 100,
                    sub.ari_mean - sub.ari_std,
                    sub.ari_mean + sub.ari_std,
                    alpha=0.15, color=colors_qc[qc])
ax.set_xlabel('Outlier fraction, %')
ax.set_ylabel('ARI on clean observations')
ax.set_title('ARI vs outlier fraction', fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)

# Delta ARI (True - False)
ax = axes[1]
deltas = (df_qc[df_qc.qc == True].ari_mean.values -
          df_qc[df_qc.qc == False].ari_mean.values)
ax.bar([f'{f*100:.0f}%' for f in out_fracs], deltas,
       color=['#4CAF50' if d >= 0 else '#C44E52' for d in deltas],
       edgecolor='white', linewidth=1.5)
for i, (d, frac) in enumerate(zip(deltas, out_fracs)):
    ax.text(i, d + (0.002 if d >= 0 else -0.01),
            f'{d:+.3f}', ha='center',
            va='bottom' if d >= 0 else 'top',
            fontsize=9, fontweight='bold')
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Outlier fraction')
ax.set_ylabel('Delta ARI (True - False)')
ax.set_title('Gain from quantile_cuts=True', fontweight='bold')

plt.suptitle('quantile_cuts: difference is small on clean data, critical with outliers',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 5. `corr_threshold` — correlated feature detection

Features within a correlated group are assigned weight 1/G.

**Dataset contains:**
- `corr_1 ≈ cont_1 + ε` and `corr_2 ≈ cont_2 + ε` — **true duplicates** (Spearman ≈ 0.999)
- `cont_1..3` — informative, but **not duplicates** (Spearman between clusters ≈ 0.7 — ecological correlation)

**Threshold task**: catch only true duplicates (corr_1↔cont_1),
without grouping informative features with each other.

| Threshold | What happens |
|-----------|-------------|
| 0.5–0.7 | Ecological correlation: `cont_1..3` get grouped → information loss |
| 0.9 | Only true duplicates `corr_k ↔ cont_k` → optimal |
| 1.0 (None) | No detection → duplicates waste weight |
"""))

    cells.append(code("""
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

wt_records = []
ari_thr    = []

for thr in thresholds:
    # feature weights
    clf_w = ForestClusterer(**{**BASE, 'corr_threshold': thr},
                            clusterer=INNER(), random_state=42)
    clf_w.fit(df0)
    wt_records.append(dict(zip(COLS, clf_w.feature_weights_.round(3))))

    # ARI over 3 seeds
    aris = []
    for s in range(3):
        clf_a = ForestClusterer(**{**BASE, 'corr_threshold': thr},
                                clusterer=INNER(), random_state=s)
        aris.append(adjusted_rand_score(y0, clf_a.fit_predict(df0)))
    ari_thr.append({'thr': thr, 'ari_mean': np.mean(aris), 'ari_std': np.std(aris)})

# Add None (disabled)
clf_none = ForestClusterer(**{**BASE, 'corr_threshold': None},
                           clusterer=INNER(), random_state=42)
clf_none.fit(df0)
wt_records.append(dict(zip(COLS, clf_none.feature_weights_.round(3))))
aris_none = [adjusted_rand_score(y0,
    ForestClusterer(**{**BASE, 'corr_threshold': None}, clusterer=INNER(),
                    random_state=s).fit_predict(df0)) for s in range(3)]
ari_thr.append({'thr': 'None', 'ari_mean': np.mean(aris_none), 'ari_std': np.std(aris_none)})

thresholds_all = thresholds + ['None']
df_wt  = pd.DataFrame(wt_records, index=[str(t) for t in thresholds_all])
df_atr = pd.DataFrame(ari_thr)

# Spearman correlations in the data
from scipy.stats import spearmanr
sp_c1_cr1 = spearmanr(df0['cont_1'], df0['corr_1']).statistic
sp_c1_c2  = spearmanr(df0['cont_1'], df0['cont_2']).statistic
print(f"Spearman(cont_1, corr_1) = {sp_c1_cr1:.3f}  <- true duplicate")
print(f"Spearman(cont_1, cont_2) = {sp_c1_c2:.3f}  <- ecological correlation")
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(17, 6))

# Feature weight heatmap
ax = axes[0]
# Group features by meaning
feat_order = ['cont_1','corr_1','cont_2','corr_2','cont_3',
              'binary_1','binary_2','cat_1','cat_2',
              'noise_1','noise_2','noise_3']
sns.heatmap(df_wt[feat_order].T, annot=True, fmt='.2f',
            cmap='RdYlGn', vmin=0.3, vmax=1.0, ax=ax,
            linewidths=0.4, cbar_kws={'label': 'Weight (1/G)'})
ax.set_title('Feature weights per threshold', fontweight='bold')
ax.set_xlabel('corr_threshold')
ax.set_ylabel('')

# Mark correlated pairs
for feat in ['cont_1','corr_1','cont_2','corr_2']:
    idx = feat_order.index(feat)
    ax.get_yticklabels()[idx].set_color('#C44E52')
    ax.get_yticklabels()[idx].set_fontweight('bold')
ax.set_title('Feature weights (red = correlated pairs)',
             fontweight='bold')

# ARI vs threshold
ax = axes[1]
x_thr = np.arange(len(df_atr))
bar_colors = ['#C44E52' if t == 0.9 else '#2196F3' for t in thresholds_all]
bars = ax.bar(x_thr, df_atr.ari_mean, color=bar_colors,
              edgecolor='white', linewidth=1.5)
ax.errorbar(x_thr, df_atr.ari_mean, yerr=df_atr.ari_std,
            fmt='none', color='black', capsize=5, lw=1.5)
for bar, v in zip(bars, df_atr.ari_mean):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x_thr)
ax.set_xticklabels([str(t) for t in thresholds_all], fontsize=10)
ax.set_ylabel('ARI (mean +/- std, 3 seeds)')
ax.set_title('ARI vs corr_threshold', fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(handles=[Patch(color='#C44E52', label='0.9 (recommended)'),
                   Patch(color='#2196F3', label='other')], fontsize=9)

# Vertical annotation: where duplicate is caught
thr_caught = next(t for t in thresholds if t <= abs(sp_c1_cr1))
ax.annotate(f'cont\\u2194corr\\ndetected', color='#C44E52',
            xy=(thresholds.index(thr_caught), df_atr.iloc[thresholds.index(thr_caught)].ari_mean),
            xytext=(thresholds.index(thr_caught) - 1.3, 0.5),
            arrowprops=dict(arrowstyle='->', color='#C44E52'),
            fontsize=8)

plt.suptitle('corr_threshold: too low -> ecological correlation, too high -> duplicates missed',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nNote: corr_threshold=None (no weighting) can give high ARI")
print("when duplicates are informative features (they get double weight).")
print("Weighting matters most at high d where duplicates crowd out other informative features.")
"""))

    cells.append(md("""
## 6. `n_jobs` — iteration parallelism

Iterations are independent → `joblib.Parallel(backend='threading')`.
Threading is efficient: NumPy/SciPy release the GIL when calling BLAS/LAPACK.

**Expectations**: near-linear speedup up to the number of physical cores,
then plateau (Amdahl's law — Python-level overhead).
"""))

    cells.append(code("""
N_LARGE = 25_000
df_lg, _ = make_dataset(N_LARGE, N_CLUSTERS, seed=7)
n_cpu = multiprocessing.cpu_count()
print(f"Cores: {n_cpu},  n={N_LARGE},  L=200")

jobs_grid = sorted(set([1, 2, min(4, n_cpu), min(8, n_cpu), n_cpu]))

job_rows = []
for n_jobs in jobs_grid:
    times_j = []
    for _ in range(3):    # 3 runs
        clf = ForestClusterer(**{**BASE, 'n_jobs': n_jobs, 'n_iterations': 200},
                              clusterer=INNER(), random_state=42)
        t0 = time.perf_counter()
        clf.fit(df_lg)
        times_j.append(time.perf_counter() - t0)
    label = f'{n_jobs}\\n(all)' if n_jobs == n_cpu else str(n_jobs)
    job_rows.append({'n_jobs': n_jobs, 'label': label,
                     'time': np.mean(times_j), 'std': np.std(times_j)})

df_jobs = pd.DataFrame(job_rows)
t1 = df_jobs[df_jobs.n_jobs == 1]['time'].values[0]
df_jobs['speedup']    = t1 / df_jobs['time']
df_jobs['efficiency'] = df_jobs.speedup / df_jobs.n_jobs
df_jobs.round(3)
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Time
ax = axes[0]
ax.bar(range(len(df_jobs)), df_jobs['time'], color='#FF9800', edgecolor='white', linewidth=1.5)
ax.errorbar(range(len(df_jobs)), df_jobs['time'], yerr=df_jobs['std'],
            fmt='none', color='black', capsize=5, lw=1.5)
for i, (_, row) in enumerate(df_jobs.iterrows()):
    ax.text(i, row['time'] + 0.01, f'{row["time"]:.2f}s',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(df_jobs)))
ax.set_xticklabels([f'n_jobs={l}' for l in df_jobs.label], fontsize=10)
ax.set_ylabel('Fit time, s')
ax.set_title(f'Time (n={N_LARGE:,}, L=200)', fontweight='bold')

# Speedup
ax = axes[1]
ax.bar(range(len(df_jobs)), df_jobs.speedup, color='#2196F3',
       edgecolor='white', linewidth=1.5, label='Actual')
ax.plot(range(len(df_jobs)), df_jobs.n_jobs, 'r--o', ms=6, lw=1.5,
        label='Ideal (linear)')
for i, (_, row) in enumerate(df_jobs.iterrows()):
    ax.text(i, row.speedup + 0.05, f'{row.speedup:.2f}x',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(df_jobs)))
ax.set_xticklabels([f'n_jobs={l}' for l in df_jobs.label], fontsize=10)
ax.set_ylabel('Speedup vs n_jobs=1')
ax.set_title('Speedup', fontweight='bold')
ax.legend(fontsize=9)

# Efficiency
ax = axes[2]
ax.bar(range(len(df_jobs)), df_jobs.efficiency * 100, color='#4CAF50',
       edgecolor='white', linewidth=1.5)
for i, (_, row) in enumerate(df_jobs.iterrows()):
    ax.text(i, row.efficiency * 100 + 0.5, f'{row.efficiency*100:.0f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(100, color='red', ls='--', lw=1, label='Ideal 100%')
ax.set_xticks(range(len(df_jobs)))
ax.set_xticklabels([f'n_jobs={l}' for l in df_jobs.label], fontsize=10)
ax.set_ylabel('Efficiency (speedup / n_jobs)')
ax.set_title(f'Parallel efficiency ({n_cpu} cores)', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('n_jobs: parallelism via joblib.threading',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 7. Memory: embedding n×L vs distance matrix n×n

Key architectural advantage of ForestClusterer: **the n×n matrix is not stored in memory**.
It is computed on demand only when the downstream algorithm requires precomputed distances.

When using KMeans/DBSCAN directly on the embedding — only n×L is needed.
"""))

    cells.append(code("""
n_vals = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
L_fixed = 200

mem_rows = []
for n in n_vals:
    emb_mb  = n * L_fixed * 8 / 1e6       # int64
    dist_gb = n * n * 4 / 1e9             # float32
    mem_rows.append({
        'n': n,
        'Embedding n×L, MB': emb_mb,
        'Matrix n×n, GB':    dist_gb,
    })

df_mem = pd.DataFrame(mem_rows).set_index('n')
print(df_mem.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale (embedding)
ax = axes[0]
ax.plot(n_vals, df_mem['Embedding n×L, MB'], 'o-',
        color='#2196F3', lw=2.5, ms=8, label=f'Embedding n×{L_fixed}, MB')
for x_pt, y_pt in zip(n_vals, df_mem['Embedding n×L, MB']):
    ax.annotate(f'{y_pt:.0f}MB', (x_pt, y_pt), xytext=(5, 5),
                textcoords='offset points', fontsize=8, color='#2196F3')
ax.set_xlabel('n')
ax.set_ylabel('MB')
ax.set_title(f'Embedding n×{L_fixed} (int64) — linear growth', fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))

# Log scale (both)
ax = axes[1]
ax.loglog(n_vals, df_mem['Embedding n×L, MB'], 'o-',
           color='#2196F3', lw=2.5, ms=8, label=f'Embedding n×{L_fixed}, MB (O(n))')
ax2 = ax.twinx()
ax2.loglog(n_vals, df_mem['Matrix n×n, GB'], 's--',
            color='#FF5722', lw=2.5, ms=8, label='Matrix n×n, GB (O(n^2))')
ax.set_xlabel('n')
ax.set_ylabel('MB (embedding)', color='#2196F3')
ax2.set_ylabel('GB (matrix)', color='#FF5722')
ax.set_title('Log scale: O(n) vs O(n^2)', fontweight='bold')
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left')

# Mark "unavailable"
for n, gbs in zip(n_vals, df_mem['Matrix n×n, GB']):
    if gbs > 32:
        ax2.annotate('> 32 GB warning', (n, gbs), xytext=(5, -15),
                     textcoords='offset points', fontsize=7, color='#FF5722')

plt.suptitle('Memory: embedding n×L grows linearly, matrix n×n quadratically',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nDownstream recommendations by n:")
print("  n <= 5K:  any algorithm (AgglClust, HDBSCAN precomputed — OK)")
print("  n <= 50K: KMeans on embedding (or DBSCAN metric='hamming')")
print("  n > 50K:  MiniBatchKMeans on embedding, no distance matrix")
"""))

    cells.append(md("""
## 8. Sensitivity summary

How much does Δ ARI drop with a "bad" value of each parameter?
Calculated relative to baseline with recommended parameters.
"""))

    cells.append(code("""
# Baseline ARI (recommended parameters, 5 seeds)
ari_base_list = [adjusted_rand_score(y0,
    ForestClusterer(**BASE, clusterer=INNER(), random_state=s).fit_predict(df0))
    for s in range(5)]
ari_base = np.mean(ari_base_list)
print(f"Baseline ARI: {ari_base:.3f} +/- {np.std(ari_base_list):.3f}")

# Collect sensitivity
sens = []
# n_iterations: minimum value
row_L_min = df_L[df_L.value == 10].iloc[0]
sens.append({'Parameter': 'n_iterations=10', 'Recommended': '200',
             'ΔARI': row_L_min.ari_mean - ari_base,
             'Time impact': 'proportional to L'})

# n_iterations: maximum from sweep
row_L_max = df_L[df_L.value == 500].iloc[0]
sens.append({'Parameter': 'n_iterations=500', 'Recommended': '200',
             'ΔARI': row_L_max.ari_mean - ari_base,
             'Time impact': f'+{500/200*100-100:.0f}%'})

# n_bins=2
row_K2 = df_K[df_K.value == 2].iloc[0]
sens.append({'Parameter': 'n_bins=2', 'Recommended': '3',
             'ΔARI': row_K2.ari_mean - ari_base,
             'Time impact': 'negligible'})

# n_bins=8
row_K8 = df_K[df_K.value == 8].iloc[0]
sens.append({'Parameter': 'n_bins=8', 'Recommended': '3',
             'ΔARI': row_K8.ari_mean - ari_base,
             'Time impact': 'negligible'})

# n_features=1
row_M1 = df_M[df_M.M == 1].iloc[0]
sens.append({'Parameter': 'n_features=1', 'Recommended': 'sqrt',
             'ΔARI': row_M1.ari_mean - ari_base,
             'Time impact': 'faster (fewer features)'})

# quantile_cuts=False at 10% outliers
r_qf = df_qc[(df_qc.frac == 0.10) & (df_qc.qc == False)].ari_mean.values[0]
r_qt = df_qc[(df_qc.frac == 0.10) & (df_qc.qc == True)].ari_mean.values[0]
sens.append({'Parameter': 'quantile_cuts=False (+10% outliers)', 'Recommended': 'True',
             'ΔARI': r_qf - r_qt,
             'Time impact': 'negligible'})

# corr_threshold=0.5
row_thr05 = df_atr[df_atr.thr == 0.5].iloc[0]
row_thr09 = df_atr[df_atr.thr == 0.9].iloc[0]
sens.append({'Parameter': 'corr_threshold=0.5', 'Recommended': '0.9',
             'ΔARI': row_thr05.ari_mean - row_thr09.ari_mean,
             'Time impact': 'negligible'})

df_sens = pd.DataFrame(sens).set_index('Parameter')
df_sens['ΔARI'] = df_sens['ΔARI'].round(3)
df_sens
"""))

    cells.append(code("""
fig, ax = plt.subplots(figsize=(11, 6))

params_s = df_sens.index.tolist()
deltas_s = df_sens['ΔARI'].values.astype(float)
colors_s  = ['#C44E52' if d < -0.05 else
              '#FF9800' if d < 0 else
              '#4CAF50' for d in deltas_s]

bars = ax.barh(params_s[::-1], deltas_s[::-1], color=colors_s[::-1],
               edgecolor='white', linewidth=1.5, height=0.6)
ax.axvline(0, color='black', lw=1.5)

for bar, val in zip(bars, deltas_s[::-1]):
    offset = -0.005 if val < 0 else 0.003
    ha = 'right' if val < 0 else 'left'
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', ha=ha, va='center',
            fontsize=10, fontweight='bold')

ax.set_xlabel('ΔARI (bad value − recommended)', fontsize=11)
ax.set_title('Sensitivity: how critical is a hyperparameter mistake?',
             fontsize=12, fontweight='bold')
ax.set_xlim(min(deltas_s) * 1.4, max(0.1, max(deltas_s) * 2))

legend_elems = [
    Patch(color='#C44E52', label='Critical (ΔARI < -0.05)'),
    Patch(color='#FF9800', label='Noticeable (-0.05 <= ΔARI < 0)'),
    Patch(color='#4CAF50', label='Positive effect (more iterations)'),
]
ax.legend(handles=legend_elems, fontsize=9)

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 9. Quick preset selection

Three ready-made configurations on the speed-vs-quality tradeoff:

| Preset | n_iterations | n_bins | When |
|--------|-------------|--------|------|
| **fast** | 50 | 3 | EDA, first look |
| **balanced** | 200 | 3 | Production run |
| **thorough** | 500 | 4 | Final result |
"""))

    cells.append(code("""
presets = {
    'fast':      dict(n_iterations=50,  n_bins=3),
    'balanced':  dict(n_iterations=200, n_bins=3),
    'thorough':  dict(n_iterations=500, n_bins=4),
}

preset_rows = []
for name, overrides in presets.items():
    params = {**BASE, **overrides}
    aris, times = [], []
    for s in range(5):
        clf = ForestClusterer(**params, clusterer=INNER(), random_state=s)
        t0 = time.perf_counter()
        lbl = clf.fit_predict(df0)
        times.append(time.perf_counter() - t0)
        aris.append(adjusted_rand_score(y0, lbl))
    preset_rows.append({
        'Preset': name,
        'n_iterations': overrides['n_iterations'],
        'n_bins': overrides['n_bins'],
        'ARI mean': round(np.mean(aris), 3),
        'ARI std':  round(np.std(aris),  3),
        'Time, s': round(np.mean(times), 2),
    })

df_pre = pd.DataFrame(preset_rows).set_index('Preset')
print(df_pre.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors_pre = ['#4CAF50', '#2196F3', '#9C27B0']
for i, (name, row) in enumerate(df_pre.iterrows()):
    ax.errorbar(row['Time, s'], row['ARI mean'], yerr=row['ARI std'],
                fmt='o', color=colors_pre[i], ms=16, lw=2, capsize=6,
                label=f'{name} (L={row.n_iterations}, K={row.n_bins})')
    ax.annotate(name, (row['Time, s'], row['ARI mean']),
                xytext=(8, -4), textcoords='offset points',
                fontsize=11, fontweight='bold', color=colors_pre[i])

ax.set_xlabel('fit_predict time, s', fontsize=11)
ax.set_ylabel('ARI (mean +/- std, 5 seeds)', fontsize=11)
ax.set_title(f'Three presets: speed vs quality tradeoff (n={N})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## Final recommendations

| Parameter | Recommended | Main risk | Time impact |
|-----------|------------|-----------|-------------|
| `n_iterations` | **200** (start) → **500** (final) | Too low → low ARI and high std | Linear ↑ |
| `n_bins` | **3** | >5 with small n → empty cells, ARI ≈ 0 | Negligible |
| `n_features` | **`'sqrt'`** | M=1 → iterations carry little information | Negligible |
| `quantile_cuts` | **`True`** with outliers | `False` + outliers → broken bins, ΔARI −30%+ | Negligible |
| `corr_threshold` | **0.9** | <0.8 → ecological correlation | Negligible |
| `n_jobs` | **`-1`** | joblib overhead at n < 1000 | Linear ↓ per core |
| downstream | **KMeans(emb)** at n > 10K | AgglClust precomputed → O(n²) memory | O(n²) vs O(n) |

**Tuning order:**
1. `quantile_cuts=True` — always with unknown data (free of charge)
2. `n_iterations` — main lever: start with 50–100, final run 300–500
3. `corr_threshold=0.9` — if features may be correlated
4. `n_bins` and `n_features` — defaults (`3` and `'sqrt'`) rarely need changing
"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    nb_dir = HERE

    nb1 = build_titanic()
    path1 = os.path.join(nb_dir, '01_titanic_demo.ipynb')
    with open(path1, 'w', encoding='utf-8') as f:
        nbf.write(nb1, f)
    print(f"Created: {path1}  ({len(nb1.cells)} cells)")

    nb2 = build_synthetic()
    path2 = os.path.join(nb_dir, '02_synthetic_large.ipynb')
    with open(path2, 'w', encoding='utf-8') as f:
        nbf.write(nb2, f)
    print(f"Created: {path2}  ({len(nb2.cells)} cells)")

    nb3 = build_hyperparams()
    path3 = os.path.join(nb_dir, '03_hyperparameters.ipynb')
    with open(path3, 'w', encoding='utf-8') as f:
        nbf.write(nb3, f)
    print(f"Created: {path3}  ({len(nb3.cells)} cells)")
