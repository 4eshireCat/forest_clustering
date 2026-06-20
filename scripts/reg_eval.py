import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from forest_clustering import ForestClusterer

DATASETS = {  # name -> (csv, target_col_index)
    "adult":     ("/home/claude/adult.csv", "income"),
    "nursery":   ("/home/claude/nursery.csv", -1),
    "car":       ("/home/claude/car.csv", -1),
    "mushroom":  ("/home/claude/mushroom.csv", 0),
    "tictactoe": ("/home/claude/tictactoe.csv", -1),
}
N_SUB = 1500

def load(name):
    path, tcol = DATASETS[name]
    header = 0 if name == "adult" else None
    df = pd.read_csv(path, header=header)
    if isinstance(tcol, str):
        y = df[tcol].astype(str).values; X = df.drop(columns=[tcol])
    else:
        tcol = df.columns[tcol] if tcol >= 0 else df.columns[tcol]
        y = df[tcol].astype(str).values; X = df.drop(columns=[tcol])
    if len(df) > N_SUB:
        rng = np.random.default_rng(0); idx = rng.choice(len(df), N_SUB, replace=False)
        X = X.iloc[idx]; y = y[idx]
    Xenc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1
                          ).fit_transform(X.astype(str)).astype(float)
    yi = pd.factorize(y)[0]
    return Xenc, yi

def run():
    out = {}
    for name in DATASETS:
        X, y = load(name)
        k = len(np.unique(y))
        res = {}
        # default centroid (KMeans)
        lab = ForestClusterer(n_iterations=150, n_bins=4, n_clusters=k,
                              corr_threshold=None, random_state=0).fit_predict(X)
        res["kmeans_ARI"] = round(float(adjusted_rand_score(y, lab)), 4)
        res["kmeans_NMI"] = round(float(normalized_mutual_info_score(y, lab)), 4)
        # louvain
        try:
            lab2 = ForestClusterer(n_iterations=150, n_bins=4, clusterer="louvain",
                                   corr_threshold=None, random_state=0).fit_predict(X)
            res["louvain_ARI"] = round(float(adjusted_rand_score(y, lab2)), 4)
            res["louvain_nclusters"] = int(len(np.unique(lab2[lab2 >= 0])))
        except Exception as e:
            res["louvain_ARI"] = f"ERR:{type(e).__name__}"
        out[name] = res
    print(json.dumps(out))

if __name__ == "__main__":
    run()
