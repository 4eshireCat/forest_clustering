"""Contrastive learning components for forest clustering.

Implements augmentation, pair generation, contrastive loss, contrastive split
evaluation, and contrastive tree building.
"""

import numpy as np


def augment_sample(x, noise_scale=0.1, dropout_prob=0.0, seed=42):
    """Augment a single sample for positive pair generation.

    Parameters
    ----------
    x : ndarray of shape (d,)
    noise_scale : float
        Std of Gaussian noise.
    dropout_prob : float
        Probability of zeroing a feature.
    seed : int
        Random seed.

    Returns
    -------
    x_aug : ndarray of shape (d,)
    """
    rng = np.random.default_rng(seed)
    x_aug = x.copy().astype(np.float64)

    if noise_scale > 0:
        x_aug = x_aug + rng.normal(0, noise_scale, size=x_aug.shape)

    if dropout_prob > 0:
        mask = rng.random(x_aug.shape) >= dropout_prob
        x_aug = x_aug * mask

    return x_aug


def generate_pairs(y, n_positive=20, n_negative=20, random_state=42):
    """Generate positive and negative pairs from labels.

    Positive pairs are pairs of indices with the same label.
    Negative pairs are pairs of indices with different labels.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Labels.
    n_positive : int
        Number of positive pairs to generate.
    n_negative : int
        Number of negative pairs to generate.
    random_state : int
        Random seed.

    Returns
    -------
    pos_pairs : ndarray of shape (n_pos, 2)
    neg_pairs : ndarray of shape (n_neg, 2)
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    n = len(y)

    if n < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    unique_labels = np.unique(y)

    # --- Positive pairs: same label ---
    # Vectorized: sample directly without O(n^2) candidate list
    pos_pairs_list = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        if len(indices) < 2:
            continue
        n_avail = len(indices) * (len(indices) - 1) // 2
        n_sample = min(n_positive // max(len(unique_labels), 1), max(n_avail, 1))
        n_sample = min(n_sample, n_avail)
        if n_sample <= 0:
            continue
        # Random pairs via sampling without full enumeration
        for _ in range(n_sample):
            pair = rng.choice(indices, size=2, replace=False)
            pos_pairs_list.append([pair[0], pair[1]])

    # --- Negative pairs: different labels ---
    neg_pairs_list = []
    if len(unique_labels) >= 2:
        n_sample_neg = min(n_negative, n * (n - 1) // 2)
        for _ in range(n_sample_neg):
            i, j = rng.integers(0, n, size=2)
            if i != j and y[i] != y[j]:
                neg_pairs_list.append([i, j])
        # If we didn't get enough, try a few more
        attempts = 0
        while len(neg_pairs_list) < min(n_negative, n) and attempts < n_negative * 2:
            i, j = rng.integers(0, n, size=2)
            if i != j and y[i] != y[j]:
                neg_pairs_list.append([i, j])
            attempts += 1

    pos_pairs = np.array(pos_pairs_list, dtype=np.int64) if pos_pairs_list else np.empty((0, 2), dtype=np.int64)
    neg_pairs = np.array(neg_pairs_list, dtype=np.int64) if neg_pairs_list else np.empty((0, 2), dtype=np.int64)

    return pos_pairs, neg_pairs


def contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5):
    """Contrastive loss for given embeddings and pair assignments.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, n_features)
    pos_pairs : ndarray of shape (n_pos, 2)
    neg_pairs : ndarray of shape (n_neg, 2)
    temperature : float

    Returns
    -------
    loss : float
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if len(pos_pairs) == 0:
        return 0.0

    z = np.asarray(embeddings, dtype=np.float64)

    eps = 1e-20
    losses = []

    # Pre-compute pairwise dot products as similarities
    # Same leaf: dot=1, sim=1; Different leaf: dot=0, sim=0
    sim_matrix = z @ z.T

    for a, p in pos_pairs:
        sim_pos = sim_matrix[a, p]

        # Find negatives for this anchor (anchor can be in either column)
        neg_mask = (neg_pairs[:, 0] == a) | (neg_pairs[:, 1] == a)
        if not np.any(neg_mask):
            losses.append(0.0)
            continue

        neg_pairs_for_a = neg_pairs[neg_mask]
        # Extract the partner (the one that is not the anchor)
        neg_partners = np.where(neg_pairs_for_a[:, 0] == a,
                                neg_pairs_for_a[:, 1],
                                neg_pairs_for_a[:, 0])
        sim_negs = sim_matrix[a, neg_partners]

        numerator = np.exp(sim_pos / temperature)
        # True NT-Xent: denominator includes both negatives and the positive pair
        denominator = np.sum(np.exp(sim_negs / temperature)) + np.exp(sim_pos / temperature)

        if denominator < eps:
            losses.append(0.0)
            continue

        fraction = numerator / denominator
        # Clip to [0, 1] for numerical stability and non-negativity
        fraction = min(fraction, 1.0)
        loss_val = -np.log(fraction + eps)
        losses.append(max(loss_val, 0.0))

    return float(np.mean(losses)) if losses else 0.0


def _entropy(labels):
    """Compute Shannon entropy."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-12))


def _info_gain_feature(X_col, mask_left):
    """Compute normalized variance-based information gain for a binary split."""
    n = len(X_col)
    n_left = np.sum(mask_left)
    n_right = n - n_left
    if n_left == 0 or n_right == 0:
        return 0.0
    var_total = np.var(X_col)
    var_left = np.var(X_col[mask_left]) if n_left > 0 else 0.0
    var_right = np.var(X_col[~mask_left]) if n_right > 0 else 0.0
    gain = (var_total - (n_left / n) * var_left - (n_right / n) * var_right) / (var_total + 1e-12)
    return max(gain, 0.0)


def _contrastive_loss_from_mask(left_mask, pos_pairs, neg_pairs, temperature=0.5):
    """Fast contrastive loss from a binary leaf-assignment mask.

    Avoids building one-hot embeddings and recomputing dot products.
    Similarity is 1.0 if two samples are in the same leaf, else 0.0.

    Parameters
    ----------
    left_mask : ndarray of shape (n,) dtype bool
    pos_pairs : ndarray of shape (n_pos, 2)
    neg_pairs : ndarray of shape (n_neg, 2)
    temperature : float

    Returns
    -------
    loss : float
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if len(pos_pairs) == 0:
        return 0.0

    eps = 1e-20
    losses = []

    for a, p in pos_pairs:
        sim_pos = 1.0 if left_mask[a] == left_mask[p] else 0.0

        neg_mask = (neg_pairs[:, 0] == a) | (neg_pairs[:, 1] == a)
        if not np.any(neg_mask):
            losses.append(0.0)
            continue

        neg_pairs_for_a = neg_pairs[neg_mask]
        neg_partners = np.where(neg_pairs_for_a[:, 0] == a,
                                neg_pairs_for_a[:, 1],
                                neg_pairs_for_a[:, 0])
        sim_negs = np.array([1.0 if left_mask[a] == left_mask[n_idx] else 0.0
                             for n_idx in neg_partners])

        numerator = np.exp(sim_pos / temperature)
        denominator = np.sum(np.exp(sim_negs / temperature)) + np.exp(sim_pos / temperature)

        if denominator < eps:
            losses.append(0.0)
            continue

        fraction = numerator / denominator
        fraction = min(fraction, 1.0)
        loss_val = -np.log(fraction + eps)
        losses.append(max(loss_val, 0.0))

    return float(np.mean(losses)) if losses else 0.0


def evaluate_split_contrastive(X, pos_pairs, neg_pairs, thresh, feature_idx,
                                temperature=1.0, lambda_info=0.1):
    """Evaluate a candidate split by contrastive score.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    pos_pairs, neg_pairs : pair arrays
    thresh : float
    feature_idx : int
    temperature : float
    lambda_info : float

    Returns
    -------
    score : float (higher is better)
    """
    n = X.shape[0]
    if n == 0:
        return -np.inf

    left_mask = X[:, feature_idx] < thresh

    # Degenerate split
    if np.all(left_mask) or np.all(~left_mask):
        return -np.inf

    loss = _contrastive_loss_from_mask(left_mask, pos_pairs, neg_pairs, temperature)

    # Compute information gain
    info_gain = _info_gain_feature(X[:, feature_idx], left_mask)

    score = -loss + lambda_info * info_gain
    return float(score)


def _random_split(X_node, rng):
    """Generate a random split for a node."""
    n, d = X_node.shape
    if n < 2:
        return None, None
    feature = rng.integers(0, d)
    col = X_node[:, feature]
    lo, hi = np.min(col), np.max(col)
    if lo >= hi:
        return None, None
    thresh = rng.uniform(lo, hi)
    return int(feature), float(thresh)


def _resolve_n_features(n_features, d):
    if n_features == "sqrt":
        return max(1, int(np.ceil(np.sqrt(d))))
    if n_features == "log2":
        return max(1, int(np.ceil(np.log2(max(d, 2)))))
    if isinstance(n_features, float) and 0 < n_features <= 1.0:
        return max(1, int(np.ceil(n_features * d)))
    return max(1, int(n_features))


class ContrastiveTree:
    """A fitted axis-aligned tree that can be applied to new data.

    Storing the split structure (rather than only the training leaf ids) is
    what makes the contrastive embedding consistent out-of-sample: ``apply``
    on the training data reproduces the leaf ids returned at fit time, and on
    new data routes each point through the same splits.  Without this, the
    clusterer's ``transform``/``partial_fit``/permutation-importance operate on
    an embedding unrelated to the one that produced ``labels_``.
    """

    __slots__ = ("root", "n_leaves")

    def __init__(self, root, n_leaves):
        self.root = root          # nested dict: leaf {'leaf':id} or split node
        self.n_leaves = int(n_leaves)

    def apply(self, X):
        """Return (n,) int64 leaf ids for X using the stored splits."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros(n, dtype=np.int64)
        if n == 0:
            return out

        def descend(idx, node):
            if "leaf" in node:
                out[idx] = node["leaf"]
                return
            go_left = X[idx, node["feature"]] < node["thresh"]
            left_idx = idx[go_left]
            right_idx = idx[~go_left]
            if len(left_idx):
                descend(left_idx, node["left"])
            if len(right_idx):
                descend(right_idx, node["right"])

        descend(np.arange(n), self.root)
        return out


def _fit_tree(X, max_depth, n_pairs, temperature, rng):
    """Fit one tree, returning (leaf_ids, root_node, n_leaves).

    Builds a contrastive tree when feasible, otherwise a random tree, and
    records the split structure so it can be replayed on new data.  The split
    objective at each node only counts pairs whose *both* endpoints fall inside
    that node (previously out-of-node points were silently lumped onto one side,
    so the score no longer measured the local split).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    next_leaf_id = [0]

    def make_leaf(node_indices, leaf_ids):
        lid = next_leaf_id[0]
        next_leaf_id[0] += 1
        leaf_ids[node_indices] = lid
        return {"leaf": lid}

    # --- decide whether to run the contrastive objective at all ---
    degenerate = (n < 10) or (n == 0) or (n > 0 and np.allclose(X, X[0]))
    pos_pairs = neg_pairs = None
    if not degenerate:
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(3, max(2, n // 5))
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto',
                            random_state=int(rng.integers(0, 2**31)), max_iter=100)
            pseudo_labels = kmeans.fit_predict(X)
        except Exception:
            pseudo_labels = rng.integers(0, max(2, n // 5), size=n)
        pos_pairs, neg_pairs = generate_pairs(
            pseudo_labels, n_positive=n_pairs, n_negative=n_pairs,
            random_state=int(rng.integers(0, 2**31)))
        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            pos_pairs = neg_pairs = None  # fall back to random splits

    n_features_per_split = _resolve_n_features('sqrt', d) if d > 0 else 1
    leaf_ids = np.zeros(n, dtype=np.int64)

    def best_contrastive_split(node_indices):
        """Return (feature, thresh) chosen by the in-node contrastive score."""
        # Restrict pairs to those fully contained in this node.
        node_set = set(int(i) for i in node_indices)

        def _in_node(pairs):
            if pairs is None or len(pairs) == 0:
                return pairs
            keep = np.fromiter(
                ((int(a) in node_set and int(b) in node_set) for a, b in pairs),
                dtype=bool, count=len(pairs),
            )
            return pairs[keep]

        pos_n = _in_node(pos_pairs)
        neg_n = _in_node(neg_pairs)
        if pos_n is None or len(pos_n) == 0:
            return None, None

        X_node = X[node_indices]
        n_feat = min(n_features_per_split, d)
        features = rng.choice(d, size=n_feat, replace=False)
        best_score, best_f, best_t = -np.inf, None, None
        for f in features:
            col = X_node[:, f]
            if np.var(col) < 1e-12:
                continue
            for pct in range(10, 100, 10):
                thresh = np.percentile(col, pct)
                if thresh <= np.min(col) or thresh >= np.max(col):
                    continue
                side_full = np.zeros(n, dtype=bool)
                side_full[node_indices] = (col < thresh)
                # in-node pairs only → out-of-node entries are never consulted
                loss = _contrastive_loss_from_mask(side_full, pos_n, neg_n, temperature)
                info_gain = _info_gain_feature(col, col < thresh)
                score = -loss + 0.1 * info_gain
                if score > best_score:
                    best_score, best_f, best_t = score, int(f), float(thresh)
        return best_f, best_t

    def build_node(node_indices, depth):
        if depth >= max_depth or len(node_indices) < 2:
            return make_leaf(node_indices, leaf_ids)

        if pos_pairs is not None:
            feat, thresh = best_contrastive_split(node_indices)
        else:
            feat, thresh = None, None

        if feat is None:
            feat, thresh = _random_split(X[node_indices], rng)
            if feat is None:
                return make_leaf(node_indices, leaf_ids)

        left_mask = X[node_indices, feat] < thresh
        left_indices = node_indices[left_mask]
        right_indices = node_indices[~left_mask]
        if len(left_indices) == 0 or len(right_indices) == 0:
            return make_leaf(node_indices, leaf_ids)

        return {
            "feature": int(feat),
            "thresh": float(thresh),
            "left": build_node(left_indices, depth + 1),
            "right": build_node(right_indices, depth + 1),
        }

    if n == 0:
        return leaf_ids, {"leaf": 0}, 1
    root = build_node(np.arange(n), 0)
    return leaf_ids, root, next_leaf_id[0]


def fit_contrastive_tree(X, max_depth=5, n_pairs=20, temperature=1.0,
                         random_state=42):
    """Fit a contrastive tree and return a :class:`ContrastiveTree`.

    ``tree.apply(X)`` reproduces the training leaf ids and can be applied to
    new data, which keeps the contrastive embedding consistent out-of-sample.
    """
    rng = np.random.default_rng(random_state)
    _, root, n_leaves = _fit_tree(X, max_depth, n_pairs, temperature, rng)
    return ContrastiveTree(root, n_leaves)


def build_contrastive_tree(X, max_depth=5, n_pairs=20, temperature=1.0,
                           random_state=42):
    """Build one contrastive tree.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    max_depth : int
    n_pairs : int
        Number of positive and negative pairs to generate.
    temperature : float
    random_state : int

    Returns
    -------
    leaf_ids : ndarray of shape (n,)
    """
    rng = np.random.default_rng(random_state)
    leaf_ids, _, _ = _fit_tree(X, max_depth, n_pairs, temperature, rng)
    return leaf_ids


def _build_random_tree(X, max_depth=5, rng=None):
    """Build a random tree (fallback). Retained for backward compatibility."""
    if rng is None:
        rng = np.random.default_rng()
    # n_pairs=0 forces the random-split path inside _fit_tree.
    leaf_ids, _, _ = _fit_tree(X, max_depth, 0, 1.0, rng)
    return leaf_ids
