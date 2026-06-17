"""Graph-based clustering using Louvain community detection on KNN graphs."""
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


class GraphLouvainClusterer:
    """Clustering via Louvain community detection on KNN similarity graphs.
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for KNN graph (default 15).
    resolution : float
        Louvain resolution parameter (default 1.0).
    weight_transform : str
        'exp' (RBF), 'linear' (1-D), 'inverse' (1/D).
    noise_strategy : str
        'mark' (-1 for outliers), 'merge' (merge singletons), 'singleton' (keep).
    mutual_knn : bool
        Use mutual KNN (both directions) for stricter connectivity.
    random_state : int or None
    """
    
    labels_ = None
    
    def __init__(self, n_neighbors=15, resolution=1.0, weight_transform='exp',
                 noise_strategy='mark', mutual_knn=False, random_state=None):
        if weight_transform not in ('exp', 'linear', 'inverse'):
            raise ValueError(f"weight_transform must be 'exp', 'linear', or 'inverse', got {weight_transform}")
        if noise_strategy not in ('mark', 'merge', 'singleton'):
            raise ValueError(f"noise_strategy must be 'mark', 'merge', or 'singleton', got {noise_strategy}")
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {n_neighbors}")
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")

        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.weight_transform = weight_transform
        self.noise_strategy = noise_strategy
        self.mutual_knn = mutual_knn
        self.random_state = random_state
        self.labels_ = None
        self._is_louvain_clusterer = True
    
    def _run_louvain_on_knn(self, knn_graph, n, D_for_merge=None):
        """Shared logic: symmetrize, weight-transform, Louvain, noise post-process.

        Parameters
        ----------
        knn_graph : scipy.sparse matrix, shape (n, n)
            Directed kNN graph with Hamming distances as edge weights.
        n : int
            Number of points.
        D_for_merge : ndarray of shape (n, n) or None
            Distance matrix used by noise_strategy='merge'.  If None,
            merging falls back to a zero matrix (no-op).
        """
        # Guard: if all KNN distances are zero (identical points),
        # scipy.sparse drops zero entries during symmetrization → empty graph.
        knn_data = knn_graph.data
        if len(knn_data) > 0 and knn_data.max() < 1e-15:
            return np.zeros(n, dtype=np.int64)

        # Symmetrize if needed
        if self.mutual_knn:
            knn_graph = knn_graph.minimum(knn_graph.T)
        else:
            knn_graph = knn_graph.maximum(knn_graph.T)

        # Convert distances to similarities
        coo = knn_graph.tocoo()

        if self.weight_transform == 'exp':
            nonzero_dists = coo.data[coo.data > 0]
            if len(nonzero_dists) > 0:
                sigma = float(np.median(nonzero_dists)) + 1e-10
            else:
                sigma = 1.0
            coo.data = np.exp(-(coo.data ** 2) / (2 * sigma ** 2))
        elif self.weight_transform == 'linear':
            max_d = coo.data.max() if len(coo.data) > 0 else 1.0
            if max_d > 1e-10:
                coo.data = 1.0 - coo.data / max_d
            else:
                coo.data = np.ones_like(coo.data)
            coo.data = np.clip(coo.data, 1e-10, 1.0)
        elif self.weight_transform == 'inverse':
            coo.data = 1.0 / (coo.data + 1e-10)

        # Build NetworkX graph
        similarity = sparse.coo_matrix((coo.data, (coo.row, coo.col)), shape=(n, n))
        G = nx.from_scipy_sparse_array(similarity, edge_attribute='weight')

        # Run Louvain
        seed = self.random_state
        communities = nx.community.louvain_communities(
            G, weight='weight', resolution=self.resolution, seed=seed
        )

        # Convert communities to labels
        labels = np.zeros(n, dtype=np.int64)
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                labels[node] = comm_id

        # Post-process noise
        if self.noise_strategy == 'mark':
            unique, counts = np.unique(labels, return_counts=True)
            singleton_comms = unique[counts == 1]
            if len(singleton_comms) == len(unique):
                labels = np.zeros(n, dtype=np.int64)
            else:
                for sc in singleton_comms:
                    labels[labels == sc] = -1
                if len(singleton_comms) > 0:
                    labels = self._renumber_labels(labels)
        elif self.noise_strategy == 'merge':
            if D_for_merge is not None:
                labels = self._merge_singletons(labels, D_for_merge)

        return labels.astype(np.int64)

    def fit(self, D, y=None):
        """Fit Louvain clustering on distance matrix.

        Parameters
        ----------
        D : ndarray of shape (n, n)
            Pairwise distance matrix.
        """
        D = np.asarray(D)
        n = D.shape[0]

        if n == 0:
            self.labels_ = np.array([], dtype=np.int64)
            return self

        if n == 1:
            self.labels_ = np.array([0], dtype=np.int64)
            return self

        # Adjust k if too large
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            k = 1

        # Build KNN graph from distance matrix
        knn_graph = kneighbors_graph(
            D, n_neighbors=k, mode='distance',
            metric='precomputed', include_self=False
        )

        self.labels_ = self._run_louvain_on_knn(knn_graph, n, D_for_merge=D)
        return self

    def fit_embedding(self, E, k=None):
        """Fit on the integer cell-id embedding directly, without a precomputed
        distance matrix.

        Builds a sparse kNN graph from the embedding using batched Hamming
        distance (number of differing iterations), then runs Louvain community
        detection.  Works for arbitrary integer embeddings (values in
        ``[0, K-1]``); a genuinely binary embedding is handled by the same path.

        Parameters
        ----------
        E : ndarray of shape (n, m)
            Integer cell-id embedding matrix from forest-clustering.
        k : int or None
            Override n_neighbors.  If None, uses ``self.n_neighbors``.
        """
        E = np.asarray(E)
        n = E.shape[0]

        if n == 0:
            self.labels_ = np.array([], dtype=np.int64)
            return self

        if n == 1:
            self.labels_ = np.array([0], dtype=np.int64)
            return self

        k = k or self.n_neighbors
        k = min(k, n - 1)
        if k < 1:
            k = 1

        # Build sparse kNN graph from embedding
        from .lsh_graph import batched_hamming_knn

        self.knn_graph_ = batched_hamming_knn(E, k=k)

        self.labels_ = self._run_louvain_on_knn(self.knn_graph_, n, D_for_merge=None)
        return self
    
    def _renumber_labels(self, labels):
        """Renumber labels so they're contiguous starting from 0, -1 stays -1."""
        unique_labels = np.unique(labels[labels >= 0])
        new_labels = np.full(len(labels), -1, dtype=np.int64)
        for new_id, old_id in enumerate(unique_labels):
            new_labels[labels == old_id] = new_id
        return new_labels
    
    def _merge_singletons(self, labels, D):
        """Merge singleton communities into their nearest non-singleton neighbor."""
        unique, counts = np.unique(labels, return_counts=True)
        singleton_comms = set(unique[counts == 1])
        if not singleton_comms:
            return labels
        
        non_singleton = [u for u in unique if u not in singleton_comms and u >= 0]
        if not non_singleton:
            return labels  # all singletons — keep as-is
        
        new_labels = labels.copy()
        for i, label in enumerate(labels):
            if label in singleton_comms:
                # Find nearest non-singleton neighbor
                mask = np.isin(labels, non_singleton)
                mask[i] = False
                if mask.any():
                    nearest = np.argmin(D[i, mask])
                    valid_indices = np.where(mask)[0]
                    new_labels[i] = labels[valid_indices[nearest]]
        
        return self._renumber_labels(new_labels)
    
    def fit_predict(self, D, y=None):
        self.fit(D, y)
        return self.labels_
