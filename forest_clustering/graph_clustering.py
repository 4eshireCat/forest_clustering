"""Graph-based clustering via Louvain or Leiden community detection on kNN graphs."""
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


class GraphLouvainClusterer:
    """Clustering via community detection on a kNN similarity graph.

    A directed kNN graph (Hamming distances on edges) is symmetrised, its
    distances are turned into similarities, and a community-detection algorithm
    partitions the graph.  Two backends are available:

    - ``'louvain'`` — NetworkX Louvain (pure Python, always available).
    - ``'leiden'``  — Leiden via ``leidenalg`` + ``igraph`` (faster and avoids
      Louvain's badly-connected-community artefacts).  Requires the optional
      ``leiden`` extra: ``pip install forest-clustering[leiden]``.

    Parameters
    ----------
    n_neighbors : int, default 15
        Number of neighbours for the kNN graph.
    resolution : float, default 1.0
        Resolution parameter; higher values yield more, smaller communities.
        Used identically by both backends (Louvain modularity / Leiden RB
        configuration).
    weight_transform : {'exp', 'linear', 'inverse'}, default 'exp'
        Distance-to-similarity transform: ``'exp'`` RBF ``exp(-d^2/2σ^2)``,
        ``'linear'`` ``1 - d/max(d)``, ``'inverse'`` ``1/(d+eps)``.
    noise_strategy : {'mark', 'merge', 'singleton'}, default 'mark'
        Post-processing of singleton communities: ``'mark'`` labels them ``-1``,
        ``'merge'`` merges them into the nearest community (needs a distance
        matrix; a no-op in the matrix-free :meth:`fit_embedding` path),
        ``'singleton'`` keeps them.
    mutual_knn : bool, default False
        If True use mutual kNN (intersection of directions) for stricter,
        sparser connectivity instead of the union.
    community_method : {'louvain', 'leiden'}, default 'louvain'
        Community-detection backend (see above).
    random_state : int or None, default None
        Seed for the community-detection algorithm.
    """

    labels_ = None

    def __init__(self, n_neighbors=15, resolution=1.0, weight_transform='exp',
                 noise_strategy='mark', mutual_knn=False,
                 community_method='louvain', random_state=None):
        if weight_transform not in ('exp', 'linear', 'inverse'):
            raise ValueError(f"weight_transform must be 'exp', 'linear', or 'inverse', got {weight_transform}")
        if noise_strategy not in ('mark', 'merge', 'singleton'):
            raise ValueError(f"noise_strategy must be 'mark', 'merge', or 'singleton', got {noise_strategy}")
        if community_method not in ('louvain', 'leiden'):
            raise ValueError(f"community_method must be 'louvain' or 'leiden', got {community_method}")
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {n_neighbors}")
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")

        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.weight_transform = weight_transform
        self.noise_strategy = noise_strategy
        self.mutual_knn = mutual_knn
        self.community_method = community_method
        self.random_state = random_state
        self.labels_ = None
        self._is_louvain_clusterer = True

    def _detect_communities(self, similarity, n):
        """Partition the similarity graph; returns an int label per node.

        Dispatches to NetworkX Louvain or leidenalg/igraph Leiden according to
        ``self.community_method``.
        """
        if self.community_method == 'leiden':
            try:
                import igraph as ig
                import leidenalg
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "community_method='leiden' requires the optional 'leiden' "
                    "extra. Install with: pip install forest-clustering[leiden]"
                ) from exc
            coo = similarity.tocoo()
            upper = coo.row < coo.col          # undirected: keep each edge once
            g = ig.Graph(n=n, edges=list(zip(coo.row[upper].tolist(),
                                              coo.col[upper].tolist())))
            part = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights=coo.data[upper].tolist(),
                resolution_parameter=self.resolution,
                seed=int(self.random_state) if self.random_state is not None else 0,
            )
            return np.asarray(part.membership, dtype=np.int64)

        # NetworkX Louvain
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "community_method='louvain' requires networkx. Install with: "
                "pip install forest-clustering[graph] (or use community_method='leiden')."
            ) from exc
        G = nx.from_scipy_sparse_array(similarity, edge_attribute='weight')
        communities = nx.community.louvain_communities(
            G, weight='weight', resolution=self.resolution, seed=self.random_state
        )
        labels = np.zeros(n, dtype=np.int64)
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                labels[node] = comm_id
        return labels

    def _run_louvain_on_knn(self, knn_graph, n, D_for_merge=None):
        """Symmetrise, transform to similarities, detect communities, denoise.

        Parameters
        ----------
        knn_graph : scipy.sparse matrix, shape (n, n)
            Directed kNN graph with Hamming distances as edge weights.
        n : int
            Number of points.
        D_for_merge : ndarray of shape (n, n) or None
            Distance matrix used by ``noise_strategy='merge'``.  If None,
            merging is a no-op.
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
        # kNN builders return unsigned-integer distances (uint16/uint32); cast to
        # float64 before any arithmetic so that e.g. -(d**2) does not wrap around
        # to a huge positive value and blow up np.exp.
        coo.data = coo.data.astype(np.float64)

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

        # Build similarity graph and detect communities (Louvain or Leiden)
        similarity = sparse.coo_matrix((coo.data, (coo.row, coo.col)), shape=(n, n))
        labels = self._detect_communities(similarity, n)

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

    def fit_embedding(self, E, k=None, method='auto', band_size=6,
                      max_bucket=150, banding_threshold=20000):
        """Fit on the integer cell-id embedding directly, without a precomputed
        distance matrix.

        Builds a sparse kNN graph from the embedding, then runs Louvain community
        detection.  Two graph builders are available:

        - ``'knn'``    : exact batched-Hamming kNN (``O(n^2 * L)`` time, but
          ``O(n*k)`` memory).  Best for small/medium ``n``.
        - ``'banding'``: LSH banding — candidate neighbours from shared cell-id
          tuples, exact Hamming on candidates only.  ``O(n*c)`` memory and
          sub-quadratic time; the large-``n`` path.
        - ``'auto'``   : ``'banding'`` when ``n > banding_threshold`` else
          ``'knn'``.

        Works for arbitrary integer embeddings (values in ``[0, K-1]``).

        Parameters
        ----------
        E : ndarray of shape (n, m)
            Integer cell-id embedding matrix from forest-clustering.
        k : int or None
            Override ``n_neighbors``.  If None, uses ``self.n_neighbors``.
        method : {'auto', 'knn', 'banding'}, default 'auto'
            Graph-construction strategy (see above).
        band_size, max_bucket : int
            LSH banding parameters (used only when banding is selected).
        banding_threshold : int, default 20000
            ``n`` above which ``'auto'`` switches to banding.
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

        if method not in ('auto', 'knn', 'banding'):
            raise ValueError(
                f"method must be 'auto', 'knn', or 'banding', got {method!r}"
            )

        from .lsh_graph import batched_hamming_knn, lsh_banding_knn

        use_banding = method == 'banding' or (
            method == 'auto' and n > banding_threshold
        )

        if use_banding:
            self.knn_graph_ = lsh_banding_knn(
                E, k=k, band_size=band_size, max_bucket=max_bucket,
                random_state=self.random_state,
            )
            # Safety net: if banding produced an (almost) empty graph (e.g. a
            # high-entropy embedding where no band collides), fall back to exact
            # kNN so we never silently return all-singletons.
            if self.knn_graph_.nnz < n:
                self.knn_graph_ = batched_hamming_knn(E, k=k)
        else:
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
        """Fit on a precomputed distance matrix ``D`` and return cluster labels.

        For the matrix-free path use :meth:`fit_embedding` instead.
        """
        self.fit(D, y)
        return self.labels_
