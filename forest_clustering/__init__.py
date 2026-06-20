from .clusterer import ForestClusterer
from .distance import pairwise_hamming, pairwise_hamming_chunked, cross_hamming
from .iteration_weights import compute_iteration_weights
from .weighted_distance import (
    pairwise_weighted_hamming,
    weighted_cross_hamming,
    pairwise_weighted_hamming_fast,
    pairwise_weighted_hamming_chunked,
    weighted_cross_hamming_fast,
)
from .transformer import ForestTransformer
from .kde_cuts import kde_peaks_cut_points
from .adaptive_bins import compute_adaptive_bins
from .correlation_aware import build_correlation_groups, select_features_correlation_aware
from .permutation_importance import compute_permutation_importance
from .graph_clustering import GraphLouvainClusterer
from .preflight import hopkins_statistic, gap_statistic, clusterability_test
from .significance import (
    permutation_test_ari,
    bootstrap_ci_ari,
    paired_permutation_test,
    cluster_significance,
    apply_multiple_testing_correction,
)
from .lsh_graph import (
    batched_hamming_knn,
    build_sparse_knn_graph,
    lsh_banding_knn,
    auto_band_size,
)
from .sparse_features import weighted_onehot_features
from .contrastive_splits import (
    augment_sample,
    generate_pairs,
    contrastive_loss,
    evaluate_split_contrastive,
    build_contrastive_tree,
)

__version__ = "0.4.0"

__all__ = [
    "ForestClusterer",
    "ForestTransformer",
    "GraphLouvainClusterer",
    "pairwise_hamming",
    "pairwise_hamming_chunked",
    "cross_hamming",
    "compute_iteration_weights",
    "pairwise_weighted_hamming",
    "weighted_cross_hamming",
    "pairwise_weighted_hamming_fast",
    "pairwise_weighted_hamming_chunked",
    "weighted_cross_hamming_fast",
    "kde_peaks_cut_points",
    "compute_adaptive_bins",
    "build_correlation_groups",
    "select_features_correlation_aware",
    "compute_permutation_importance",
    "hopkins_statistic",
    "gap_statistic",
    "clusterability_test",
    "permutation_test_ari",
    "bootstrap_ci_ari",
    "paired_permutation_test",
    "cluster_significance",
    "apply_multiple_testing_correction",
    "batched_hamming_knn",
    "build_sparse_knn_graph",
    "lsh_banding_knn",
    "auto_band_size",
    "weighted_onehot_features",
    "augment_sample",
    "generate_pairs",
    "contrastive_loss",
    "evaluate_split_contrastive",
    "build_contrastive_tree",
    "__version__",
]
