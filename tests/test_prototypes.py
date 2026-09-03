import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from forest_clustering import PrototypeSampler, SubsampledClusterer


def make_mixed_redundant():
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for c, center in enumerate([0.0, 5.0, 10.0]):
        for i in range(40):
            rows.append({
                "x": center + rng.normal(0, 0.15),
                "y": center + rng.normal(0, 0.15),
                "cat": f"group_{c}",
            })
            labels.append(c)
    # Duplicate a few rows exactly to force compressible buckets.
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:30]], ignore_index=True)
    labels = np.r_[labels, labels[:30]]
    return df, labels


def test_leaf_signature_sampler_resamples_with_weights_and_expands_labels():
    X, _ = make_mixed_redundant()
    sampler = PrototypeSampler(
        method="leaf_signature",
        n_partitions=24,
        n_prototypes="auto",
        compression=0.4,
        signature_depth=2,
        preserve_rare=False,
        random_state=0,
    )
    Xp, w = sampler.fit_resample(X)

    assert len(Xp) == len(w) == sampler.n_prototypes_
    assert sampler.inverse_assignment_.shape == (len(X),)
    assert np.isclose(w.sum(), len(X))
    assert sampler.compression_summary()["compression_ratio"] < 1.0

    proto_labels = np.arange(sampler.n_prototypes_)
    full_labels = sampler.expand_labels(proto_labels)
    assert full_labels.shape == (len(X),)
    assert np.all(full_labels == proto_labels[sampler.inverse_assignment_])


def test_leaf_signature_transform_assigns_training_rows_to_known_prototypes():
    X, _ = make_mixed_redundant()
    sampler = PrototypeSampler(
        method="leaf_signature",
        n_partitions=16,
        compression=0.5,
        signature_depth=2,
        preserve_rare=False,
        random_state=1,
    ).fit(X)
    assignment = sampler.transform(X)
    assert assignment.shape == (len(X),)
    assert assignment.min() >= 0
    assert assignment.max() < sampler.n_prototypes_


def test_preserve_rare_keeps_singletons_as_weight_one():
    X = pd.DataFrame({"x": [0, 0, 0, 10, 20], "cat": ["a", "a", "a", "rare1", "rare2"]})
    sampler = PrototypeSampler(
        method="leaf_signature",
        n_partitions=8,
        signature_depth=1,
        preserve_rare=True,
        rare_bucket_min_size=2,
        random_state=2,
    ).fit(X)
    assert np.any(sampler.sample_weight_ == 1.0)
    assert sampler.compression_summary()["rare_bucket_count"] >= 1


def test_birch_sampler_returns_valid_weights_and_assignments():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.20, random_state=0)
    sampler = PrototypeSampler(method="birch", compression=0.25, birch_threshold=0.35, random_state=0)
    Xp, w = sampler.fit_resample(X)
    assert len(Xp) == len(w) == sampler.n_prototypes_
    assert np.isclose(w.sum(), X.shape[0])
    assert sampler.transform(X[:5]).shape == (5,)
    assert sampler.compression_summary()["reconstruction_error_mean"] >= 0


def test_subsampled_clusterer_expands_prototype_labels_to_full_data():
    X, y = make_blobs(n_samples=180, centers=3, cluster_std=0.25, random_state=3)
    model = SubsampledClusterer(
        sampler=PrototypeSampler(method="birch", compression=0.25, birch_threshold=0.30, random_state=3),
        clusterer=KMeans(n_clusters=3, n_init=10, random_state=3),
        random_state=3,
    )
    labels = model.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert len(model.prototype_labels_) == model.sampler_.n_prototypes_
    assert model.compression_summary()["compression_ratio"] < 1.0
    assert adjusted_rand_score(y, labels) > 0.9
    assert model.predict(X[:10]).shape == (10,)


def test_invalid_sampler_parameters_raise_clear_errors():
    with pytest.raises(ValueError):
        PrototypeSampler(method="unknown").fit([[1], [2]])
    with pytest.raises(ValueError):
        PrototypeSampler(compression=1.5).fit([[1], [2]])
    with pytest.raises(ValueError):
        PrototypeSampler(representative="bad").fit([[1], [2]])


def test_leaf_signature_medoid_uses_hamming_geometry():
    """Integer leaf ids are nominal, so their magnitudes must not affect a medoid."""
    signatures = np.array([
        [300_000_000_000_000, 0, 0, 0, 0],
        [300_000_000_000_000, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1_000_000_000_000_000, 1, 1, 1, 1],
    ], dtype=np.int64)
    sampler = PrototypeSampler(method="leaf_signature", representative="medoid")

    representative = sampler._representative_index(
        np.arange(len(signatures)), signatures
    )

    assert representative == 2


def test_subsampled_clusterer_classifier_assignment_predicts():
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.25, random_state=8)
    model = SubsampledClusterer(
        sampler=PrototypeSampler(method="birch", compression=0.3, birch_threshold=0.35, random_state=8),
        clusterer=KMeans(n_clusters=3, n_init=5, random_state=8),
        assignment="classifier",
        random_state=8,
    ).fit(X)
    pred = model.predict(X[:7])
    assert pred.shape == (7,)
