"""Shared pytest fixtures for forest-clustering weighted embedding tests."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Return a small synthetic mixed-type DataFrame for clustering."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "num_b": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "cat_a": ["x", "y", "x", "y", "x", "y", "x", "y"],
        "cat_b": ["a", "a", "b", "b", "a", "a", "b", "b"],
    })


@pytest.fixture
def sample_embedding():
    """Return a synthetic (n, L) int64 embedding matrix with known structure.

    n=5 samples, L=4 iterations.  Cell IDs are small enough that differences
    are easy to reason about manually.
    """
    return np.array([
        [0, 1, 2, 3],   # sample 0
        [0, 1, 0, 1],   # sample 1
        [1, 0, 2, 3],   # sample 2
        [1, 1, 0, 1],   # sample 3
        [0, 0, 2, 3],   # sample 4
    ], dtype=np.int64)


@pytest.fixture
def uniform_weights():
    """Return a uniform weight vector of length L=4 (all ones)."""
    return np.ones(4, dtype=np.float64)


@pytest.fixture
def perfect_separation_embedding():
    """Return an embedding where every sample lands in its own unique cell.

    This maximizes entropy / Gini impurity for every iteration.
    n=5, L=3 — each column has 5 unique values.
    """
    return np.array([
        [0, 5, 10],
        [1, 6, 11],
        [2, 7, 12],
        [3, 8, 13],
        [4, 9, 14],
    ], dtype=np.int64)


@pytest.fixture
def no_separation_embedding():
    """Return an embedding where every sample lands in the same cell.

    This minimizes entropy / Gini impurity to zero for every iteration.
    n=5, L=3 — all values identical in each column.
    """
    return np.zeros((5, 3), dtype=np.int64)


@pytest.fixture
def mixed_quality_embedding():
    """Return an embedding with varying iteration quality.

    Column 0: perfect separation (5 unique cells) — best.
    Column 1: moderate separation (3 unique cells).
    Column 2: no separation (all same cell) — worst.
    """
    return np.array([
        [0, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
        [3, 2, 0],
        [4, 2, 0],
    ], dtype=np.int64)
