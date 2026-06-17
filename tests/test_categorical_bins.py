"""
Tests for Fix #4: Categorical Bin Count Auto-Adjustment.

When adaptive_bins=False, categorical features must get
    n_bins_eff = clip(n_unique, min_bins, B_max)
instead of the fixed n_bins parameter.

Numerical features must remain unchanged (fixed n_bins).

B_max = min(max_bins, Sturges(n)) where Sturges(n) = ceil(log2(max(n,2)) + 1)

Test Classes (12 classes, 160+ test cases):
    TestCategoricalBinsCore        — Binary/multi-level categorical → correct n_unique bins
    TestCategoricalBinsCap         — B_max cap enforcement (Sturges + max_bins)
    TestNumericalUnchanged         — Numerical features unaffected by fix
    TestMixedTypes                 — Mixed cat+num datasets
    TestCovertypeRegression        — 40 binary soil_type columns → 2 bins each
    TestNurseryRegression          — 2-5 level categorical features
    TestEdgeCases                  — Constant cat, min_bins wins, tiny datasets
    TestInvariants                 — Property-based: P1-P5 invariants via parametrize
    TestBackwardCompatibility      — Pure numeric datasets unchanged
    TestEquivalenceWithAdaptiveBins — adaptive_bins=True vs False match for cats
    TestAdaptiveBinsMapContent     — adaptive_bins_map_ has correct entries
    TestStringCategoricals         — String-labeled categorical features
    TestApplyCategoricalBinCap     — Direct unit tests for apply_categorical_bin_cap()
                                       (TC-01 through TC-15 from spec)

Total: ~162 tests (1 skipped due to parameter preconditions).
"""

import sys
import math
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, '/mnt/agents/output/forest_clustering_weighted')
from forest_clustering import ForestClusterer
from forest_clustering.adaptive_bins import compute_adaptive_bins
from forest_clustering.partitioner import apply_categorical_bin_cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sturges(n):
    """Sturges rule: ceil(log2(max(n, 2)) + 1)."""
    return int(np.ceil(np.log2(max(n, 2)) + 1))


def _b_max(n, max_bins=10):
    """Global effective maximum B_max = min(max_bins, Sturges(n))."""
    return min(max_bins, _sturges(n))


def _collect_cat_k_by_col_idx(specs):
    """From a list of IterationSpec, collect all (col_idx, K) pairs for
    categorical BinSpecs.  Returns dict[col_idx] -> set of observed K values."""
    result = {}
    for spec in specs:
        for bs in spec.bin_specs:
            if bs.type == "categorical":
                result.setdefault(bs.col_idx, set()).add(bs.K)
    return result


def _collect_num_k_by_col_idx(specs):
    """From a list of IterationSpec, collect all (col_idx, K) pairs for
    numerical BinSpecs.  Returns dict[col_idx] -> set of observed K values."""
    result = {}
    for spec in specs:
        for bs in spec.bin_specs:
            if bs.type == "numerical":
                result.setdefault(bs.col_idx, set()).add(bs.K)
    return result


def _get_feature_index_map(fc, X):
    """Return a dict mapping column names (or indices) to internal encoded
    column indices.  We use fc.encoder_.feature_names_."""
    if hasattr(fc.encoder_, 'feature_names_'):
        return {name: idx for idx, name in enumerate(fc.encoder_.feature_names_)}
    return {i: i for i in range(X.shape[1])}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Core correctness: categorical features get n_unique bins
# ---------------------------------------------------------------------------

class TestCategoricalBinsCore:
    """Categorical features must receive n_bins = min(n_unique, B_max)."""

    def test_binary_categorical_gets_2_bins(self):
        """TC-01 [E2]: Binary categorical with n_bins=5 must get 2 bins."""
        X = pd.DataFrame({
            "a": [0, 1, 0, 1],
            "b": [0, 1, 0, 1],
        })
        fc = ForestClusterer(
            n_iterations=5,
            n_bins=5,
            n_features=2,          # select ALL features every iteration
            adaptive_bins=False,
            feature_types={"a": "categorical", "b": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"Expected binary cat 'a' to get K=2, got {cat_ks[0]}"
        assert cat_ks[1] == {2}, f"Expected binary cat 'b' to get K=2, got {cat_ks[1]}"

    def test_binary_categorical_explicit_feature_types(self):
        """Same as above but using integer column indices."""
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            feature_types={0: "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"Expected K=2 for binary cat, got {cat_ks[0]}"

    def test_multi_level_categorical_exact_match(self):
        """TC-04 [E4]: 5-level categorical with B_max >= 5 must get 5 bins."""
        X = pd.DataFrame({"a": list(range(10)) + list(range(10))})  # 10 unique values
        n = len(X)
        max_bins = 10
        expected = min(10, _b_max(n, max_bins))

        fc = ForestClusterer(
            n_iterations=5,
            n_bins=3,               # request 3, but categorical should get 10
            n_features=1,
            adaptive_bins=False,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {expected}, (
            f"Expected {expected} bins for 10-level cat (B_max={_b_max(n, max_bins)}), "
            f"got {cat_ks[0]}"
        )

    def test_categorical_4_levels_default_params(self):
        """Categorical with 4 unique, n=4 -> Sturges(4)=3, B_max=3,
        so K = max(min(4, 3), 2) = 3 (Sturges cap applies)."""
        X = pd.DataFrame({"a": [0, 1, 2, 3]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=3,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        # n=4 -> Sturges=3, B_max=3, n_unique=4 -> K=max(min(4,3), 2)=3
        assert cat_ks[0] == {3}, f"Expected 3 bins (Sturges cap), got {cat_ks[0]}"

    def test_categorical_3_levels_n_bins_5(self):
        """3-level categorical with n_bins=5 must get 3 bins."""
        X = pd.DataFrame({"a": [0, 1, 2, 0, 1, 2]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {3}, f"Expected 3 bins for 3-level cat, got {cat_ks[0]}"


# ---------------------------------------------------------------------------
# 2. Cap enforcement: n_unique > B_max
# ---------------------------------------------------------------------------

class TestCategoricalBinsCap:
    """Categorical features must be capped at B_max."""

    def test_categorical_above_max_bins_cap(self):
        """TC-05 [E5]: Categorical with 100 unique, max_bins=10 -> capped at B_max."""
        X = pd.DataFrame({"a": list(range(100))})
        n = 100
        max_bins = 10
        expected = _b_max(n, max_bins)

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=50,              # request 50, but should be capped
            n_features=1,
            adaptive_bins=False,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {expected}, (
            f"Expected {expected} bins (B_max={_b_max(n, max_bins)}), got {cat_ks[0]}"
        )

    def test_categorical_above_sturges_cap(self):
        """Even with very large max_bins, Sturges cap still applies."""
        X = pd.DataFrame({"a": list(range(50))})
        n = 50
        max_bins = 1000  # huge, won't be the limiting factor
        expected = _b_max(n, max_bins)  # Sturges(50) = 7

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=50,
            n_features=1,
            adaptive_bins=False,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {expected}, (
            f"Expected {expected} bins (Sturges cap), got {cat_ks[0]}"
        )

    @pytest.mark.parametrize("n_unique,n,max_bins,expected", [
        (100, 50, 10, 7),     # Sturges(50)=7, min(10,7)=7
        (100, 50, 5, 5),      # max_bins=5 wins
        (20, 20, 10, 6),      # Sturges(20)=6, min(10,6)=6
        (3, 100, 10, 3),      # n_unique=3 < B_max=8
        (2, 1000, 10, 2),     # binary
    ])
    def test_categorical_cap_various(self, n_unique, n, max_bins, expected):
        """Parametrized: categorical bin count respects B_max cap."""
        rng = np.random.default_rng(42)
        values = list(range(n_unique)) * (n // n_unique + 1)
        values = values[:n]
        rng.shuffle(values)
        X = pd.DataFrame({"a": values})

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=100,
            n_features=1,
            adaptive_bins=False,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {expected}, (
            f"n_unique={n_unique}, n={n}, max_bins={max_bins}: "
            f"expected {expected}, got {cat_ks[0]}"
        )


# ---------------------------------------------------------------------------
# 3. Numerical features: unchanged
# ---------------------------------------------------------------------------

class TestNumericalUnchanged:
    """Numerical features must still get fixed n_bins."""

    def test_numerical_unchanged(self):
        """TC-08 [N1]: Numerical features must use fixed n_bins."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=3,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        num_ks = _collect_num_k_by_col_idx(fc.specs_)
        assert num_ks[0] == {3}, f"Expected numerical to get K=3 (fixed n_bins), got {num_ks[0]}"

    def test_numerical_different_n_bins(self):
        """Numerical with n_bins=7 must get 7 bins."""
        X = pd.DataFrame({"a": np.linspace(0, 1, 50)})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=7,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        num_ks = _collect_num_k_by_col_idx(fc.specs_)
        assert num_ks[0] == {7}, f"Expected K=7, got {num_ks[0]}"

    def test_numerical_n_bins_1(self):
        """Numerical with n_bins=1 must get 1 bin."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=1,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        num_ks = _collect_num_k_by_col_idx(fc.specs_)
        assert num_ks[0] == {1}, f"Expected K=1, got {num_ks[0]}"


# ---------------------------------------------------------------------------
# 4. Mixed types
# ---------------------------------------------------------------------------

class TestMixedTypes:
    """Mixed datasets: categorical gets n_unique, numerical gets n_bins."""

    def test_mixed_cat_and_num(self):
        """TC-10: cat with 3 unique -> 3 bins, num -> 5 bins."""
        X = pd.DataFrame({
            "cat": [0, 1, 2, 0, 1, 2],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        fc = ForestClusterer(
            n_iterations=5,
            n_bins=5,
            n_features=2,
            adaptive_bins=False,
            feature_types={"cat": "categorical", "num": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        num_ks = _collect_num_k_by_col_idx(fc.specs_)

        assert cat_ks[0] == {3}, f"Expected cat K=3, got {cat_ks[0]}"
        assert num_ks[1] == {5}, f"Expected num K=5, got {num_ks[1]}"

    def test_mixed_multiple_cats_and_nums(self):
        """Multiple categorical and numerical columns."""
        X = pd.DataFrame({
            "cat_a": [0, 1, 0, 1],       # 2 unique
            "cat_b": [0, 1, 2, 3],       # 4 unique
            "num_a": [1.0, 2.0, 3.0, 4.0],
            "num_b": [0.5, 1.5, 2.5, 3.5],
        })
        fc = ForestClusterer(
            n_iterations=5,
            n_bins=5,
            n_features=4,
            adaptive_bins=False,
            feature_types={
                "cat_a": "categorical",
                "cat_b": "categorical",
                "num_a": "numerical",
                "num_b": "numerical",
            },
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        num_ks = _collect_num_k_by_col_idx(fc.specs_)

        assert cat_ks[0] == {2}, f"Expected cat_a K=2, got {cat_ks[0]}"
        # n=4 -> Sturges(4)=3, B_max=3, cat_b has 4 unique -> K=max(min(4,3), 2)=3
        assert cat_ks[1] == {3}, f"Expected cat_b K=3 (Sturges cap), got {cat_ks[1]}"
        assert num_ks[2] == {5}, f"Expected num_a K=5, got {num_ks[2]}"
        assert num_ks[3] == {5}, f"Expected num_b K=5, got {num_ks[3]}"


# ---------------------------------------------------------------------------
# 5. Regression: Covertype-like binary soil_type features
# ---------------------------------------------------------------------------

class TestCovertypeRegression:
    """Covertype dataset: 40 binary soil_type columns must each get 2 bins."""

    def test_covertype_40_binary_soil_columns(self):
        """TC-11 [P8]: 40 binary categorical columns → 2 bins each."""
        n = 500
        rng = np.random.default_rng(42)
        data = {}
        for i in range(40):
            data[f"soil_{i}"] = rng.integers(0, 2, size=n)

        X = pd.DataFrame(data)
        feature_types = {c: "categorical" for c in X.columns}

        fc = ForestClusterer(
            n_iterations=5,
            n_bins=3,               # default; binary cats should NOT get 3
            n_features=40,
            adaptive_bins=False,
            feature_types=feature_types,
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        for col_idx in range(40):
            assert col_idx in cat_ks, f"Column {col_idx} not found in any iteration"
            assert cat_ks[col_idx] == {2}, (
                f"Expected binary soil_type column {col_idx} to get K=2, "
                f"got {cat_ks[col_idx]}"
            )

    def test_covertype_mixed_with_numerical(self):
        """Covertype-like with binary soil columns + one numerical."""
        n = 500
        rng = np.random.default_rng(42)
        data = {}
        for i in range(10):
            data[f"soil_{i}"] = rng.integers(0, 2, size=n)
        data["elevation"] = rng.uniform(0, 4000, size=n)

        X = pd.DataFrame(data)
        feature_types = {c: "categorical" for c in X.columns if c.startswith("soil_")}
        feature_types["elevation"] = "numerical"

        fc = ForestClusterer(
            n_iterations=5,
            n_bins=5,
            n_features=11,
            adaptive_bins=False,
            feature_types=feature_types,
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        num_ks = _collect_num_k_by_col_idx(fc.specs_)

        for col_idx in range(10):
            assert cat_ks[col_idx] == {2}, f"Soil {col_idx}: expected K=2, got {cat_ks[col_idx]}"
        assert num_ks[10] == {5}, f"Elevation: expected K=5, got {num_ks[10]}"


# ---------------------------------------------------------------------------
# 6. Regression: Nursery-like multi-level categorical features
# ---------------------------------------------------------------------------

class TestNurseryRegression:
    """Nursery dataset: all categorical, 2-5 levels per feature."""

    def test_nursery_like_2_to_5_levels(self):
        """TC: Nursery-like features with 2,3,4,5 levels → 2,3,4,5 bins."""
        n = 100
        rng = np.random.default_rng(42)
        data = {
            "parents":   rng.integers(0, 2, size=n),   # 2 unique
            "has_nurs":  rng.integers(0, 3, size=n),   # 3 unique
            "form":      rng.integers(0, 4, size=n),   # 4 unique
            "children":  rng.integers(0, 5, size=n),   # 5 unique
        }
        X = pd.DataFrame(data)
        feature_types = {c: "categorical" for c in X.columns}

        fc = ForestClusterer(
            n_iterations=5,
            n_bins=3,               # old behavior would give 3 to all
            n_features=4,
            adaptive_bins=False,
            feature_types=feature_types,
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"parents (2-level): expected K=2, got {cat_ks[0]}"
        assert cat_ks[1] == {3}, f"has_nurs (3-level): expected K=3, got {cat_ks[1]}"
        assert cat_ks[2] == {4}, f"form (4-level): expected K=4, got {cat_ks[2]}"
        assert cat_ks[3] == {5}, f"children (5-level): expected K=5, got {cat_ks[3]}"


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for categorical bin count."""

    def test_constant_categorical(self):
        """TC-01 [E1]: Constant categorical (1 unique) → min_bins."""
        X = pd.DataFrame({"a": [0, 0, 0, 0]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            min_bins=2,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"Expected constant cat to get K=min_bins=2, got {cat_ks[0]}"

    def test_constant_categorical_min_bins_1(self):
        """TC [G2]: Constant categorical with min_bins=1 → 1 bin."""
        X = pd.DataFrame({"a": [0, 0, 0, 0]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            min_bins=1,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {1}, f"Expected K=1 with min_bins=1, got {cat_ks[0]}"

    def test_binary_cat_high_min_bins(self):
        """TC-03 [E3]: Binary cat with min_bins=3 → 3 bins (min_bins wins)."""
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            min_bins=3,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {3}, f"Expected K=3 (min_bins wins), got {cat_ks[0]}"

    def test_tiny_dataset_sturges_cap(self):
        """TC-07 [E7]: Very small n → Sturges cap is very low."""
        # n=2: Sturges(2)=2, B_max=2, n_unique=1 -> K=max(min(1,2), 2)=2
        X = pd.DataFrame({"a": [0, 0]})
        n = 2
        expected = _b_max(n, max_bins=10)  # Sturges(2)=2, B_max=2

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=10,
            n_features=1,
            n_clusters=1,  # avoid KMeans error with tiny datasets
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        # n_unique=1, min_bins=2, B_max=2 -> K=max(min(1,2), 2)=2
        assert cat_ks[0] == {expected}, (
            f"Tiny dataset: expected K={expected} (B_max={_b_max(n)}), got {cat_ks[0]}"
        )

    def test_n_unique_0_no_valid_categories(self):
        """All values are NaN/invalid for categorical → n_unique effectively 0 or 1."""
        # When all values are -1 (unknown marker for categoricals),
        # build_col_stats sets n_unique = 1 (since len(valid)==0 → n_unique=1)
        X = pd.DataFrame({"a": [0, 0, 0, 0]})  # simplest case: constant
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        # n_unique=1, min_bins=2 → should get 2
        assert 2 in cat_ks[0] or cat_ks[0] == {2}, (
            f"Constant/empty cat: expected K=2 (min_bins), got {cat_ks[0]}"
        )


# ---------------------------------------------------------------------------
# 8. Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Property-based invariants that must hold for all categorical features."""

    @pytest.mark.parametrize("n_unique", [1, 2, 3, 5, 10, 20, 50])
    @pytest.mark.parametrize("n", [5, 10, 50, 100, 1000])
    @pytest.mark.parametrize("max_bins", [5, 10, 20])
    def test_invariant_categorical_range(self, n_unique, n, max_bins):
        """P1: For every categorical feature: min_bins <= K <= B_max."""
        rng = np.random.default_rng(42)
        values = list(range(n_unique)) * (n // n_unique + 1)
        values = values[:n]
        rng.shuffle(values)
        X = pd.DataFrame({"a": values})

        min_bins = 2
        bmax = _b_max(n, max_bins)
        expected = max(min_bins, min(n_unique, bmax))

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=100,  # large fixed value, should be ignored for cats
            n_features=1,
            adaptive_bins=False,
            min_bins=min_bins,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        observed = list(cat_ks[0])[0]

        assert observed == expected, (
            f"Invariant P1 violated: n_unique={n_unique}, n={n}, max_bins={max_bins}: "
            f"expected {expected}, got {observed}"
        )
        assert min_bins <= observed <= bmax, (
            f"Invariant P1 range violated: {min_bins} <= {observed} <= {bmax}"
        )

    @pytest.mark.parametrize("n_unique", [1, 2, 3, 5, 10])
    def test_invariant_exact_match_low_cardinality(self, n_unique):
        """P3: If min_bins <= n_unique <= B_max, then K == n_unique."""
        n = 1000
        max_bins = 10
        bmax = _b_max(n, max_bins)  # Sturges(1000) = 11, B_max = 10
        min_bins = 2

        if n_unique < min_bins or n_unique > bmax:
            pytest.skip(f"n_unique={n_unique} outside [min_bins, B_max] range for this test")

        rng = np.random.default_rng(42)
        values = list(range(n_unique)) * (n // n_unique + 1)
        values = values[:n]
        rng.shuffle(values)
        X = pd.DataFrame({"a": values})

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=3,
            n_features=1,
            adaptive_bins=False,
            min_bins=min_bins,
            max_bins=max_bins,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        observed = list(cat_ks[0])[0]
        assert observed == n_unique, (
            f"P3 exact match failed: expected {n_unique}, got {observed}"
        )

    def test_invariant_binary_exactness(self):
        """P4: Binary categorical (n_unique=2, min_bins<=2) → K=2."""
        X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=10,
            n_features=1,
            adaptive_bins=False,
            min_bins=2,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"P4 binary exactness failed: expected 2, got {cat_ks[0]}"

    def test_invariant_constant_feature(self):
        """P5: Constant categorical (n_unique=1) → K=min_bins."""
        X = pd.DataFrame({"a": [0, 0, 0, 0, 0]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=10,
            n_features=1,
            adaptive_bins=False,
            min_bins=2,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"P5 constant feature failed: expected min_bins=2, got {cat_ks[0]}"


# ---------------------------------------------------------------------------
# 9. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Pure numeric datasets must behave identically before and after fix."""

    def test_pure_numeric_empty_bins_map(self):
        """P7: Pure numeric dataset → adaptive_bins_map_ should be empty/None."""
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [0.5, 1.5, 2.5, 3.5],
        })
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=3,
            n_features=2,
            adaptive_bins=False,
            feature_types={"a": "numerical", "b": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        # With no categoricals, adaptive_bins_map_ should be empty or None
        assert not fc.adaptive_bins_map_, (
            f"Expected empty/None adaptive_bins_map_ for pure numeric, "
            f"got {fc.adaptive_bins_map_}"
        )

    def test_pure_numeric_fixed_bins(self):
        """P7: All numerical features get fixed n_bins."""
        X = pd.DataFrame({
            "a": np.linspace(0, 1, 20),
            "b": np.linspace(10, 20, 20),
        })
        fc = ForestClusterer(
            n_iterations=5,
            n_bins=7,
            n_features=2,
            adaptive_bins=False,
            feature_types={"a": "numerical", "b": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        num_ks = _collect_num_k_by_col_idx(fc.specs_)
        assert num_ks[0] == {7}, f"Expected K=7 for 'a', got {num_ks[0]}"
        assert num_ks[1] == {7}, f"Expected K=7 for 'b', got {num_ks[1]}"


# ---------------------------------------------------------------------------
# 10. Equivalence with adaptive_bins=True for pure categorical
# ---------------------------------------------------------------------------

class TestEquivalenceWithAdaptiveBins:
    """apply_categorical_bin_cap output must match compute_adaptive_bins
    for categorical entries."""

    def test_equivalence_with_adaptive_bins_single_categorical(self):
        """P10: Categorical bin count with adaptive_bins=True vs adaptive_bins=False
        should produce the same K for categorical features."""
        X = pd.DataFrame({
            "cat_2": [0, 1, 0, 1],
            "cat_5": [0, 1, 2, 3],
            "num": [1.0, 2.0, 3.0, 4.0],
        })

        # adaptive_bins=True
        fc_adaptive = ForestClusterer(
            n_iterations=5,
            n_bins=3,
            n_features=3,
            adaptive_bins=True,
            feature_types={
                "cat_2": "categorical",
                "cat_5": "categorical",
                "num": "numerical",
            },
            random_state=42,
        )
        fc_adaptive.fit(X)

        # adaptive_bins=False (with fix)
        fc_fixed = ForestClusterer(
            n_iterations=5,
            n_bins=3,
            n_features=3,
            adaptive_bins=False,
            feature_types={
                "cat_2": "categorical",
                "cat_5": "categorical",
                "num": "numerical",
            },
            random_state=42,
        )
        fc_fixed.fit(X)

        # Compare categorical K values
        cat_ks_adaptive = _collect_cat_k_by_col_idx(fc_adaptive.specs_)
        cat_ks_fixed = _collect_cat_k_by_col_idx(fc_fixed.specs_)

        for col_idx in [0, 1]:
            assert cat_ks_adaptive[col_idx] == cat_ks_fixed[col_idx], (
                f"Equivalence failed for col {col_idx}: "
                f"adaptive={cat_ks_adaptive[col_idx]}, "
                f"fixed={cat_ks_fixed[col_idx]}"
            )


# ---------------------------------------------------------------------------
# 11. Integration: checking via adaptive_bins_map_
# ---------------------------------------------------------------------------

class TestAdaptiveBinsMapContent:
    """Check that adaptive_bins_map_ contains correct entries."""

    def test_bins_map_contains_categorical_entries(self):
        """When categoricals present, adaptive_bins_map_ should have entries."""
        X = pd.DataFrame({
            "cat": [0, 1, 2],
            "num": [1.0, 2.0, 3.0],
        })
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=2,
            adaptive_bins=False,
            feature_types={"cat": "categorical", "num": "numerical"},
            random_state=42,
        )
        fc.fit(X)

        # After fix: adaptive_bins_map_ should contain entry for categorical column
        # (Before fix it was None)
        assert fc.adaptive_bins_map_ is not None, (
            "adaptive_bins_map_ should not be None when categoricals present"
        )
        assert 0 in fc.adaptive_bins_map_, (
            "Categorical column 0 should have entry in adaptive_bins_map_"
        )
        # Numerical column should NOT have entry (uses default n_bins)
        assert 1 not in fc.adaptive_bins_map_, (
            "Numerical column 1 should NOT have entry in adaptive_bins_map_"
        )

    def test_bins_map_values_correct(self):
        """Check actual values in adaptive_bins_map_."""
        X = pd.DataFrame({
            "cat_2": [0, 1, 0, 1, 0],
            "cat_5": [0, 1, 2, 3, 4],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        n = len(X)
        bmax = _b_max(n)

        fc = ForestClusterer(
            n_iterations=3,
            n_bins=10,
            n_features=3,
            adaptive_bins=False,
            feature_types={
                "cat_2": "categorical",
                "cat_5": "categorical",
                "num": "numerical",
            },
            random_state=42,
        )
        fc.fit(X)

        expected_0 = min(2, bmax)  # 2 unique for cat_2
        expected_1 = min(5, bmax)  # 5 unique for cat_5

        assert fc.adaptive_bins_map_[0] == expected_0, (
            f"cat_2: expected {expected_0}, got {fc.adaptive_bins_map_[0]}"
        )
        assert fc.adaptive_bins_map_[1] == expected_1, (
            f"cat_5: expected {expected_1}, got {fc.adaptive_bins_map_[1]}"
        )


# ---------------------------------------------------------------------------
# 12. String-labeled categoricals
# ---------------------------------------------------------------------------

class TestStringCategoricals:
    """Categorical features with string labels."""

    def test_string_categorical_binary(self):
        """Binary string categorical → 2 bins."""
        X = pd.DataFrame({"a": ["x", "y", "x", "y", "x", "y"]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=5,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {2}, f"Expected 2 bins for binary string cat, got {cat_ks[0]}"

    def test_string_categorical_multi_level(self):
        """String categorical with 4 levels → 4 bins."""
        X = pd.DataFrame({"a": ["a", "b", "c", "d", "a", "b", "c", "d"]})
        fc = ForestClusterer(
            n_iterations=3,
            n_bins=3,
            n_features=1,
            adaptive_bins=False,
            feature_types={"a": "categorical"},
            random_state=42,
        )
        fc.fit(X)

        cat_ks = _collect_cat_k_by_col_idx(fc.specs_)
        assert cat_ks[0] == {4}, f"Expected 4 bins for 4-level string cat, got {cat_ks[0]}"


# ---------------------------------------------------------------------------
# 13. Direct unit tests for apply_categorical_bin_cap()
# ---------------------------------------------------------------------------

class TestApplyCategoricalBinCap:
    """Direct unit tests for the apply_categorical_bin_cap function.
    
    These mirror the test cases TC-01 through TC-15 from the spec.
    """

    def test_tc01_constant_cat(self):
        """TC-01 [E1]: Constant categorical (n_unique=1) → min_bins."""
        col_stats = [{"type": "categorical", "n_unique": 1}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        assert result == {0: 2}

    def test_tc02_binary_cat(self):
        """TC-02 [E2]: Binary categorical → 2 bins."""
        col_stats = [{"type": "categorical", "n_unique": 2}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        assert result == {0: 2}

    def test_tc03_binary_cat_high_min(self):
        """TC-03 [E3]: Binary cat with min_bins=3 → 3 bins."""
        col_stats = [{"type": "categorical", "n_unique": 2}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=3, max_bins=10, n=100)
        assert result == {0: 3}

    def test_tc04_moderate_unique(self):
        """TC-04 [E4]: n_unique=4, B_max=8 → 4 bins (exact match)."""
        col_stats = [{"type": "categorical", "n_unique": 4}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        # Sturges(100)=8, B_max=min(10,8)=8, n_unique=4 < 8 -> K=4
        assert result == {0: 4}

    def test_tc05_above_cap(self):
        """TC-05 [E5]: n_unique=100, n=50, B_max=7 → 7 bins."""
        col_stats = [{"type": "categorical", "n_unique": 100}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=50)
        # Sturges(50)=7, B_max=7
        assert result == {0: 7}

    def test_tc06_empty_col_n_unique_0(self):
        """TC-06 [E6]: n_unique=0 → min_bins."""
        col_stats = [{"type": "categorical", "n_unique": 0}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=10)
        assert result == {0: 2}

    def test_tc07_tiny_dataset(self):
        """TC-07 [E7]: n=1 → Sturges=2, B_max=2 → K=2."""
        col_stats = [{"type": "categorical", "n_unique": 5}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=1)
        assert result == {0: 2}

    def test_tc08_numeric_unchanged(self):
        """TC-08 [N1]: Numerical features → empty dict."""
        col_stats = [{"type": "numerical", "min": 0, "max": 10, "n_unique": 5}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        assert result == {}

    def test_tc09_mixed_no_cat(self):
        """TC-09 [P7]: All numerical → empty dict."""
        col_stats = [
            {"type": "numerical", "min": 0, "max": 10, "n_unique": 5},
            {"type": "numerical", "min": 0, "max": 10, "n_unique": 5},
            {"type": "numerical", "min": 0, "max": 10, "n_unique": 5},
        ]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        assert result == {}

    def test_tc10_mixed_with_cat(self):
        """TC-10 [P7]: Mixed → only cat columns in result."""
        col_stats = [
            {"type": "numerical", "min": 0, "max": 10, "n_unique": 5},
            {"type": "categorical", "n_unique": 3},
            {"type": "numerical", "min": 0, "max": 10, "n_unique": 5},
            {"type": "categorical", "n_unique": 2},
        ]
        result = apply_categorical_bin_cap(col_stats, n_bins=5, min_bins=2, max_bins=10, n=100)
        # Sturges(100)=8, B_max=8, cat(3)->3, cat(2)->2
        assert result == {1: 3, 3: 2}

    def test_tc11_covertype_like(self):
        """TC-11 [P8]: 40 binary categorical columns → all 2 bins."""
        col_stats = [{"type": "categorical", "n_unique": 2} for _ in range(40)]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=50000)
        expected = {j: 2 for j in range(40)}
        assert result == expected

    def test_tc14_fallback_n_categories_key(self):
        """TC-14: Fallback to 'n_categories' key if 'n_unique' missing."""
        col_stats = [{"type": "categorical", "n_categories": 3}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        assert result == {0: 3}

    def test_tc15_missing_key_raises(self):
        """TC-15: Missing both n_unique and n_categories → ValueError."""
        col_stats = [{"type": "categorical"}]
        with pytest.raises(ValueError, match="n_unique.*n_categories"):
            apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)

    def test_tc13_invalid_params_min_gt_max(self):
        """TC-13 [G1]: min_bins > max_bins → ValueError."""
        col_stats = [{"type": "categorical", "n_unique": 3}]
        with pytest.raises(ValueError, match="min_bins.*max_bins"):
            apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=5, max_bins=2, n=100)

    def test_n_bins_0_raises(self):
        """n_bins < 1 → ValueError."""
        col_stats = [{"type": "categorical", "n_unique": 3}]
        with pytest.raises(ValueError, match="n_bins"):
            apply_categorical_bin_cap(col_stats, n_bins=0, min_bins=2, max_bins=10, n=100)

    def test_n_0_raises(self):
        """n < 1 → ValueError."""
        col_stats = [{"type": "categorical", "n_unique": 3}]
        with pytest.raises(ValueError, match="n must be >= 1"):
            apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=0)

    def test_g2_min_bins_1(self):
        """TC [G2]: min_bins=1, constant cat → 1 bin."""
        col_stats = [{"type": "categorical", "n_unique": 1}]
        result = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=1, max_bins=10, n=100)
        assert result == {0: 1}

    def test_p6_sturges_monotonicity(self):
        """P6: B_max is non-decreasing in n (for fixed max_bins)."""
        col_stats = [{"type": "categorical", "n_unique": 100}]
        result_100 = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=100)
        result_1000 = apply_categorical_bin_cap(col_stats, n_bins=3, min_bins=2, max_bins=10, n=1000)
        assert result_100[0] <= result_1000[0], "B_max should be non-decreasing in n"

    def test_p10_equivalence_with_compute_adaptive_bins_categorical_only(self):
        """P10: For pure categorical stats, apply_categorical_bin_cap and
        compute_adaptive_bins produce identical results for categorical entries."""
        rng = np.random.default_rng(42)
        for trial in range(20):
            n_unique = rng.integers(1, 15)
            n = rng.integers(10, 200)
            max_bins = rng.integers(3, 15)
            min_bins = rng.integers(1, 3)

            col_stats = [{"type": "categorical", "n_unique": int(n_unique)}]

            result_apply = apply_categorical_bin_cap(
                col_stats, n_bins=3, min_bins=int(min_bins), max_bins=int(max_bins), n=int(n)
            )
            result_adaptive = compute_adaptive_bins(
                col_stats, n=int(n), min_bins=int(min_bins), max_bins=int(max_bins)
            )
            assert result_apply == result_adaptive, (
                f"Trial {trial}: n_unique={n_unique}, n={n}, max_bins={max_bins}, "
                f"min_bins={min_bins}: apply={result_apply} != adaptive={result_adaptive}"
            )
