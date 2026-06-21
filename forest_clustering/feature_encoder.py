import numpy as np
import pandas as pd
from dataclasses import dataclass


_RARE_SENTINEL_BASE = "__forest_clustering_rare__"


@dataclass
class ColumnMeta:
    original_idx: int
    name: str
    type: str  # 'numerical' | 'categorical'
    cat_encoder: dict | None = None  # original_value -> int_code
    n_categories: int = 0
    role: str = "original"  # 'original' | 'missing_indicator'
    numeric_coerced: bool = False
    rare_value: object | None = None


def _safe_issubdtype(dtype, supertype) -> bool:
    """np.issubdtype that returns False for non-numpy extension dtypes."""
    try:
        return bool(np.issubdtype(dtype, supertype))
    except (TypeError, ValueError):
        return False


def _normalize_feature_overrides(overrides):
    if overrides is None:
        return {}
    if isinstance(overrides, dict):
        return dict(overrides)
    if isinstance(overrides, (list, tuple)):
        return {i: v for i, v in enumerate(overrides)}
    raise TypeError("feature_types_override/feature_types must be a dict, list, tuple, or None")


def _missing_mask(s: pd.Series) -> pd.Series:
    return pd.isna(s)


class DataEncoder:
    """Converts DataFrame/ndarray to a float64 internal matrix.

    Categorical columns are label-encoded to non-negative integers. Unknown /
    missing values become -1 unless rare-category grouping is enabled, in which
    case unseen non-missing categories are mapped to the learned rare bucket.
    Optional missing indicators add stable binary categorical features.
    """

    def __init__(
        self,
        feature_types_override: dict | list | tuple | None = None,
        cat_threshold: int = 10,
        add_missing_indicators: bool = False,
        rare_category_min_count: int | None = None,
        rare_category_min_freq: float | None = None,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        self.feature_types_override = _normalize_feature_overrides(feature_types_override)
        self.cat_threshold = cat_threshold
        self.add_missing_indicators = add_missing_indicators
        self.rare_category_min_count = rare_category_min_count
        self.rare_category_min_freq = rare_category_min_freq
        self.coerce_numeric_strings = coerce_numeric_strings
        self.numeric_string_min_fraction = numeric_string_min_fraction
        self.columns_: list[ColumnMeta] = []
        self.input_columns_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return self._fit_df(X)
        return self._fit_array(np.asarray(X))

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return self._transform_df(X)
        return self._transform_array(np.asarray(X))

    @property
    def feature_types_(self) -> list[str]:
        return [m.type for m in self.columns_]

    @property
    def d_(self) -> int:
        return len(self.columns_)

    # ------------------------------------------------------------------
    # DataFrame path
    # ------------------------------------------------------------------

    def _fit_df(self, df: pd.DataFrame) -> np.ndarray:
        self.columns_ = []
        self.input_columns_ = [str(c) for c in df.columns]
        cols = []
        for i, col_name in enumerate(df.columns):
            s = df[col_name]
            override = self._get_override(col_name, i)
            coerced = self._should_coerce_numeric_string(s, override)
            is_num = self._is_numerical(s, override, coerced=coerced)
            if is_num:
                meta = ColumnMeta(i, str(col_name), "numerical", numeric_coerced=coerced or not self._is_native_numeric(s))
                cols.append(self._to_numeric_array(s))
            else:
                encoder, encoded, rare_value = self._fit_categorical(s)
                meta = ColumnMeta(i, str(col_name), "categorical", encoder, len(encoder), rare_value=rare_value)
                cols.append(encoded)
            self.columns_.append(meta)
            if self.add_missing_indicators and _missing_mask(s).any():
                ind_meta = ColumnMeta(i, f"{col_name}__missing", "categorical", role="missing_indicator", n_categories=2)
                self.columns_.append(ind_meta)
                cols.append(_missing_mask(s).to_numpy(dtype=np.float64))
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    def _transform_df(self, df: pd.DataFrame) -> np.ndarray:
        expected_names = getattr(self, "input_columns_", None)
        if expected_names is not None:
            actual_names = [str(c) for c in df.columns]
            if expected_names != actual_names:
                raise ValueError(
                    f"Column names do not match fit() columns. Expected: {expected_names}, got: {actual_names}"
                )
        cols = []
        for meta in self.columns_:
            s = df.iloc[:, meta.original_idx]
            if meta.role == "missing_indicator":
                cols.append(_missing_mask(s).to_numpy(dtype=np.float64))
            elif meta.type == "numerical":
                cols.append(self._to_numeric_array(s))
            else:
                cols.append(self._apply_categorical(s, meta))
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    # ------------------------------------------------------------------
    # ndarray path
    # ------------------------------------------------------------------

    def _fit_array(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like object")
        self.columns_ = []
        self.input_columns_ = [f"f{i}" for i in range(X.shape[1])]
        n, d = X.shape
        cols = []
        for i in range(d):
            s = pd.Series(X[:, i])
            override = self._get_override(i, i)
            coerced = self._should_coerce_numeric_string(s, override)
            is_num = self._is_numerical(s, override, coerced=coerced)
            if is_num:
                meta = ColumnMeta(i, f"f{i}", "numerical", numeric_coerced=coerced or not self._is_native_numeric(s))
                cols.append(self._to_numeric_array(s))
            else:
                encoder, encoded, rare_value = self._fit_categorical(s)
                meta = ColumnMeta(i, f"f{i}", "categorical", encoder, len(encoder), rare_value=rare_value)
                cols.append(encoded)
            self.columns_.append(meta)
            if self.add_missing_indicators and _missing_mask(s).any():
                ind_meta = ColumnMeta(i, f"f{i}__missing", "categorical", role="missing_indicator", n_categories=2)
                self.columns_.append(ind_meta)
                cols.append(_missing_mask(s).to_numpy(dtype=np.float64))
        return np.column_stack(cols) if cols else np.empty((n, 0))

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like object")
        if X.shape[1] != len(getattr(self, "input_columns_", [])):
            raise ValueError(f"X has {X.shape[1]} features, expected {len(self.input_columns_)}")
        cols = []
        for meta in self.columns_:
            s = pd.Series(X[:, meta.original_idx])
            if meta.role == "missing_indicator":
                cols.append(_missing_mask(s).to_numpy(dtype=np.float64))
            elif meta.type == "numerical":
                cols.append(self._to_numeric_array(s))
            else:
                cols.append(self._apply_categorical(s, meta))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_override(self, col_name, idx: int):
        return self.feature_types_override.get(col_name) or self.feature_types_override.get(str(col_name)) or self.feature_types_override.get(idx)

    @staticmethod
    def _is_native_numeric(s: pd.Series) -> bool:
        return s.dtype.kind in ("i", "u", "f") or _safe_issubdtype(s.dtype, np.number)

    def _to_numeric_array(self, s: pd.Series) -> np.ndarray:
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

    def _numeric_string_fraction(self, s: pd.Series) -> float:
        vals = s.dropna()
        if len(vals) == 0:
            return 0.0
        parsed = pd.to_numeric(vals, errors="coerce")
        return float(np.isfinite(parsed.to_numpy(dtype=np.float64)).mean())

    def _should_coerce_numeric_string(self, s: pd.Series, override: str | None = None) -> bool:
        if override == "numerical" and not self._is_native_numeric(s):
            return True
        if override == "categorical" or not self.coerce_numeric_strings:
            return False
        if self._is_native_numeric(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return False
        return self._numeric_string_fraction(s) >= float(self.numeric_string_min_fraction)

    def _is_numerical(self, s: pd.Series, override: str | None, coerced: bool = False) -> bool:
        if override == "numerical":
            return True
        if override == "categorical":
            return False
        if coerced:
            return True
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            return False
        if s.dtype.kind in ("i", "u", "f"):
            return s.nunique(dropna=True) > self.cat_threshold
        return False

    def _rare_sentinel(self, values) -> str:
        sentinel = _RARE_SENTINEL_BASE
        existing = set(values)
        if sentinel not in existing:
            return sentinel
        k = 1
        while f"{sentinel}_{k}" in existing:
            k += 1
        return f"{sentinel}_{k}"

    def _fit_categorical(self, s: pd.Series) -> tuple[dict, np.ndarray, object | None]:
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        non_missing = s.dropna()
        values = list(non_missing.unique())
        rare_value = None
        keep_values = values
        if self.rare_category_min_count is not None or self.rare_category_min_freq is not None:
            counts = non_missing.value_counts(dropna=True)
            min_count = 1 if self.rare_category_min_count is None else int(self.rare_category_min_count)
            min_freq = 0.0 if self.rare_category_min_freq is None else float(self.rare_category_min_freq)
            denom = max(len(non_missing), 1)
            keep_values = [v for v in values if counts.get(v, 0) >= min_count and counts.get(v, 0) / denom >= min_freq]
            if len(keep_values) < len(values):
                rare_value = self._rare_sentinel(values)
                keep_values = list(keep_values) + [rare_value]
        unique_vals = sorted(keep_values, key=lambda x: (str(type(x)), str(x)))
        encoder = {v: j for j, v in enumerate(unique_vals)}
        encoded = self._encode_categorical_series(s, encoder, rare_value)
        return encoder, encoded, rare_value

    @staticmethod
    def _encode_categorical_series(s: pd.Series, encoder: dict, rare_value=None) -> np.ndarray:
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        mapped = s.map(encoder)
        if rare_value is not None and rare_value in encoder:
            rare_code = encoder[rare_value]
            # Non-missing values not in the frequent-category encoder become rare.
            mapped = mapped.where(mapped.notna() | s.isna(), rare_code)
        return mapped.fillna(-1).to_numpy(dtype=np.float64)

    @staticmethod
    def _apply_categorical(s: pd.Series, meta: ColumnMeta) -> np.ndarray:
        return DataEncoder._encode_categorical_series(s, meta.cat_encoder or {}, meta.rare_value)

    # ------------------------------------------------------------------
    # Feature type auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_feature_types(
        df,
        strategy='naive',
        known_types=None,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        """Auto-detect feature types.

        Object/string columns can optionally be treated as numerical when most
        non-missing values parse cleanly as numbers. This prevents columns such
        as ``"12.5"`` / ``"7"`` from being clustered as arbitrary categories.
        """
        if strategy not in ('naive', 'smart'):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        is_ndarray = isinstance(df, np.ndarray)
        if is_ndarray:
            df = pd.DataFrame(df, columns=[f'f{i}' for i in range(df.shape[1])])

        known_types = _normalize_feature_overrides(known_types)
        result = {}

        for i, col in enumerate(df.columns):
            key = i if is_ndarray else col
            if key in known_types:
                result[key] = known_types[key]
                continue
            result[key] = DataEncoder._detect_single_column(
                df[col],
                str(col),
                strategy,
                coerce_numeric_strings=coerce_numeric_strings,
                numeric_string_min_fraction=numeric_string_min_fraction,
            )

        return result

    @staticmethod
    def _detect_single_column(
        series,
        col_name,
        strategy,
        coerce_numeric_strings: bool = False,
        numeric_string_min_fraction: float = 0.90,
    ):
        """Detect type for a single column."""
        n = len(series)
        n_unique = series.nunique(dropna=True)
        dtype = series.dtype

        # Numeric-string coercion must run before the generic object ->
        # categorical rule. We still leave binary-looking numeric strings as
        # categorical under the usual low-cardinality rule.
        if coerce_numeric_strings and not _safe_issubdtype(dtype, np.number) and not isinstance(dtype, pd.CategoricalDtype):
            non_missing = series.dropna()
            if len(non_missing):
                parsed = pd.to_numeric(non_missing, errors="coerce")
                frac = float(np.isfinite(parsed.to_numpy(dtype=np.float64)).mean())
                if frac >= numeric_string_min_fraction:
                    parsed_unique = parsed.dropna().nunique()
                    if parsed_unique > 2:
                        return 'numerical'

        # Trivial cases (both strategies)
        if n_unique <= 1:
            return 'categorical'
        if n_unique == 2:
            return 'categorical'  # binary
        if dtype == bool or dtype.name == 'bool':
            return 'categorical'
        if dtype.name == 'category':
            return 'categorical'
        if dtype == object:
            return 'categorical'

        if strategy == 'naive':
            return 'numerical' if _safe_issubdtype(dtype, np.number) else 'categorical'

        # Smart detection
        if _safe_issubdtype(dtype, np.datetime64):
            return 'numerical'

        col_lower = str(col_name).lower()
        id_keywords = ('_id', 'id_', 'userid', 'user_id', 'itemid', 'item_id',
                       'code_', '_code', 'idx', 'index')
        has_id_name = any(kw in col_lower for kw in id_keywords)

        finite_vals = series.dropna()
        if len(finite_vals) == 0:
            return 'categorical'

        if _safe_issubdtype(dtype, np.integer):
            cardinality_threshold = max(10, int(np.sqrt(n)))
            if has_id_name and n_unique > max(5, n // 20):
                return 'numerical'
            if n_unique <= cardinality_threshold:
                return 'categorical'
            if n_unique / n > 0.5:
                return 'numerical'
            return 'categorical'

        if _safe_issubdtype(dtype, np.floating):
            vals = finite_vals.values
            all_integers = np.all(np.isclose(vals, np.round(vals), rtol=0, atol=1e-9))
            if all_integers and n_unique <= max(10, int(np.sqrt(n))):
                return 'categorical'
            if n_unique <= max(5, int(np.sqrt(n) / 2)):
                return 'categorical'
            return 'numerical'

        if dtype == object or dtype.name == 'object':
            return 'categorical'

        return 'numerical' if _safe_issubdtype(dtype, np.number) else 'categorical'
