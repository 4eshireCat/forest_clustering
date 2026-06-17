import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ColumnMeta:
    original_idx: int
    name: str
    type: str  # 'numerical' | 'categorical'
    cat_encoder: dict | None = None  # original_value → int_code
    n_categories: int = 0


def _safe_issubdtype(dtype, supertype) -> bool:
    """np.issubdtype that returns False for non-numpy (e.g. pandas StringDtype)
    extension dtypes instead of raising (pandas >= 3 compatibility)."""
    try:
        return bool(np.issubdtype(dtype, supertype))
    except (TypeError, ValueError):
        return False


class DataEncoder:
    """Converts DataFrame/ndarray to a float64 internal matrix.

    Categorical columns are label-encoded to non-negative integers.
    Unknown / NaN values become -1.
    """

    def __init__(
        self,
        feature_types_override: dict | None = None,
        cat_threshold: int = 10,
    ):
        self.feature_types_override = feature_types_override or {}
        self.cat_threshold = cat_threshold
        self.columns_: list[ColumnMeta] = []

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
        cols = []
        for i, col_name in enumerate(df.columns):
            s = df[col_name]
            override = self.feature_types_override.get(col_name) or self.feature_types_override.get(i)
            is_num = self._is_numerical(s, override)
            if is_num:
                meta = ColumnMeta(i, str(col_name), "numerical")
                cols.append(s.values.astype(np.float64))
            else:
                encoder, encoded = self._fit_categorical(s)
                meta = ColumnMeta(i, str(col_name), "categorical", encoder, len(encoder))
                cols.append(encoded)
            self.columns_.append(meta)
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    def _transform_df(self, df: pd.DataFrame) -> np.ndarray:
        # Validate column names match fit() columns
        if hasattr(self, 'columns_') and self.columns_:
            expected_names = [meta.name for meta in self.columns_]
            actual_names = list(df.columns)
            if expected_names != actual_names:
                raise ValueError(
                    f"Column names do not match fit() columns. "
                    f"Expected: {expected_names}, got: {actual_names}"
                )
        cols = []
        for meta in self.columns_:
            s = df.iloc[:, meta.original_idx]
            if meta.type == "numerical":
                cols.append(s.values.astype(np.float64))
            else:
                cols.append(self._apply_categorical(s, meta.cat_encoder))
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    # ------------------------------------------------------------------
    # ndarray path
    # ------------------------------------------------------------------

    def _fit_array(self, X: np.ndarray) -> np.ndarray:
        self.columns_ = []
        n, d = X.shape
        cols = []
        for i in range(d):
            s = pd.Series(X[:, i])
            override = self.feature_types_override.get(i)
            is_num = self._is_numerical(s, override)
            if is_num:
                meta = ColumnMeta(i, f"f{i}", "numerical")
                cols.append(X[:, i].astype(np.float64))
            else:
                encoder, encoded = self._fit_categorical(s)
                meta = ColumnMeta(i, f"f{i}", "categorical", encoder, len(encoder))
                cols.append(encoded)
            self.columns_.append(meta)
        return np.column_stack(cols) if cols else np.empty((n, 0))

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for meta in self.columns_:
            s = pd.Series(X[:, meta.original_idx])
            if meta.type == "numerical":
                cols.append(X[:, meta.original_idx].astype(np.float64))
            else:
                cols.append(self._apply_categorical(s, meta.cat_encoder))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_numerical(self, s: pd.Series, override: str | None) -> bool:
        if override == "numerical":
            return True
        if override == "categorical":
            return False
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            return False
        if s.dtype.kind in ("i", "u", "f"):
            return s.nunique() > self.cat_threshold
        return False

    @staticmethod
    def _fit_categorical(s: pd.Series) -> tuple[dict, np.ndarray]:
        # Convert CategoricalDtype to object to avoid fillna issues
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        unique_vals = sorted(s.dropna().unique(), key=lambda x: (str(type(x)), x))
        encoder = {v: j for j, v in enumerate(unique_vals)}
        encoded = s.map(encoder).fillna(-1).values.astype(np.float64)
        return encoder, encoded

    @staticmethod
    def _apply_categorical(s: pd.Series, encoder: dict) -> np.ndarray:
        # Convert CategoricalDtype to object to avoid fillna issues
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        return s.map(encoder).fillna(-1).values.astype(np.float64)

    # ------------------------------------------------------------------
    # Feature type auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_feature_types(df, strategy='naive', known_types=None):
        """Auto-detect feature types.

        Parameters
        ----------
        df : pd.DataFrame or ndarray
        strategy : 'naive' | 'smart'
        known_types : dict or None
            Column name -> 'categorical' | 'numerical' overrides.

        Returns
        -------
        types : dict
            {col_name: 'categorical' | 'numerical'}
        """
        if strategy not in ('naive', 'smart'):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        is_ndarray = isinstance(df, np.ndarray)
        if is_ndarray:
            df = pd.DataFrame(df, columns=[f'f{i}' for i in range(df.shape[1])])

        known_types = known_types or {}
        result = {}

        for i, col in enumerate(df.columns):
            key = i if is_ndarray else col
            if key in known_types:
                result[key] = known_types[key]
                continue

            result[key] = DataEncoder._detect_single_column(df[col], col, strategy)

        return result

    @staticmethod
    def _detect_single_column(series, col_name, strategy):
        """Detect type for a single column."""
        n = len(series)
        n_unique = series.nunique(dropna=True)
        dtype = series.dtype

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
            # Naive: only dtype-based
            return 'numerical' if _safe_issubdtype(dtype, np.number) else 'categorical'

        # Smart detection

        # Rule: datetime → numerical (timestamp)
        if _safe_issubdtype(dtype, np.datetime64):
            return 'numerical'

        # Rule: semantic hint — column name suggests ID
        col_lower = col_name.lower()
        id_keywords = ('_id', 'id_', 'userid', 'user_id', 'itemid', 'item_id',
                       'code_', '_code', 'idx', 'index')
        has_id_name = any(kw in col_lower for kw in id_keywords)

        # Get finite values for analysis
        finite_vals = series.dropna()
        if len(finite_vals) == 0:
            return 'categorical'  # all NaN

        # Integer analysis
        if _safe_issubdtype(dtype, np.integer):
            cardinality_threshold = max(10, int(np.sqrt(n)))
            # Check ID semantic hint FIRST (before cardinality)
            if has_id_name and n_unique > max(5, n // 20):
                return 'numerical'
            # Low-cardinality integer → categorical (encoding)
            if n_unique <= cardinality_threshold:
                return 'categorical'
            # High-cardinality integer, unique ratio > 0.5 → numerical (likely ID)
            if n_unique / n > 0.5:
                return 'numerical'
            # Medium-cardinality integer → categorical
            return 'categorical'

        # Float analysis
        if _safe_issubdtype(dtype, np.floating):
            # Check if all values are actually integers (e.g., 1.0, 2.0, 3.0)
            vals = finite_vals.values
            all_integers = np.all(np.isclose(vals, np.round(vals), rtol=0, atol=1e-9))
            if all_integers and n_unique <= max(10, int(np.sqrt(n))):
                return 'categorical'  # float-encoded categorical

            # Few unique float values → categorical (e.g., ratings 1.0-5.0)
            if n_unique <= max(5, int(np.sqrt(n) / 2)):
                return 'categorical'

            # Many unique continuous values → numerical
            return 'numerical'

        # Object/string analysis
        if dtype == object or dtype.name == 'object':
            return 'categorical'

        # Fallback
        return 'numerical' if _safe_issubdtype(dtype, np.number) else 'categorical'
