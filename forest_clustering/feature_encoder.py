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
        unique_vals = sorted(s.dropna().unique(), key=lambda x: (str(type(x)), x))
        encoder = {v: j for j, v in enumerate(unique_vals)}
        encoded = s.map(encoder).fillna(-1).values.astype(np.float64)
        return encoder, encoded

    @staticmethod
    def _apply_categorical(s: pd.Series, encoder: dict) -> np.ndarray:
        return s.map(encoder).fillna(-1).values.astype(np.float64)
