"""Tibble utilities for the polars backend.

Provides:
- Tibble: a pl.DataFrame subclass carrying _datar metadata (eager)
- LazyTibble: a pl.LazyFrame subclass carrying _datar metadata (lazy)
- lazy() / collect(): standalone conversion utilities
- as_tibble() / reconstruct_tibble(): metadata helpers
"""

from __future__ import annotations

from typing import Any

import polars as pl

_DATAR_KEYS = ("groups", "rownames", "backend")


class Tibble(pl.DataFrame):
    """Eager Tibble — a pl.DataFrame subclass with _datar metadata.

    Args:
        data: A pl.DataFrame, pl.LazyFrame, or dict-like to wrap.
        _datar: Metadata dict with keys like 'groups', 'rownames', 'backend'.
    """

    def __new__(cls, data=None, _datar=None):
        if data is None:
            obj = super().__new__(cls)
            obj._df = pl.DataFrame()._df
            object.__setattr__(obj, "_datar", _datar or {})
            return obj

        if isinstance(data, pl.DataFrame):
            pydf = data._df
        elif isinstance(data, pl.LazyFrame):
            pydf = data.collect()._df
        else:
            pydf = pl.DataFrame(data)._df

        obj = super().__new__(cls)
        obj._df = pydf
        object.__setattr__(obj, "_datar", _datar or {})
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def _from_pydf(self, pydf):
        """Override to preserve _datar on internal DataFrame operations."""
        new = super()._from_pydf(pydf)
        if hasattr(self, "_datar"):
            object.__setattr__(new, "_datar", self._datar.copy())
        return new

    def __getitem__(self, key):
        """Support nested tibble columns stored with $ prefix convention.

        When pick() or other verbs return a DataFrame inside mutate(),
        it is expanded into sub-columns like ``sel$z``, ``sel$x1``.
        Accessing ``tibble["sel"]`` rebuilds the nested tibble.

        Struct columns are returned via _StructAccessor so that
        ``df["y"]["x"]`` delegates to ``series.struct["x"]``.
        """
        if isinstance(key, str):
            try:
                result = super().__getitem__(key)
            except pl.exceptions.ColumnNotFoundError:
                prefix = f"{key}$"
                sub_cols = [c for c in self.columns if c.startswith(prefix)]
                if sub_cols:
                    result = self.select(sub_cols)
                    rename = {c: c[len(prefix):] for c in sub_cols}
                    result = result.rename(rename)
                    _datar = getattr(self, "_datar", {}).copy()
                    return Tibble(result, _datar=_datar)
                raise
            if isinstance(result, pl.Series) and result.dtype == pl.Struct:
                # _StructAccessor is defined below
                return _struct_accessor(result)
            return result
        return super().__getitem__(key)

    def __getattr__(self, name: str) -> pl.Series | _StructAccessor:
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            return self.get_column(name)
        except pl.exceptions.ColumnNotFoundError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __dir__(self) -> list[str]:
        base = list(super().__dir__())
        return base + [c for c in self.columns if c not in base]

    def get_column(self, name: str) -> pl.Series | _StructAccessor:
        """Return a single column, wrapping struct Series for field access."""
        result = super().get_column(name)
        if isinstance(result, pl.Series) and result.dtype == pl.Struct:
            return _struct_accessor(result)
        return result

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            prefix = f"{key}$"
            return any(c.startswith(prefix) for c in self.columns)
        return False

    def collect(self, **kwargs: Any) -> Tibble:
        """Backward-compatible collect: return self (already eager)."""
        return self

    def lazy(self) -> LazyTibble:
        """Convert to LazyTibble, preserving _datar."""
        lf = super().lazy()
        _datar = getattr(self, "_datar", {}).copy()
        return LazyTibble(lf, _datar=_datar)

    @property
    def shape(self) -> tuple[int, int]:
        return super().shape


class LazyTibble(pl.LazyFrame):
    """Lazy Tibble — a pl.LazyFrame subclass with _datar metadata.

    All polars lazy operations (filter, select, with_columns, etc.)
    auto-preserve _datar via the overridden _from_pyldf method.

    Args:
        data: A pl.LazyFrame, pl.DataFrame, or dict-like to wrap.
        _datar: Metadata dict with keys like 'groups', 'rownames', 'backend'.
    """

    def __new__(cls, data=None, _datar=None):
        if data is None:
            obj = super().__new__(cls)
            obj._ldf = pl.LazyFrame()._ldf
            object.__setattr__(obj, "_datar", _datar or {})
            return obj

        if isinstance(data, pl.LazyFrame):
            lf = data
        elif isinstance(data, pl.DataFrame):
            lf = data.lazy()
        else:
            lf = pl.LazyFrame(data)

        obj = super().__new__(cls)
        obj._ldf = lf._ldf
        object.__setattr__(obj, "_datar", _datar or {})
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def _from_pyldf(self, pyldf):
        """Override to preserve _datar across all lazy operations."""
        new = super()._from_pyldf(pyldf)
        if hasattr(self, "_datar"):
            object.__setattr__(new, "_datar", self._datar.copy())
        return new

    def collect(self, **kwargs: Any) -> Tibble:
        """Materialize to Tibble, preserving _datar."""
        df = super().collect(**kwargs)
        _datar = getattr(self, "_datar", {}).copy()
        return Tibble(df, _datar=_datar)


# ---- Standalone conversion utilities ----------------------------------------


def to_lazy(data: Any) -> LazyTibble:
    """Convert data to a LazyTibble, preserving _datar if present."""
    if isinstance(data, LazyTibble):
        return data
    _datar = getattr(data, "_datar", {})
    if isinstance(data, pl.LazyFrame):
        return LazyTibble(data, _datar=_datar)
    if isinstance(data, (Tibble, pl.DataFrame)):
        lf = data.lazy()
        return LazyTibble(lf, _datar=_datar)
    df = pl.DataFrame(data) if not isinstance(data, pl.DataFrame) else data
    return LazyTibble(df.lazy())


def to_eager(data: Any) -> Tibble | Any:
    """Materialize a LazyTibble/LazyFrame to a Tibble, preserving _datar."""
    if isinstance(data, Tibble):
        return data
    _datar = getattr(data, "_datar", {})
    if isinstance(data, (LazyTibble, pl.LazyFrame)):
        df = data.collect()
        return Tibble(df, _datar=_datar)
    if isinstance(data, pl.DataFrame):
        return Tibble(data, _datar=_datar)
    return data


# ---- Metadata helpers -------------------------------------------------------


def as_tibble(x: Any, _datar: dict | None = None) -> Tibble:
    """Convert x to a Tibble (eager) with _datar metadata.

    If x is already a Tibble with _datar, returns as-is.
    If x is a LazyTibble or LazyFrame, collects then wraps.
    If x is a plain DataFrame, wraps directly.

    Args:
        x: Input data to convert.
        _datar: Optional metadata dict. Defaults to
            {'groups': None, 'rownames': None, 'backend': 'polars'}.

    Returns:
        A Tibble with _datar metadata.
    """
    if _datar is None:
        _datar = {"groups": None, "rownames": None, "backend": "polars"}

    if isinstance(x, Tibble):
        if not hasattr(x, "_datar") or not x._datar:
            object.__setattr__(x, "_datar", _datar)
        return x

    if isinstance(x, (LazyTibble, pl.LazyFrame)):
        return Tibble(x.collect(), _datar=_datar)

    if isinstance(x, pl.DataFrame):
        return Tibble(x, _datar=_datar)

    if isinstance(x, dict):
        return Tibble(pl.DataFrame(x), _datar=_datar)

    if hasattr(x, "to_dict"):
        return Tibble(
            pl.DataFrame(x.to_dict(orient="records")), _datar=_datar
        )

    if isinstance(x, (list, tuple)):
        return Tibble(
            pl.DataFrame(x, strict=False, orient="row"), _datar=_datar
        )

    return Tibble(pl.DataFrame(x), _datar=_datar)


def reconstruct_tibble(
    data: Tibble | LazyTibble | pl.DataFrame | pl.LazyFrame,
    old_data: Tibble | LazyTibble | None = None,
) -> Tibble | LazyTibble:
    """Ensure data is a Tibble or LazyTibble with proper _datar metadata.

    Preserves the lazy/eager nature of the data if already
    Tibble or LazyTibble. Copies metadata from old_data if available.

    Args:
        data: The DataFrame/LazyFrame to ensure metadata on.
        old_data: Optional Tibble/LazyTibble to copy metadata from.

    Returns:
        Tibble or LazyTibble with _datar attribute set.
    """
    existing_datar = getattr(data, "_datar", None)

    if not isinstance(data, (Tibble, LazyTibble)):
        if isinstance(data, pl.LazyFrame):
            data = LazyTibble(data)
        elif isinstance(data, pl.DataFrame):
            data = Tibble(data)
        else:
            data = Tibble(data)

    if existing_datar and isinstance(existing_datar, dict) and existing_datar:
        meta = dict(existing_datar)
    elif old_data is not None and hasattr(old_data, "_datar"):
        meta = dict(old_data._datar)
    else:
        meta = {}

    meta.setdefault("groups", None)
    meta.setdefault("rownames", None)
    meta.setdefault("backend", "polars")

    object.__setattr__(data, "_datar", meta)
    return data


class _StructAccessor:
    """Wraps a struct Series so that ``s["field"]`` delegates to
    ``series.struct["field"]``, returning a regular Polars Series.

    Used by Tibble.__getitem__ so that ``df["y"]["x"]`` works on
    struct columns.
    """

    def __init__(self, series: pl.Series) -> None:
        self._s = series

    def __getitem__(self, key: str) -> pl.Series:
        if isinstance(key, str):
            return self._s.struct[key]
        return self._s[key]

    def __iter__(self):
        return iter(self._s)

    def __len__(self) -> int:
        return len(self._s)

    def to_list(self):
        return self._s.to_list()

    def __repr__(self) -> str:
        return repr(self._s)


def _struct_accessor(series: pl.Series) -> _StructAccessor:
    """Return a _StructAccessor for the given struct Series."""
    return _StructAccessor(series)
