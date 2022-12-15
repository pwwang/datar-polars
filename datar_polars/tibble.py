from __future__ import annotations

from copy import copy
from functools import singledispatch
from itertools import chain
from typing import Any, Callable, Mapping, Sequence

from polars import DataFrame, LazyFrame, DataType, Series, lit, all as pl_all
from pipda import evaluate_expr
from datar.core.names import repair_names

from .collections import Collection
from .utils import name_of, is_scalar


class LazyTibble(LazyFrame):
    @classmethod
    @property
    def _dataframe_class(cls):
        return Tibble


class LazyTibbleRowwise(LazyFrame):
    @classmethod
    @property
    def _dataframe_class(cls):
        return TibbleRowwise


class Tibble(DataFrame):
    _lazyframe_class = LazyTibble
    # metadata
    _datar = {}

    def __init__(self, data=None, columns=None, orient=None, *, meta=None):
        if meta is None and isinstance(data, Tibble):
            meta = data._datar

        self._datar.update(meta or {})
        if isinstance(data, DataFrame):
            self._df = data._df
            return

        super().__init__(data, columns, orient)

    def copy(self, copy_meta=True):
        """Copy the tibble"""
        out = copy(self)
        if copy_meta:
            out._datar = self._datar.copy()
        return out

    def select(self, exprs) -> Tibble:
        out = super().select(exprs)
        return Tibble(out)

    def with_columns(self, exprs, **named_exprs) -> Tibble:
        out = super().with_columns(exprs, **named_exprs)
        return Tibble(out)

    def with_column(self, column) -> Tibble:
        out = super().with_column(column)
        return Tibble(out)

    def __setitem__(self, key, value: Any) -> None:
        if isinstance(key, str):
            if is_scalar(value):
                value = lit(value).alias(key)
            else:
                value = Series(name=key, values=value)
            self._df = self.with_column(value)._df
            return
        return super().__setitem__(key, value)

    @classmethod
    def from_pairs(
        cls,
        names: Sequence[str],
        values: Sequence[Any],
        _name_repair: str | Callable = "check_unique",
        _dtypes: DataType | Mapping[str, DataType] = None,
    ) -> Tibble:
        """Construct a tibble with name-value pairs

        Instead of do `**kwargs`, this allows duplicated names

        Args:
            names: The names of the data to be construct a tibble
            values: The data to construct a tibble, must have the same length
                with the names
            _name_repair: How to repair names
            _dtypes: The dtypes for post conversion
        """
        # from .collections import Collection
        from .contexts import Context

        if len(names) != len(values):
            raise ValueError(
                "Lengths of `names` and `values` are not the same."
            )
        if _name_repair == "minimal":
            raise ValueError(
                "Repair names using `minimal` is not supported for "
                "`polars` backend"
            )
        names = repair_names(names, _name_repair)

        out = None
        for name, value in zip(names, values):
            value = evaluate_expr(value, out, Context.EVAL_DATA)
            dtype = _dtypes.get(name) if isinstance(_dtypes, dict) else _dtypes
            if isinstance(value, Collection):
                value.expand()

            out = add_to_tibble(
                out,
                name,
                value,
                broadcast_tbl=True,
                dtype=dtype,
            )

        return Tibble() if out is None else out

    @classmethod
    def from_args(
        cls,
        *args,
        _name_repair: str | Callable = "check_unique",
        _dtypes: DataType | Mapping[str, DataType] = None,
        **kwargs,
    ):
        if not args and not kwargs:
            return DataFrame()

        names = [name_of(arg) for arg in args] + list(kwargs)
        return cls.from_pairs(
            names,
            list(chain(args, kwargs.values())),
            _name_repair=_name_repair,
            _dtypes=_dtypes,
        )


class TibbleRowwise(Tibble):
    _lazyframe_class = LazyTibbleRowwise


@singledispatch
def init_tibble_from(value, name: str, dtype: DataType) -> Tibble:
    """Initialize a tibble from a value"""
    try:
        df = Tibble({name: value})
    except ValueError:
        df = Tibble({name: [value]})

    return df.select(pl_all().cast(dtype)) if dtype else df


@init_tibble_from.register(Series)
def _init_tibble_from_series(
    value: Series,
    name: str,
    dtype: DataType,
) -> Tibble:
    # Deprecate warning, None will be used as series name in the future
    # So use 0 as default here
    name = name or value.name or "0"
    if dtype:
        value = value.cast(dtype)
    return Tibble(value.to_frame(), columns=[name])


@init_tibble_from.register(DataFrame)
def _init_tibble_frame_df(
    value: DataFrame,
    name: str,
    dtype: DataType,
) -> Tibble:
    out = Tibble(value)
    if name:
        out.columns = [f"{name}${col}" for col in out.columns]
    return out


@init_tibble_from.register(Tibble)
def _init_tibble_frame_tibble(
    value: Tibble, name: str, dtype: DataType
) -> Tibble:
    out = value.__class__(value, meta=value._datar)
    if name:
        out.columns = [f"{name}${col}" for col in out.columns]
    return out


def add_to_tibble(
    tbl: DataFrame,
    name: str,
    value: Any,
    broadcast_tbl: bool = False,
    dtype: DataType = None,
) -> Tibble:
    """Add data to tibble"""
    if value is None:
        return tbl

    if tbl is None:
        return init_tibble_from(value, name, dtype)

    if broadcast_tbl:
        tbl = broadcast_base(value, tbl, name)

    if not name and isinstance(value, DataFrame):
        for cl in value.columns:
            tbl = add_to_tibble(tbl, cl, value[cl], dtype=dtype)

        return tbl

    if is_scalar(value):
        val = lit(value)
        if dtype:
            val = val.cast(dtype)
        return tbl.with_columns(val.alias(name))

    # dtype=int not working on float
    # val = Series(name=name, values=value, dtype=dtype)
    val = Series(name=name, values=value)
    if dtype:
        val = val.cast(dtype)
    return tbl.with_column(val)


def broadcast_base(
    value,
    base: Tibble,
    name: str = None,
):
    """Broadcast the base dataframe when value has more elements

    Args:
        value: The value
        base: The base data frame

    Returns:
        A tuple of the transformed value and base
    """
    # plain arrays, scalars, np.array(True)
    if is_scalar(value) or len(value) == 1:
        return base

    name = name or name_of(value) or str(value)
    # The length should be [1, len(value)]
    if base.shape[0] == len(value):
        return base

    if base.shape[0] == 1:
        return base.sample(n=len(value), with_replacement=True)

    # Let polars handle the error
    return base
