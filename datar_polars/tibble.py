from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatch, wraps
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from polars import (
    DataFrame,
    DataType,
    Series,
    Expr,
    concat_list,
    all as pl_all,
    col as pl_col,
    lit,
)
from polars.exceptions import NotFoundError
from pipda import evaluate_expr
from datar.core.names import repair_names

from .collections import Collection
from .utils import name_of, to_expr, is_scalar

if TYPE_CHECKING:
    from .extended import Grouper


def _keep_ns(method: Callable) -> Callable:
    """Decorator to keep the namespace of the method"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        # method() may turn self back to DataFrame
        out = self.__class__(out)
        out.datar.update_from(self)
        return out

    return wrapper


def _series_grouped_agg(method: Callable) -> Callable:
    """Decorator to make sure grouped return SeriesAgg or TableAgg object"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        col = pl_col(self.name)
        agg = getattr(col, method.__name__)
        out = self.datar.grouper.gf.agg([agg(*args, **kwargs)]).to_series(-1)
        return out.datar.as_agg(self.datar.grouper)

    return wrapper


class Tibble(DataFrame):
    # Not working with polars 0.15
    # _lazyframe_class = LazyTibble

    def __init__(self, data=None, columns=None, orient=None):
        if isinstance(data, DataFrame):
            self._df = data._df
            self.datar.update_from(data)
            return

        super().__init__(data, columns, orient)

    select = _keep_ns(DataFrame.select)
    drop = _keep_ns(DataFrame.drop)
    filter = _keep_ns(DataFrame.filter)
    rename = _keep_ns(DataFrame.rename)
    sort = _keep_ns(DataFrame.sort)
    with_column = _keep_ns(DataFrame.with_column)
    with_columns = _keep_ns(DataFrame.with_columns)

    def __setitem__(self, key, value: Any) -> None:
        # allow df['a'] = 1
        if isinstance(key, str):
            value = to_expr(value, name=key)
            self._df = self.with_column(value)._df
            return
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            result = super().__getitem__(key)
        except NotFoundError:
            subdf_cols = [
                col for col in self.columns if col.startswith(f"{key}$")
            ]
            if not subdf_cols:
                raise

            result = self.select(subdf_cols)
            result.columns = [col[len(key) + 1 :] for col in subdf_cols]

        return result

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
            value = evaluate_expr(value, out, Context.EVAL)
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


class TibbleGrouped(Tibble):
    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(key, str):
            return SeriesGrouped(key, out, grouper=self.datar.grouper)
        return out

    def __str__(self) -> str:
        return f"{self.datar.grouper.str_()}\n{super().__str__()}"

    def _repr_html_(self) -> str:
        html = super()._repr_html_()
        return html.replace(
            '<table border="1" class="dataframe">\n',
            '<table border="1" class="dataframe">\n'
            f"<small>{self.datar.grouper.html()}</small>\n<br />\n",
        )


class TibbleRowwise(Tibble):
    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(key, str):
            return SeriesRowwise(key, out, grouper=self.datar.grouper)
        return out

    def __str__(self) -> str:
        return f"{self.datar.grouper.str_()}\n{super().__str__()}"

    def _repr_html_(self) -> str:
        html = super()._repr_html_()
        return html.replace(
            '<table border="1" class="dataframe">\n',
            '<table border="1" class="dataframe">\n'
            f"<small>{self.datar.grouper.html()}</small>\n<br />\n",
        )


class Aggregated(ABC):

    def _get_broadcasted_indexes(
        self,
        new_sizes: int | Sequence[int] | np.ndarray | Grouper,
    ) -> np.ndarray:
        """Broadcast the series to a new sizes"""
        if isinstance(new_sizes, int):
            new_sizes = [new_sizes]

        from .extended import Grouper
        if isinstance(new_sizes, Grouper):
            # agg_keys:
            #  *gvars, _size
            df = self.datar.agg_keys.join(
                new_sizes.rows_size.rename({"_size": "_new_sizes"}),
                how="left",
                on=self.datar.agg_keys.columns[:-1],
            )
        else:
            df = self.datar.agg_keys.with_column(
                Series("_new_sizes", new_sizes)
            )

        # df
        # *gvars, _size, _new_sizes
        # get the broadcasted indexes
        indexes_df = (
            df
            .lazy()
            .with_columns(
                [
                    (pl_col("_new_sizes") / pl_col("_size"))
                    .cast(int)
                    .alias("_factor"),
                    pl_col("_size").cumsum().alias("_index_end"),
                ]
            )
            .with_column(
                (pl_col("_index_end") - pl_col("_size")).alias("_index_start")
            )
            .with_column(
                concat_list(pl_col(["_index_start", "_index_end"]))
                .apply(lambda l: list(range(*l)))
                .alias("_indexes")
            )
            .collect()
        )

        indexes = np.concatenate(
            [
                list(idx) * fct
                for idx, fct in zip(
                    indexes_df["_indexes"],
                    indexes_df["_factor"],
                )
            ]
        )

        return indexes.take(np.concatenate(df["_rows"]))

    @abstractmethod
    def broadcast_to(
        self,
        new_sizes: int | Sequence[int] | np.ndarray | Grouper,
    ) -> SeriesBase | Tibble:
        ...


class Transformed(Aggregated, ABC):
    ...


class SeriesAgg(Series, Aggregated):

    def broadcast_to(
        self,
        new_sizes: int | Sequence[int] | np.ndarray | Grouper,
    ) -> SeriesBase | Tibble:
        from .extended import Grouper
        broadcasted_indexes = self._get_broadcasted_indexes(new_sizes)
        out = self.take(broadcasted_indexes)
        if isinstance(new_sizes, Grouper):
            out = SeriesGrouped(self.name, out, grouper=new_sizes)
        return out

    def to_frame(self, *args, **kwargs) -> TibbleAgg:
        out = TibbleAgg(super().to_frame(*args, **kwargs))
        out.datar._agg_keys = self.datar._agg_keys
        return out


class SeriesTransformed(SeriesAgg):
    ...


class TibbleAgg(Tibble, Aggregated):

    def broadcast_to(
        self,
        new_sizes: int | Sequence[int] | np.ndarray | Grouper,
    ) -> SeriesBase | Tibble:
        from .extended import Grouper

        broadcasted_indexes = self._get_broadcasted_indexes(new_sizes)
        out = self[broadcasted_indexes, :]
        if isinstance(new_sizes, Grouper):
            out = TibbleGrouped(out)
            out.datar._grouper = new_sizes
        return out


class TibbleTransformed(TibbleAgg):
    ...


class SeriesBase(Series, ABC):

    def __init__(self, *args, grouper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datar.grouper = grouper


class SeriesGrouped(SeriesBase):

    # make sure these methods return SeriesAgg
    all = _series_grouped_agg(Series.all)
    any = _series_grouped_agg(Series.any)
    max = _series_grouped_agg(Series.max)
    min = _series_grouped_agg(Series.min)
    mean = _series_grouped_agg(Series.mean)
    median = _series_grouped_agg(Series.median)
    nan_max = _series_grouped_agg(Series.nan_max)
    nan_min = _series_grouped_agg(Series.nan_min)
    null_count = _series_grouped_agg(Series.null_count)
    sum = _series_grouped_agg(Series.sum)

    def to_frame(self, *args, **kwargs) -> TibbleGrouped:
        out = TibbleGrouped(super().to_frame(*args, **kwargs))
        out.datar._grouper = self.datar.grouper
        return out


class SeriesRowwise(SeriesBase):
    ...


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
    if name:
        value = value.alias(name)
    if dtype:
        value = value.cast(dtype)
    return Tibble(value.to_frame())


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
    out = value.datar.copy()
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

    if broadcast_tbl and not isinstance(value, Expr):
        tbl = broadcast_base(value, tbl, name)

    if not name and isinstance(value, DataFrame):
        for cl in value.columns:
            tbl = add_to_tibble(tbl, cl, value[cl], dtype=dtype)

        return tbl

    if isinstance(value, DataFrame) and value.shape[1] == 0:
        value = None

    if isinstance(value, Expr):
        if dtype:
            value = value.cast(dtype)
        return tbl.with_column(value.alias(name))

    if is_scalar(value):
        val = lit(value)
        if dtype:
            val = val.cast(dtype)
        return tbl.with_columns(val.alias(name))

    if isinstance(value, DataFrame):
        for col in value.columns:
            tbl = add_to_tibble(tbl, f"{name}${col}", value[col], dtype=dtype)
        return tbl

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
    if len(value) in (0, base.shape[0]):
        return base

    if base.shape[0] == 1:
        return base[[0] * len(value), :]

    # Let polars handle the error
    return base
