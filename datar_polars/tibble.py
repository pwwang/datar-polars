from __future__ import annotations

from functools import singledispatch, wraps
from itertools import chain
from typing import Any, Callable, Mapping, Sequence

from polars import DataFrame, DataType, Series, Expr, lit, all as pl_all
from polars.exceptions import NotFoundError
from pipda import evaluate_expr
from datar.core.names import repair_names

from .collections import Collection
from .utils import name_of, to_expr, is_scalar


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


class Tibble(DataFrame):
    # Not working with polars 0.15
    # _lazyframe_class = LazyTibble

    def __init__(self, data=None, columns=None, orient=None):
        if isinstance(data, DataFrame):
            self._df = data._df
            self.datar.update_from(data)
            return

        super().__init__(data, columns, orient)

    def copy(self, copy_meta=True):
        """Copy the tibble"""
        out = self.clone()
        out.datar.update_from(self)
        return out

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


class TibbleGrouped(Tibble):
    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(key, str):
            return SeriesGrouped(out, grouper=self.datar.grouper)
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
            return SeriesRowwise(out, grouper=self.datar.grouper)
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


class SeriesGrouped(Series):
    def __init__(self, *args, grouper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datar.grouper = grouper


class SeriesRowwise(Series):
    def __init__(self, *args, grouper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datar.grouper = grouper


class SeriesAgg(Series):
    def __init__(self, *args, grouper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datar.grouper = grouper


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
    out = value.copy()
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
