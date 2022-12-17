"""Constructing tibbles"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Mapping

from polars import DataFrame
from pipda import Expression, evaluate_expr, ReferenceAttr, ReferenceItem
from datar.apis.tibble import (
    tibble,
    tibble_,
    tibble_row,
    tribble,
    as_tibble,
)

from ...utils import ExpressionWrapper, name_of
from ...tibble import Tibble
from ...contexts import Context

if TYPE_CHECKING:
    from polars import DataType


@tibble.register(backend="polars")
def _tibble(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: int = None,
    _dtypes: DataType | Mapping[str, DataType] = None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> DataFrame:
    """Construct a tibble/dataframe from a list of columns.

    Args:
        *args: Columns to be added to the tibble.
        _name_repair: How to repair the column names. See `tibble_`.
        _rows: Number of rows to be added to the tibble.
        _dtypes: Data types of the columns.
        _drop_index: Whether to drop the index column (not supported by polars).
        _index: Index of the tibble (not supported by polars).
        **kwargs: Columns to be added to the tibble.

    Returns:
        A tibble.
    """
    # Scan the args, kwargs see if we have any reference to self
    eval_data = {}
    evaled_args = []
    evaled_kws = {}
    for val in args:
        key = name_of(val)
        if isinstance(val, Expression):
            try:
                # Check if any refers to self (args, kwargs)
                evaluate_expr(val, eval_data, Context.EVAL_DATA)
            except (KeyError, NotImplementedError):
                # If not, it refers to external data
                evaled_args.append(val)
            else:
                # Otherwise, wrap it so that it gets bypass the evaluation
                evaled_args.append(ExpressionWrapper(val))
        else:
            evaled_args.append(val)

        if isinstance(val, DataFrame):
            for col in val.columns:
                eval_data[col] = 1
        else:
            eval_data[key] = 1

    for key, val in kwargs.items():
        if isinstance(val, Expression):
            try:
                evaluate_expr(val, eval_data, Context.EVAL_DATA)
            except (KeyError, NotImplementedError):
                evaled_kws[key] = val
            else:
                evaled_kws[key] = ExpressionWrapper(val)
        else:
            evaled_kws[key] = val

        eval_data[key] = 1

    return tibble_(
        *evaled_args,
        _name_repair=_name_repair,
        _rows=_rows,
        _dtypes=_dtypes,
        __ast_fallback="normal",
        __backend="polars",
        **evaled_kws,
    )


@tibble_.register(object, backend="polars", context=Context.EVAL_DATA)
def _tibble_(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: int = None,
    _dtypes: DataType | Mapping[str, DataType] = None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> DataFrame:
    if _rows is not None:
        raise ValueError(
            "Using `_rows` to create a tibble/dataframe is "
            "not supported by polars backend."
        )

    return Tibble.from_args(
        *args,
        **kwargs,
        _name_repair=_name_repair,
        _dtypes=_dtypes,
    )


@tribble.register(backend="polars")
def _tribble(
    *dummies: Any,
    _name_repair: str | Callable = "check_unique",
    _dtypes: DataType | Mapping[str, DataType] = None,
) -> Tibble:
    columns = []
    data = []
    for i, dummy in enumerate(dummies):
        # columns
        if (
            isinstance(dummy, (ReferenceAttr, ReferenceItem))
            and dummy._pipda_level == 1
        ):
            columns.append(dummy._pipda_ref)

        elif not columns:
            raise ValueError(
                "Must specify at least one column using the `f.<name>` syntax."
            )

        else:
            ncols = len(columns)
            if not data:
                data = [[] for _ in range(ncols)]

            data[i % ncols].append(dummy)

    # only columns provided
    if not data:
        data = [[] for _ in range(len(columns))]

    if len(data[-1]) != len(data[0]):
        raise ValueError(
            "Data must be rectangular. "
            f"{sum(len(dat) for dat in data)} cells is not an integer "
            f"multiple of {len(columns)} columns."
        )

    return Tibble.from_pairs(
        columns,
        data,
        _name_repair=_name_repair,
        _dtypes=_dtypes,
    )


@tibble_row.register(backend="polars")
def _tibble_row(
    *args: Any,
    _name_repair: str | Callable = "check_unique",
    _dtypes: DataType | Mapping[str, DataType] = None,
    **kwargs: Any,
) -> Tibble:
    """Constructs a data frame that is guaranteed to occupy one row.
    Scalar values will be wrapped with `[]`
    Args:
        *args: and
        **kwargs: A set of name-value pairs.
        _name_repair: treatment of problematic column names:
            - "minimal": No name repair or checks, beyond basic existence,
            - "unique": Make sure names are unique and not empty,
            - "check_unique": (default value), no name repair,
                but check they are unique,
            - "universal": Make the names unique and syntactic
            - a function: apply custom name repair
    Returns:
        A constructed dataframe
    """
    if not args and not kwargs:
        df = Tibble()  # can't have 0-column frame
    else:
        df = Tibble.from_args(
            *args,
            **kwargs,
            _name_repair=_name_repair,
            _dtypes=_dtypes,
        )

    if df.shape[0] > 1:
        raise ValueError("All arguments must be size one, use `[]` to wrap.")

    return df


@as_tibble.register(
    (dict, DataFrame),
    context=Context.EVAL_DATA,
    backend="polars",
)
def _as_tibble_df(df: DataFrame | dict) -> Tibble:
    return Tibble(df)


@as_tibble.register(Tibble, context=Context.EVAL_DATA, backend="polars")
def _as_tibble_tbl(df: Tibble) -> Tibble:
    """Convert a polars DataFrame object to Tibble object"""
    return df
