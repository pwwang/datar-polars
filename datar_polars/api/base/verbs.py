"""Function from R-base that can be used as verbs"""
from typing import Any, List, Sequence, Tuple

import numpy as np
import polars as pl
from polars import DataFrame, Series
from polars.internals.dataframe.groupby import GroupBy
from datar.core.utils import arg_match
from datar.apis.base import (
    colnames,
    rownames,
    set_colnames,
    set_rownames,
    dim,
    nrow,
    ncol,
    diag,
    t,
    duplicated,
    max_col,
    complete_cases,
    head,
    tail,
)

from ...utils import is_scalar, unique as _unique


@colnames.register(DataFrame, backend="polars")
def _colnames(df: DataFrame, nested: bool = True) -> List[str]:
    has_nest = any("$" in str(col) for col in df.columns)
    if not has_nest or not nested:
        return df.columns

    # x, y, y -> x, y
    names = [col.split("$", 1)[0] for col in df.columns]
    names = _unique(names)
    return names if nested else df.columns


@colnames.register(GroupBy, backend="polars")
def _colnames_groupby(df: GroupBy, nested: bool = True) -> List[str]:
    columns = df._df.columns()
    has_nest = any("$" in str(col) for col in columns)
    if not has_nest or not nested:
        return columns

    # x, y, y -> x, y
    names = [col.split("$", 1)[0] for col in columns]
    names = _unique(names)
    return names if nested else columns


@set_colnames.register(DataFrame, backend="polars")
def _set_colnames(
    df: DataFrame,
    names: Sequence[str],
    nested: bool = True,
) -> DataFrame:
    df = df.clone()
    if not nested:
        df.columns = names
        return df

    # x, y$a, y$b with names m, n -> m, n$a, n$b
    old_names = colnames(
        df,
        nested=True,
        __ast_fallback="normal",
        __backend="polars",
    )
    mapping = dict(zip(old_names, names))
    names = []
    for col in df.columns:
        strcol = str(col)
        if "$" in strcol:
            prefix, suffix = strcol.split("$", 1)
            new_prefix = mapping[prefix]
            names.append(f"{new_prefix}${suffix}")
        else:
            names.append(mapping[col])
    df.columns = names
    return df


@rownames.register(DataFrame, backend="polars")
def _rownames(df: DataFrame) -> List[int]:
    return np.arange(df.shape[0]).tolist()


@rownames.register(GroupBy, backend="polars")
def _rownames_groupby(df: GroupBy) -> List[int]:
    return np.arange(df._df.shape()[0]).tolist()


@set_rownames.register(DataFrame, backend="polars")
def _set_rownames(df: DataFrame, names: Any):
    raise ValueError("Cannot set row names for polars DataFrame")


@dim.register(DataFrame, backend="polars")
def _dim(x: DataFrame, nested: bool = True) -> Tuple[int, int]:
    return (
        nrow(x, __ast_fallback="normal", __backend="polars"),
        ncol(x, nested, __ast_fallback="normal", __backend="polars"),
    )


@nrow.register(DataFrame, backend="polars")
def _nrow(_data: DataFrame) -> int:
    return _data.shape[0]


@nrow.register(GroupBy, backend="polars")
def _nrow_groupby(_data: GroupBy) -> int:
    return _data._df.shape()[0]


@ncol.register(DataFrame, backend="polars")
def _ncol(_data: DataFrame, nested: bool = True) -> int:
    if not nested:
        return _data.shape[1]

    return len(
        colnames(
            _data,
            nested=nested,
            __ast_fallback="normal",
            __backend="polars",
        )
    )


@ncol.register(GroupBy, backend="polars")
def _ncol_groupby(_data: GroupBy, nested: bool = True) -> int:
    if not nested:
        return _data._df.shape()[1]

    return len(
        colnames(
            _data,
            nested=nested,
            __ast_fallback="normal",
            __backend="polars",
        )
    )


@diag.register(object, backend="polars")
def _diag(x=1, nrow=None, ncol=None):
    if nrow is None and isinstance(x, int):
        nrow = x
        x = 1
    if ncol is None:
        ncol = nrow
    if is_scalar(x):
        nmax = max(nrow, ncol)
        x = [x] * nmax
    elif nrow is not None:
        nmax = max(nrow, ncol)
        nmax = nmax // len(x)
        x = x * nmax

    x = np.array(x)
    ret = DataFrame(np.diag(x))
    return ret[:nrow, :ncol]


@diag.register(DataFrame, backend="polars")
def _diag_df(
    x,
    nrow=None,
    ncol=None,
):
    if nrow is not None and ncol is not None:
        raise ValueError("Extra arguments received for diag.")

    if nrow is not None:
        np.fill_diagonal(x.to_numpy(), nrow)
        return x

    return np.diag(x)


@t.register(DataFrame, backend="polars")
def _t(_data: DataFrame, copy=False):
    return _data.transpose(copy=copy)


@duplicated.register(DataFrame, backend="polars")
def _duplicated(
    x: DataFrame,
    incomparables=None,
    from_last=False,
):
    keep = "first" if not from_last else "last"
    return x.duplicated(keep=keep).to_numpy()


@max_col.register(DataFrame, backend="polars")
def _max_col(df, ties_method="random"):
    ties_method = arg_match(
        ties_method, "ties_method", ["random", "first", "last"]
    )

    def which_max_with_ties(ser):
        """Find index with max if ties happen"""
        indices = np.flatnonzero(ser == max(ser))
        if len(indices) == 1 or ties_method == "first":
            return indices[0]
        if ties_method == "random":
            return np.random.choice(indices)
        return indices[-1]

    return df.select(
        pl
        .concat_list(pl.all())
        .map(lambda s: s.apply(which_max_with_ties))
        .alias("max_col")
    )["max_col"].to_numpy()


@complete_cases.register(DataFrame, backend="polars")
def _complete_cases(_data):
    return _data.select(
        pl
        .concat_list(pl.all())
        .map(lambda s: s.apply(lambda x: x.null_count() == 0))
        .alias("complete_cases")
    )["complete_cases"].to_numpy()


# actually from R::utils
@head.register((DataFrame, Series), backend="polars")
def _head(_data, n=6):
    return _data.head(n)


@tail.register((DataFrame, Series), backend="polars")
def _tail(_data, n=6):
    return _data.tail(n)
