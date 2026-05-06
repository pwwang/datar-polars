"""Base sequence functions for the polars backend.

Implements: seq_along, seq_len, rep, rev, sample, c_ (combine),
length, lengths, sort, rank, order, match.
"""

from __future__ import annotations

import random as _random
from typing import Any

import polars as pl

from datar.apis.base import (
    c_,
    expand_grid,
    length,
    lengths,
    match,
    order,
    rank,
    rep,
    rev,
    sample,
    seq,
    seq_along,
    seq_len,
    sort,
)

from ...collections import Collection
from ...contexts import Context
from ...tibble import Tibble, LazyTibble


# ---- seq_along ----------------------------------------------------------


@seq_along.register(pl.Expr, context=Context.EVAL, backend="polars")
def _seq_along_expr(x: pl.Expr) -> pl.Expr:
    """Return 0-based index range matching length of `x`."""
    return pl.int_range(1, x.len() + 1)


@seq_along.register(object, context=Context.EVAL, backend="polars")
def _seq_along_obj(x: Any) -> Any:
    """Return 0-based index range matching length of `x`."""
    if isinstance(x, pl.Series):
        return pl.int_range(1, x.len() + 1, eager=True)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return list(range(1, len(list(x)) + 1))
    return [1]  # scalar is length 1


# ---- seq_len ------------------------------------------------------------


@seq_len.register(pl.Expr, context=Context.EVAL, backend="polars")
def _seq_len_expr(x: pl.Expr) -> pl.Expr:
    """Return range from 0 to x-1 as an Expr."""
    return pl.int_ranges(0, x.cast(pl.Int64))


@seq_len.register(object, context=Context.EVAL, backend="polars")
def _seq_len_obj(x: Any) -> Any:
    """Return range from 0 to x-1."""
    if isinstance(x, pl.Series):
        n = x[0] if len(x) > 0 else 0
        return list(range(int(n)))
    if isinstance(x, pl.Expr):
        return list(range(int(x)))  # shouldn't happen but safety
    return list(range(int(x)))


# ---- rep ----------------------------------------------------------------


@rep.register(pl.Expr, context=Context.EVAL, backend="polars")
def _rep_expr(
    x: pl.Expr,
    times: int = 1,
    length: int | None = None,
    each: int = 1,
) -> pl.Expr:
    """Repeat elements of an Expr.

    Uses repeat_by to produce a list column — each row gets a list
    of repeated values.  Does NOT explode, so row count is preserved.
    """
    result = x
    if each > 1:
        result = result.repeat_by(pl.lit(each))
    if times > 1:
        result = result.repeat_by(pl.lit(times))
    if length is not None:
        result = result.list.slice(0, length)
    return result


@rep.register(object, context=Context.EVAL, backend="polars")
def _rep_obj(
    x: Any,
    times: int = 1,
    length: int | None = None,
    each: int = 1,
) -> Any:
    """Repeat elements of an object."""
    if isinstance(x, pl.Series):
        vals = x.to_list()
    elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        vals = list(x)
    else:
        vals = [x]

    # Apply `each`
    if each > 1:
        vals = [v for v in vals for _ in range(each)]

    # Apply `times`
    if isinstance(times, (int, float)):
        vals = vals * int(times)
    elif hasattr(times, "__iter__") and not isinstance(times, (str, bytes)):
        times_list = list(times)
        if len(times_list) == 1:
            vals = vals * int(times_list[0])
        else:
            result = []
            for v, t in zip(vals, times_list):
                result.extend([v] * int(t))
            vals = result
    else:
        vals = vals * int(times)

    # Apply `length`
    if length is not None:
        vals = vals[:length]

    if isinstance(x, pl.Series):
        return pl.Series(x.name, vals, dtype=x.dtype)
    return vals


# ---- rev ----------------------------------------------------------------


@rev.register(pl.Expr, context=Context.EVAL, backend="polars")
def _rev_expr(x: pl.Expr) -> pl.Expr:
    """Reverse an Expr."""
    return x.reverse()


@rev.register(object, context=Context.EVAL, backend="polars")
def _rev_obj(x: Any) -> Any:
    """Reverse an object."""
    if isinstance(x, pl.Series):
        return x.reverse()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return list(reversed(list(x)))
    return x  # scalar cannot be reversed


# ---- sample -------------------------------------------------------------


@sample.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sample_expr(
    x: pl.Expr,
    size: int | None = None,
    replace: bool = False,
    prob: Any = None,
) -> pl.Expr:
    """Random sample from an Expr.

    When size differs from the column length, wraps the result in
    concat_list so it broadcasts as a single list value (no row-count
    mismatch).
    """
    if size is None:
        return x.shuffle()
    sampled = x.shuffle().head(size).implode()
    return sampled


@sample.register(object, context=Context.EVAL, backend="polars")
def _sample_obj(
    x: Any,
    size: int | None = None,
    replace: bool = False,
    prob: Any = None,
) -> Any:
    """Random sample from an object."""
    if isinstance(x, pl.Series):
        vals = x.to_list()
    elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        vals = list(x)
    else:
        vals = [x]

    n = len(vals)
    if n == 0:
        return pl.Series([], dtype=pl.Int64) if isinstance(x, pl.Series) else []

    if size is None:
        size = n
    size = int(size)

    if replace:
        if prob is not None:
            result = _random.choices(vals, weights=prob, k=size)
        else:
            result = _random.choices(vals, k=size)
    else:
        if size > n:
            size = n
        result = _random.sample(vals, size)

    if isinstance(x, pl.Series):
        return pl.Series(x.name, result)
    return result


# ---- c_ (combine) --------------------------------------------------------

@c_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _c_expr(*args: Any) -> Collection:
    """Combine Expr arguments into a Collection."""
    return Collection(*args)


@c_.register(object, context=Context.EVAL, backend="polars")
def _c_obj(*args: Any) -> Collection:
    """Combine arguments into a Collection.

    Handles pl.Series, scalars, and other types.
    Delegates to the polars-aware Collection constructor.
    """
    return Collection(*args)


# ---- length -------------------------------------------------------------


@length.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _length_tibble(x: Tibble) -> int:
    """Number of elements in a Tibble (rows * columns)."""
    nrow_val = x.collect().height
    ncol_val = len(x.columns)
    return nrow_val * ncol_val


@length.register(object, context=Context.EVAL, backend="polars")
def _length_obj(x: Any) -> int:
    """Number of elements."""
    if isinstance(x, pl.Series):
        return len(x)
    if isinstance(x, pl.Expr):
        raise TypeError("Cannot determine length of a bare expression.")
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        return pdf.height * pdf.width
    if isinstance(x, (str, bytes)):
        return 1
    if hasattr(x, "__len__"):
        return len(x)
    return 1


# ---- lengths ------------------------------------------------------------


@lengths.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lengths_expr(x: pl.Expr) -> pl.Expr:
    """Lengths of each element (for list columns)."""
    return x.list.len()


@lengths.register(object, context=Context.EVAL, backend="polars")
def _lengths_obj(x: Any) -> Any:
    """Lengths of each element."""
    if isinstance(x, pl.Series):
        if x.dtype == pl.List:
            return x.list.len()
        return pl.Series("lengths", [1] * len(x), dtype=pl.Int64)
    if isinstance(x, pl.Expr):
        return x.list.len()
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        return pl.Series("lengths", [pdf.width] * pdf.height, dtype=pl.Int64)
    if hasattr(x, "__len__"):
        return len(x)
    return 1


# ---- sort ---------------------------------------------------------------


@sort.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sort_expr(
    x: pl.Expr,
    decreasing: bool = False,
    na_last: bool = True,
) -> pl.Expr:
    """Sort an expression."""
    return x.sort(descending=decreasing, nulls_last=na_last)


@sort.register(object, context=Context.EVAL, backend="polars")
def _sort_obj(
    x: Any,
    decreasing: bool = False,
    na_last: bool = True,
) -> Any:
    """Sort a vector/iterable."""
    if isinstance(x, pl.Series):
        return x.sort(descending=decreasing, nulls_last=na_last)
    if isinstance(x, pl.Expr):
        return x.sort(descending=decreasing, nulls_last=na_last)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        items = list(x)
        nulls = [v for v in items if v is None]
        non_nulls = [v for v in items if v is not None]
        non_nulls.sort(reverse=decreasing)
        if na_last:
            return non_nulls + nulls
        else:
            return nulls + non_nulls
    return x


# ---- rank ---------------------------------------------------------------


@rank.register(pl.Expr, context=Context.EVAL, backend="polars")
def _rank_expr(
    x: pl.Expr,
    na_last: bool = True,
    ties_method: str = "average",
) -> pl.Expr:
    """Rank an expression."""
    method_map = {
        "average": "average",
        "first": "ordinal",
        "random": "random",
        "max": "max",
        "min": "min",
    }
    polars_method = method_map.get(ties_method, "average")
    return x.rank(method=polars_method, descending=False)


@rank.register(object, context=Context.EVAL, backend="polars")
def _rank_obj(
    x: Any,
    na_last: bool = True,
    ties_method: str = "average",
) -> Any:
    """Rank a vector/iterable."""
    if isinstance(x, pl.Series):
        method_map = {
            "average": "average",
            "first": "ordinal",
            "random": "random",
            "max": "max",
            "min": "min",
        }
        polars_method = method_map.get(ties_method, "average")
        return x.rank(method=polars_method, descending=False)
    if isinstance(x, pl.Expr):
        return _rank_expr(x, na_last=na_last, ties_method=ties_method)
    from scipy import stats as scipy_stats

    return scipy_stats.rankdata(x, method=ties_method)


# ---- order --------------------------------------------------------------


@order.register(pl.Expr, context=Context.EVAL, backend="polars")
def _order_expr(
    x: pl.Expr,
    decreasing: bool = False,
    na_last: bool = True,
) -> pl.Expr:
    """Order indices of an expression."""
    return x.arg_sort(descending=decreasing, nulls_last=na_last)


@order.register(object, context=Context.EVAL, backend="polars")
def _order_obj(
    x: Any,
    decreasing: bool = False,
    na_last: bool = True,
) -> Any:
    """Order indices of a vector/iterable."""
    if isinstance(x, pl.Series):
        return x.arg_sort(descending=decreasing, nulls_last=na_last)
    if isinstance(x, pl.Expr):
        return x.arg_sort(descending=decreasing, nulls_last=na_last)
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if not na_last:
        arr = np.where(np.isnan(arr), -np.inf, arr)
    if decreasing:
        return list(np.argsort(-arr))
    return list(np.argsort(arr))


# ---- match --------------------------------------------------------------


@match.register(pl.Expr, context=Context.EVAL, backend="polars")
def _match_expr(
    x: pl.Expr,
    table: Any,
    nomatch: int = -1,
) -> pl.Expr:
    """Match positions of x in table (Expr version)."""
    if isinstance(table, pl.Expr):
        raise TypeError("match() does not support Expr for table.")
    if isinstance(table, pl.Series):
        table_vals = table.to_list()
    elif hasattr(table, "__iter__") and not isinstance(table, (str, bytes)):
        table_vals = list(table)
    else:
        table_vals = [table]
    return x.is_in(table_vals).cast(pl.Int64)


@match.register(object, context=Context.EVAL, backend="polars")
def _match_obj(
    x: Any,
    table: Any,
    nomatch: int = -1,
) -> Any:
    """Match positions of x in table."""
    if isinstance(table, pl.Series):
        table_vals = table.to_list()
    elif hasattr(table, "__iter__") and not isinstance(table, (str, bytes)):
        table_vals = list(table)
    else:
        table_vals = [table]

    if isinstance(x, pl.Series):
        result = []
        for val in x.to_list():
            try:
                idx = table_vals.index(val) + 1  # 1-based
                result.append(idx)
            except ValueError:
                result.append(nomatch)
        return pl.Series("match", result, dtype=pl.Int64)
    if isinstance(x, pl.Expr):
        return _match_expr(x, table, nomatch)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        result = []
        for val in x:
            try:
                idx = table_vals.index(val) + 1
                result.append(idx)
            except ValueError:
                result.append(nomatch)
        return result
    try:
        return table_vals.index(x) + 1
    except ValueError:
        return nomatch


# ---- seq -----------------------------------------------------------------


@seq.register(pl.Expr, context=Context.EVAL, backend="polars")
def _seq_expr(
    from_: Any = None,
    to: Any = None,
    by: Any = None,
    length_out: Any = None,
    along_with: Any = None,
) -> pl.Expr:
    if along_with is not None:
        n = along_with.len()
        return pl.int_range(1, n + 1)
    if length_out is not None:
        n = length_out if isinstance(length_out, int) else int(length_out)
        if to is not None and from_ is not None:
            step = (to - from_) / (n - 1) if n > 1 else 0.0
            return pl.int_range(0, n).cast(pl.Float64) * step + from_
        return pl.int_range(0, n)
    if from_ is not None and to is not None:
        step = by if by is not None else 1
        if by is None and from_ > to:
            step = -1
        return pl.int_range(from_, to + step, step)
    if from_ is not None:
        return pl.int_range(from_, from_ + 1)
    raise ValueError("seq() requires at least `from_` and `to`, or `length_out`.")


@seq.register(object, context=Context.EVAL, backend="polars")
def _seq_obj(
    from_: Any = None,
    to: Any = None,
    by: Any = None,
    length_out: Any = None,
    along_with: Any = None,
) -> Any:
    if along_with is not None:
        if isinstance(along_with, pl.Series):
            return list(range(1, len(along_with) + 1))
        if hasattr(along_with, "__iter__") and not isinstance(
            along_with, (str, bytes)
        ):
            return list(range(1, len(list(along_with)) + 1))
        return [1]
    if length_out is not None:
        n = int(length_out)
        if to is not None and from_ is not None:
            import numpy as np
            return np.linspace(from_, to, n).tolist()
        if from_ is not None:
            return list(range(int(from_), int(from_) + n))
        if to is not None:
            return list(range(int(to) - n + 1, int(to) + 1))
        return list(range(n))
    if from_ is not None and to is not None:
        step = by if by is not None else 1
        # auto-detect descending: R's seq(3, 1) → c(3, 2, 1)
        if by is None and from_ > to:
            step = -1
        result = []
        val = from_
        while (step > 0 and val <= to) or (step < 0 and val >= to):
            result.append(val)
            val += step
        return result
    if from_ is not None:
        return [from_]
    raise ValueError("seq() requires at least `from_` and `to`, or `length_out`.")


# ---- expand_grid ---------------------------------------------------------


@expand_grid.register(object, context=Context.EVAL, backend="polars")
def _expand_grid(x: Any, *args: Any, **kwargs: Any) -> Tibble:
    """Create a data frame from all combinations of inputs."""
    import itertools

    all_inputs = {}
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        for c in pdf.columns:
            all_inputs[c] = pdf[c].to_list()
    elif isinstance(x, pl.Series):
        all_inputs[x.name or "Var1"] = x.to_list()
    elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        all_inputs["Var1"] = list(x)
    else:
        all_inputs["Var1"] = [x]

    for i, arg in enumerate(args):
        if isinstance(arg, (pl.DataFrame, pl.LazyFrame)):
            pdf = arg.collect() if isinstance(arg, pl.LazyFrame) else arg
            for c in pdf.columns:
                all_inputs[c] = pdf[c].to_list()
        elif isinstance(arg, pl.Series):
            all_inputs[arg.name or f"Var{i + 2}"] = arg.to_list()
        elif hasattr(arg, "__iter__") and not isinstance(arg, (str, bytes)):
            all_inputs[f"Var{i + 2}"] = list(arg)
        else:
            all_inputs[f"Var{i + 2}"] = [arg]

    for k, v in kwargs.items():
        if isinstance(v, pl.Series):
            all_inputs[k] = v.to_list()
        elif hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            all_inputs[k] = list(v)
        else:
            all_inputs[k] = [v]

    keys = list(all_inputs.keys())
    values = list(all_inputs.values())
    combinations = list(itertools.product(*values))
    if not combinations:
        return Tibble(pl.DataFrame({}))
    return Tibble(pl.DataFrame({k: [c[i] for c in combinations] for i, k in enumerate(keys)}))
