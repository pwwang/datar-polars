"""DataFrame-level utility functions for the polars backend.

Implements: nrow, ncol, dim, colnames, rownames,
set_colnames, set_rownames, head, tail, diag, t,
duplicated, unique, complete_cases.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    colnames,
    complete_cases,
    diag,
    dim,
    duplicated,
    head,
    max_col,
    ncol,
    nrow,
    rownames,
    set_colnames,
    set_rownames,
    t,
    tail,
    unique,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble


# ---- nrow ---------------------------------------------------------------


@nrow.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _nrow_tibble(x: Tibble) -> int:
    return x.collect().height


@nrow.register(object, context=Context.EVAL, backend="polars")
def _nrow_obj(x: Any) -> int:
    if isinstance(x, pl.Series):
        return len(x)
    if isinstance(x, pl.Expr):
        raise TypeError(
            "Cannot determine nrow of an expression without a context."
        )
    if hasattr(x, "__len__"):
        return len(x)
    return 0


# ---- ncol ---------------------------------------------------------------


@ncol.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _ncol_tibble(x: Tibble) -> int:
    return len(x.columns)


@ncol.register(object, context=Context.EVAL, backend="polars")
def _ncol_obj(x: Any) -> int:
    if isinstance(x, pl.Series):
        return 1
    if hasattr(x, "columns"):
        return len(getattr(x, "columns"))
    if hasattr(x, "__len__"):
        return len(x)
    return 0


# ---- dim ----------------------------------------------------------------


@dim.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _dim_tibble(x: Tibble) -> tuple[int, int]:
    return (x.collect().height, len(x.columns))


@dim.register(object, context=Context.EVAL, backend="polars")
def _dim_obj(x: Any) -> tuple[int, int] | None:
    if isinstance(x, pl.Series):
        return (len(x),)
    if hasattr(x, "shape"):
        return getattr(x, "shape")
    if hasattr(x, "__len__"):
        return (len(x),)
    return None


# ---- colnames -----------------------------------------------------------


@colnames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _colnames_tibble(x: Tibble) -> list[str]:
    return x.columns


@colnames.register(object, context=Context.EVAL, backend="polars")
def _colnames_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return [x.name] if x.name else [""]
    if hasattr(x, "columns"):
        return list(getattr(x, "columns"))
    return None


# ---- rownames -----------------------------------------------------------


@rownames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rownames_tibble(x: Tibble) -> list[str] | None:
    if x._datar.get("rownames"):
        return x._datar["rownames"]
    return None


@rownames.register(object, context=Context.EVAL, backend="polars")
def _rownames_obj(x: Any) -> Any:
    if hasattr(x, "_datar") and getattr(x, "_datar", {}).get("rownames"):
        return x._datar["rownames"]
    if hasattr(x, "index"):
        return list(getattr(x, "index"))
    return None


# ---- set_colnames -------------------------------------------------------


@set_colnames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _set_colnames_tibble(x: Tibble, names: list[str]) -> Tibble:
    old_cols = x.columns
    if len(names) != len(old_cols):
        raise ValueError(
            f"Length of new names ({len(names)}) must match "
            f"number of columns ({len(old_cols)})."
        )
    return x.rename(dict(zip(old_cols, names)))


@set_colnames.register(object, context=Context.EVAL, backend="polars")
def _set_colnames_obj(x: Any, names: list[str]) -> Any:
    if isinstance(x, pl.Series):
        return x.alias(names[0])
    if isinstance(x, pl.Expr):
        return x.alias(names[0])
    if hasattr(x, "rename"):
        return x.rename(columns=dict(zip(getattr(x, "columns", []), names)))
    return x


# ---- set_rownames -------------------------------------------------------


@set_rownames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _set_rownames_tibble(x: Tibble, names: list[str]) -> Tibble:
    x = reconstruct_tibble(x)
    x._datar["rownames"] = names
    return x


@set_rownames.register(object, context=Context.EVAL, backend="polars")
def _set_rownames_obj(x: Any, names: list[str]) -> Any:
    if hasattr(x, "_datar"):
        x._datar["rownames"] = names
    return x


# ---- head ---------------------------------------------------------------


@head.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _head_tibble(x: Tibble, n: int = 6) -> pl.DataFrame:
    return x.limit(n).collect()


@head.register(object, context=Context.EVAL, backend="polars")
def _head_obj(x: Any, n: int = 6) -> Any:
    if isinstance(x, pl.Series):
        return x.head(n)
    if isinstance(x, pl.Expr):
        return x.head(n)
    if hasattr(x, "__getitem__"):
        return x[:n]
    return x


# ---- tail ---------------------------------------------------------------


@tail.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _tail_tibble(x: Tibble, n: int = 6) -> pl.DataFrame:
    return x.tail(n).collect()


@tail.register(object, context=Context.EVAL, backend="polars")
def _tail_obj(x: Any, n: int = 6) -> Any:
    if isinstance(x, pl.Series):
        return x.tail(n)
    if isinstance(x, pl.Expr):
        return x.tail(n)
    if hasattr(x, "__getitem__"):
        return x[-n:]
    return x


# ---- duplicated ---------------------------------------------------------


@duplicated.register(pl.Expr, context=Context.EVAL, backend="polars")
def _duplicated_expr(
    x: pl.Expr, incomparables: Any = None, from_last: bool = False
) -> pl.Expr:
    """R-compatible duplicated: marks subsequent occurrences after first as True.

    Uses cum_count().over(x) > 1 instead of is_duplicated() because polars
    is_duplicated() marks ALL occurrences of a duplicate value, while R's
    duplicated() only marks the second and later occurrences.
    """
    if from_last:
        return x.reverse().cum_count().over(x.reverse()).reverse() > 1
    return x.cum_count().over(x) > 1


@duplicated.register(object, context=Context.EVAL, backend="polars")
def _duplicated_obj(
    x: Any, incomparables: Any = None, from_last: bool = False
) -> Any:
    if isinstance(x, pl.Series):
        df = x.to_frame()
        col = pl.col(x.name)
        if from_last:
            result = df.select(
                _dup=col.reverse().cum_count().over(col.reverse()).reverse() > 1,
            )["_dup"]
        else:
            result = df.select(_dup=col.cum_count().over(col) > 1)["_dup"]
        return result
    if isinstance(x, pl.Expr):
        if from_last:
            return x.reverse().cum_count().over(x.reverse()).reverse() > 1
        return x.cum_count().over(x) > 1
    if hasattr(x, "duplicated"):
        return x.duplicated()
    incomparables = incomparables or []
    dups: set = set()
    result: list[bool] = []
    vals = reversed(x) if from_last else x
    for v in vals:
        if v in incomparables:
            result.append(False)
        elif v in dups or (isinstance(v, float) and v != v):
            result.append(True)
        else:
            dups.add(v)
            result.append(False)
    if from_last:
        result = list(reversed(result))
    return result


# ---- unique -------------------------------------------------------------


@unique.register(pl.Expr, context=Context.EVAL, backend="polars")
def _unique_expr(x: pl.Expr) -> pl.Expr:
    return x.unique()


@unique.register(object, context=Context.EVAL, backend="polars")
def _unique_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.unique()
    if isinstance(x, pl.Expr):
        return x.unique()
    if hasattr(x, "unique"):
        return x.unique()
    seen: set = set()
    result: list = []
    for v in x:
        if v not in seen:
            result.append(v)
            seen.add(v)
    return result


# ---- complete_cases -----------------------------------------------------


@complete_cases.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _complete_cases_tibble(x: Tibble) -> pl.Series:
    return (
        x.select(pl.all_horizontal(pl.all().is_not_null()))
        .collect()
        .to_series()
    )


@complete_cases.register(object, context=Context.EVAL, backend="polars")
def _complete_cases_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.is_not_null()
    if isinstance(x, pl.Expr):
        return x.is_not_null()
    return x is not None


# ---- diag ---------------------------------------------------------------


@diag.register(pl.Expr, context=Context.EVAL, backend="polars")
def _diag_expr(
    x: pl.Expr, nrow: Any = None, ncol: Any = None
) -> pl.Expr:
    """Diagonal of an expression (no-op at lazy Expr level)."""
    return x


@diag.register(object, context=Context.EVAL, backend="polars")
def _diag_obj(x: Any, nrow: Any = None, ncol: Any = None) -> Any:
    """Extract or construct a diagonal."""
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        n = min(pdf.height, pdf.width)
        result = [pdf[row, row] for row in range(n)]
        return pl.Series("diag", result)
    if isinstance(x, pl.Series):
        n = len(x)
        if nrow is not None:
            n = nrow
        if ncol is not None:
            n = ncol
        vals = x.to_list()
        rows = []
        for i in range(n):
            row = [0] * n
            if i < len(vals):
                row[i] = vals[i]
            rows.append(row)
        return pl.DataFrame(rows)
    if isinstance(x, pl.Expr):
        return x
    import numpy as np

    # R's diag(x, nrow, ncol): if x is a scalar, create diagonal matrix
    if nrow is not None or ncol is not None:
        n = nrow if nrow is not None else ncol
        return np.eye(n, n if ncol is None else ncol) * x

    return np.diag(x, k=0)


# ---- t ------------------------------------------------------------------


@t.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _t_tibble(x: Tibble) -> Tibble:
    """Transpose a Tibble."""
    pdf = x.collect()
    cols = pdf.columns
    data = [pdf[c].to_list() for c in cols]
    if data:
        transposed = list(zip(*data))
    else:
        transposed = []
    new_cols = [str(i) for i in range(pdf.height)]
    result = pl.DataFrame(transposed, schema=new_cols)
    return Tibble(result)


@t.register(object, context=Context.EVAL, backend="polars")
def _t_obj(x: Any) -> Any:
    """Transpose a matrix/DataFrame."""
    if isinstance(x, pl.Series):
        return pl.DataFrame(
            [x.to_list()],
            schema=[f"col_{i}" for i in range(len(x))],
        )
    if isinstance(x, pl.Expr):
        raise TypeError(
            "Cannot transpose a bare Expr; use within a DataFrame context."
        )
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        cols = pdf.columns
        data = [pdf[c].to_list() for c in cols]
        transposed = list(zip(*data)) if data else []
        new_cols = [str(i) for i in range(len(pdf))]
        return pl.DataFrame(transposed, schema=new_cols)
    import numpy as np

    return np.transpose(x)


# ---- max_col -------------------------------------------------------------


@max_col.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _max_col_tibble(
    x: Tibble, ties_method: str = "random", nested: bool = True
) -> pl.Series:
    """Find column index with the max value per row (1-based)."""
    pdf = x.collect()
    ncols = len(pdf.columns)
    if ncols == 0:
        return pl.Series([], dtype=pl.Int64)

    numeric_cols = [
        c for c in pdf.columns
        if pdf[c].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
    ]
    if not numeric_cols:
        numeric_cols = list(pdf.columns)

    result = []
    for i in range(len(pdf)):
        row_vals = [pdf[c][i] for c in numeric_cols]
        if all(v is None for v in row_vals):
            result.append(None)
        else:
            max_val = max(v for v in row_vals if v is not None)
            if ties_method == "first":
                idx = next(j for j, v in enumerate(row_vals) if v == max_val) + 1
            elif ties_method == "last":
                idx = (
                    len(row_vals)
                    - next(j for j, v in enumerate(reversed(row_vals)) if v == max_val)
                )
            else:
                candidates = [j for j, v in enumerate(row_vals) if v == max_val]
                import random

                idx = random.choice(candidates) + 1
            result.append(idx)
    return pl.Series("max_col", result, dtype=pl.Int64)


@max_col.register(object, context=Context.EVAL, backend="polars")
def _max_col_obj(
    x: Any, ties_method: str = "random", nested: bool = True
) -> Any:
    """Find column index with max value per row from a numpy array or matrix."""
    import numpy as np

    arr = np.asarray(x)
    if arr.ndim == 1:
        return 1
    return np.argmax(arr, axis=1) + 1
