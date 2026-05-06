"""Table functions for the polars backend.

Implements: table, tabulate.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import table, tabulate

from ...contexts import Context
from ...tibble import Tibble


# ---- table --------------------------------------------------------------


@table.register(object, context=Context.EVAL, backend="polars")
def _table(
    x: Any,
    *more: Any,
    exclude: Any = None,
    use_na: str = "no",
    dnn: Any = None,
    deparse_level: int = 1,
) -> Any:
    """Build a contingency table of counts."""
    if not more:
        if isinstance(x, pl.Series):
            counts = x.value_counts(sort=True)
            return Tibble(counts)
        if isinstance(x, pl.Expr):
            raise TypeError(
                "table() requires materialized data, not bare Expr."
            )
        if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
            pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
            col = pdf.columns[0]
            counts = pdf[col].value_counts(sort=True)
            return Tibble(counts)
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            from collections import Counter

            cnt = Counter(x)
            keys = sorted(cnt.keys())
            result = pl.DataFrame(
                {"value": keys, "count": [cnt[k] for k in keys]}
            )
            return Tibble(result)
        from collections import Counter

        return Counter([x])

    # Two-variable case
    if len(more) == 1:
        y = more[0]
        if isinstance(x, pl.Series) and isinstance(y, pl.Series):
            df = pl.DataFrame({"x": x, "y": y})
            result = df.group_by(["x", "y"]).len().sort(["x", "y"])
            return Tibble(result)
        if isinstance(x, pl.Series):
            y_vals = list(y) if hasattr(y, "__iter__") else [y]
            df = pl.DataFrame({"x": x, "y": y_vals[: len(x)]})
            result = df.group_by(["x", "y"]).len().sort(["x", "y"])
            return Tibble(result)
        x_vals = list(x) if hasattr(x, "__iter__") else [x]
        y_vals = list(y) if hasattr(y, "__iter__") else [y]
        n = max(len(x_vals), len(y_vals))
        x_vals = (x_vals * ((n // max(len(x_vals), 1)) + 1))[:n]
        y_vals = (y_vals * ((n // max(len(y_vals), 1)) + 1))[:n]
        df = pl.DataFrame({"x": x_vals, "y": y_vals})
        result = df.group_by(["x", "y"]).len().sort(["x", "y"])
        return Tibble(result)
    raise ValueError("table() supports at most 2 variables.")


# ---- tabulate -----------------------------------------------------------


@tabulate.register(object, context=Context.EVAL, backend="polars")
def _tabulate(bin: Any, nbins: Any = None) -> Any:
    """Count occurrences of each integer value."""
    if isinstance(bin, pl.Series):
        vals = bin.drop_nulls().to_list()
    elif isinstance(bin, pl.Expr):
        raise TypeError("tabulate() requires materialized data.")
    elif hasattr(bin, "__iter__") and not isinstance(bin, (str, bytes)):
        vals = [v for v in bin if v is not None]
    else:
        vals = [bin] if bin is not None else []
    if not vals:
        return []
    int_vals = [int(v) for v in vals]
    if nbins is None:
        nbins = max(1, max(int_vals) if int_vals else 0)
    from collections import Counter

    cnt = Counter(int_vals)
    return [cnt.get(i, 0) for i in range(1, nbins + 1)]
