"""Set operations.

https://github.com/tidyverse/dplyr/blob/master/R/sets.r
"""

from __future__ import annotations

from typing import Any

from datar.apis.dplyr import (
    ungroup,
    bind_rows,
    intersect,
    union,
    setdiff,
    union_all,
    setequal,
    symdiff,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...common import (
    setdiff as _setdiff,
    union as _union,
    intersect as _intersect,
)


def _check_xy(x: Tibble, y: Tibble) -> None:
    """Check that x and y have compatible columns."""
    if len(list(x.columns)) != len(list(y.columns)):
        raise ValueError(
            "not compatible:\n"
            f"- different number of columns: "
            f"{len(list(x.columns))} vs {len(list(y.columns))}"
        )

    in_y_not_x = _setdiff(y.columns, x.columns)
    in_x_not_y = _setdiff(x.columns, y.columns)
    if in_y_not_x or in_x_not_y:
        msg = ["not compatible:"]
        if in_y_not_x:
            msg.append(f"- Cols in `y` but not `x`: {list(in_y_not_x)}.")
        if in_x_not_y:
            msg.append(f"- Cols in `x` but not `y`: {list(in_x_not_y)}.")
        raise ValueError("\n".join(msg))


# ── union_all ───────────────────────────────────────────────────────────────


@union_all.register(object, backend="polars")
def _union_all_obj(x: Any, y: Any) -> Any:
    """Union of two objects."""
    import polars as pl

    if isinstance(x, pl.Expr) or isinstance(y, pl.Expr):
        raise ValueError("union_all not supported on lazy expressions")
    return pl.concat([pl.Series(x), pl.Series(y)]).to_list()


@union_all.register((Tibble, LazyTibble), backend="polars")
def _union_all(
    x: Tibble,
    y: Tibble,
) -> Tibble:
    """Union all rows of two Tibbles."""
    _check_xy(x, y)
    import polars as pl

    result = pl.concat([x.collect(), y.collect()], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── symdiff ─────────────────────────────────────────────────────────────────


@symdiff.register(object, backend="polars")
def _symdiff_obj(x: Any, y: Any) -> Any:
    """Symmetric difference of two vectors."""
    a = _setdiff(_union(x, y), _intersect(x, y))
    return a


@symdiff.register((Tibble, LazyTibble), backend="polars")
def _symdiff_df(x: Tibble, y: Tibble) -> Tibble:
    """Symmetric difference of two Tibbles."""
    _check_xy(x, y)

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    # Get all column names
    cols = list(x_df.columns)

    # Concatenate and find non-duplicated rows
    combined = pl.concat([x_df, y_df], how="diagonal_relaxed")
    # Symmetric diff = rows that appear in exactly one
    result = combined.unique(maintain_order=True).join(
        combined.group_by(cols).agg(pl.len().alias("_n")),
        on=cols,
        how="left",
    ).filter(pl.col("_n") == 1).drop("_n")

    return Tibble(result.lazy())


# ── intersect ───────────────────────────────────────────────────────────────


@intersect.register((Tibble, LazyTibble), backend="polars")
def _intersect_df(x: Tibble, y: Tibble) -> Tibble:
    """Intersection of two Tibbles."""
    _check_xy(x, y)
    import polars as pl

    result = x.collect().join(y.collect(), on=list(x.columns), how="inner").unique(
        maintain_order=True
    )
    return Tibble(result.lazy())


# ── union ───────────────────────────────────────────────────────────────────


@union.register((Tibble, LazyTibble), backend="polars")
def _union_df(x: Tibble, y: Tibble) -> Tibble:
    """Union of two Tibbles."""
    _check_xy(x, y)
    import polars as pl

    result = pl.concat(
        [x.collect(), y.collect()], how="diagonal_relaxed"
    ).unique(maintain_order=True)
    return Tibble(result.lazy())


# ── setdiff ─────────────────────────────────────────────────────────────────


@setdiff.register((Tibble, LazyTibble), backend="polars")
def _setdiff_df(x: Tibble, y: Tibble) -> Tibble:
    """Set difference of two Tibbles (rows in x not in y)."""
    _check_xy(x, y)
    import polars as pl

    result = x.collect().join(
        y.collect(), on=list(x.columns), how="anti"
    )
    return Tibble(result.lazy())


# ── setequal ────────────────────────────────────────────────────────────────


@setequal.register((Tibble, LazyTibble), backend="polars")
def _set_equal_df(x: Tibble, y: Tibble, equal_na: bool = True) -> bool:
    """Check if two Tibbles have the same rows (ignoring order)."""
    _check_xy(x, y)

    import polars as pl

    cols = list(x.columns)
    x_sorted = x.collect().sort(cols)
    y_sorted = y.collect().sort(cols)

    if x_sorted.shape != y_sorted.shape:
        return False

    return x_sorted.equals(y_sorted)
