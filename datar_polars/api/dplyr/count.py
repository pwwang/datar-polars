"""Count observations by group

See source https://github.com/tidyverse/dplyr/blob/master/R/count-tally.R
"""

from __future__ import annotations

from typing import Any, Optional

from datar import options_context
from datar.core.defaults import f
from datar.core.utils import logger
from datar.apis.dplyr import (
    n,
    group_by,
    ungroup,
    group_by_drop_default,
    group_vars,
    mutate,
    summarise,
    arrange,
    desc,
    count,
    add_count,
    tally,
    add_tally,
)

import polars as pl

from ...polars import DataFrame
from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble


def _count_expr(wt: Any = None) -> Any:
    """Build the count expression for summarise."""
    if wt is None:
        return pl.len().cast(pl.Int64)
    if isinstance(wt, pl.Expr):
        return wt.sum()
    # Handle pipda references (e.g. f.column from wt=f.w)
    from pipda import ReferenceAttr, ReferenceItem

    if isinstance(wt, (ReferenceAttr, ReferenceItem)):
        return pl.col(wt._pipda_ref).sum()
    # scalar/Series: sum it
    return pl.lit(wt).sum()


def _check_name(name: Optional[str], invars: Any) -> str:
    """Check if count name is valid."""
    if name is None:
        name = _n_name(invars)
        if name != "n":
            logger.warning(
                "Storing counts in `%s`, as `n` already present in input. "
                'Use `name="new_name" to pick a new name.`',
                name,
            )
    elif not isinstance(name, str):
        raise ValueError("`name` must be a single string.")
    return name


def _n_name(invars: Any) -> str:
    """Make sure that name does not exist in invars."""
    name = "n"
    while name in invars:
        name = "n" + name
    return name


# ── count ───────────────────────────────────────────────────────────────────


@count.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _count(
    x: Tibble,
    *args: Any,
    wt: Any = None,
    sort: bool = False,
    name: Optional[str] = None,
    _drop: Optional[bool] = None,
    **kwargs: Any,
) -> Tibble:
    if _drop is None:
        _drop = group_by_drop_default(x)

    if args or kwargs:
        from pipda.function import FunctionCall

        # Separate expression kwargs (need mutate) from simple ref kwargs
        expr_kwargs = {
            k: v for k, v in kwargs.items() if isinstance(v, FunctionCall)
        }
        simple_kwargs = {
            k: v for k, v in kwargs.items() if not isinstance(v, FunctionCall)
        }

        # Evaluate expression kwargs via mutate first
        if expr_kwargs:
            x = mutate(
                x,
                **expr_kwargs,
                __ast_fallback="normal",
                __backend="polars",
            )

        # For simple ref kwargs (count(count=f.x)), the key names the count
        # column and the value is the grouping column.
        if name is None and simple_kwargs:
            name = list(simple_kwargs.keys())[0]

        out = group_by(
            x,
            *args,
            *simple_kwargs.values(),
            *expr_kwargs.keys(),
            _add=True,
            _drop=_drop,
            __ast_fallback="normal",
            __backend="polars",
        )
    else:
        out = x

    out = tally(
        out,
        wt=wt,
        sort=sort,
        name=name,
        __ast_fallback="normal",
        __backend="polars",
    )

    return reconstruct_tibble(out, x)


# ── tally ───────────────────────────────────────────────────────────────────


@tally.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _tally(
    x: Tibble,
    wt: Any = None,
    sort: bool = False,
    name: Optional[str] = None,
) -> Tibble:
    name = _check_name(
        name,
        group_vars(x, __ast_fallback="normal", __backend="polars"),
    )

    with options_context(dplyr_summarise_inform=False):
        out = summarise(
            x,
            __ast_fallback="normal",
            __backend="polars",
            **{name: _count_expr(wt)},
        )

    if sort:
        out = arrange(
            ungroup(out, __ast_fallback="normal", __backend="polars"),
            desc(f[name]),
            __ast_fallback="normal",
            __backend="polars",
        )
        return reconstruct_tibble(out, x)

    return out


# ── add_count ───────────────────────────────────────────────────────────────


@add_count.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _add_count(
    x: Tibble,
    *args: Any,
    wt: Any = None,
    sort: bool = False,
    name: str = "n",
    **kwargs: Any,
) -> Tibble:
    if args or kwargs:
        out = group_by(
            x,
            *args,
            **kwargs,
            _add=True,
            __ast_fallback="normal",
            __backend="polars",
        )
    else:
        out = x

    out = add_tally(
        out,
        wt=wt,
        sort=sort,
        name=name,
        __ast_fallback="normal",
        __backend="polars",
    )
    return out


# ── add_tally ───────────────────────────────────────────────────────────────


@add_tally.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _add_tally(
    x: Tibble,
    wt: Any = None,
    sort: bool = False,
    name: str = "n",
) -> Tibble:
    name = _check_name(name, x.collect_schema().names())

    expr = _count_expr(wt)
    # On grouped frames, wrap in .over() for per-group windowed counts
    gvars = group_vars(x, __ast_fallback="normal", __backend="polars")
    if gvars and isinstance(expr, pl.Expr):
        expr = expr.over(gvars)

    out = mutate(
        x,
        **{name: expr},
        __ast_fallback="normal",
        __backend="polars",
    )

    if sort:
        sort_ed = arrange(
            ungroup(out, __ast_fallback="normal", __backend="polars"),
            desc(f[name]),
            __ast_fallback="normal",
            __backend="polars",
        )
        return reconstruct_tibble(sort_ed, x)

    return out
