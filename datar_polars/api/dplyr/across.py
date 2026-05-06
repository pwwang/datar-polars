"""Apply a function (or functions) across multiple columns

See source https://github.com/tidyverse/dplyr/blob/master/R/across.R
"""

from __future__ import annotations

from typing import Any, Optional

from pipda import evaluate_expr
from datar.apis.dplyr import across, c_across, if_any, if_all

from ...collections import Collection
from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...utils import vars_select, name_of
from ...common import is_scalar
from .context import _MultiValueExpr
from .summarise import _frame_to_expr_items


def _get_gvars(data: Tibble) -> list:
    """Return grouping variable names from a Tibble."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _resolve_cols(data: Tibble, cols: Any) -> list:
    """Resolve column selection to a list of column names."""
    if cols is None:
        all_columns = list(data.collect_schema().names())
        gvars = _get_gvars(data)
        return [c for c in all_columns if c not in gvars]
    all_columns = list(data.collect_schema().names())

    # Unwrap slice from Collection (e.g., c[f.a :] → Collection-wrapped slice)
    if isinstance(cols, Collection) and len(cols) == 1:
        inner = cols[0]
        if isinstance(inner, slice):
            cols = inner

    # Pipda evaluate_expr converts Collection to plain list — unwrap if needed
    if (
        isinstance(cols, (list, tuple))
        and not isinstance(cols, Collection)
        and len(cols) == 1
        and isinstance(cols[0], slice)
    ):
        cols = cols[0]

    # Handle slice selections (e.g., f[f.a : f.b] → slice("a", "b", None))
    if isinstance(cols, slice):
        start = cols.start
        stop = cols.stop
        # Resolve ReferenceAttr / ReferenceItem to column name string
        if hasattr(start, "_pipda_ref"):
            start = start._pipda_ref
        if hasattr(stop, "_pipda_ref"):
            stop = stop._pipda_ref
        try:
            start_idx = all_columns.index(start) if start else 0
            stop_idx = (
                all_columns.index(stop) if stop else len(all_columns) - 1
            )
            return all_columns[start_idx : stop_idx + 1]
        except ValueError:
            pass

    # Empty column list — no columns selected.
    # Collection subclasses (Intersect, Inverted, etc.) have len==0 until
    # expand() is called, so skip them here — let vars_select expand them.
    if (
        isinstance(cols, (list, tuple))
        and not isinstance(cols, Collection)
        and len(cols) == 0
    ):
        return []

    idx = vars_select(all_columns, cols, raise_nonexists=False)
    return [all_columns[i] for i in idx]


@across.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _across(
    _data: Tibble,
    *args: Any,
    _names: Optional[str] = None,
    _fn_context: Context = Context.EVAL,
    **kwargs: Any,
) -> Any:
    """Apply functions across a selection of columns.

    Returns a list of pl.Expr for use inside verbs like mutate/summarise.
    """
    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)

    _cols, _fns, *rest = args
    other_args = tuple(rest)

    # Only evaluate lazy expressions; Collection objects are already resolved.
    # pipda.evaluate_expr() wraps list subclasses in a generator that
    # Collection.__init__ cannot unpack correctly.
    if not isinstance(_cols, Collection):
        _cols = evaluate_expr(_cols, _data, Context.SELECT)

    selected = _resolve_cols(_data, _cols)

    # No columns selected → no-op
    if not selected:
        return None

    if selected and isinstance(selected[0], (str,)):
        pass  # selected is already a list of column names

    if _fns is None:
        # No function specified — return selected column expressions
        import polars as pl

        exprs = [pl.col(c) for c in selected]
        if _names is not None:
            # Apply naming scheme
            pass  # basic support
        if len(exprs) == 1:
            return (exprs[0],)
        return exprs
    import polars as pl

    def _apply_fn(fn, col_expr, *fn_args, **fn_kwargs):
        """Apply a function to a column expression, with fallback for Python
        built-ins that don't support pl.Expr (e.g. round, abs)."""
        try:
            return fn(col_expr, *fn_args, **fn_kwargs)
        except TypeError:
            if hasattr(fn, "__name__") and hasattr(col_expr, fn.__name__):
                return getattr(col_expr, fn.__name__)(*fn_args, **fn_kwargs)
            raise

    def _wrap_result(result, name: str) -> pl.Expr:
        """Wrap a function result as a named pl.Expr.

        Handles: pl.Expr → .alias(name),
                list/tuple of pl.Expr → pl.concat_list(...).alias(name),
                anything else → pl.lit(...).alias(name).
        """
        if isinstance(result, pl.Expr):
            return result.alias(name)
        if isinstance(result, (Tibble, LazyTibble, pl.DataFrame, pl.LazyFrame)):
            return _frame_to_expr_items(result, prefix=name)
        if isinstance(result, (list, tuple)):
            if result and all(isinstance(e, pl.Expr) for e in result):
                return pl.concat_list(list(result)).alias(name)
            # Plain list of scalars — let pl.lit handle it
        return pl.lit(result).alias(name)

    exprs = []

    # Single function
    if callable(_fns):
        _fn_name = _fns.__name__ if hasattr(_fns, "__name__") else "fn"
        for col in selected:
            result = _apply_fn(_fns, pl.col(col), *other_args, **kwargs)
            name = (
                f"{col}"
                if _names is None
                else _names.replace("{_col}", col)
                .replace("{_fn1}", _fn_name)
                .replace("{_fn0}", _fn_name)
                .replace("{_fn}", _fn_name)
            )
            wrapped = _wrap_result(result, name)
            if isinstance(wrapped, list):
                exprs.extend(wrapped)
            else:
                exprs.append(wrapped)
    elif isinstance(_fns, dict):
        for col in selected:
            for fn_key, fn in _fns.items():
                result = _apply_fn(fn, pl.col(col), *other_args, **kwargs)
                name = (
                    f"{col}_{fn_key}"
                    if _names is None
                    else _names.replace("{_col}", col)
                    .replace("{_fn1}", str(fn_key))
                    .replace("{_fn0}", str(fn_key))
                    .replace("{_fn}", str(fn_key))
                )
                wrapped = _wrap_result(result, name)
                if isinstance(wrapped, list):
                    exprs.extend(wrapped)
                else:
                    exprs.append(wrapped)
    elif isinstance(_fns, list):
        for i, fn in enumerate(_fns):
            for col in selected:
                result = _apply_fn(fn, pl.col(col), *other_args, **kwargs)
                name = (
                    f"{col}_{i}"
                    if _names is None
                    else _names.replace("{_col}", col)
                    .replace("{_fn1}", str(i + 1))
                    .replace("{_fn0}", str(i))
                    .replace("{_fn}", str(i))
                )
                wrapped = _wrap_result(result, name)
                if isinstance(wrapped, list):
                    exprs.extend(wrapped)
                else:
                    exprs.append(wrapped)
    else:
        # Function-like object from pipda
        for col in selected:
            result = _fns(pl.col(col), *other_args, **kwargs)
            wrapped = _wrap_result(result, col)
            if isinstance(wrapped, list):
                exprs.extend(wrapped)
            else:
                exprs.append(wrapped)

    if len(exprs) == 1:
        # Return as 1-tuple so that mutate/summarise can distinguish
        # across results (which use expr output names) from raw
        # expressions (which use the kwarg/positional key).
        return (exprs[0],)
    return exprs


@c_across.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _c_across(
    _data: Tibble,
    _cols: Any = None,
) -> Any:
    """Select columns for rowwise operations (c_across).

    Returns a list of pl.Expr for the selected columns.
    """
    selected = _resolve_cols(_data, _cols)
    import polars as pl

    return [pl.col(c) for c in selected]


def _combine_predicates(exprs, combine: str = "any") -> Any:
    """Combine predicate expressions with AND (all) or OR (any)."""
    import polars as pl

    if not exprs:
        return pl.lit(True)
    if combine == "any":
        result = exprs[0]
        for e in exprs[1:]:
            result = result | e
    else:
        result = exprs[0]
        for e in exprs[1:]:
            result = result & e
    return result


@if_any.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _if_any(
    _data: Tibble,
    *args: Any,
    _names: Optional[str] = None,
    _context: Optional[Context] = None,
    **kwargs: Any,
) -> Any:
    """Apply predicate to columns, return True if any column is True."""
    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)

    _cols, _fns, *rest = args
    other_args = tuple(rest)

    selected = _resolve_cols(_data, _cols)

    import polars as pl

    if _fns is None:
        # Use column values directly as booleans
        if not selected:
            return pl.lit(False)
        exprs = [pl.col(c).cast(pl.Boolean) for c in selected]
        return _combine_predicates(exprs, "any")

    if callable(_fns):
        exprs = [_fns(pl.col(c), *other_args, **kwargs) for c in selected]
    else:
        # pipda expression
        exprs = [
            evaluate_expr(_fns(pl.col(c), *other_args, **kwargs), _data, Context.EVAL)
            for c in selected
        ]

    return _combine_predicates(exprs, "any")


@if_all.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _if_all(
    _data: Tibble,
    *args: Any,
    _names: Optional[str] = None,
    _context: Optional[Context] = None,
    **kwargs: Any,
) -> Any:
    """Apply predicate to columns, return True if all columns are True."""
    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)

    _cols, _fns, *rest = args
    other_args = tuple(rest)

    selected = _resolve_cols(_data, _cols)

    import polars as pl

    if _fns is None:
        if not selected:
            return pl.lit(True)
        exprs = [pl.col(c).cast(pl.Boolean) for c in selected]
        return _combine_predicates(exprs, "all")

    if callable(_fns):
        exprs = [_fns(pl.col(c), *other_args, **kwargs) for c in selected]
    else:
        exprs = [
            evaluate_expr(_fns(pl.col(c), *other_args, **kwargs), _data, Context.EVAL)
            for c in selected
        ]

    return _combine_predicates(exprs, "all")
