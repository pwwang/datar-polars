"""Tidyr API implementations for the polars backend.

Provides drop_na, replace_na, fill, pivot_longer, pivot_wider, separate,
unite, extract, expand, complete, nest, unnest, chop, unchop, pack, unpack,
expand_grid, nesting, crossing, separate_rows, hoist, full_seq, uncount.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Union
import itertools
import re
import math

import polars as pl
import numpy as np

from datar.core.names import repair_names
from datar.core.utils import arg_match, logger
from datar.apis.tidyr import (
    drop_na,
    replace_na,
    fill,
    pivot_longer,
    pivot_wider,
    separate,
    unite,
    extract,
    expand,
    complete,
    nest,
    unnest,
    chop,
    unchop,
    pack,
    unpack,
    expand_grid,
    nesting,
    crossing,
    full_seq,
    separate_rows,
    uncount,
)
from pipda import register_verb

from ..contexts import Context
from ..tibble import Tibble, LazyTibble, reconstruct_tibble, as_tibble
from ..collections import Collection
from ..common import is_scalar, setdiff, union, intersect
from ..utils import (
    vars_select,
    to_series,
    DEFAULT_COLUMN_PREFIX,
    replace_na_with_none,
)
from .base.seq import _seq_obj
from ..polars import DataFrame


# ═══════════════════════════════════════════════════════════════════════════════
# drop_na — drop rows with NA values
# ═══════════════════════════════════════════════════════════════════════════════


@drop_na.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _drop_na(
    _data: Tibble,
    *columns: str,
    _how: str = "any",
) -> Tibble:
    """Drop rows containing missing values."""
    arg_match(_how, "_how", ["any", "all"])

    if columns:
        all_columns = _data.collect_schema().names()
        sel_columns = [all_columns[i] for i in vars_select(all_columns, *columns)]
        result = _data.drop_nulls(subset=sel_columns)
        # also drop nan if the column is numeric, since R treats NaN as NA in drop_na
        for col in sel_columns:
            if _data.collect_schema().get(col).is_numeric():
                result = result.filter(~pl.col(col).is_nan())
    else:
        if _how == "all":
            # Drop rows where ALL columns are null
            all_null_expr = pl.all_horizontal(
                [
                    pl.col(c).is_null() | (
                        pl.col(c).is_nan()
                        if _data.collect_schema().get(c).is_numeric()
                        else pl.lit(False)
                    )
                    for c in _data.collect_schema().names()
                ]
            )
            result = _data.filter(~all_null_expr)
        else:
            result = _data.drop_nulls()
            # Also drop rows where any numeric column is NaN, since R treats NaN as NA in drop_na
            numeric_cols = [c for c in _data.collect_schema().names() if _data.collect_schema().get(c).is_numeric()]
            for col in numeric_cols:
                result = result.filter(~pl.col(col).is_nan())

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# replace_na — replace NA values
# ═══════════════════════════════════════════════════════════════════════════════


@replace_na.register(pl.Series, context=Context.EVAL, backend="polars")
def _replace_na_series(data: pl.Series, data_or_replace=None, replace=None) -> pl.Series:
    """Replace NA in a Series."""
    if data_or_replace is None and replace is None:
        return data
    if replace is None:
        replace = data_or_replace
    else:
        if data_or_replace is not None:
            data = data_or_replace
    if data.dtype.is_float():
        data = data.fill_nan(replace)
    return data.fill_null(replace)


@replace_na.register(pl.Expr, context=Context.EVAL, backend="polars")
def _replace_na_expr(data: pl.Expr, data_or_replace=None, replace=None) -> pl.Expr:
    """Replace NA in an Expr."""
    if data_or_replace is None and replace is None:
        return data
    if replace is None:
        replace = data_or_replace
    else:
        if data_or_replace is not None:
            data = data_or_replace
    # fill_nan is only valid for float-typed expressions; use fill_null for others
    return data.fill_nan(replace).fill_null(replace)


@replace_na.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _replace_na_tibble(
    data: Tibble,
    data_or_replace=None,
    replace=None,
) -> Tibble:
    """Replace NA in a Tibble."""
    if data_or_replace is None and replace is None:
        return data
    if replace is None:
        replace = data_or_replace
    else:
        if data_or_replace is not None:
            data = data_or_replace

    if isinstance(replace, dict):
        # Per-column replacement
        exprs = []
        schema = data.collect_schema()
        for col in schema.names():
            if col in replace:
                e = pl.col(col)
                if schema[col].is_float():
                    e = e.fill_nan(replace[col])
                exprs.append(e.fill_null(replace[col]))
            else:
                exprs.append(pl.col(col))
        result = data.with_columns(exprs)
    else:
        schema = data.collect_schema()
        exprs = []
        for col in schema.names():
            e = pl.col(col)
            if schema[col].is_float():
                e = e.fill_nan(replace)
            exprs.append(e.fill_null(replace))
        result = data.with_columns(exprs)

    return reconstruct_tibble(result, data)


@replace_na.register(object, context=Context.EVAL, backend="polars")
def _replace_na_obj(data: Any, data_or_replace=None, replace=None) -> Any:
    """Replace NA for any object."""
    if data_or_replace is None and replace is None:
        return data
    if replace is None:
        replace = data_or_replace
    else:
        if data_or_replace is not None:
            data = data_or_replace

    if isinstance(data, list):
        return [replace if v is None else v for v in data]
    if data is None:
        return replace
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# fill — fill missing values
# ═══════════════════════════════════════════════════════════════════════════════


@fill.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _fill(
    _data: Tibble,
    *columns: Union[str, int],
    _direction: str = "down",
) -> Tibble:
    """Fill missing values with previous or next entry."""
    if not columns:
        columns = tuple(_data.collect_schema().names())

    all_columns = _data.collect_schema().names()
    col_names = [all_columns[i] for i in vars_select(all_columns, *columns)]

    schema = _data.collect_schema()
    groups = ((_data._datar or {}).get("groups") or [])
    group_cols = [pl.col(g) for g in groups]

    def _fill_expr(col_name: str, direction: str) -> pl.Expr:
        e = pl.col(col_name)
        # Convert NaN to null so forward/backward fill works on them.
        # Only float types can hold NaN; is_not_nan is unsupported on str.
        if schema[col_name].is_float():
            e = e.fill_nan(None)
        if direction == "down":
            e = e.forward_fill()
        elif direction == "up":
            e = e.backward_fill()
        elif direction == "downup":
            e = e.forward_fill().backward_fill()
        elif direction == "updown":
            e = e.backward_fill().forward_fill()
        else:
            raise ValueError(f"Unknown direction: {_direction}")
        if group_cols:
            e = e.over(*group_cols)
        return e

    exprs = []
    for col in _data.collect_schema().names():
        if col in col_names:
            exprs.append(_fill_expr(col, _direction))
        else:
            exprs.append(pl.col(col))

    result = _data.with_columns(exprs)
    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# pivot_longer — wide to long
# ═══════════════════════════════════════════════════════════════════════════════


@pivot_longer.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _pivot_longer(
    _data: Tibble,
    cols,
    names_to="name",
    names_prefix: Optional[str] = None,
    names_sep: Optional[str] = None,
    names_pattern: Optional[str] = None,
    names_dtypes=None,
    names_transform=None,
    names_repair="check_unique",
    values_to: str = "value",
    values_drop_na: bool = False,
    values_dtypes=None,
    values_transform=None,
) -> Tibble:
    """Pivot data from wide to long format."""
    from ..collections import Collection, Inverted, Negated

    all_columns = _data.collect_schema().names()
    # Resolve cols to column names.
    # If cols is a Collection (or subclass like Inverted/Negated), re-expand
    # with the actual pool so negation selectors work correctly.
    if isinstance(cols, Collection):
        coll = cols if cols.pool is not None else Collection(cols, pool=list(all_columns))
        if not list.__len__(coll) or cols.pool is None:
            coll = Collection(cols, pool=list(all_columns))
        value_vars = list(coll)
    elif isinstance(cols, (list, tuple)):
        value_vars = list(cols)
    elif hasattr(cols, "_pipda_ref"):
        value_vars = [cols._pipda_ref]
    elif isinstance(cols, str):
        value_vars = [cols]
    else:
        # Try to resolve via f[...] or f[...:...]
        try:
            coll = Collection(cols, pool=list(all_columns))
            value_vars = list(coll)
        except Exception:
            value_vars = [cols]

    # Resolve actual column names from the original columns
    resolved_vars = []
    for v in value_vars:
        if isinstance(v, str) and v in all_columns:
            resolved_vars.append(v)
        elif isinstance(v, int) and 0 <= v < len(all_columns):
            resolved_vars.append(all_columns[v])

    if not resolved_vars:
        # Fall back to all columns
        id_vars = []
        resolved_vars = list(all_columns)
    else:
        id_vars = [c for c in all_columns if c not in resolved_vars]

    # Handle names_prefix
    prefix_removed_vars = resolved_vars
    if names_prefix:
        prefix_removed_vars = [
            re.sub(f"^{names_prefix}", "", v) for v in resolved_vars
        ]

    var_name = names_to if isinstance(names_to, str) else names_to[0]

    # Add row index for R-style ordering (interleaved, not grouped by var)
    _data = _data.with_columns(pl.int_range(0, pl.len()).alias("_row_idx"))

    result = _data.unpivot(
        index=id_vars + ["_row_idx"] if id_vars else ["_row_idx"],
        on=resolved_vars,
        variable_name=var_name,
        value_name=values_to,
    )

    # Sort by original row index to get R-style interleaved ordering
    # Cast variable column to ordered categorical so sort respects col order
    _var_order = pl.Enum(prefix_removed_vars)
    _var_col = pl.col(var_name)
    if names_prefix:
        # Strip prefix from the variable name column before casting
        _var_col = _var_col.str.strip_prefix(names_prefix)
    result = result.with_columns(
        _var_col.cast(_var_order)
    ).sort(by=["_row_idx", var_name]).drop("_row_idx")

    # Handle names_sep for multi-column names_to
    if names_sep and isinstance(names_to, list) and len(names_to) > 1:
        for i, name_col in enumerate(names_to):
            if name_col is None:
                continue
            result = result.with_columns(
                pl.col(var_name)
                .str.split(names_sep)
                .list.get(i)
                .alias(name_col)
            )

    if values_drop_na:
        result = result.filter(pl.col(values_to).is_not_null())

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# pivot_wider — long to wide
# ═══════════════════════════════════════════════════════════════════════════════


@pivot_wider.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _pivot_wider(
    _data: Tibble,
    id_cols=None,
    names_from=None,
    names_prefix: str = "",
    names_sep: str = "_",
    names_glue=None,
    names_sort: bool = False,
    names_repair: str = "check_unique",
    values_from=None,
    values_fill=None,
    values_fn=None,
    unused_fn=None,
) -> Tibble:
    """Pivot data from long to wide format."""
    all_columns = _data.collect_schema().names()

    # Resolve names_from and values_from
    if names_from is None:
        names_from = all_columns[0]
    if values_from is None:
        values_from = [c for c in all_columns if c != names_from][0]

    if isinstance(values_from, (list, tuple)):
        values_from = values_from[0]

    if isinstance(names_from, str):
        names_col = names_from
    else:
        names_col = names_from

    if isinstance(values_from, str):
        values_col = values_from
    else:
        values_col = values_from

    # Determine index columns
    if id_cols is None:
        index_cols = [c for c in all_columns if c not in (names_col, values_col)]
    else:
        index_cols = id_cols if isinstance(id_cols, list) else [id_cols]

    if values_fn is None:
        aggregate_function = pl.element().first()
    elif isinstance(values_fn, dict):
        aggregate_function = pl.element().map_elements(
            lambda x: values_fn[values_col](x)
        )
    else:
        fn_name = getattr(values_fn, "__name__", str(values_fn))
        aggregate_function = getattr(pl.element(), fn_name)()

    result = _data.pivot(
        index=index_cols if index_cols else None,
        on=names_col,
        values=values_col,
        aggregate_function=aggregate_function,
    )

    # Apply names_prefix
    if names_prefix:
        current_cols = result.collect_schema().names()
        new_cols = {}
        for c in current_cols:
            if c not in index_cols:
                new_cols[c] = names_prefix + str(c)
        if new_cols:
            result = result.rename(new_cols)

    if values_fill is not None:
        value_cols = [c for c in result.collect_schema().names() if c not in index_cols]
        for vc in value_cols:
            result = result.with_columns(pl.col(vc).fill_null(values_fill))

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# separate — split a column into multiple columns
# ═══════════════════════════════════════════════════════════════════════════════


def _separate_col(val, n_pieces: int, sep: str, extra: str, fill: str):
    """Split one string value into a fixed-length list, respecting
    extra/fill strategies.  Returns a list of length n_pieces."""
    import re
    import math

    if val is None or (isinstance(val, float) and math.isnan(val)):
        return [None] * n_pieces

    val = str(val)
    if extra == "merge":
        # Limit splits so the last piece absorbs the remainder.
        pieces = re.split(sep, val, maxsplit=n_pieces - 1)
    else:
        pieces = re.split(sep, val)
        # "drop" / "warn" — truncate excess pieces.
        if len(pieces) > n_pieces:
            pieces = pieces[:n_pieces]

    if len(pieces) < n_pieces:
        shortfall = n_pieces - len(pieces)
        if fill in ("right", "warn"):
            pieces = pieces + [None] * shortfall
        else:  # fill="left"
            pieces = [None] * shortfall + pieces

    return pieces


@separate.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _separate(
    _data: Tibble,
    col: Union[str, int],
    into,
    sep: str = r"[^a-zA-Z0-9]",
    remove: bool = True,
    convert=False,
    extra: str = "warn",
    fill: str = "right",
) -> Tibble:
    """Separate a column into multiple columns."""
    import math
    from functools import partial

    all_columns = _data.collect_schema().names()
    if isinstance(col, int):
        col = all_columns[col]

    if isinstance(into, str):
        into = [into]

    n_pieces = len(into)

    # Build a temporary list column via map_elements so we can apply
    # proper extra/fill semantics and full regex support.
    _tmp = "__sep_list__"
    split_fn = partial(
        _separate_col,
        n_pieces=n_pieces,
        sep=sep,
        extra=extra,
        fill=fill,
    )
    result = _data.with_columns(
        pl.col(col)
        .map_elements(split_fn, return_dtype=pl.List(pl.String))
        .alias(_tmp)
    )

    exprs = []
    for i, new_col in enumerate(into):
        if new_col is None:
            continue
        # NA in datar is float('nan') — treat as "ignore this piece"
        try:
            if isinstance(new_col, float) and math.isnan(new_col):
                continue
        except (TypeError, ValueError):
            pass
        e = pl.col(_tmp).list.get(i, null_on_oob=True)
        if convert:
            if isinstance(convert, type):
                e = e.cast(convert)
            elif isinstance(convert, dict) and new_col in convert:
                dtype = convert[new_col]
                if dtype is float:
                    dtype = pl.Float64
                elif dtype is int:
                    dtype = pl.Int64
                elif dtype is bool:
                    dtype = pl.Boolean
                e = e.cast(dtype)
        exprs.append(e.alias(new_col))

    result = result.with_columns(exprs).drop(_tmp)
    if remove:
        result = result.drop(col)

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# unite — combine multiple columns into one
# ═══════════════════════════════════════════════════════════════════════════════


@unite.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _unite(
    _data: Tibble,
    col: str,
    *cols: Union[str, int],
    sep: str = "_",
    remove: bool = True,
    na_rm: bool = False,
) -> Tibble:
    """Unite multiple columns into one."""
    all_columns = _data.collect_schema().names()
    selected_cols = cols
    if not selected_cols:
        selected_cols = tuple(all_columns)
    elif (
        len(selected_cols) == 1
        and hasattr(selected_cols[0], "__iter__")
        and not isinstance(selected_cols[0], (str, bytes))
    ):
        # Support unite("z", [0, 1]) and similar iterable column specs.
        selected_cols = tuple(selected_cols[0])

    col_names = []
    for c in selected_cols:
        if isinstance(c, int):
            col_names.append(all_columns[c])
        else:
            col_names.append(c)

    if na_rm:
        # Skip null pieces entirely and add separators only between
        # non-null values, matching tidyr::unite(na_rm=True).
        parts = [pl.col(cn).cast(pl.Utf8) for cn in col_names]
        result = _data.with_columns(
            pl.concat_str(parts, separator=sep, ignore_nulls=True)
            .fill_null("")
            .alias(col)
        )
    else:
        result = _data.with_columns(
            pl.concat_str([pl.col(cn) for cn in col_names], separator=sep).alias(col)
        )

    if remove:
        ordered_cols = [col] + [
            name for name in all_columns if name not in col_names and name != col
        ]
    else:
        ordered_cols = [col] + [name for name in all_columns if name != col]

    result = result.select(ordered_cols)

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# extract — extract capture groups into new columns
# ═══════════════════════════════════════════════════════════════════════════════


@extract.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _extract(
    data: Tibble,
    col: Union[str, int],
    into,
    regex: str = r"(\w+)",
    remove: bool = True,
    convert=False,
) -> Tibble:
    """Extract capture groups from a column into new columns."""
    all_columns = data.collect_schema().names()
    if isinstance(col, int):
        col = all_columns[col]

    if isinstance(into, str):
        into = [into]

    # Group capture-group indices by target column name.
    # Duplicate names → concat the captured groups into one column.
    name_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(into):
        if name is not None:
            name_to_indices.setdefault(name, []).append(i)

    exprs = []
    for new_col, indices in name_to_indices.items():
        if len(indices) == 1:
            e = pl.col(col).str.extract(regex, group_index=indices[0] + 1)
        else:
            parts = [
                pl.col(col).str.extract(regex, group_index=i + 1)
                for i in indices
            ]
            e = pl.concat_str(parts)
        if convert:
            e = e.cast(convert) if isinstance(convert, type) else e
        exprs.append(e.alias(new_col))

    if remove:
        result = data.with_columns(exprs).drop(col)
    else:
        result = data.with_columns(exprs)

    return reconstruct_tibble(result, data)


# ═══════════════════════════════════════════════════════════════════════════════
# expand — generate all combinations
# ═══════════════════════════════════════════════════════════════════════════════


@expand.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _expand(
    data: Tibble,
    *args,
    _name_repair: str = "check_unique",
    **kwargs,
) -> Tibble:
    """Generate all combinations of variables."""
    all_columns = data.collect_schema().names()
    df = data.collect()

    # Collect expansion "units": each unit is either a list of values (for
    # a single column) or a pl.DataFrame (for nested column groups from
    # nesting()). Units participate in cross-product with each other.
    units: list = []  # list of (list_of_keys, list_of_value_lists | pl.DataFrame)
    enum_dtypes: dict = {}
    for col_name in all_columns:
        dt = df.get_column(col_name).dtype
        if isinstance(dt, pl.Enum):
            enum_dtypes[col_name] = dt
    for arg in args:
        if isinstance(arg, _FullSeqRequest):
            if arg.column in all_columns:
                col_vals = (
                    df.get_column(arg.column).drop_nulls().to_list()
                )
                if col_vals:
                    full_vals = _full_seq_any(col_vals, arg.period, arg.tol)
                    units.append(([arg.column], [full_vals]))
                else:
                    units.append(([arg.column], [col_vals]))
        elif hasattr(arg, "_pipda_ref"):
            col_name = arg._pipda_ref
            if col_name in all_columns:
                col = df.get_column(col_name)
                if isinstance(col.dtype, (pl.Enum, pl.Categorical)):
                    units.append(([col_name], [col.cat.get_categories().to_list()]))
                else:
                    units.append(([col_name], [col.unique(maintain_order=True).to_list()]))
        elif hasattr(arg, "columns") and hasattr(arg, "collect"):
            # Tibble/DataFrame from nesting() — unique row combinations
            nesting_cols = arg.collect().columns
            valid_cols = [c for c in nesting_cols if c in all_columns]
            if valid_cols:
                nesting_df = df.select(valid_cols).unique(maintain_order=True)
                units.append((valid_cols, nesting_df))
        elif isinstance(arg, str) and arg in all_columns:
            col = df.get_column(arg)
            if isinstance(col.dtype, (pl.Enum, pl.Categorical)):
                units.append(([arg], [col.cat.get_categories().to_list()]))
            else:
                units.append(([arg], [col.unique(maintain_order=True).to_list()]))
        elif arg is not None:
            # Direct list of values
            vals = list(arg) if hasattr(arg, "__iter__") else [arg]
            units.append(([f"{DEFAULT_COLUMN_PREFIX}{len(units)}"], [vals]))

    for key, val in kwargs.items():
        vals = list(val) if hasattr(val, "__iter__") else [val]
        units.append(([key], [vals]))

    if not units:
        return as_tibble({})

    # Cross product of units. Each unit contributes either:
    #   - a list of columns with individual value lists (for single-column units)
    #   - a DataFrame whose rows are the combinations (for nesting units)
    result_cols = {}
    # Build row-oriented cross product: each "row" in the product space is a
    # tuple of unit-indices, where each unit contributes either a scalar index
    # into its value lists or a row index into its DataFrame.
    unit_ranges = []
    for _keys, values in units:
        if isinstance(values, pl.DataFrame):
            unit_ranges.append(range(values.height))
        else:
            unit_ranges.append(range(len(values[0])))

    for combination in itertools.product(*unit_ranges):
        for (keys, values), idx in zip(units, combination):
            if isinstance(values, pl.DataFrame):
                row = values.row(idx, named=True)
                for k in keys:
                    result_cols.setdefault(k, []).append(row[k])
            else:
                for k, vals in zip(keys, values):
                    result_cols.setdefault(k, []).append(vals[idx])

    result_df = pl.DataFrame(result_cols)
    for col, dtype in enum_dtypes.items():
        if col in result_df.columns:
            result_df = result_df.with_columns(
                result_df[col].cast(dtype)
            )
    return reconstruct_tibble(Tibble(result_df), data)


# ═══════════════════════════════════════════════════════════════════════════════
# expand_grid — create a data frame from all combinations
# ═══════════════════════════════════════════════════════════════════════════════


@expand_grid.register(object, backend="polars")
def _expand_grid(*args, _name_repair="check_unique", **kwargs) -> Tibble:
    """Create a Tibble from all combinations of factors."""
    all_items = {}
    for arg in args:
        if isinstance(arg, dict):
            all_items.update(arg)
        elif isinstance(arg, (list, tuple, pl.Series, range)):
            all_items[f"{DEFAULT_COLUMN_PREFIX}{len(all_items)}"] = list(arg)
        elif hasattr(arg, "to_list"):
            all_items[f"{DEFAULT_COLUMN_PREFIX}{len(all_items)}"] = arg.to_list()

    all_items.update(kwargs)

    if not all_items:
        return Tibble(pl.DataFrame())

    keys = list(all_items.keys())
    values = []
    for value in all_items.values():
        vals = (
            list(value)
            if hasattr(value, "__iter__") and not isinstance(value, str)
            else [value]
        )
        values.append(replace_na_with_none(vals))
    combinations = list(itertools.product(*values))

    result_df = pl.DataFrame(
        {k: [row[i] for row in combinations] for i, k in enumerate(keys)},
        strict=False,
    )

    # Apply name repair
    cols = result_df.columns
    new_cols = repair_names(list(cols), repair=_name_repair)
    if list(new_cols) != list(cols):
        result_df = result_df.rename(dict(zip(cols, new_cols)))

    return Tibble(result_df)


# ═══════════════════════════════════════════════════════════════════════════════
# nesting — find combinations present in data
# ═══════════════════════════════════════════════════════════════════════════════


@nesting.register(object, backend="polars")
def _nesting(
    *args,
    _name_repair: str = "check_unique",
    **kwargs,
) -> Tibble:
    """A helper that only finds combinations already present in the data."""
    # Collect column names from args/kwargs
    cols = {}
    for arg in args:
        if isinstance(arg, str):
            cols[arg] = pl.Series([], dtype=pl.Null)
        elif hasattr(arg, "_pipda_ref"):
            cols[arg._pipda_ref] = pl.Series([], dtype=pl.Null)
        elif isinstance(arg, dict):
            for k, v in arg.items():
                cols[k] = pl.Series(list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else [v])
        elif hasattr(arg, "__iter__") and not isinstance(arg, str):
            cols[f"{DEFAULT_COLUMN_PREFIX}{len(cols)}"] = pl.Series(list(arg))

    for key, val in kwargs.items():
        cols[key] = pl.Series(list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else [val])

    if not cols:
        return Tibble(pl.DataFrame())

    return Tibble(pl.DataFrame(cols))


# ═══════════════════════════════════════════════════════════════════════════════
# crossing — expand_grid with dedup and sort
# ═══════════════════════════════════════════════════════════════════════════════


@crossing.register(object, backend="polars")
def _crossing(
    *args,
    _name_repair: str = "check_unique",
    **kwargs,
) -> Tibble:
    """A wrapper around expand_grid that de-duplicates and sorts inputs."""
    all_items = {}
    for arg in args:
        if isinstance(arg, dict):
            for k, v in arg.items():
                all_items[k] = sorted(set(v))
        elif isinstance(arg, (list, tuple, pl.Series, range)):
            items = list(arg)
            all_items[f"{DEFAULT_COLUMN_PREFIX}{len(all_items)}"] = sorted(set(items))
        elif hasattr(arg, "to_list"):
            items = arg.to_list()
            all_items[f"{DEFAULT_COLUMN_PREFIX}{len(all_items)}"] = sorted(set(items))

    for key, val in kwargs.items():
        items = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else [val]
        all_items[key] = sorted(set(items))

    return _expand_grid(**all_items, _name_repair=_name_repair)


# ═══════════════════════════════════════════════════════════════════════════════
# complete — complete missing combinations
# ═══════════════════════════════════════════════════════════════════════════════


@complete.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _complete(
    data: Tibble,
    *args,
    fill=None,
    explicit: bool = True,
) -> Tibble:
    """Complete a data frame with missing combinations."""
    all_columns = data.collect_schema().names()
    df = data.collect()

    # Each dim is (column_names, row_value_tuples).
    # Nested columns (from nesting()) move together as a unit.
    dims: list[tuple[list[str], list[tuple]]] = []

    for arg in args:
        if isinstance(arg, Tibble):
            # nesting() result — find unique combinations in data
            nest_cols = arg.collect_schema().names()
            valid = [c for c in nest_cols if c in all_columns]
            if valid:
                unique_rows = df.select(valid).unique(maintain_order=True)
                rows = [
                    tuple(unique_rows.get_column(c).to_list()[i]
                          for c in valid)
                    for i in range(unique_rows.height)
                ]
                dims.append((valid, rows))
        elif hasattr(arg, "_pipda_ref"):
            col_name = arg._pipda_ref
            if col_name in all_columns:
                vals = df.get_column(col_name).unique(maintain_order=True).to_list()
                dims.append(([col_name], [(v,) for v in vals]))
        elif isinstance(arg, str) and arg in all_columns:
            vals = df.get_column(arg).unique(maintain_order=True).to_list()
            dims.append(([arg], [(v,) for v in vals]))
        elif arg is not None:
            vals = list(arg) if hasattr(arg, "__iter__") and not isinstance(arg, str) else [arg]
            key = f"{DEFAULT_COLUMN_PREFIX}{len(dims)}"
            dims.append(([key], [(v,) for v in vals]))

    if not dims:
        return reconstruct_tibble(data)

    # Build grid from product of dimensions
    grid_cols: dict[str, list] = {}
    for combo in itertools.product(*[rows for _, rows in dims]):
        for (col_names, _), row_values in zip(dims, combo):
            for cn, rv in zip(col_names, row_values):
                grid_cols.setdefault(cn, []).append(rv)

    keys = list(grid_cols.keys())
    grid_df = pl.DataFrame(grid_cols)

    # Join with original data
    result_df = grid_df.join(df, on=keys, how="left")

    if fill is not None:
        for col, val in fill.items():
            if col in result_df.columns:
                result_df = result_df.with_columns(pl.col(col).fill_null(val))

    return reconstruct_tibble(Tibble(result_df), data)


# ═══════════════════════════════════════════════════════════════════════════════
# nest — create list-column of data frames
# ═══════════════════════════════════════════════════════════════════════════════


@nest.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _nest(
    _data: Tibble,
    _names_sep: Optional[str] = None,
    **cols: Union[str, int],
) -> Tibble:
    """Nest columns into list-columns."""
    all_columns = _data.collect_schema().names()
    if not cols:
        return reconstruct_tibble(_data)

    # Resolve each key to its column names
    nested_cols = set()
    key_to_cols = {}
    for key, val in cols.items():
        if isinstance(val, Collection):
            val.expand(pool=all_columns)
            resolved = [all_columns[c] if isinstance(c, int) else c for c in val]
        elif isinstance(val, (list, tuple)):
            resolved = [all_columns[c] if isinstance(c, int) else c for c in val]
        else:
            resolved = [all_columns[val] if isinstance(val, int) else val]
        key_to_cols[key] = resolved
        nested_cols.update(resolved)

    other_cols = [c for c in all_columns if c not in nested_cols]

    agg_exprs = [
        pl.struct(col_names).alias(key) for key, col_names in key_to_cols.items()
    ]

    if not other_cols:
        result = _data.select(*agg_exprs)
    else:
        result = _data.group_by(other_cols, maintain_order=True).agg(*agg_exprs)

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# unnest — expand list-columns
# ═══════════════════════════════════════════════════════════════════════════════


@unnest.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _unnest(
    data: Tibble,
    *cols: Union[str, int],
    keep_empty: bool = False,
    dtypes=None,
    names_sep: Optional[str] = None,
    names_repair: str = "check_unique",
) -> Tibble:
    """Unnest list-columns into regular columns."""
    from ..collections import Collection

    all_columns = data.collect_schema().names()
    if not cols:
        cols = tuple(all_columns)

    # Resolve each arg — a Collection from c(f.a, f.b) needs to be
    # expanded against the pool of real column names
    col_names: list[str] = []
    for c in cols:
        if isinstance(c, Collection):
            coll = (
                c if c.pool is not None
                else Collection(c, pool=list(all_columns))
            )
            for item in coll:
                if isinstance(item, int):
                    col_names.append(all_columns[item])
                elif isinstance(item, str):
                    col_names.append(item)
        elif isinstance(c, int):
            col_names.append(all_columns[c])
        else:
            col_names.append(c)

    schema = data.collect_schema()

    # Object dtype columns contain Python lists or DataFrames
    # that need manual expansion
    if any(schema[c] == pl.Object for c in col_names):
        materialized = data.collect()
        n_rows = materialized.height
        col_values = {
            c: materialized[c].to_list() for c in all_columns
        }

        # Check for nested DataFrames among unchop columns
        has_df = False
        for c in col_names:
            for v in col_values[c]:
                if isinstance(v, pl.DataFrame):
                    has_df = True
                    break
            if has_df:
                break

        if has_df:
            # ---- DataFrame path: expand into struct columns ----
            # Compute union of inner column names per unchop column
            inner_cols_map: dict = {}
            for oc in col_names:
                union_inner: list = []
                for v in col_values[oc]:
                    if isinstance(v, pl.DataFrame):
                        for ic in v.collect_schema().names():
                            if ic not in union_inner:
                                union_inner.append(ic)
                inner_cols_map[oc] = union_inner

            # Non-unchop columns stay as-is (repeated by sizes)
            key_cols = [c for c in all_columns if c not in col_names]

            # Gather expanded values row by row
            expanded_key = {c: [] for c in key_cols}
            expanded_inner: dict = {}
            for oc in col_names:
                for ic in inner_cols_map[oc]:
                    expanded_inner[(oc, ic)] = []

            for i in range(n_rows):
                sizes: list = []
                for oc in col_names:
                    val = col_values[oc][i]
                    if isinstance(val, pl.DataFrame):
                        sizes.append(val.height)
                    elif val is None:
                        sizes.append(1 if keep_empty else 0)
                    else:
                        sizes.append(1)

                expand_len = max(sizes) if sizes else 1

                if expand_len == 0:
                    continue

                # Repeat key columns
                for kc in key_cols:
                    expanded_key[kc].extend(
                        [col_values[kc][i]] * expand_len
                    )

                # Expand inner fields
                for oc in col_names:
                    val = col_values[oc][i]
                    if isinstance(val, pl.DataFrame):
                        for ic in inner_cols_map[oc]:
                            if ic in val.collect_schema().names():
                                expanded_inner[(oc, ic)].extend(
                                    val[ic].to_list()
                                )
                            else:
                                expanded_inner[(oc, ic)].extend(
                                    [None] * val.height
                                )
                    else:
                        for ic in inner_cols_map[oc]:
                            expanded_inner[(oc, ic)].append(None)

            # Build result columns
            result_series = []
            for kc in key_cols:
                vals = expanded_key[kc]
                col_dtype = _resolve_dtype(kc, dtypes, col_names)
                if col_dtype is not None:
                    result_series.append(
                        pl.Series(kc, vals, dtype=col_dtype)
                    )
                else:
                    result_series.append(pl.Series(kc, vals))

            # Build struct columns for each unchop DF column
            for oc in col_names:
                inner_cols = inner_cols_map[oc]
                if not inner_cols:
                    continue
                struct_df_data = {}
                for ic in inner_cols:
                    vals = expanded_inner[(oc, ic)]
                    struct_df_data[ic] = vals
                struct_s = pl.DataFrame(struct_df_data).to_struct(oc)
                result_series.append(struct_s)

            result = pl.DataFrame(result_series)
            return reconstruct_tibble(result, data)

        # ---- Plain list path ----
        expanded = {c: [] for c in all_columns}
        for i in range(n_rows):
            lengths = []
            for c in col_names:
                val = col_values[c][i]
                if isinstance(val, list):
                    lengths.append(len(val))
                else:
                    lengths.append(1)

            # Use max length so mixed scalar/list columns can be unnested
            # together by repeating scalar values across expanded rows.
            expand_len = max(lengths) if lengths else 0

            if expand_len == 0:
                if keep_empty:
                    for c in all_columns:
                        if c in col_names:
                            expanded[c].append(None)
                        else:
                            expanded[c].append(col_values[c][i])
                continue

            for c in all_columns:
                if c in col_names:
                    val = col_values[c][i]
                    if isinstance(val, list):
                        if len(val) == expand_len:
                            expanded[c].extend(val)
                        elif len(val) == 1 and expand_len > 1:
                            expanded[c].extend(val * expand_len)
                        elif len(val) == 0:
                            expanded[c].extend([None] * expand_len)
                        else:
                            # Fallback for ragged lengths: pad with None.
                            expanded[c].extend(val + [None] * (expand_len - len(val)))
                    else:
                        expanded[c].extend([val] * expand_len)
                else:
                    expanded[c].extend(
                        [col_values[c][i]] * expand_len
                    )

        result_series = []
        for c in all_columns:
            vals = expanded[c]
            col_dtype = _resolve_dtype(c, dtypes, col_names)
            if col_dtype is not None:
                result_series.append(
                    pl.Series(c, vals, dtype=col_dtype)
                )
            elif vals and any(v is not None for v in vals):
                types_set = {type(v) for v in vals if v is not None}
                if len(types_set) > 1:
                    result_series.append(
                        pl.Series(c, vals, dtype=pl.Object)
                    )
                else:
                    result_series.append(pl.Series(c, vals))
            else:
                result_series.append(pl.Series(c, vals))
        result = pl.DataFrame(result_series)
        return reconstruct_tibble(result, data)

    result = data.explode(col_names)

    # Apply dtype casts for native List columns
    if dtypes is not None:
        for c in col_names:
            col_dtype = _resolve_dtype(c, dtypes, col_names)
            if col_dtype is not None:
                result = result.with_columns(
                    pl.col(c).cast(col_dtype)
                )

    if not keep_empty:
        result = result.drop_nulls(subset=col_names)

    return reconstruct_tibble(result, data)


def _resolve_dtype(
    col_name: str,
    dtypes,
    unchop_cols: list,
):
    """Resolve the dtype for a column from the dtypes specification.

    Args:
        col_name: Name of the column
        dtypes: Single type or dict mapping column names to types
        unchop_cols: List of column names being unchopped

    Returns:
        The resolved dtype, or None if no type is specified
    """
    if dtypes is None:
        return None
    if not isinstance(dtypes, dict):
        # Single type applies to all unchop columns
        return dtypes if col_name in unchop_cols else None
    # Dict: try exact match, then prefix match for nested columns
    if col_name in dtypes:
        return dtypes[col_name]
    for key, val in dtypes.items():
        if col_name.startswith(f"{key}$"):
            return val
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# chop — convert rows to list-columns within groups
# ═══════════════════════════════════════════════════════════════════════════════


@chop.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _chop(
    data: Tibble,
    cols=None,
) -> Tibble:
    """Chop rows into list-columns within groups (inverse of unchop).

    Groups by the columns NOT being chopped, then aggregates the
    chopped columns into list-columns.
    """
    all_columns = data.collect_schema().names()
    if cols is None:
        return reconstruct_tibble(data)

    # Resolve column names from the selection
    col_indices = vars_select(all_columns, cols)
    chop_cols = [all_columns[i] for i in col_indices]

    if not chop_cols:
        return reconstruct_tibble(data)

    # Key columns are those NOT being chopped (implicit groups)
    key_cols = [c for c in all_columns if c not in chop_cols]

    if key_cols:
        result = data.group_by(key_cols, maintain_order=True).agg(
            [pl.col(c) for c in chop_cols]
        )
    else:
        # No key columns — add a dummy group to collapse to single row
        result = (
            data.with_columns(pl.lit(1).alias("_dummy"))
            .group_by("_dummy", maintain_order=True)
            .agg([pl.col(c) for c in chop_cols])
            .drop("_dummy")
        )

    return reconstruct_tibble(result, data)


# ═══════════════════════════════════════════════════════════════════════════════
# unchop — expand list-columns
# ═══════════════════════════════════════════════════════════════════════════════


@unchop.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _unchop(
    data: Tibble,
    cols=None,
    keep_empty: bool = False,
    dtypes=None,
) -> Tibble:
    """Unchop list-columns, similar to unnest."""
    return _unnest(data, *cols if cols else (), keep_empty=keep_empty,
                   dtypes=dtypes)


# ═══════════════════════════════════════════════════════════════════════════════
# pack — collapse columns into a single df-column
# ═══════════════════════════════════════════════════════════════════════════════


@pack.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _pack(
    _data: Tibble,
    _names_sep: Optional[str] = None,
    **cols: Union[str, int],
) -> Tibble:
    """Pack columns into a struct column."""
    all_columns = _data.collect_schema().names()
    if not cols:
        return reconstruct_tibble(_data)

    for key, val in cols.items():
        if isinstance(val, (list, tuple)):
            col_names = [v if isinstance(v, str) else all_columns[v] for v in val]
        else:
            col_names = [val if isinstance(val, str) else all_columns[val]]

        # Create struct column
        _data = _data.with_columns(pl.struct(col_names).alias(key))
        # Drop the original columns
        _data = _data.drop([c for c in col_names if c != key])

    return reconstruct_tibble(_data, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# unpack — expand struct columns
# ═══════════════════════════════════════════════════════════════════════════════


@unpack.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _unpack(
    data: Tibble,
    cols,
    names_sep: Optional[str] = None,
    names_repair: str = "check_unique",
) -> Tibble:
    """Unpack struct columns into individual columns."""
    all_columns = data.collect_schema().names()
    if isinstance(cols, (str, int)):
        cols = [cols]

    col_names = []
    for c in cols:
        if isinstance(c, int):
            col_names.append(all_columns[c])
        else:
            col_names.append(c)

    result = data
    for cn in col_names:
        if cn in result.collect_schema().names():
            # Unnest struct fields
            result = result.unnest(cn)

    # Apply names_sep if needed
    if names_sep:
        new_cols = {}
        for c in result.collect_schema().names():
            if c not in all_columns:
                parts = c.split("_")
                if len(parts) > 1:
                    new_c = names_sep.join(parts)
                else:
                    new_c = c
                new_cols[c] = new_c
        if new_cols:
            result = result.rename(new_cols)

    return reconstruct_tibble(result, data)


# ═══════════════════════════════════════════════════════════════════════════════
# full_seq — full sequence of values
# ═══════════════════════════════════════════════════════════════════════════════


class _FullSeqRequest:
    """Sentinel: signals _expand to compute full_seq from data for a column."""
    __slots__ = ("column", "period", "tol")

    def __init__(self, column: str, period, tol: float = 1e-6) -> None:
        self.column = column
        self.period = period
        self.tol = tol


@full_seq.register(pl.Series, backend="polars")
def _full_seq_series(x, period, tol=1e-6):
    """Full sequence from a Series."""
    vals = x.drop_nulls().to_list()
    if not vals:
        return []
    return _full_seq_any(vals, period, tol)


@full_seq.register(object, backend="polars")
def _full_seq_any(x, period, tol=1e-6):
    """Create the full sequence of values in a vector."""
    if isinstance(x, str):
        # Column name from Context.SELECT — defer to expand/data
        return _FullSeqRequest(x, period, tol)

    x = to_series(x)
    x = x.sort().drop_nulls()
    if len(x) == 0:
        return []

    minx = x[0]
    maxx = x[-1]
    if (((x - minx) % period > tol) & (period - ((x - minx) % period) > tol)).any():
        raise ValueError("`x` is not a regular sequence.")

    if period - ((maxx - minx) % period) <= tol:
        maxx += tol

    return _seq_obj(minx, maxx, period)


# ═══════════════════════════════════════════════════════════════════════════════
# separate_rows — separate rows by splitting a column
# ═══════════════════════════════════════════════════════════════════════════════


@separate_rows.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _separate_rows(
    _data: Tibble,
    *cols: Union[str, int],
    sep: str = r"[^a-zA-Z0-9]",
    convert=False,
) -> Tibble:
    """Separate a column into multiple rows."""
    import re as _re

    all_columns = _data.collect_schema().names()
    if not cols:
        return reconstruct_tibble(_data)

    col_names = []
    for c in cols:
        if isinstance(c, int):
            col_names.append(all_columns[c])
        else:
            col_names.append(c)

    # Split each column into a list using regex, then explode all at once
    # so rows stay aligned.
    split_exprs = [
        pl.col(cn).map_elements(
            lambda v, s=sep: _re.split(s, str(v)) if v is not None else [None],
            return_dtype=pl.List(pl.String),
        ).alias(cn)
        for cn in col_names
    ]
    result = _data.with_columns(split_exprs).explode(col_names)

    # Apply convert
    if convert:
        cast_exprs = []
        if isinstance(convert, dict):
            for cn, dtype in convert.items():
                if dtype is float:
                    dtype = pl.Float64
                elif dtype is int:
                    dtype = pl.Int64
                elif dtype is bool:
                    dtype = pl.Boolean
                cast_exprs.append(pl.col(cn).cast(dtype))
        elif isinstance(convert, type):
            for cn in col_names:
                cast_exprs.append(pl.col(cn).cast(convert))
        if cast_exprs:
            result = result.with_columns(cast_exprs)

    return reconstruct_tibble(result, _data)


# ═══════════════════════════════════════════════════════════════════════════════
# uncount — expand rows by count
# ═══════════════════════════════════════════════════════════════════════════════


@uncount.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _uncount(
    _data: Tibble,
    weights,
    _remove: bool = True,
    _id: Optional[str] = None,
) -> Tibble:
    """Expand rows by count (weights)."""
    all_columns = _data.collect_schema().names()

    # Resolve weights: Expr → Series, str → column name
    if isinstance(weights, pl.Expr):
        weight_series = _data.collect().select(weights.alias("__w__")).get_column("__w__")
        weight_col = None  # expression-based, no column to remove
    elif isinstance(weights, str):
        weight_col = weights
        weight_series = None
    elif hasattr(weights, "_pipda_ref"):
        weight_col = weights._pipda_ref
        weight_series = None
    else:
        weight_col = None
        weight_series = weights  # scalar

    # Validate weight column exists
    if isinstance(weight_col, str) and weight_col not in all_columns:
        raise ValueError(
            f"`weights` must evaluate to numerics, "
            f"column `{weight_col}` not found."
        )

    df = _data.collect()
    columns = list(df.columns)

    # Repeat each row by its weight
    rows = []
    for i in range(df.shape[0]):
        row_dict = df.row(i, named=True)
        if weight_series is not None:
            if isinstance(weight_series, pl.Series):
                weight = weight_series[i]
            else:
                weight = weight_series
        elif isinstance(weight_col, str):
            weight = row_dict.get(weight_col)
        else:
            weight = weight_col

        if weight is None:
            continue

        if not isinstance(weight, (int, float)):
            raise ValueError("`weights` must evaluate to numerics.")
        if isinstance(weight, float) and weight != int(weight):
            raise ValueError("`weights` must evaluate to integer.")
        if weight < 0:
            raise ValueError("`weights` must be >= 0.")

        weight_int = int(weight)
        if weight_int == 0:
            continue

        for j in range(weight_int):
            row_copy = dict(row_dict)
            if _id:
                row_copy[_id] = i
            rows.append(row_copy)

    if _remove and isinstance(weight_col, str) and weight_col in columns:
        rows = [
            {k: v for k, v in row.items() if k != weight_col}
            for row in rows
        ]
        columns = [c for c in columns if c != weight_col]

    if _id and _id not in columns:
        columns = columns + [_id]

    if rows:
        result_df = pl.DataFrame(rows, schema=columns)
    else:
        result_df = pl.DataFrame({c: [] for c in columns})

    return reconstruct_tibble(Tibble(result_df), _data)


# ═══════════════════════════════════════════════════════════════════════════════
# expand_grid already handled above
# ═══════════════════════════════════════════════════════════════════════════════
