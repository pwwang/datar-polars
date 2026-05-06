"""Extract a single column

https://github.com/tidyverse/dplyr/blob/master/R/pull.R
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl
from datar.core.utils import arg_match
from datar.apis.dplyr import pull

from ...polars import DataFrame
from ...contexts import Context
from ...tibble import Tibble
from ...common import is_scalar


@pull.register(
    Tibble,
    context=Context.SELECT,
    kw_context={"name": Context.EVAL},
    backend="polars",
)
def _pull(
    _data: Tibble,
    var: Any = -1,
    *,
    name: Optional[Any] = None,
    to: Optional[str] = None,
) -> Any:
    """Pull a single column from the data frame.

    Args:
        _data: The Tibble to extract from.
        var: Column name or position (-1 = last column).
        name: Optional names for dict/series output.
        to: Output format — 'list', 'array', 'frame', 'series', 'dict', or None.
    """
    to = arg_match(to, "to", ["list", "array", "frame", "series", "dict", None])

    columns = _data.collect_schema().names()

    if isinstance(var, int):
        var = columns[var]
        var = var.split("$", 1)[0]

    # Collect var column — handle pl.Expr from f.col references
    if isinstance(var, pl.Expr):
        result_df = _data.select(var).collect()
        col_series = result_df.get_column(result_df.columns[0])
    else:
        col_series = _data.select(var).collect().get_column(var)

    # Collect name column if it's an Expr (e.g., f.name)
    if isinstance(name, pl.Expr):
        name_df = _data.select(name).collect()
        name = name_df.get_column(name_df.columns[0]).to_list()
    elif name is not None and is_scalar(name):
        name = [name]

    if to is None:
        if name is not None and len(name) == len(col_series):
            to = "dict"
        else:
            to = "series"

    if to == "dict":
        if name is None or len(name) != len(col_series):
            raise ValueError(
                "No `name` provided or length mismatches with the values."
            )
        return dict(zip(name, col_series.to_list()))

    if to == "list":
        return col_series.to_list()

    if to == "array":
        return col_series.to_numpy()

    if to == "frame":
        df = col_series.to_frame()
        if name and len(name) != df.shape[1]:
            raise ValueError(
                f"Expect {df.shape[1]} names but got {len(name)}."
            )
        if name:
            df = df.rename({df.columns[0]: name[0]})
        return df

    # to == 'series'
    if name:
        col_series = col_series.alias(name[0])
    return col_series
