"""Pick columns by name

See source https://github.com/tidyverse/dplyr/blob/master/R/across.R
"""

from __future__ import annotations

from typing import Any

from datar.apis.base import c
from datar.apis.dplyr import pick, group_vars

from ...contexts import Context
from ...tibble import Tibble, LazyTibble


@pick.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _pick(_data: Tibble, *args: Any, **kwargs: Any) -> Tibble:
    """Pick columns by name within dplyr verbs.

    Args:
        _data: The data frame.
        *args: Columns to pick.

    Returns:
        The selected columns as a Tibble.
    """
    if not args:
        raise ValueError("must pick at least one column")

    cols = c(*args)
    out = _data.select(cols)
    return out
