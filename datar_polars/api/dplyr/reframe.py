"""Reframe a data frame for the polars backend.

Reframe is like summarise but can return any number of rows per group.
"""

from __future__ import annotations

from typing import Any

from datar.apis.dplyr import reframe

from ...contexts import Context
from ...tibble import Tibble, LazyTibble
from .summarise import _reframe as _summarise_reframe


@reframe.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _reframe(
    _data: Tibble,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    return _summarise_reframe(_data, *args, **kwargs)
