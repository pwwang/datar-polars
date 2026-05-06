"""Glimpse a data frame.

https://github.com/tidyverse/dplyr/blob/master/R/glimpse.R
"""

from __future__ import annotations

import html as _html
import textwrap
from shutil import get_terminal_size
from typing import Any, Callable, Optional

from datar.apis.dplyr import glimpse

from ...contexts import Context
from ...tibble import Tibble, LazyTibble


class Glimpse:
    """Glimpse object with string and HTML representations."""

    def __init__(
        self,
        x: Tibble,
        width: Optional[int] = None,
        formatter: Optional[Callable] = None,
    ) -> None:
        self._x = x
        self._width = width or get_terminal_size((100, 20)).columns
        self._formatter = formatter or (lambda v: str(v))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        df = self._x.collect()
        rows, cols = df.shape
        lines = [f"Rows: {rows}", f"Columns: {cols}"]
        for col_name in df.columns:
            series = df[col_name]
            dtype = str(series.dtype)
            # format first few values
            first_vals = series.head(10).to_list()
            formatted = ", ".join(self._formatter(v) for v in first_vals)
            formatted = textwrap.shorten(
                formatted,
                width=self._width - 4 - len(col_name) - len(dtype),
                placeholder="...",
            )
            lines.append(f". {col_name} <{dtype}> {formatted}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        df = self._x.collect()
        rows, cols = df.shape
        out = [f"<div><i>Rows: {rows}, Columns: {cols}</i></div>", "<table>"]
        for col_name in df.columns:
            series = df[col_name]
            dtype = str(series.dtype)
            first_vals = series.head(10).to_list()
            formatted = ", ".join(self._formatter(v) for v in first_vals)
            formatted = _html.escape(textwrap.shorten(
                formatted,
                width=self._width - 4 - len(col_name) - len(dtype),
                placeholder="...",
            ))
            out.append(
                f'<tr><th style="text-align: left">. <b>{col_name}</b></th>'
                f'<td style="text-align: left"><i>&lt;{dtype}&gt;</i></td>'
                f'<td style="text-align: left">{formatted}</td></tr>'
            )
        out.append("</table>")
        return "\n".join(out)


@glimpse.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _glimpse(
    x: Tibble,
    width: Optional[int] = None,
    formatter: Optional[Callable] = None,
) -> Glimpse:
    """Return a Glimpse object for a Tibble."""
    return Glimpse(x, width=width, formatter=formatter)
