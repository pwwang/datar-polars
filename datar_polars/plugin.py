"""Polars backend plugin for datar

Implements all simplug hooks to register polars as a datar backend.
"""

from typing import TYPE_CHECKING, Mapping

from datar.core.plugin import plugin

# Attach version to the plugin for simplug version tracking
from .version import __version__  # noqa: F401
from .polars import read_csv

if TYPE_CHECKING:
    from .polars import DataFrame


@plugin.impl
def setup():
    """Initialize the polars backend"""
    import polars as pl
    import datar.all as datar_all
    import datar.dplyr as datar_dplyr
    from datar.apis import dplyr as datar_dplyr_apis
    from datar.core.options import add_option

    add_option("dplyr_summarise_inform", True)

    # Patch pl.Series.__getitem__ to accept boolean masks (like pandas does).
    # datar code commonly uses x[bool_mask] which fails on native polars.
    _original_series_getitem = pl.Series.__getitem__

    def _patched_getitem(self, key):
        if isinstance(key, pl.Series) and key.dtype == pl.Boolean:
            return self.filter(key)
        if isinstance(key, (list, tuple)) and len(key) > 0 and isinstance(key[0], bool):
            return self.filter(key)
        return _original_series_getitem(self, key)

    pl.Series.__getitem__ = _patched_getitem

    if not hasattr(datar_dplyr, "slice"):
        datar_dplyr.slice = datar_dplyr.slice_
    if not hasattr(datar_dplyr_apis, "slice"):
        datar_dplyr_apis.slice = datar_dplyr_apis.slice_
    if not hasattr(datar_all, "slice"):
        datar_all.slice = datar_dplyr.slice_
        if hasattr(datar_all, "__all__") and "slice" not in datar_all.__all__:
            datar_all.__all__.append("slice")


@plugin.impl
def get_versions():
    """Return the versions of polars and datar-polars"""
    import polars

    return {
        "datar-polars": __version__,
        "polars": polars.__version__,
    }


@plugin.impl
def load_dataset(name: str, metadata: Mapping) -> "DataFrame":
    """Load a dataset as a LazyTibble via read_csv"""
    if name not in metadata:
        raise AttributeError(
            f"No such dataset: {name}. "
            "Use datar.data.descr_datasets() to see all available datasets."
        )

    meta = metadata[name]
    df = read_csv(
        meta.source,
        null_values=getattr(meta, "null_values", ["NA", "nan", "NaN", "null"]),
    )
    from .tibble import Tibble

    return Tibble(df, _datar={"backend": "polars"})


@plugin.impl
def dplyr_api():
    """Register dplyr API implementations"""
    from .api.dplyr import (  # noqa: F401
        across,
        arrange,
        bind,
        context,
        count,
        desc,
        distinct,
        filter_,
        funs,
        glimpse,
        group_by,
        group_data,
        group_iter,
        if_else,
        join,
        mutate,
        order_by,
        pick,
        pull,
        rank,
        recode,
        reframe,
        relocate,
        rename,
        rows,
        select,
        sets,
        slice_,
        summarise,
        tidyselect,
    )


@plugin.impl
def base_api():
    """Register base API implementations"""
    from .api.base import (  # noqa: F401
        verbs,
        asis,
        string,
        cum,
        trig,
        rank,
        arithm,
        stats,
        table,
        sets,
        which,
        seq,
        types,
        bessel,
        complex,
        random,
        special,
        factor,
    )


@plugin.impl
def tibble_api():
    """Register tibble API implementations"""
    from .api import tibble  # noqa: F401


@plugin.impl
def tidyr_api():
    """Register tidyr API implementations"""
    from .api import tidyr  # noqa: F401


@plugin.impl
def forcats_api():
    """Register forcats API implementations"""
    from .api import forcats  # noqa: F401


@plugin.impl
def misc_api():
    """Register misc API implementations"""
    from .api import misc  # noqa: F401


@plugin.impl
def c_getitem(item):
    """Get item for c() collection"""
    from .collections import Collection

    return Collection(item)


@plugin.impl
def operate(op, x, y=None):
    """Operate on x and y"""
    from .operators import operate as operate_

    return operate_(op, x, y)
