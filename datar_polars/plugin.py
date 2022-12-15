from typing import TYPE_CHECKING, Mapping

from simplug import Simplug

from polars import read_csv, DataFrame

plugin = Simplug("datar")


# @plugin.impl
# def setup():
#   from datar.core.options import add_option
#   pdtypes.patch()
#   add_option("use_modin", False)
#   add_option("dplyr_summarise_inform", True)


@plugin.impl
def load_dataset(name: str, metadata: Mapping) -> DataFrame:
    if name not in metadata:
        raise AttributeError(
            f"No such dataset: {name}. "
            "Use datar.data.descr_datasets() to see all available datasets."
        )

    meta = metadata[name]
    return read_csv(meta.source)


# @plugin.impl
# def base_api():
#     from .api.base import (
#         arithm,
#         asis,
#         bessel,
#         complex,
#         cum,
#         factor,
#         funs,
#         null,
#         random,
#         seq,
#         special,
#         string,
#         table,
#         trig,
#         verbs,
#         which,
#     )
#     from .api.base.asis import as_pd_date

#     return {"as_pd_date": as_pd_date}


@plugin.impl
def dplyr_api():
    from .api.dplyr import (
        # across,
        # arrange,
        # bind,
        # context,
        # count_tally,
        # desc,
        # distinct,
        # filter_,
        # funs,
        # glimpse,
        group_by,
        group_data,
        # group_iter,
        # if_else,
        # join,
        # lead_lag,
        mutate,
        # order_by,
        # pull,
        # rank,
        # recode,
        relocate,
        # rename,
        # rows,
        select,
        # sets,
        # slice_,
        # summarise,
        # tidyselect,
    )


@plugin.impl
def tibble_api():
    # from .api.tibble import tibble, verbs
    from .api.tibble import tibble


# @plugin.impl
# def tidyr_api():
#     from .api.tidyr import (
#         chop,
#         complete,
#         drop_na,
#         expand,
#         extract,
#         fill,
#         funs,
#         nest,
#         pack,
#         pivot_long,
#         pivot_wide,
#         replace_na,
#         separate,
#         uncount,
#         unite,
#     )


# @plugin.impl
# def forcats_api():
#     from .api.forcats import (
#         fct_multi,
#         lvl_addrm,
#         lvl_order,
#         lvl_value,
#         lvls,
#         misc,
#     )


# @plugin.impl
# def other_api():
#     from .api.other import (
#         itemgetter,
#         attrgetter,
#         pd_cat,
#         pd_dt,
#         pd_str,
#         flatten,
#     )
#     return {
#         "itemgetter": itemgetter,
#         "attrgetter": attrgetter,
#         "pd_cat": pd_cat,
#         "pd_dt": pd_dt,
#         "pd_str": pd_str,
#         "flatten": flatten,
#     }


@plugin.impl
def get_versions():
    import polars
    from .version import __version__

    out = {
        "datar-polars": __version__,
        "polars": polars.__version__,
    }

    return out


# @plugin.impl
# def c_getitem(item):
#     from .collections import Collection
#     return Collection(item)


# @plugin.impl
# def operate(op, x, y=None):
#     from .operators import operate as operate_
#     return operate_(op, x, y)
