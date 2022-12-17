
from polars import Expr, Series

from datar.apis.base import (
    # ceiling,
    # cov,
    # floor,
    mean,
    # median,
    # pmax,
    # pmin,
    # sqrt,
    # var,
    # scale,
    # min_,
    # max_,
    # round_,
    sum_,
    # abs_,
    # prod,
    # sign,
    # signif,
    # trunc,
    # exp,
    # log,
    # log2,
    # log10,
    # log1p,
    # sd,
    # weighted_mean,
    # quantile,
    # proportions,
    # col_sums,
    # row_sums,
    # col_sds,
    # row_sds,
    # col_means,
    # row_means,
    # col_medians,
    # row_medians,
)
import datar_numpy.api.arithm  # noqa: F401


@mean.register((Series, Expr), backend="polars")
def _mean(x, na_rm=False):
    return x.mean()


@sum_.register((Series, Expr), backend="polars")
def _sum(x, na_rm=False):
    return x.sum()
