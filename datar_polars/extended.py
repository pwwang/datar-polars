from __future__ import annotations
from abc import ABC, abstractproperty
from typing import Any, List, Mapping

import numpy as np
import polars as pl
from polars.internals.dataframe.groupby import GroupBy

from .tibble import Tibble, TibbleGrouped, TibbleRowwise


class Grouper(ABC):
    __slots__ = [
        "_df",
        "_gf",
        "_sort",
        "_group_data",
        "_group_keys",
        "_group_indices",
        "_group_rows",
        "_group_size",
        "_n_groups",
        "_group_vars",
    ]

    def __init__(
        self,
        df: pl.DataFrame | GroupBy,
        by: List[str] = None,
        sort: bool = True,
    ):
        self._group_data = None
        self._group_keys = None
        self._group_indices = None
        self._group_rows = None
        self._group_size = None
        self._group_vars = by
        self._n_groups = None
        self._df = df
        self._gf = None
        self._sort = sort

    @abstractproperty
    def df(self) -> pl.DataFrame:
        """Get the data frame"""

    @abstractproperty
    def gf(self) -> GroupBy:
        """Get the grouped frame"""

    @property
    def group_vars(self) -> List[str]:
        return self._group_vars or []

    @property
    def sort(self) -> bool:
        return self._sort

    @abstractproperty
    def group_data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""

    @abstractproperty
    def group_rows(self) -> np.ndarray:
        """The row indexes in each group"""

    @abstractproperty
    def group_keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""

    @abstractproperty
    def group_indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""

    @abstractproperty
    def group_size(self) -> np.ndarray:
        """Get the sizes of each group"""

    @abstractproperty
    def n_groups(self) -> int:
        """Get the number of groups"""

    def str_(self) -> str:
        """Get the string representation"""
        return str(self.df)  # pragma: no cover

    def html(self) -> str:
        """Get the html representation"""
        return self.df._repr_html_()  # pragma: no cover


class GFGrouper(Grouper):

    def __init__(
        self,
        df: pl.DataFrame | GroupBy,
        by: List[str] = None,
        sort: bool = True,
    ):
        super().__init__(df, by, sort)
        if isinstance(df, GroupBy):
            if by:
                raise ValueError("Cannot specify `by` when `df` is a GroupBy")
            self._df = None
            self._gf = df
            self._group_vars = df.by
        else:
            self._df = df

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            self._df = self.gf.head(self.gf._df.shape()[0])
        return self._df

    @property
    def gf(self) -> GroupBy:
        if self._gf is None:
            self._gf = self.df.groupby(
                self.group_vars,
                maintain_order=self.sort,
            )
        return self._gf

    @property
    def group_data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._group_data is None:
            self._group_data = (
                self
                .gf._groups()
                .rename({"groups": "_rows"})
                # In order to keep the order of rows
                .join(self.group_keys, on=self.group_vars, how="inner")
            )
        return self._group_data

    @property
    def group_rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._group_rows is None:
            self._group_rows = self.group_data["_rows"].to_numpy()
        return self._group_rows

    @property
    def group_keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        # self.gf._groups() does not keep the order of rows
        return self.gf.agg([])

    @property
    def group_indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._group_indices is None:
            self._group_indices = np.repeat(
                np.arange(self.n_groups),
                self.group_size,
            )
        return self._group_indices

    @property
    def group_size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._group_size is None:
            self._group_size = self.gf.count()["count"].to_numpy()
        return self._group_size

    @property
    def n_groups(self) -> int:
        """Get the number of groups"""
        if self._n_groups is None:
            self._n_groups = self.group_data.shape[0]
        return self._n_groups

    def str_(self) -> str:
        """Get the string representation"""
        return f"grouped: ({', '.join(self.group_vars)}), n={self.n_groups}"

    def html(self) -> str:
        """Get the html representation"""
        return f"grouped: ({', '.join(self.group_vars)}), n={self.n_groups}"


class DFGrouper(Grouper):
    def __init__(
        self,
        df: pl.DataFrame | GroupBy,
        by: List[str] = None,
        sort: bool = True,
    ):
        if by:
            raise ValueError("`by` is not supported for DFGrouper.")
        super().__init__(df)

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def gf(self) -> GroupBy:
        return None

    @property
    def group_data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._group_data is None:
            self._group_data = Tibble(
                {"_rows": [np.arange(self.df.shape[0]).tolist()]}
            )
        return self._group_data

    @property
    def group_rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._group_rows is None:
            self._group_rows = np.array(
                [np.arange(self.df.shape[0]).tolist()],
                dtype=object,
            )
        return self._group_rows

    @property
    def group_keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        return Tibble()

    @property
    def group_indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._group_indices is None:
            self._group_indices = np.repeat(0, self.df.shape[0])
        return self._group_indices

    @property
    def group_size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._group_size is None:
            self._group_size = np.array([self.df.shape[0]])
        return self._group_size

    @property
    def n_groups(self) -> int:
        """Get the number of groups"""
        if self._n_groups is None:
            self._n_groups = 1
        return self._n_groups


class RFGrouper(Grouper):

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def gf(self) -> GroupBy:
        return None

    @property
    def group_data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._group_data is None:
            if not self.group_vars:
                self._group_data = Tibble(
                    {"_rows": [[i] for i in range(self.df.shape[0])]}
                )
            else:
                self._group_data = self.group_keys.with_column(
                    pl.Series(
                        name="_rows",
                        values=[[i] for i in range(self.df.shape[0])],
                    )
                )
        return self._group_data

    @property
    def group_rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._group_rows is None:
            self._group_rows = np.array(
                [[i] for i in range(self.df.shape[0])],
                dtype=object,
            )
        return self._group_rows

    @property
    def group_keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        if self.group_vars:
            return self.df[self.group_vars]
        return Tibble()

    @property
    def group_indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._group_indices is None:
            self._group_indices = np.arange(self.df.shape[0])
        return self._group_indices

    @property
    def group_size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._group_size is None:
            self._group_size = np.repeat(1, self.df.shape[0])
        return self._group_size

    @property
    def n_groups(self) -> int:
        """Get the number of groups"""
        if self._n_groups is None:
            self._n_groups = self.df.shape[0]
        return self._n_groups

    def str_(self) -> str:
        """Get the string representation"""
        return f"rowwise: ({', '.join(self.group_vars)})"

    def html(self) -> str:
        """Get the html representation"""
        return f"rowwise: ({', '.join(self.group_vars)})"


@pl.api.register_expr_namespace("datar")
class ExprDatarNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr
        self._is_rowwise = False

    @property
    def is_rowwise(self) -> pl.Expr:
        return self._is_rowwise

    @is_rowwise.setter
    def is_rowwise(self, value: bool):
        self._is_rowwise = value


@pl.api.register_dataframe_namespace("datar")
class DFDatarNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df
        self._grouper = None
        self._meta = {}

    def update_from(self, df: pl.DataFrame) -> None:
        self._grouper = df.datar.grouper
        self._meta = df.datar.meta.copy()

    def group_by(
        self,
        col: str,
        *cols: str,
        sort: bool = True,
    ) -> TibbleGrouped:
        out = TibbleGrouped(self._df)
        out.datar._grouper = GFGrouper(self._df, [col, *cols], sort=sort)
        return out

    def rowwise(self, *cols: str) -> pl.DataFrame:
        out = TibbleRowwise(self._df)
        out.datar._grouper = RFGrouper(self._df, list(cols))
        return out

    def ungroup(self, inplace: bool = False) -> pl.DataFrame:
        if inplace:
            self._grouper = None
            return

        out = Tibble(self._df)
        out.datar.ungroup(inplace=True)
        return out

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta

    @meta.setter
    def meta(self, value: Mapping[str, Any]):
        self._meta = value

    @property
    def grouper(self) -> Grouper:
        return self._grouper


@pl.api.register_series_namespace("datar")
class SeriesDatarNamespace:
    def __init__(self, series: pl.Series):
        self._series = series
        self._grouper = None
        self._is_rowwise = False

    @property
    def is_rowwise(self) -> bool:
        return self._is_rowwise

    @is_rowwise.setter
    def is_rowwise(self, value: bool):
        self._is_rowwise = value

    @property
    def grouper(self) -> bool:
        return self._grouper

    @grouper.setter
    def grouper(self, grouper: Grouper):
        self._grouper = grouper

    def ungroup(self):
        self._grouper = None
        self._is_rowwise = False


# @pl.api.register_lazyframe_namespace("datar")
# class LazyFrameDatarNamespace:
#     def __init__(self, lf: pl.LazyFrame):
#         self._lf = lf
#         self._is_grouped = False
#         self._is_rowwise = False
#         self._group_vars = []
#         self._meta = {}

#     def group_by(self, col: str, *cols: str) -> pl.LazyFrame:
#         self._is_grouped = True
#         self._group_vars = [col, *cols]
#         return self._lf

#     def rowwise(self, *cols: str) -> pl.LazyFrame:
#         self._is_rowwise = True
#         self._group_vars = cols
#         return self._lf

#     def ungroup(self) -> pl.LazyFrame:
#         self._is_grouped = False
#         self._is_rowwise = False
#         self._group_vars = []
#         return self._lf

#     @property
#     def meta(self) -> Mapping[str, Any]:
#         return self._meta

#     @meta.setter
#     def meta(self, value: Mapping[str, Any]):
#         self._meta = value

#     @property
#     def is_grouped(self) -> bool:
#         return self._is_grouped

#     @property
#     def is_rowwise(self) -> bool:
#         return self._is_rowwise

#     @property
#     def group_vars(self) -> List[str]:
#         return self._group_vars
