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
        "_data",
        "_keys",
        "_indices",
        "_rows",
        "_size",
        "_n",
        "_vars",
    ]

    def __init__(
        self,
        df: pl.DataFrame | GroupBy,
        by: List[str] = None,
        sort: bool = True,
    ):
        self._data = None
        self._keys = None
        self._indices = None
        self._rows = None
        self._size = None
        self._vars = by
        self._n = None
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
    def vars(self) -> List[str]:
        return self._vars or []

    @property
    def sort(self) -> bool:
        return self._sort

    @abstractproperty
    def data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""

    @abstractproperty
    def rows(self) -> np.ndarray:
        """The row indexes in each group"""

    @abstractproperty
    def keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""

    @abstractproperty
    def indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""

    @abstractproperty
    def size(self) -> np.ndarray:
        """Get the sizes of each group"""

    @abstractproperty
    def n(self) -> int:
        """Get the number of groups"""

    def compatible_with(self, other: Grouper) -> bool:
        """Check if two groupers are compatible"""
        if self is other:
            return True

        if self.vars != other.vars:
            return False

        if self.keys.unique().frame_equal(self.keys.unique()):
            return False

        return (
            (self.size == 1)
            | (other.size == 1)
            | (self.size == other.size)
        ).all()

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
            self._vars = df.by
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
                self.vars,
                maintain_order=self.sort,
            )
        return self._gf

    @property
    def data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._data is None:
            self._data = (
                self
                .gf._groups()
                .rename({"groups": "_rows"})
                # In order to keep the order of rows
                .join(self.keys, on=self.vars, how="inner")
            )
        return self._data

    @property
    def rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._rows is None:
            self._rows = self.data["_rows"].to_numpy()
        return self._rows

    @property
    def keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        # self.gf._groups() does not keep the order of rows
        return self.gf.agg([])

    @property
    def indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._indices is None:
            self._indices = np.repeat(
                np.arange(self.n),
                self.size,
            )
        return self._indices

    @property
    def size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._size is None:
            self._size = self.gf.count()["count"].to_numpy()
        return self._size

    @property
    def n(self) -> int:
        """Get the number of groups"""
        if self._n is None:
            self._n = self.data.shape[0]
        return self._n

    def str_(self) -> str:
        """Get the string representation"""
        return f"grouped: ({', '.join(self.vars)}), n={self.n}"

    def html(self) -> str:
        """Get the html representation"""
        return f"grouped: ({', '.join(self.vars)}), n={self.n}"


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
    def data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._data is None:
            self._data = Tibble(
                {"_rows": [np.arange(self.df.shape[0]).tolist()]}
            )
        return self._data

    @property
    def rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._rows is None:
            self._rows = np.array(
                [np.arange(self.df.shape[0]).tolist()],
                dtype=object,
            )
        return self._rows

    @property
    def keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        return Tibble()

    @property
    def indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._indices is None:
            self._indices = np.repeat(0, self.df.shape[0])
        return self._indices

    @property
    def size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._size is None:
            self._size = np.array([self.df.shape[0]])
        return self._size

    @property
    def n(self) -> int:
        """Get the number of groups"""
        if self._n is None:
            self._n = 1
        return self._n


class RFGrouper(Grouper):

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def gf(self) -> GroupBy:
        return None

    @property
    def data(self) -> pl.DataFrame:
        """A data frame with groupvar columns and _rows as group row indexes"""
        if self._data is None:
            if not self.vars:
                self._data = Tibble(
                    {"_rows": [[i] for i in range(self.df.shape[0])]}
                )
            else:
                self._data = self.keys.with_column(
                    pl.Series(
                        name="_rows",
                        values=[[i] for i in range(self.df.shape[0])],
                    )
                )
        return self._data

    @property
    def rows(self) -> np.ndarray:
        """The row indexes in each group"""
        if self._rows is None:
            self._rows = np.array(
                [[i] for i in range(self.df.shape[0])],
                dtype=object,
            )
        return self._rows

    @property
    def keys(self) -> pl.DataFrame:
        """A subset of the dataframe, with the groupvar columns only"""
        if self.vars:
            return self.df[self.vars]
        return Tibble()

    @property
    def indices(self) -> np.ndarray:
        """Use indices to mark the group of each row"""
        if self._indices is None:
            self._indices = np.arange(self.df.shape[0])
        return self._indices

    @property
    def size(self) -> np.ndarray:
        """Get the sizes of each group"""
        if self._size is None:
            self._size = np.repeat(1, self.df.shape[0])
        return self._size

    @property
    def n(self) -> int:
        """Get the number of groups"""
        if self._n is None:
            self._n = self.df.shape[0]
        return self._n

    def str_(self) -> str:
        """Get the string representation"""
        return f"rowwise: ({', '.join(self.vars)})"

    def html(self) -> str:
        """Get the html representation"""
        return f"rowwise: ({', '.join(self.vars)})"


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

    @property
    def grouper(self) -> bool:
        return self._grouper

    @grouper.setter
    def grouper(self, grouper: Grouper):
        self._grouper = grouper

    def ungroup(self) -> pl.Series:
        self._grouper = None
        return self._series
