import polars as pl


class Factor(pl.Series):

    def __init__(self, *args, levels=None, **kwargs):
        s = pl.Series(*args, **kwargs).cast(pl.Utf8).cast(pl.Categorical)
        super().__init__(s)
        self._codes = None
        self._levels = None
        if levels is not None:
            self.set_levels(levels)

    def copy(self):
        return Factor(self)

    @property
    def codes(self):
        if self._codes is None:
            self._codes = self.to_physical()
        return self._codes

    @property
    def levels(self):
        if self._levels is None:
            self._levels = self[self.codes[self.codes.is_not_null()].unique()]
        return self._levels

    @levels.setter
    def levels(self, lvls):
        ...

    def __str__(self) -> str:
        return f"{self.datar.grouper.str_()}\n{super().__str__()}"
