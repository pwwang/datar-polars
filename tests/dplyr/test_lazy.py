"""Tests for lazy() and collect() verbs."""

import polars as pl

from datar import f
from datar.dplyr import filter_
from datar_polars import lazy, collect
from datar_polars.tibble import Tibble, LazyTibble, as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data)


class TestLazy:
    """Tests for the lazy() verb."""

    def test_lazy_on_raw_dataframe_returns_lazytibble(self):
        """lazy() on a raw pl.DataFrame returns a LazyTibble."""
        df = _df({"x": [1, 2, 3]})
        out = df >> lazy(__backend="polars")
        assert isinstance(out, LazyTibble)
        out_collected = out.collect()
        assert isinstance(out_collected, Tibble)
        assert_df_equal(out_collected, df)

    def test_lazy_on_tibble_converts_to_lazytibble(self):
        """lazy() on a Tibble converts to LazyTibble."""
        tb = as_tibble(_df({"x": [1, 2, 3]}))
        out = tb >> lazy(__backend="polars")
        assert isinstance(out, LazyTibble)

    def test_lazy_in_pipeline_with_filter(self):
        """lazy() can be used in a pipeline with filter_."""
        df = _df({"x": [1, 2, 3, 4]})
        out = (
            df
            >> lazy(__backend="polars")
            >> filter_(f.x > 2)
        )
        assert isinstance(out, LazyTibble)
        result = out.collect()
        assert isinstance(result, Tibble)
        assert result["x"].to_list() == [3, 4]

    def test_lazy_direct_call(self):
        """lazy() can be called directly (not just in a pipeline)."""
        df = _df({"x": [1, 2, 3]})
        out = lazy(df)
        assert isinstance(out, LazyTibble)
        assert out.collect()["x"].to_list() == [1, 2, 3]


class TestCollect:
    """Tests for the collect() verb."""

    def test_collect_on_tibble_returns_tibble(self):
        """collect() on a Tibble returns it as-is (already eager)."""
        tb = as_tibble(_df({"x": [1, 2, 3]}))
        out = tb >> collect(__backend="polars")
        assert isinstance(out, Tibble)
        assert out["x"].to_list() == [1, 2, 3]

    def test_collect_in_pipeline(self):
        """Pipeline: df >> lazy >> filter_ >> collect returns Tibble."""
        df = _df({"x": [1, 2, 3, 4]})
        out = (
            df
            >> lazy(__backend="polars")
            >> filter_(f.x > 2)
            >> collect(__backend="polars")
        )
        assert isinstance(out, Tibble)
        assert out["x"].to_list() == [3, 4]

    def test_collect_on_already_collected_dataframe(self):
        """collect() on an already-collected pl.DataFrame wraps as Tibble."""
        df = _df({"x": [1, 2, 3]})
        out = df >> collect(__backend="polars")
        assert isinstance(out, Tibble)
        assert out["x"].to_list() == [1, 2, 3]

    def test_collect_direct_call(self):
        """collect() can be called directly (not just in a pipeline)."""
        tb = as_tibble(_df({"x": [1, 2, 3]}))
        out = collect(tb)
        assert isinstance(out, Tibble)
        assert out["x"].to_list() == [1, 2, 3]
