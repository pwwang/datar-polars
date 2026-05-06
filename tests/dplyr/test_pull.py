"""Tests for dplyr pull verb.
"""

import polars as pl
from datar import f
from datar.dplyr import pull
from datar.data import starwars
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


class TestPull:
    def test_pull_default_last_column(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        result = pull(df)
        assert result.to_list() == [3, 4]

    def test_pull_by_name(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        result = pull(df, "a")
        assert result.to_list() == [1, 2]

    def test_pull_by_index(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        result = pull(df, 0)
        assert result.to_list() == [1, 2]
        result = pull(df, 1)
        assert result.to_list() == [3, 4]

    def test_pull_to_list(self):
        df = _df({"x": [10, 20, 30]})
        result = pull(df, to="list")
        assert result == [10, 20, 30]

    def test_pull_to_array(self):
        import numpy as np

        df = _df({"x": [1, 2, 3]})
        result = pull(df, to="array")
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_pull_to_dict(self):
        df = _df({"x": [10, 20]})
        result = pull(
            df, to="dict", name=["a", "b"]
        )
        assert result == {"a": 10, "b": 20}

    def test_pull_negative_index(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        result = pull(df, -1)
        assert result.to_list() == [3, 4]

    def test_pull_starwars(self):
        out = starwars >> pull(f.height, name=f.name)
        assert out['Luke Skywalker'] == 172.0
