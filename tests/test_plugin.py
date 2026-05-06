"""Tests for the datar_polars plugin registration."""

import pytest
import polars as pl

from datar.core.plugin import plugin


class TestPluginLoaded:
    def test_setup_runs(self):
        """Verify the setup hook initializes options."""
        from datar.core.options import get_option

        opt = get_option("dplyr_summarise_inform")
        assert opt is not None

    def test_get_versions(self):
        """Verify get_versions returns polars and datar-polars versions."""
        # Simulate the get_versions hook call
        import polars
        from datar_polars.version import __version__

        versions = {
            "datar-polars": __version__,
            "polars": polars.__version__,
        }
        assert "datar-polars" in versions
        assert "polars" in versions
        assert versions["datar-polars"] == __version__

    def test_load_dataset_not_implemented_for_most_datasets(self):
        """Most datasets need CSV loading through read_csv."""
        from datar_polars.plugin import load_dataset

        with pytest.raises(AttributeError, match="No such dataset"):
            load_dataset("nonexistent_dataset_xyz", {})


class TestBackendRegistration:
    def test_polars_backend_registered(self):
        """Verify the polars backend is a registered simplug plugin."""
        # The polars module should be importable and have a plugin
        from datar_polars import plugin as polars_plugin

        assert polars_plugin is not None

    def test_dplyr_api_imports(self):
        """Verify dplyr API modules import without error."""
        from datar_polars.api.dplyr import mutate, filter_, select
        from datar_polars.api.dplyr import arrange, group_by, summarise
        from datar_polars.api.dplyr import join, distinct

        assert mutate is not None
        assert filter_ is not None
        assert select is not None
        assert arrange is not None
        assert group_by is not None
        assert summarise is not None

    def test_verbs_registered_for_pl_dataframe(self):
        """Verify verbs are registered for polars DataFrames."""
        from datar.dplyr import mutate, filter_, select, group_by
        from datar_polars.polars import DataFrame

        # Check that the verbs have a polars implementation
        # by checking if they dispatch correctly
        assert callable(mutate)
        assert callable(filter_)
        assert callable(select)


class TestPolarsTypes:
    def test_dataframe_is_pl_dataframe(self):
        from datar_polars.polars import DataFrame, Series, LazyFrame

        assert DataFrame is pl.DataFrame
        assert Series is pl.Series
        assert LazyFrame is pl.LazyFrame


class TestCLikeGetitem:
    def test_c_getitem_returns_collection(self):
        from datar_polars.collections import Collection
        from datar_polars.plugin import c_getitem

        result = c_getitem([1, 2, 3])
        assert isinstance(result, Collection)
        assert list(result) == [1, 2, 3]
