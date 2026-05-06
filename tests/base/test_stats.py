"""Tests for base stats functions: cov, diff, scale, weighted_mean, quantile."""

import polars as pl
import pytest
from datar.base import cov, diff, scale, weighted_mean, quantile
from datar.dplyr import mutate, summarise
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── cov ─────────────────────────────────────────────────────────────────


class TestCov:
    def test_cov_two_series(self):
        s1 = pl.Series("a", [1, 2, 3, 4, 5])
        s2 = pl.Series("b", [2, 4, 6, 8, 10])
        result = cov(s1, s2)
        assert result == pytest.approx(5.0)

    def test_cov_series_with_list(self):
        s1 = pl.Series("a", [1, 2, 3, 4, 5])
        # y as list handled differently
        result = cov(s1, [2, 4, 6, 8, 10])
        assert result == pytest.approx(5.0)

    def test_cov_series_no_y_raises(self):
        s1 = pl.Series("a", [1, 2, 3])
        with pytest.raises(ValueError):
            cov(s1)

    def test_cov_scalars(self):
        result = cov([1, 2, 3], [4, 5, 6])
        assert result == pytest.approx(1.0)

    def test_cov_dataframe(self):
        df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
        result = cov(df)
        assert result.shape == (2, 2)

    def test_cov_dataframe_with_y_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        with pytest.raises(ValueError):
            cov(df, [1, 2, 3])

    def test_cov_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        from datar import f
        out = df >> mutate(z=cov(f.x, f.y))
        assert out is not None


# ── diff ────────────────────────────────────────────────────────────────


class TestDiff:
    def test_diff_series_lag1(self):
        s = pl.Series("x", [1, 3, 6, 10, 15])
        result = diff(s)
        assert result.to_list() == [None, 2, 3, 4, 5]

    def test_diff_series_lag2(self):
        s = pl.Series("x", [1, 3, 6, 10, 15])
        result = diff(s, lag=2)
        assert result.to_list() == [None, None, 5, 7, 9]

    def test_diff_series_diff2(self):
        s = pl.Series("x", [1, 3, 6, 10, 15])
        result = diff(s, differences=2)
        assert result.to_list() == [None, None, 1, 1, 1]

    def test_diff_list(self):
        result = diff([1, 4, 9, 16])
        assert result.to_list() == [None, 3, 5, 7]

    def test_diff_scalar(self):
        result = diff(5)
        assert result is None

    def test_diff_in_mutate(self):
        df = _df({"x": [1, 3, 6, 10]})
        from datar import f
        out = df >> mutate(y=diff(f.x))
        vals = out.get_column("y").to_list()
        assert vals[0] is None
        assert vals[1:] == [2, 3, 4]


# ── scale ───────────────────────────────────────────────────────────────


class TestScale:
    def test_scale_series_default(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = scale(s)
        vals = result.to_list()
        assert pytest.approx(vals[2]) == 0.0
        expected = [-1.2649, -0.6325, 0.0, 0.6325, 1.2649]
        for v, e in zip(vals, expected):
            assert v == pytest.approx(e, rel=1e-3)

    def test_scale_series_center_only(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = scale(s, center=True, scale_=False)
        assert result.to_list() == [-2, -1, 0, 1, 2]

    def test_scale_series_scale_only(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = scale(s, center=False, scale_=True)
        vals = result.to_list()
        assert sum(vals) != pytest.approx(0.0)  # not centered

    def test_scale_series_noop(self):
        s = pl.Series("x", [1, 2, 3])
        result = scale(s, center=False, scale_=False)
        assert result.to_list() == [1, 2, 3]

    def test_scale_list(self):
        result = scale([1, 2, 3, 4, 5])
        assert result[2] == pytest.approx(0.0)

    def test_scale_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        from datar import f
        out = df >> mutate(y=scale(f.x))
        vals = out.get_column("y").to_list()
        assert vals[2] == pytest.approx(0.0)


# ── weighted_mean ──────────────────────────────────────────────────────


class TestWeightedMean:
    def test_weighted_mean_series(self):
        x = pl.Series("x", [1, 2, 3, 4])
        w = pl.Series("w", [1, 1, 1, 1])
        result = weighted_mean(x, w)
        assert result == pytest.approx(2.5)

    def test_weighted_mean_unequal_weights(self):
        x = pl.Series("x", [1, 2, 3, 4])
        w = pl.Series("w", [1, 2, 3, 4])
        result = weighted_mean(x, w)
        assert result == pytest.approx(3.0)

    def test_weighted_mean_no_weights(self):
        x = pl.Series("x", [1, 2, 3, 4])
        result = weighted_mean(x)
        assert result == pytest.approx(2.5)

    def test_weighted_mean_list(self):
        result = weighted_mean([1, 2, 3, 4], [1, 2, 3, 4])
        assert result == pytest.approx(3.0)

    def test_weighted_mean_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4], "w": [1, 2, 3, 4]})
        from datar import f
        out = df >> mutate(
            y=weighted_mean(f.x, f.w)
        )
        assert out is not None


# ── quantile ────────────────────────────────────────────────────────────


class TestQuantile:
    def test_quantile_series_single(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = quantile(s, probs=0.5)
        assert result == 3.0

    def test_quantile_series_multiple(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = quantile(s, probs=[0.25, 0.5, 0.75])
        assert result == [2.0, 3.0, 4.0]

    def test_quantile_series_default(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = quantile(s)
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_quantile_list(self):
        result = quantile([1, 2, 3, 4, 5], probs=0.5)
        assert result == 3.0

    def test_quantile_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        from datar import f
        out = df >> mutate(
            y=quantile(f.x, probs=0.5)
        )
        assert out is not None
