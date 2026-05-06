"""Tests for base string functions: chartr, grep, grepl, gsub, strsplit,
strtoi, sub, substr, sprintf, trimws, startswith, endswith.
"""

import polars as pl
from datar import f
from datar.base import (
    chartr, grep, grepl, gsub, strsplit, strtoi, sub, substr, substring,
    sprintf, trimws, startswith, endswith,
)
from datar.dplyr import mutate, filter_
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── chartr ───────────────────────────────────────────────────────────────


class TestChartr:
    def test_chartr_in_mutate(self):
        df = _df({"x": ["abc", "bac", "cab"]})
        out = df >> mutate(
            y=chartr("ab", "xy", f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["xyc", "yxc", "cxy"]

    def test_chartr_scalar(self):
        assert chartr("ab", "xy", "abc") == "xyc"


# ── grep ─────────────────────────────────────────────────────────────────


class TestGrep:
    def test_grep_basic(self):
        result = grep("he", pl.Series(["hello", "world", "help", "nope"]))
        assert result == [0, 2]

    def test_grep_regex(self):
        result = grep(".", ["ab", "c.d"])
        assert result == [0, 1]

    def test_grep_regex_fixed(self):
        result = grep(".", ["ab", "c.d"], fixed=True)
        assert result == [1]

    def test_grep_invert(self):
        result = grep(
            "he", pl.Series(["hello", "world", "help", "nope"]),
            invert=True,
        )
        assert result == [1, 3]

    def test_grep_ignore_case(self):
        result = grep(
            "HE", pl.Series(["hello", "world", "HELP"]),
            ignore_case=True,
        )
        assert result == [0, 2]

    def test_grep_fixed(self):
        # fixed=True means literal, not regex
        result = grep(
            ".", pl.Series(["a.b", "axb", "ab"]),
            fixed=True,
        )
        assert result == [0]

    def test_grep_scalar(self):
        assert grep("he", "hello") == [0]
        assert grep("he", "world") == []


# ── grepl ────────────────────────────────────────────────────────────────


class TestGrepl:
    def test_grepl_in_mutate(self):
        df = _df({"x": ["hello", "world", "help"]})
        out = df >> mutate(
            y=grepl("he", f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [True, False, True]

    def test_grepl_scalar(self):
        assert grepl("he", "hello") is True
        assert grepl("he", "world") is False

    def test_grepl_ignore_case(self):
        df = _df({"x": ["Hello", "WORLD", "help"]})
        out = df >> mutate(
            y=grepl("he", f.x, ignore_case=True),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [True, False, True]


# ── gsub ─────────────────────────────────────────────────────────────────


class TestGsub:
    def test_gsub_in_mutate(self):
        df = _df({"x": ["abac", "abcabc", "xyz"]})
        out = df >> mutate(
            y=gsub("a", "X", f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["XbXc", "XbcXbc", "xyz"]

    def test_gsub_scalar(self):
        assert gsub("a", "X", "abac") == "XbXc"

    def test_gsub_regex(self):
        result = gsub(".", "x", ["ab", "c.d.e"])
        assert result == ["xx", "xxxxx"]

    def test_gsub_fixed(self):
        # fixed=True: literal, not regex
        assert gsub(".", "X", "a.b", fixed=True) == "aXb"

        result = gsub(".", "x", ["ab", "c.d.e"], fixed=True)
        assert result == ["ab", "cxdxe"]

    def test_gsub_replace_with_ref(self):
        # In R, you can use \\1 to refer to the first capture group in the replacement.
        # In datar-polars, we use \1 instead.
        result = gsub(r"(\w)(\w)", r"\2\1", "abcd")
        assert result == "badc"

        result = gsub(r"(\w)(\d)", r"\2\1", ["a1", "b2"])
        assert result == ["1a", "2b"]


# ── strsplit ─────────────────────────────────────────────────────────────


class TestStrsplit:
    def test_strsplit_in_mutate(self):
        df = _df({"x": ["a,b,c", "d,e"]})
        out = df >> mutate(
            y=strsplit(f.x, ","),
            __backend="polars",
        )
        result = out.get_column("y").to_list()
        assert result == [["a", "b", "c"], ["d", "e"]]

    def test_strsplit_scalar(self):
        assert strsplit("a,b,c", ",") == ["a", "b", "c"]

    def test_strsplit_list(self):
        result = strsplit(["a,b", "c,d,e"], ",")
        assert result == [["a", "b"], ["c", "d", "e"]]


# ── strtoi ───────────────────────────────────────────────────────────────


class TestStrtoi:
    def test_strtoi_in_mutate(self):
        df = _df({"x": ["10", "20", "30"]})
        out = df >> mutate(
            y=strtoi(f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [10, 20, 30]

    def test_strtoi_scalar(self):
        assert strtoi("42") == 42

    def test_strtoi_base(self):
        assert strtoi("ff", base=16) == 255


# ── sub ──────────────────────────────────────────────────────────────────


class TestSub:
    def test_sub_in_mutate(self):
        df = _df({"x": ["abac", "abcabc", "xyz"]})
        out = df >> mutate(
            y=sub("a", "X", f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["Xbac", "Xbcabc", "xyz"]

    def test_sub_scalar(self):
        assert sub("a", "X", "abac") == "Xbac"

    def test_sub_regex(self):
        result = sub(".", "x", ["ab", "c.d.e"])
        assert result == ["xb", "x.d.e"]

        result = sub(".", "x", pl.Series(["ab", "c.d.e"]))
        assert result.to_list() == ["xb", "x.d.e"]

    def test_sub_fixed(self):
        # fixed=True means literal, not regex
        result = sub(".", "x", ["ab", "c.d.e"], fixed=True)
        assert result == ["ab", "cxd.e"]


# ── substr ───────────────────────────────────────────────────────────────


class TestSubstr:
    def test_substr_in_mutate(self):
        df = _df({"x": ["abcdef", "ghijkl"]})
        out = df >> mutate(
            y=substr(f.x, 1, 3),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["bc", "hi"]

    def test_substr_scalar(self):
        assert substr("abcdef", 1, 3) == "bc"


# ── substring ───────────────────────────────────────────────────────────────


class TestSubstring:
    def test_substring_in_mutate(self):
        df = _df({"x": ["abcdef", "ghijkl"]})
        out = df >> mutate(
            y=substring(f.x, 1, 4),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["bcd", "hij"]

    def test_substring_scalar(self):
        assert substring("abcdef", 1, 4) == "bcd"


# ── sprintf ──────────────────────────────────────────────────────────────


class TestSprintf:
    def test_sprintf_in_mutate(self):
        df = _df({"name": ["Alice", "Bob"]})
        out = df >> mutate(
            y=sprintf("Hello %s", f.name),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["Hello Alice", "Hello Bob"]

    def test_sprintf_scalar(self):
        assert sprintf("Hello %s", "Alice") == "Hello Alice"

    def test_sprintf_two_args(self):
        df = _df({"name": ["Alice", "Bob"], "score": [95, 87]})
        out = df >> mutate(
            y=sprintf("%s: %d", f.name, f.score),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["Alice: 95", "Bob: 87"]


# ── trimws ───────────────────────────────────────────────────────────────


class TestTrimws:
    def test_trimws_in_mutate(self):
        df = _df({"x": ["  hello  ", "\t test\t", "abc"]})
        out = df >> mutate(
            y=trimws(f.x),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["hello", "test", "abc"]

    def test_trimws_scalar(self):
        assert trimws("  hello  ") == "hello"

    def test_trimws_left(self):
        assert trimws("  hello  ", which="left") == "hello  "

    def test_trimws_right(self):
        assert trimws("  hello  ", which="right") == "  hello"


# ── startswith ───────────────────────────────────────────────────────────


class TestStartswith:
    def test_startswith_in_mutate(self):
        df = _df({"x": ["hello", "world", "help"]})
        out = df >> mutate(
            y=startswith(f.x, "he"),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [True, False, True]

    def test_startswith_scalar(self):
        assert startswith("hello", "he") is True
        assert startswith("world", "he") is False


# ── endswith ─────────────────────────────────────────────────────────────


class TestEndswith:
    def test_endswith_in_mutate(self):
        df = _df({"x": ["hello", "world", "held"]})
        out = df >> mutate(
            y=endswith(f.x, "ld"),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [False, True, True]

    def test_endswith_scalar(self):
        assert endswith("hello", "lo") is True
        assert endswith("world", "he") is False
