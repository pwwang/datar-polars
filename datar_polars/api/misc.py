from polars import DataFrame, LazyFrame
from pipda import register_verb


@register_verb(DataFrame)
def flatten(_data: DataFrame, bycol: bool = False):
    """Flatten a dataframe into a 1-d python list

    Args:
        _data: The dataframe

    Returns:
        The flattened list
    """
    if bycol:
        return _data.transpose().to_numpy().flatten().tolist()
    return _data.to_numpy().flatten().tolist()


@register_verb(DataFrame)
def lazy(_data: DataFrame):
    """Return a lazy dataframe

    Args:
        _data: The dataframe

    Returns:
        The lazy dataframe
    """
    raise NotImplementedError("Lazy dataframe will be supported in the future")
    # return _data.lazy()


@register_verb(LazyFrame)
def collect(_data: LazyFrame):
    """Collect the result of a lazy dataframe

    Args:
        _data: The lazy dataframe

    Returns:
        The dataframe
    """
    raise NotImplementedError("Lazy dataframe will be supported in the future")
    # return _data.collect()
