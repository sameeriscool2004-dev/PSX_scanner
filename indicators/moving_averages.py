"""Moving-average indicators for ordered numeric price data."""

from collections.abc import Sequence
from math import isfinite

Number = int | float


def _values(values: Sequence[Number]) -> list[float]:
    result = [float(value) for value in values]
    if any(not isfinite(value) for value in result):
        raise ValueError("values must contain only finite numbers")
    return result


def _period(period: int, length: int) -> None:
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period must be a positive integer")
    if period > length:
        raise ValueError("period cannot be greater than the number of values")


def sma(values: Sequence[Number], period: int) -> list[float | None]:
    """Return a simple moving average, aligned with ``values``."""
    data = _values(values)
    _period(period, len(data))
    result: list[float | None] = [None] * (period - 1)
    window_sum = sum(data[:period])
    result.append(window_sum / period)

    for index in range(period, len(data)):
        window_sum += data[index] - data[index - period]
        result.append(window_sum / period)
    return result


def ema(values: Sequence[Number], period: int) -> list[float | None]:
    """Return an exponential moving average seeded by the first SMA."""
    data = _values(values)
    _period(period, len(data))
    result: list[float | None] = [None] * (period - 1)
    average = sum(data[:period]) / period
    result.append(average)
    multiplier = 2 / (period + 1)

    for value in data[period:]:
        average = (value - average) * multiplier + average
        result.append(average)
    return result


def wma(values: Sequence[Number], period: int) -> list[float | None]:
    """Return a linearly weighted moving average."""
    data = _values(values)
    _period(period, len(data))
    weights = range(1, period + 1)
    denominator = period * (period + 1) / 2
    result: list[float | None] = [None] * (period - 1)

    for end in range(period, len(data) + 1):
        window = data[end - period:end]
        result.append(sum(value * weight for value, weight in zip(window, weights)) / denominator)
    return result
