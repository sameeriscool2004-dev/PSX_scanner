"""Momentum indicators for ordered numeric price data."""

from collections.abc import Sequence

from .moving_averages import Number, _period, _values, ema


def rsi(values: Sequence[Number], period: int = 14) -> list[float | None]:
    """Return the Relative Strength Index using Wilder smoothing."""
    data = _values(values)
    _period(period, len(data))
    result: list[float | None] = [None] * period
    gains = [max(data[index] - data[index - 1], 0.0) for index in range(1, len(data))]
    losses = [max(data[index - 1] - data[index], 0.0) for index in range(1, len(data))]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def relative_strength_index() -> float:
        if average_gain == 0 and average_loss == 0:
            return 50.0
        if average_loss == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + average_gain / average_loss))

    result.append(relative_strength_index())
    for index in range(period, len(gains)):
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        result.append(relative_strength_index())
    return result


def roc(values: Sequence[Number], period: int = 12) -> list[float | None]:
    """Return the percentage Rate of Change."""
    data = _values(values)
    _period(period, len(data))
    result: list[float | None] = [None] * period
    for index in range(period, len(data)):
        previous = data[index - period]
        result.append(None if previous == 0 else (data[index] - previous) / previous * 100.0)
    return result


def macd(
    values: Sequence[Number],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return aligned MACD, signal, and histogram lines."""
    data = _values(values)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")
    _period(fast_period, len(data))
    _period(slow_period, len(data))
    if not isinstance(signal_period, int) or isinstance(signal_period, bool) or signal_period <= 0:
        raise ValueError("signal_period must be a positive integer")

    fast = ema(data, fast_period)
    slow = ema(data, slow_period)
    line: list[float | None] = [
        None if fast_value is None or slow_value is None else fast_value - slow_value
        for fast_value, slow_value in zip(fast, slow)
    ]
    defined_line = [value for value in line if value is not None]
    signal_values = ema(defined_line, signal_period) if len(defined_line) >= signal_period else []
    signal: list[float | None] = [None] * len(line)
    first_signal_index = slow_period - 1
    for offset, value in enumerate(signal_values):
        signal[first_signal_index + offset] = value
    histogram = [
        None if line_value is None or signal_value is None else line_value - signal_value
        for line_value, signal_value in zip(line, signal)
    ]
    return line, signal, histogram
