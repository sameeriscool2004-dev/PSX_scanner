"""Composite scoring utilities for trend-following signals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from indicators import ema, macd, roc, rsi

Number = int | float


def _values(values: Sequence[Number]) -> list[float]:
    """Normalize input values to finite floats."""
    result = [float(value) for value in values]
    if not result:
        raise ValueError("values must contain at least one item")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("values must contain only finite numbers")
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def score_scan_components(
    *,
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
    close: float,
    rsi_value: float | None,
    volume_ratio: float,
    setup: str,
    bullish_candle: str,
    market_filter: str,
) -> float:
    """Calculate the scanner's weighted score on a 0-100 scale."""
    breakdown = score_scan_breakdown(
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        close=close,
        rsi_value=rsi_value,
        volume_ratio=volume_ratio,
        setup=setup,
        bullish_candle=bullish_candle,
        market_filter=market_filter,
    )
    return round(_clamp(sum(breakdown.values()), 0.0, 100.0), 2)


def score_scan_breakdown(
    *,
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
    close: float,
    rsi_value: float | None,
    volume_ratio: float,
    setup: str,
    bullish_candle: str,
    market_filter: str,
) -> dict[str, float]:
    """Return each weighted scanner score component before final rounding."""
    trend_points = 25.0 if sma20 is not None and sma50 is not None and sma200 is not None and sma20 > sma50 > sma200 else (
        12.5 if sma20 is not None and sma50 is not None and sma20 > sma50 else 0.0
    )
    sma_points = sum(5.0 for average in (sma20, sma50, sma200) if average is not None and close > average)
    rsi_points = 0.0 if rsi_value is None else _clamp((rsi_value - 50.0) / 20.0 * 10.0, 0.0, 10.0)
    volume_points = _clamp(volume_ratio / 1.5 * 15.0, 0.0, 15.0)
    setup_points = {"BREAKOUT": 20.0, "RETEST": 20.0, "EARLY_BREAKOUT": 10.0}.get(setup, 0.0)
    candle_points = 10.0 if bullish_candle != "NONE" else 0.0
    market_points = 5.0 if market_filter == "BULLISH" else 0.0
    return {
        "Trend": trend_points,
        "SMAs": sma_points,
        "RSI": rsi_points,
        "Volume": volume_points,
        "Breakout_Retest": setup_points,
        "Bullish_Candle": candle_points,
        "Market_Filter": market_points,
    }


def classify_scan_signal(score: float) -> str:
    """Translate a 0-100 scanner score into the requested signal bands."""
    if score >= 90:
        return "EXCEPTIONAL_BUY"
    if score >= 80:
        return "STRONG_BUY"
    if score >= 70:
        return "WATCHLIST"
    if score >= 60:
        return "DEVELOPING"
    return "NO_BUY"


@dataclass(frozen=True)
class StrategySignal:
    """A compact representation of a signal score and its interpretation."""

    score: float
    direction: str
    confidence: str

    @property
    def is_bullish(self) -> bool:
        return self.direction == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "bearish"


def score_prices(
    values: Sequence[Number],
    short_period: int = 10,
    long_period: int = 30,
    rsi_period: int = 14,
) -> float:
    """Return a composite trend score in the range [-100, 100].

    The score combines EMA trend alignment, RSI momentum, rate-of-change, and MACD
    confirmation into one normalized signal.
    """
    data = _values(values)
    if not isinstance(short_period, int) or isinstance(short_period, bool) or short_period <= 0:
        raise ValueError("short_period must be a positive integer")
    if not isinstance(long_period, int) or isinstance(long_period, bool) or long_period <= 0:
        raise ValueError("long_period must be a positive integer")
    if short_period >= long_period:
        raise ValueError("short_period must be less than long_period")
    if not isinstance(rsi_period, int) or isinstance(rsi_period, bool) or rsi_period <= 0:
        raise ValueError("rsi_period must be a positive integer")

    minimum_length = max(short_period, long_period, rsi_period, 12, 26, 9)
    if len(data) < minimum_length:
        raise ValueError("values are too short for the configured strategy parameters")

    short_ema = ema(data, short_period)
    long_ema = ema(data, long_period)
    short_value = short_ema[-1]
    long_value = long_ema[-1]

    trend_gap = 0.0 if long_value in (None, 0.0) else (short_value - long_value) / abs(long_value)
    trend_score = 100.0 * trend_gap

    rsi_values = rsi(data, rsi_period)
    last_rsi = rsi_values[-1] if rsi_values and rsi_values[-1] is not None else 50.0
    momentum_score = 2.0 * (last_rsi - 50.0)

    roc_values = roc(data, 12)
    last_roc = roc_values[-1] if roc_values and roc_values[-1] is not None else 0.0

    macd_line, signal_line, _ = macd(data, fast_period=12, slow_period=26, signal_period=9)
    last_macd = macd_line[-1] if macd_line and macd_line[-1] is not None else 0.0
    last_signal = signal_line[-1] if signal_line and signal_line[-1] is not None else 0.0
    macd_gap = 0.0 if last_signal in (None, 0.0) else (last_macd - last_signal) / abs(last_signal)
    macd_score = 100.0 * macd_gap

    composite_score = (0.45 * trend_score) + (0.35 * momentum_score) + (0.15 * last_roc) + (0.25 * macd_score)
    return round(_clamp(composite_score, -100.0, 100.0), 2)


def score_signal(
    values: Sequence[Number],
    short_period: int = 10,
    long_period: int = 30,
    rsi_period: int = 14,
) -> float:
    """Backward-compatible alias for score_prices."""
    return score_prices(values, short_period=short_period, long_period=long_period, rsi_period=rsi_period)


def classify_signal(score: float) -> str:
    """Translate a numeric score into a market bias label."""
    if score >= 60.0:
        return "strong_buy"
    if score >= 25.0:
        return "buy"
    if score <= -60.0:
        return "strong_sell"
    if score <= -25.0:
        return "sell"
    return "hold"


def evaluate_signal(
    values: Sequence[Number],
    short_period: int = 10,
    long_period: int = 30,
    rsi_period: int = 14,
) -> StrategySignal:
    """Create a structured score and interpretation for a price series."""
    score = score_prices(values, short_period=short_period, long_period=long_period, rsi_period=rsi_period)
    if score >= 60.0:
        direction = "bullish"
        confidence = "strong"
    elif score >= 25.0:
        direction = "bullish"
        confidence = "moderate"
    elif score <= -60.0:
        direction = "bearish"
        confidence = "strong"
    elif score <= -25.0:
        direction = "bearish"
        confidence = "moderate"
    else:
        direction = "neutral"
        confidence = "low"
    return StrategySignal(score=score, direction=direction, confidence=confidence)
