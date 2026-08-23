"""Price-action pattern helpers used by the scanner."""

from __future__ import annotations

import pandas as pd


def breakout_setup(
    history: pd.DataFrame,
    lookback: int = 50,
    early_threshold: float = 0.03,
    retest_window: int = 20,
    retest_tolerance: float = 0.03,
) -> tuple[float | None, float | None, str]:
    """Classify the latest bar using resistance calculated from earlier bars."""
    if len(history) <= lookback:
        return None, None, "NONE"

    closes = history["Close"].astype(float).tolist()
    highs = history.get("High", history["Close"]).astype(float).tolist()
    resistance_series = pd.Series(highs).rolling(lookback).max().shift(1)
    resistance = resistance_series.iloc[-1]
    if pd.isna(resistance) or resistance <= 0:
        return None, None, "NONE"

    close = closes[-1]
    distance = (close - resistance) / resistance * 100.0
    latest_breakout: tuple[int, float] | None = None
    retest_start = max(lookback, len(history) - retest_window)
    for index in range(retest_start, len(history) - 1):
        prior_resistance = resistance_series.iloc[index]
        if not pd.isna(prior_resistance) and closes[index] > prior_resistance:
            latest_breakout = (index, float(prior_resistance))

    if close > resistance:
        return float(resistance), distance, "BREAKOUT"
    if latest_breakout:
        level = latest_breakout[1]
        recent_low = float(history["Low"].iloc[-1]) if "Low" in history else close
        if close >= level * (1.0 - retest_tolerance) and recent_low <= level * (1.0 + retest_tolerance):
            return float(resistance), distance, "RETEST"
    if close >= resistance * (1.0 - early_threshold):
        return float(resistance), distance, "EARLY_BREAKOUT"
    return float(resistance), distance, "NONE"


def bullish_candle(history: pd.DataFrame) -> str:
    """Return the strongest recognized bullish reversal candle on the latest bar."""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(history.columns) or len(history) < 2:
        return "NONE"

    current = history.iloc[-1]
    previous = history.iloc[-2]
    current_open = float(current["Open"])
    current_high = float(current["High"])
    current_low = float(current["Low"])
    current_close = float(current["Close"])
    previous_open = float(previous["Open"])
    previous_close = float(previous["Close"])
    body = abs(current_close - current_open)
    candle_range = current_high - current_low
    if candle_range <= 0:
        return "NONE"

    if (
        current_close > current_open
        and previous_close < previous_open
        and current_open <= previous_close
        and current_close >= previous_open
    ):
        return "BULLISH_ENGULFING"

    lower_shadow = min(current_open, current_close) - current_low
    upper_shadow = current_high - max(current_open, current_close)
    if body > 0 and lower_shadow >= 2.0 * body and upper_shadow <= body and current_close >= current_low + candle_range * 0.55:
        return "HAMMER"

    if current_close > current_open and body / candle_range >= 0.6:
        return "STRONG_BULLISH"
    return "NONE"