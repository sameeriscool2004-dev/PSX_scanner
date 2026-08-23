"""Real PSX historical OHLCV provider."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd
from urllib import error, request

from .universe import normalize_symbol


class PSXHistoricalProvider(ABC):
    """Provider interface for retrieving OHLCV data for a PSX symbol."""

    @abstractmethod
    def fetch_historical(self, symbol: str, start_date: str | date | None = None, end_date: str | date | None = None) -> pd.DataFrame:
        """Return OHLCV data as a DataFrame."""


class DPSHistoricalProvider(PSXHistoricalProvider):
    """Public PSX DPS provider for EOD historical data."""

    base_url = "https://dps.psx.com.pk/timeseries/eod/{symbol}"

    def fetch_historical(self, symbol: str, start_date: str | date | None = None, end_date: str | date | None = None) -> pd.DataFrame:
        normalized = normalize_symbol(symbol)
        if not normalized:
            raise ValueError("symbol cannot be empty")

        url = self.base_url.format(symbol=normalized)
        try:
            response = request.urlopen(request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30)
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        except (error.URLError, error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to fetch historical data for '{normalized}' from the live PSX endpoint: {url}."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected response format from PSX endpoint for '{normalized}'.")

        raw_rows = payload.get("data")
        if not isinstance(raw_rows, list):
            raise RuntimeError(f"Unexpected response payload for '{normalized}'; expected a data list.")

        records: list[dict[str, Any]] = []
        for row in raw_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                continue
            try:
                timestamp, close, volume, open_value = row[0], row[1], row[2], row[3]
                date_value = pd.to_datetime(timestamp, unit="s", utc=True).tz_localize(None)
                open_price = float(open_value)
                close_price = float(close)
                volume_value = float(volume)
            except (TypeError, ValueError):
                continue

            if not pd.notna(open_price) or not pd.notna(close_price) or not pd.notna(volume_value):
                continue

            records.append(
                {
                    "Date": date_value,
                    "Open": open_price,
                    "High": max(open_price, close_price),
                    "Low": min(open_price, close_price),
                    "Close": close_price,
                    "Volume": volume_value,
                }
            )

        if not records:
            raise RuntimeError(f"No valid historical OHLCV rows were returned for '{normalized}'.")

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError(f"Historical data table is empty for '{normalized}'.")

        df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
        df["Date"] = pd.to_datetime(df["Date"])

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
        df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] >= 0)]

        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            df = df[df["Date"] >= start_ts]
        if end_date is not None:
            end_ts = pd.Timestamp(end_date)
            df = df[df["Date"] <= end_ts]

        if len(df) < 300:
            raise ValueError(
                f"Symbol '{normalized}' has insufficient historical data ({len(df)} rows). "
                "At least 300 trading days are required for the strategy."
            )

        return df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def get_historical_data(
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    provider: PSXHistoricalProvider | None = None,
) -> pd.DataFrame:
    """Retrieve real daily PSX historical data for a symbol.

    The caller may provide a custom provider, otherwise the live DPS endpoint is
    used. This keeps the data-source logic modular.
    """
    normalized = normalize_symbol(symbol)
    if not normalized:
        raise ValueError("symbol cannot be empty")

    active_provider = provider or DPSHistoricalProvider()
    return active_provider.fetch_historical(normalized, start_date=start_date, end_date=end_date)


fetch_symbol_data = get_historical_data

def download_market_data(symbols: str | list[str] | tuple[str, ...], start_date: str | date | None = None, end_date: str | date | None = None) -> dict[str, pd.DataFrame]:
    """Download real historical data for one or more symbols."""
    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        raise ValueError("at least one symbol is required")

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        result[normalize_symbol(symbol)] = get_historical_data(symbol, start_date=start_date, end_date=end_date)
    return result


load_market_data = download_market_data


if __name__ == "__main__":
    df = get_historical_data("HBL")
    print("Symbol:", "HBL")
    print("Rows:", len(df))
    print("First date:", df["Date"].iloc[0].date())
    print("Last date:", df["Date"].iloc[-1].date())
    print("Latest Close:", df["Close"].iloc[-1])
    print("Latest Volume:", int(df["Volume"].iloc[-1]))
