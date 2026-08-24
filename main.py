"""Live PSX scanner entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pandas as pd

try:
    import Strategy as strategy_module
except ImportError:  # pragma: no cover - fallback for lowercase package names
    import strategy as strategy_module

from data import get_historical_data, get_psx_symbols
from indicators import ema, rsi, sma

TEST_LIMIT: int | None = None
MIN_TRADING_DAYS = 250
MIN_AVG_VOLUME = 100_000
MIN_AVG_VALUE_TRADED = 10_000_000
MAX_EMA9_EXTENSION = 0.05
OUTPUT_COLUMNS = [
    "Symbol", "Close", "SMA20", "SMA50", "SMA200", "RSI",
    "EMA9", "EMA20", "EMA50", "Distance_From_EMA9", "Distance_From_EMA20",
    "EMA_Alignment", "EMA_Slope",
    "EMA_Aligned", "EMA_Slope_Passed", "Close_Above_EMA9", "EMA9_Extension_Passed",
    "Pullback_Passed", "Bullish_Confirmation_Passed",
    "Volume_Ratio", "Resistance", "Distance_To_Resistance", "Setup",
    "Bullish_Candle", "Market_Filter", "Score", "Signal",
    "Trend_Passed", "RSI_Passed", "Volume_Passed", "Breakout_Passed",
    "Candle_Passed", "Market_Passed", "Average_Volume_20",
    "Average_Value_Traded_20", "Liquidity_Passed", "Volume_Status",
    "Conditions_Partial", "Conditions_Failed", "Confirmation_Missing",
    "Conditions_Lost",
]


class PSXDataProvider:
    """Adapter connecting the scanner to the live PSX data providers."""

    def list_symbols(self) -> list[str]:
        return get_psx_symbols()

    def fetch_ohlcv(self, symbol: str):
        """Return live historical OHLCV data for a symbol."""
        return get_historical_data(symbol)


PlaceholderDataProvider = PSXDataProvider


class PSXScanner:
    """Small scanner that discovers symbols, fetches OHLCV, computes indicators,
    applies the strategy score, and ranks stocks from strongest to weakest.
    """

    def __init__(
        self,
        symbol_provider: Callable[[], list[str]] = get_psx_symbols,
        data_provider: Callable[[str], pd.DataFrame] = get_historical_data,
        market_data_provider: Callable[[str], pd.DataFrame] | None = None,
        market_symbol: str = "KSE100",
        resistance_lookback: int = 50,
        retest_window: int = 20,
        retest_tolerance: float = 0.03,
        min_avg_volume: float = MIN_AVG_VOLUME,
        min_avg_value_traded: float = MIN_AVG_VALUE_TRADED,
        max_workers: int = 4,
        max_retries: int = 1,
        request_delay: float = 0.1,
        results_dir: str | Path = "results",
    ) -> None:
        self.symbol_provider = symbol_provider
        self.data_provider = data_provider
        self.market_data_provider = market_data_provider or (
            data_provider if data_provider is get_historical_data else None
        )
        self.market_symbol = market_symbol
        self.resistance_lookback = resistance_lookback
        self.retest_window = retest_window
        self.retest_tolerance = retest_tolerance
        self.min_avg_volume = min_avg_volume
        self.min_avg_value_traded = min_avg_value_traded
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.results_dir = Path(results_dir)
        self.market_verification: dict[str, Any] = {
            "symbol": market_symbol,
            "endpoint": "UNKNOWN",
            "close": None,
            "sma200": None,
            "filter": "UNKNOWN",
        }

    @staticmethod
    def _clean_history(data: pd.DataFrame) -> pd.DataFrame:
        missing = {"Close", "Volume"} - set(data.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        clean = data.copy()
        clean["Close"] = pd.to_numeric(clean["Close"], errors="coerce")
        clean["Volume"] = pd.to_numeric(clean["Volume"], errors="coerce")
        clean = clean.dropna(subset=["Close", "Volume"])
        clean = clean[(clean["Close"] > 0) & (clean["Volume"] >= 0)]
        if "Date" in clean.columns:
            clean = clean.sort_values("Date")
        return clean.reset_index(drop=True)

    def _fetch_with_retry(self, symbol: str) -> pd.DataFrame:
        """Fetch one symbol with bounded retries for temporary provider failures."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.request_delay:
                    sleep(self.request_delay * (2 ** attempt if attempt else 1))
                return self.data_provider(symbol)
            except (RuntimeError, OSError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
        if last_error is None:
            raise RuntimeError(f"Unable to fetch historical data for '{symbol}'.")
        raise last_error

    def scan(
        self,
        symbols: Sequence[str] | None = None,
        test_limit: int | None = TEST_LIMIT,
    ) -> pd.DataFrame:
        """Return successfully evaluated symbols ranked by strategy score."""
        universe = list(symbols) if symbols is not None else self.symbol_provider()
        discovered_total = len(universe)
        if test_limit is not None:
            if not isinstance(test_limit, int) or test_limit <= 0:
                raise ValueError("test_limit must be a positive integer or None")
            universe = universe[:test_limit]

        market_filter = "UNKNOWN"
        print(
            f"Breakout resistance lookback: {self.resistance_lookback} trading days "
            "(current candle excluded)"
        )
        if self.market_data_provider is not None:
            try:
                market_history = self._clean_history(self.market_data_provider(self.market_symbol))
                market_closes = market_history["Close"].tolist()
                market_sma200 = sma(market_closes, 200)[-1]
                if market_sma200 is not None:
                    market_filter = "BULLISH" if market_closes[-1] > market_sma200 else "BEARISH"
                    endpoint = (
                        f"https://dps.psx.com.pk/timeseries/eod/{self.market_symbol}"
                        if self.market_data_provider is get_historical_data
                        else "custom market-data provider"
                    )
                    self.market_verification = {
                        "symbol": self.market_symbol,
                        "endpoint": endpoint,
                        "close": market_closes[-1],
                        "sma200": market_sma200,
                        "filter": market_filter,
                    }
            except (RuntimeError, ValueError, KeyError, TypeError, OSError):
                market_filter = "UNKNOWN"
        print(
            "KSE-100 verification: "
            f"symbol={self.market_verification['symbol']}, "
            f"endpoint={self.market_verification['endpoint']}, "
            f"Close={self.market_verification['close']}, "
            f"SMA200={self.market_verification['sma200']}, "
            f"Market_Filter={self.market_verification['filter']}"
        )

        rows: list[dict[str, Any]] = []
        illiquid_count = 0
        failed_downloads = 0
        insufficient_history = 0
        successful_scans = 0
        total = len(universe)
        scan_started_at = perf_counter()
        downloaded: dict[str, pd.DataFrame] = {}
        download_errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_with_retry, symbol): symbol for symbol in universe
            }
            for index, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                elapsed = perf_counter() - scan_started_at
                print(
                    f"Downloaded {index}/{total}: {symbol} | successful={successful_scans} "
                    f"failures={failed_downloads} | elapsed={elapsed:.1f}s"
                )

                try:
                    downloaded[symbol] = future.result()
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                    download_errors[symbol] = exc
                    if "insufficient historical data" in str(exc).lower():
                        insufficient_history += 1
                    else:
                        failed_downloads += 1

        for index, symbol in enumerate(universe, start=1):
            print(
                f"Scanning {index}/{total}: {symbol} | successful={successful_scans} "
                f"failures={failed_downloads} | elapsed={perf_counter() - scan_started_at:.1f}s"
            )
            if symbol in download_errors:
                print(f"  Skipped {symbol}: {download_errors[symbol]}")
                continue
            try:
                history = self._clean_history(downloaded[symbol])
                if len(history) < MIN_TRADING_DAYS:
                    insufficient_history += 1
                    print(f"  Skipped {symbol}: only {len(history)} valid trading days")
                    continue

                closes = history["Close"].tolist()
                volumes = history["Volume"].tolist()
                sma20 = sma(closes, 20)[-1]
                sma50 = sma(closes, 50)[-1]
                sma200 = sma(closes, 200)[-1]
                ema9_values = ema(closes, 9)
                ema20_values = ema(closes, 20)
                ema50_values = ema(closes, 50)
                ema9 = ema9_values[-1]
                ema20 = ema20_values[-1]
                ema50 = ema50_values[-1]
                ema9_previous = ema9_values[-2]
                ema20_previous = ema20_values[-2]
                ema50_previous = ema50_values[-2]
                distance_from_ema9 = (closes[-1] - ema9) / ema9
                distance_from_ema20 = (closes[-1] - ema20) / ema20
                rsi14 = rsi(closes, 14)[-1]
                average_volume = sum(volumes[-20:]) / 20
                average_value_traded = sum(
                    close * volume for close, volume in zip(closes[-20:], volumes[-20:])
                ) / 20
                current_volume = volumes[-1]
                volume_ratio = current_volume / average_volume if average_volume else 0.0
                resistance, distance, setup = strategy_module.breakout_setup(
                    history,
                    lookback=self.resistance_lookback,
                    early_threshold=self.retest_tolerance,
                    retest_window=self.retest_window,
                    retest_tolerance=self.retest_tolerance,
                )
                candle = strategy_module.bullish_candle(history)
                ema_aligned = ema9 > ema20 > ema50
                ema_slope_passed = ema9 > ema9_previous and ema20 > ema20_previous
                close_above_ema9 = closes[-1] > ema9
                ema9_extension_passed = 0.0 <= distance_from_ema9 <= MAX_EMA9_EXTENSION
                pullback_passed = setup == "RETEST"
                bullish_confirmation_passed = candle != "NONE"
                trend_passed = sma20 is not None and sma50 is not None and sma200 is not None and sma20 > sma50 > sma200
                rsi_passed = rsi14 is not None and 50 < rsi14 < 70
                volume_passed = current_volume >= 1.5 * average_volume if average_volume else False
                if volume_ratio >= 1.5:
                    volume_status = "PASSED"
                elif volume_ratio >= 1.0:
                    volume_status = "PARTIAL"
                else:
                    volume_status = "FAILED"
                breakout_passed = setup in {"BREAKOUT", "RETEST", "EARLY_BREAKOUT"}
                candle_passed = candle != "NONE"
                market_passed = market_filter == "BULLISH"
                liquidity_passed = (
                    average_volume >= self.min_avg_volume
                    and average_value_traded >= self.min_avg_value_traded
                )
                if not liquidity_passed:
                    illiquid_count += 1
                conditions = {
                    "Trend": trend_passed,
                    "RSI": rsi_passed,
                    "Volume": volume_passed,
                    "Breakout": breakout_passed,
                    "Candle": candle_passed,
                    "Market": market_passed,
                    "Liquidity": liquidity_passed,
                    "EMA Alignment": ema_aligned,
                    "EMA Slope": ema_slope_passed,
                    "Close > EMA9": close_above_ema9,
                    "EMA Extension": ema9_extension_passed,
                    "Pullback/Retest": pullback_passed,
                    "Bullish Confirmation": bullish_confirmation_passed,
                }
                partial_conditions = ["Volume"] if volume_status == "PARTIAL" else []
                failed_conditions = [name for name, passed in conditions.items() if not passed]
                if "Volume" in failed_conditions and volume_status == "PARTIAL":
                    failed_conditions.remove("Volume")
                score_breakdown = strategy_module.score_scan_breakdown(
                    sma20=sma20, sma50=sma50, sma200=sma200, close=closes[-1],
                    rsi_value=rsi14, volume_ratio=volume_ratio, setup=setup,
                    bullish_candle=candle, market_filter=market_filter,
                    ema9=ema9, ema20=ema20, ema50=ema50,
                    ema9_previous=ema9_previous, ema20_previous=ema20_previous,
                    ema50_previous=ema50_previous,
                )
                strict_confirmation = {
                    "Trend": trend_passed,
                    "RSI": rsi_passed,
                    "Volume": volume_ratio >= 1.5,
                    "Breakout/Retest": setup in {"BREAKOUT", "RETEST"},
                    "Liquidity": liquidity_passed,
                    "EMA Alignment": ema_aligned,
                    "EMA Slope": ema_slope_passed,
                    "Close > EMA9": close_above_ema9,
                    "EMA Extension": ema9_extension_passed,
                    "Pullback/Retest": pullback_passed,
                    "Bullish Confirmation": bullish_confirmation_passed,
                }
                confirmation_missing = [name for name, passed in strict_confirmation.items() if not passed]
                lost_conditions = failed_conditions + [
                    f"{name} (partial: {score_breakdown['Volume']:.2f}/15)"
                    for name in partial_conditions
                ]
                score = strategy_module.score_scan_components(
                    sma20=sma20, sma50=sma50, sma200=sma200, close=closes[-1],
                    rsi_value=rsi14, volume_ratio=volume_ratio, setup=setup,
                    bullish_candle=candle, market_filter=market_filter,
                    ema9=ema9, ema20=ema20, ema50=ema50,
                    ema9_previous=ema9_previous, ema20_previous=ema20_previous,
                    ema50_previous=ema50_previous,
                )
                ema_structure_passed = ema_aligned and ema_slope_passed
                if not bullish_confirmation_passed:
                    score = min(score, 89)
                if not pullback_passed:
                    score = min(score, 79)
                if not ema_structure_passed:
                    score = min(score, 69)
                signal = strategy_module.classify_scan_signal(score)
                if signal in {"EXCEPTIONAL_BUY", "STRONG_BUY"} and confirmation_missing:
                    signal = "WATCHLIST"
                if not liquidity_passed:
                    signal = "ILLIQUID"
                rows.append({
                    "Symbol": symbol,
                    "Close": closes[-1],
                    "SMA20": sma20,
                    "SMA50": sma50,
                    "SMA200": sma200,
                    "RSI": rsi14,
                    "EMA9": ema9,
                    "EMA20": ema20,
                    "EMA50": ema50,
                    "Distance_From_EMA9": distance_from_ema9,
                    "Distance_From_EMA20": distance_from_ema20,
                    "EMA_Alignment": score_breakdown["EMA_Alignment"],
                    "EMA_Slope": score_breakdown["EMA_Slope"],
                    "EMA_Aligned": ema_aligned,
                    "EMA_Slope_Passed": ema_slope_passed,
                    "Close_Above_EMA9": close_above_ema9,
                    "EMA9_Extension_Passed": ema9_extension_passed,
                    "Pullback_Passed": pullback_passed,
                    "Bullish_Confirmation_Passed": bullish_confirmation_passed,
                    "Volume_Ratio": volume_ratio,
                    "Resistance": resistance,
                    "Distance_To_Resistance": distance,
                    "Setup": setup,
                    "Bullish_Candle": candle,
                    "Market_Filter": market_filter,
                    "Score": score,
                    "Signal": signal,
                    "Trend_Passed": trend_passed,
                    "RSI_Passed": rsi_passed,
                    "Volume_Passed": volume_passed,
                    "Breakout_Passed": breakout_passed,
                    "Candle_Passed": candle_passed,
                    "Market_Passed": market_passed,
                    "Average_Volume_20": average_volume,
                    "Average_Value_Traded_20": average_value_traded,
                    "Liquidity_Passed": liquidity_passed,
                    "Volume_Status": volume_status,
                    "Conditions_Partial": ", ".join(partial_conditions) or "NONE",
                    "Conditions_Failed": ", ".join(failed_conditions) or "NONE",
                    "Confirmation_Missing": ", ".join(confirmation_missing) or "NONE",
                    "Conditions_Lost": ", ".join(lost_conditions) or "NONE",
                })
                successful_scans += 1
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                if isinstance(exc, (RuntimeError, OSError)):
                    failed_downloads += 1
                elif "insufficient historical data" in str(exc).lower():
                    insufficient_history += 1
                else:
                    failed_downloads += 1
                print(f"  Skipped {symbol}: {exc}")

        print(
            f"Liquidity thresholds: Average_Volume_20 >= {self.min_avg_volume:,.0f}, "
            f"Average_Value_Traded_20 >= {self.min_avg_value_traded:,.0f}"
        )
        print(f"Symbols marked ILLIQUID: {illiquid_count}")
        print(f"Total symbols attempted: {total}")
        print(f"Successfully processed: {len(rows)}")
        print(f"Failed downloads: {failed_downloads}")
        print(f"Insufficient history: {insufficient_history}")
        result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS) if rows else pd.DataFrame(columns=OUTPUT_COLUMNS)
        signal_counts = result["Signal"].value_counts() if not result.empty else {}
        for signal in ("EXCEPTIONAL_BUY", "STRONG_BUY", "WATCHLIST", "DEVELOPING", "NO_BUY"):
            print(f"Number of {signal}: {signal_counts.get(signal, 0)}")
        result = result.sort_values("Score", ascending=False, ignore_index=True) if rows else result
        self.results_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.results_dir / "latest_scan.csv", index=False)
        metadata = {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_symbols_discovered": discovered_total,
            "successfully_processed": successful_scans,
            "failed": failed_downloads,
            "insufficient_history": insufficient_history,
            "illiquid": illiquid_count,
            "total_scan_duration": round(perf_counter() - scan_started_at, 2),
        }
        (self.results_dir / "latest_scan_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if not rows:
            return result
        return result


def run_scan(test_limit: int | None = TEST_LIMIT) -> pd.DataFrame:
    """Run the live scanner with the configured test limit."""
    return PSXScanner().scan(test_limit=test_limit)


if __name__ == "__main__":
    started_at = perf_counter()
    results = run_scan(test_limit=TEST_LIMIT)
    print("\nTop 30 liquid stocks ranked by Score:")
    liquid_results = results[results["Liquidity_Passed"]].head(30) if not results.empty else results
    display_columns = [
        "Symbol", "Close", "RSI", "Volume_Ratio", "Volume_Status", "Setup",
        "Bullish_Candle", "Liquidity_Passed", "Score", "Signal",
        "Confirmation_Missing", "Conditions_Partial", "Conditions_Failed",
    ]
    print("No liquid stocks produced valid scan results." if liquid_results.empty else liquid_results[display_columns].to_string(index=False))
    print(f"Total execution time: {perf_counter() - started_at:.2f} seconds")
