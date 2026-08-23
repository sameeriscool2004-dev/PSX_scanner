"""Live PSX symbol discovery and normalization helpers."""

from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence
from urllib import error, request

_INVALID_TICKERS = {
    "",
    "DPS",
    "PSX",
    "KSE",
    "INDEX",
    "TEST",
    "DEMO",
    "NULL",
    "N/A",
}


class PSXSymbolProvider(ABC):
    """Provider interface for retrieving the current PSX symbol list."""

    @abstractmethod
    def fetch_symbols(self) -> list[str]:
        """Return raw symbol strings from a public PSX source."""


class DPSPSXSymbolProvider(PSXSymbolProvider):
    """Public DPS screener provider for live PSX ticker discovery."""

    def __init__(self, url: str = "https://dps.psx.com.pk/screener"):
        self.url = url

    def fetch_symbols(self) -> list[str]:
        try:
            req = request.Request(self.url, headers={"User-Agent": "Mozilla/5.0"})
            with request.urlopen(req, timeout=20) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except (error.URLError, error.HTTPError, TimeoutError, ValueError) as exc:
            raise RuntimeError(
                "Unable to reach the public PSX screener for symbol discovery. "
                "Please check your network connection or try again later."
            ) from exc

        symbol_matches = re.findall(
            r'<a[^>]*class="tbl__symbol"[^>]*>\s*<strong>\s*([A-Za-z0-9]+)\s*</strong>',
            html,
            flags=re.IGNORECASE,
        )
        if not symbol_matches:
            raise RuntimeError(
                "The public PSX screener did not expose any usable ticker symbols. "
                "The source format may have changed."
            )
        return symbol_matches


def normalize_symbol(symbol: str) -> str:
    """Normalize a PSX ticker into a clean uppercase form."""
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    cleaned = re.sub(r"[^A-Za-z0-9]", "", symbol.strip().upper())
    if not cleaned:
        raise ValueError("symbol cannot be empty")
    return cleaned


def _is_valid_symbol(symbol: str) -> bool:
    if not symbol or symbol in _INVALID_TICKERS:
        return False
    if symbol.isdigit():
        return False
    if len(symbol) < 2 or len(symbol) > 12:
        return False
    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        return False
    return True


def discover_psx_symbols(symbols: Sequence[str] | Iterable[str] | None = None) -> list[str]:
    """Return a deduplicated list of valid PSX stock symbols."""
    source = list(symbols) if symbols is not None else []
    discovered: list[str] = []
    seen: set[str] = set()

    for raw_symbol in source:
        try:
            normalized = normalize_symbol(raw_symbol)
        except (TypeError, ValueError):
            continue
        if not _is_valid_symbol(normalized):
            continue
        if normalized not in seen:
            seen.add(normalized)
            discovered.append(normalized)

    return discovered


def get_psx_symbols(provider: PSXSymbolProvider | None = None) -> list[str]:
    """Retrieve currently listed PSX symbols from a public source.

    This keeps the symbol-source logic modular so the provider can be swapped for
    another PSX data source without changing the scanner interface.
    """
    active_provider = provider or DPSPSXSymbolProvider()
    raw_symbols = active_provider.fetch_symbols()
    symbols = discover_psx_symbols(raw_symbols)
    return sorted(symbols)


def load_symbols_from_csv(path: str | Path) -> list[str]:
    """Load symbols from a CSV file with a single column named ``symbol``."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"symbol file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row")
        names = {name.lower() for name in reader.fieldnames}
        if "symbol" not in names:
            raise ValueError("CSV file must contain a 'symbol' column")
        return discover_psx_symbols(row["symbol"] for row in reader if row.get("symbol"))


get_psx_universe = get_psx_symbols


if __name__ == "__main__":
    symbols = get_psx_symbols()
    print(f"Total symbols discovered: {len(symbols)}")
    print("First 20 symbols:")
    for symbol in symbols[:20]:
        print(symbol)
