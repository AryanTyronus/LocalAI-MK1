"""
Reactive stock quote fetcher.

No background polling; single request per invocation.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict

import requests


class StockFetcher:
    """Fetch delayed quote data from a public CSV endpoint."""

    SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")

    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def _normalize_symbol(self, symbol: str) -> str:
        value = (symbol or "").strip().upper()
        if not self.SYMBOL_RE.match(value):
            raise ValueError("Invalid symbol format.")
        return value

    def execute(self, params: Dict) -> Dict:
        symbol = self._normalize_symbol(str(params.get("symbol", "")))
        # Stooq free delayed endpoint
        url = f"https://stooq.com/q/l/?s={symbol.lower()}.us&i=d"

        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(response.text)))
            if not rows:
                return {"ok": False, "error": "No data returned.", "symbol": symbol}
            row = rows[0]
            close = row.get("Close")
            if not close or close == "N/D":
                return {"ok": False, "error": "No quote available for symbol.", "symbol": symbol}
            return {
                "ok": True,
                "symbol": symbol,
                "close": close,
                "date": row.get("Date"),
                "time": row.get("Time"),
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "volume": row.get("Volume"),
                "source": "stooq",
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": f"Network/quote fetch failure: {exc}", "symbol": symbol}

