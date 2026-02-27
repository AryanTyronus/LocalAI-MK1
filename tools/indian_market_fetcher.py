"""
Reactive Indian stock market fetcher.

Lifecycle:
spawn request -> fetch market data -> return -> terminate
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import requests


class IndianMarketFetcher:
    """Fetch Indian market indices and NSE/BSE stock quotes via Yahoo Finance."""

    YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

    def __init__(self, timeout_seconds: int = 6):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def _fetch_quotes(self, symbols: List[str]) -> List[Dict]:
        response = requests.get(
            self.YAHOO_QUOTE_URL,
            params={"symbols": ",".join(symbols)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return ((payload.get("quoteResponse") or {}).get("result")) or []

    def _normalize_indian_symbol(self, raw: str) -> str:
        symbol = (raw or "").strip().upper()
        if not symbol:
            return ""
        if symbol.startswith("^"):
            return symbol
        if "." in symbol:
            return symbol
        # Default to NSE if not specified.
        return f"{symbol}.NS"

    def _to_row(self, quote: Dict) -> Dict:
        ts = quote.get("regularMarketTime")
        market_time = ""
        if isinstance(ts, (int, float)):
            market_time = datetime.fromtimestamp(ts).astimezone().isoformat()
        return {
            "symbol": quote.get("symbol"),
            "name": quote.get("longName") or quote.get("shortName") or quote.get("symbol"),
            "price": quote.get("regularMarketPrice"),
            "change": quote.get("regularMarketChange"),
            "change_percent": quote.get("regularMarketChangePercent"),
            "day_high": quote.get("regularMarketDayHigh"),
            "day_low": quote.get("regularMarketDayLow"),
            "market_time": market_time,
            "currency": quote.get("currency"),
            "exchange": quote.get("fullExchangeName") or quote.get("exchange"),
        }

    def execute(self, params: Dict) -> Dict:
        symbol = self._normalize_indian_symbol(str(params.get("symbol", "")))

        try:
            if symbol:
                results = self._fetch_quotes([symbol])
                if not results and symbol.endswith(".NS"):
                    # Fallback to BSE if NSE symbol not found
                    bse_symbol = symbol[:-3] + ".BO"
                    results = self._fetch_quotes([bse_symbol])
                if not results:
                    return {"ok": False, "error": "No quote found for symbol.", "symbol": symbol}
                return {
                    "ok": True,
                    "mode": "symbol",
                    "symbol": symbol,
                    "quote": self._to_row(results[0]),
                    "source": "yahoo_finance",
                    "fetched_at": datetime.now().astimezone().isoformat(),
                }

            index_symbols = ["^NSEI", "^BSESN", "^NSEBANK"]
            results = self._fetch_quotes(index_symbols)
            rows = [self._to_row(r) for r in results]
            return {
                "ok": True,
                "mode": "overview",
                "indices": rows,
                "source": "yahoo_finance",
                "fetched_at": datetime.now().astimezone().isoformat(),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "symbol": symbol}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid market response: {exc}", "symbol": symbol}
