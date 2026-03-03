"""
Reactive Indian stock market fetcher.

Lifecycle:
spawn request -> fetch market data -> return -> terminate
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
import time

import requests


class IndianMarketFetcher:
    """Fetch Indian market indices and NSE/BSE stock quotes via Yahoo Finance."""

    YAHOO_QUOTE_URLS = [
        "https://query1.finance.yahoo.com/v7/finance/quote",
        "https://query2.finance.yahoo.com/v7/finance/quote",
    ]
    YAHOO_CHART_URLS = [
        "https://query1.finance.yahoo.com/v8/finance/chart",
        "https://query2.finance.yahoo.com/v8/finance/chart",
    ]

    def __init__(self, timeout_seconds: int = 6, max_retries: int = 2):
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

    def _fetch_quotes(self, symbols: List[str]) -> List[Dict]:
        last_error: Exception | None = None
        params = {"symbols": ",".join(symbols)}

        for url in self.YAHOO_QUOTE_URLS:
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(
                        url,
                        params=params,
                        headers=self._headers,
                        timeout=self.timeout_seconds,
                    )

                    if response.status_code == 429:
                        retry_after_raw = (response.headers or {}).get("Retry-After", "").strip()
                        if retry_after_raw.isdigit():
                            wait_seconds = min(5, max(1, int(retry_after_raw)))
                        else:
                            wait_seconds = min(5, 2 ** attempt)
                        if attempt < self.max_retries:
                            time.sleep(wait_seconds)
                            continue
                        response.raise_for_status()

                    response.raise_for_status()
                    payload = response.json()
                    return ((payload.get("quoteResponse") or {}).get("result")) or []
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(min(5, 2 ** attempt))
                        continue
                    break
                except ValueError as exc:
                    # Invalid JSON / payload decoding should stop retry loop for this URL.
                    last_error = exc
                    break

        if last_error:
            raise last_error
        return []

    def _fetch_quote_via_chart(self, symbol: str) -> Dict:
        """
        Fallback for single-symbol fetch when quote endpoint is rate-limited.
        """
        params = {"interval": "1d", "range": "1d"}
        last_error: Exception | None = None

        for base_url in self.YAHOO_CHART_URLS:
            url = f"{base_url}/{symbol}"
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(
                        url,
                        params=params,
                        headers=self._headers,
                        timeout=self.timeout_seconds,
                    )
                    if response.status_code == 429:
                        if attempt < self.max_retries:
                            time.sleep(min(5, 2 ** attempt))
                            continue
                        response.raise_for_status()

                    response.raise_for_status()
                    payload = response.json()
                    result = (((payload.get("chart") or {}).get("result")) or [None])[0]
                    if not isinstance(result, dict):
                        raise ValueError("Chart response missing result data.")

                    meta = result.get("meta") or {}
                    if not isinstance(meta, dict):
                        raise ValueError("Chart response meta is invalid.")

                    regular_price = meta.get("regularMarketPrice")
                    previous_close = meta.get("previousClose")
                    change_value = None
                    try:
                        if regular_price is not None and previous_close is not None:
                            change_value = float(regular_price) - float(previous_close)
                    except Exception:
                        change_value = None

                    return {
                        "symbol": meta.get("symbol") or symbol,
                        "longName": meta.get("longName"),
                        "shortName": meta.get("shortName"),
                        "regularMarketPrice": regular_price,
                        "regularMarketChange": change_value,
                        "regularMarketChangePercent": meta.get("regularMarketChangePercent"),
                        "regularMarketDayHigh": meta.get("regularMarketDayHigh"),
                        "regularMarketDayLow": meta.get("regularMarketDayLow"),
                        "regularMarketTime": meta.get("regularMarketTime"),
                        "currency": meta.get("currency"),
                        "fullExchangeName": meta.get("exchangeName"),
                        "exchange": meta.get("exchangeName"),
                    }
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(min(5, 2 ** attempt))
                        continue
                    break
                except ValueError as exc:
                    last_error = exc
                    break

        if last_error:
            raise last_error
        raise ValueError("No chart data returned.")

    def _normalize_indian_symbol(self, raw: str) -> str:
        symbol = (raw or "").strip().upper()
        if not symbol:
            return ""
        alias_to_index = {
            "NIFTY": "^NSEI",
            "NIFTY50": "^NSEI",
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK",
            "NIFTYBANK": "^NSEBANK",
            "NIFTY BANK": "^NSEBANK",
        }
        if symbol in alias_to_index:
            return alias_to_index[symbol]
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
                    # Fallback for persistent rate limits on quote endpoint.
                    try:
                        chart_quote = self._fetch_quote_via_chart(symbol)
                        results = [chart_quote]
                    except Exception:
                        pass
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
            msg = str(exc)
            if "429" in msg:
                msg = "Rate-limited by Yahoo Finance. Please retry in 30-60 seconds."
            return {"ok": False, "error": msg, "symbol": symbol}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid market response: {exc}", "symbol": symbol}
