"""
Reactive current-affairs fetcher.

Lifecycle:
spawn request -> fetch live fact -> return -> terminate
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict
from urllib.parse import quote

import requests


class CurrentAffairsFetcher:
    """Fetch live current-affairs facts from Wikipedia summaries."""

    WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, timeout_seconds: int = 6):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def _is_us_president_query(self, query: str) -> bool:
        q = (query or "").strip().lower()
        return bool(
            re.search(
                r"\b(who\s+is\s+the\s+president\s+of\s+(america|the\s+united\s+states|usa)|president\s+of\s+america|president\s+of\s+the\s+united\s+states)\b",
                q,
            )
        )

    def _extract_current_president_name(self, text: str) -> str:
        patterns = [
            r"current president is ([A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+)+)",
            r"and current president, ([A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+)+)",
            r"president is ([A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+)+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "")
            if match:
                return match.group(1).strip()
        return ""

    def execute(self, params: Dict) -> Dict:
        query = str(params.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "Query is required."}

        if not self._is_us_president_query(query):
            return {
                "ok": False,
                "error": "Unsupported current-affairs query. Try asking about the U.S. president.",
                "query": query,
            }

        page = "President_of_the_United_States"
        url = self.WIKI_SUMMARY_URL + quote(page)
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            extract = str(payload.get("extract", "")).strip()
            source = (((payload.get("content_urls") or {}).get("desktop") or {}).get("page")) or f"https://en.wikipedia.org/wiki/{page}"

            name = self._extract_current_president_name(extract)
            if name:
                fact = f"The current president of the United States is {name}."
            else:
                fact = extract[:320] if extract else "Live source did not return a usable summary."

            return {
                "ok": True,
                "query": query,
                "fact": fact,
                "source": source,
                "fetched_at": datetime.now().astimezone().isoformat(),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "query": query}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid current-affairs response: {exc}", "query": query}
