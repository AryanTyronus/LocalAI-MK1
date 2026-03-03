"""
Reactive person lookup fetcher.

Lifecycle:
spawn request -> fetch live person summary -> return -> terminate
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict
from urllib.parse import quote

import requests


class PersonLookupFetcher:
    """Fetch live person summaries from Wikipedia."""

    WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
    REQUEST_HEADERS = {
        # Wikipedia can reject generic python-requests clients without a descriptive UA.
        "User-Agent": "LocalAI/1.0 (local assistant; contact: local@localhost)",
        "Accept": "application/json",
    }

    def __init__(self, timeout_seconds: int = 6):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def _clean_query(self, raw: str) -> str:
        text = (raw or "").strip()
        text = re.sub(r"^(who\s+is|tell\s+me\s+about|information\s+on|info\s+on)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(from\s+the\s+internet|from\s+internet|online|latest)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[?!.]+$", "", text).strip()
        return text

    def execute(self, params: Dict) -> Dict:
        raw_query = str(params.get("query", "")).strip()
        if not raw_query:
            return {"ok": False, "error": "Query is required."}

        person = self._clean_query(raw_query)
        if not person:
            return {"ok": False, "error": "Could not extract a person name from the query.", "query": raw_query}

        url = self.WIKI_SUMMARY_URL + quote(person.replace(" ", "_"))
        try:
            response = requests.get(url, headers=self.REQUEST_HEADERS, timeout=self.timeout_seconds)
            if response.status_code == 404:
                return {
                    "ok": False,
                    "error": f"No Wikipedia summary found for '{person}'.",
                    "query": raw_query,
                }
            if response.status_code == 403:
                return self._fallback_action_api(raw_query, person)
            response.raise_for_status()
            payload = response.json()

            summary = str(payload.get("extract", "")).strip()
            title = str(payload.get("title", person)).strip() or person
            source = (((payload.get("content_urls") or {}).get("desktop") or {}).get("page")) or ""
            page_type = str(payload.get("type", "")).strip().lower()

            if not summary:
                return {
                    "ok": False,
                    "error": f"Live source returned no summary for '{title}'.",
                    "query": raw_query,
                }

            return {
                "ok": True,
                "query": raw_query,
                "person": title,
                "summary": summary,
                "page_type": page_type,
                "source": source,
                "fetched_at": datetime.now().astimezone().isoformat(),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "query": raw_query}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid person lookup response: {exc}", "query": raw_query}

    def _fallback_action_api(self, raw_query: str, person: str) -> Dict:
        """Fallback when REST summary endpoint is blocked."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "inprop": "url",
            "titles": person,
        }
        try:
            response = requests.get(
                self.WIKI_ACTION_API,
                params=params,
                headers=self.REQUEST_HEADERS,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json() or {}
            pages = ((payload.get("query") or {}).get("pages") or {})
            if not pages:
                return {"ok": False, "error": f"No Wikipedia page found for '{person}'.", "query": raw_query}

            page = next(iter(pages.values()))
            page_id = page.get("pageid")
            if str(page_id) == "-1":
                return {"ok": False, "error": f"No Wikipedia page found for '{person}'.", "query": raw_query}

            title = str(page.get("title", person)).strip() or person
            summary = str(page.get("extract", "")).strip()
            source = str(page.get("fullurl", "")).strip() or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if not summary:
                return {"ok": False, "error": f"Live source returned no summary for '{title}'.", "query": raw_query}

            return {
                "ok": True,
                "query": raw_query,
                "person": title,
                "summary": summary,
                "page_type": "standard",
                "source": source,
                "fetched_at": datetime.now().astimezone().isoformat(),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "query": raw_query}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid fallback response: {exc}", "query": raw_query}
