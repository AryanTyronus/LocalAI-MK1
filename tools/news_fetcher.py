"""
Reactive news fetcher.

Lifecycle:
spawn request -> fetch latest headlines -> return -> terminate
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


class NewsFetcher:
    """Fetch latest headlines from Google News RSS."""

    def __init__(self, timeout_seconds: int = 6, max_items: int = 5):
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_items = max(1, int(max_items))

    def _build_url(self, topic: str) -> str:
        cleaned = (topic or "").strip()
        if not cleaned:
            return "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        return f"https://news.google.com/rss/search?q={quote_plus(cleaned)}&hl=en-US&gl=US&ceid=US:en"

    def _parse_items(self, xml_text: str, limit: int) -> List[Dict]:
        root = ET.fromstring(xml_text)
        items: List[Dict] = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = ""
            source_node = item.find("source")
            if source_node is not None and source_node.text:
                source = source_node.text.strip()
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published_at": pub_date,
                    "source": source,
                }
            )
        return items

    def execute(self, params: Dict) -> Dict:
        topic = str(params.get("topic", "")).strip()
        limit_raw = params.get("limit", self.max_items)
        try:
            limit = max(1, min(int(limit_raw), 10))
        except Exception:
            limit = self.max_items

        url = self._build_url(topic)
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            headlines = self._parse_items(response.text, limit=limit)
            return {
                "ok": True,
                "topic": topic or "top headlines",
                "count": len(headlines),
                "headlines": headlines,
                "fetched_at": datetime.now().astimezone().isoformat(),
                "source": "google_news_rss",
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "topic": topic}
        except ET.ParseError as exc:
            return {"ok": False, "error": f"RSS parse error: {exc}", "topic": topic}
