"""
Reactive weather fetcher.

Lifecycle:
spawn request -> fetch current weather -> return -> terminate
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict
from urllib.parse import quote_plus

import requests


class WeatherFetcher:
    """Fetch weather updates for a location using wttr.in JSON endpoint."""

    def __init__(self, timeout_seconds: int = 6):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def execute(self, params: Dict) -> Dict:
        location = str(params.get("location", "")).strip()
        if not location:
            return {"ok": False, "error": "Location is required."}

        url = f"https://wttr.in/{quote_plus(location)}?format=j1"
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()

            current = (payload.get("current_condition") or [{}])[0]
            nearest = (payload.get("nearest_area") or [{}])[0]
            area_name = ((nearest.get("areaName") or [{}])[0]).get("value") or location
            country = ((nearest.get("country") or [{}])[0]).get("value") or ""
            region = ((nearest.get("region") or [{}])[0]).get("value") or ""

            weather_desc = ((current.get("weatherDesc") or [{}])[0]).get("value") or "Unknown"
            temp_c = current.get("temp_C")
            feels_c = current.get("FeelsLikeC")
            humidity = current.get("humidity")
            wind_kmph = current.get("windspeedKmph")

            return {
                "ok": True,
                "location": area_name,
                "region": region,
                "country": country,
                "condition": weather_desc,
                "temp_c": temp_c,
                "feels_like_c": feels_c,
                "humidity_pct": humidity,
                "wind_kmph": wind_kmph,
                "fetched_at": datetime.now().astimezone().isoformat(),
                "source": "wttr.in",
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "location": location}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid weather response: {exc}", "location": location}
