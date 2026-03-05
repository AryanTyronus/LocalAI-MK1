"""
Capability registry for safe automation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Capability:
    name: str
    tool_name: str
    allow_path: bool = False
    dangerous: bool = False


class CapabilityRegistry:
    def __init__(self):
        self._caps: Dict[str, Capability] = {
            "search_web": Capability("search_web", "news_fetcher"),
            "read_file": Capability("read_file", "file_reader", allow_path=True),
            "write_file": Capability("write_file", "write_file", allow_path=True, dangerous=True),
            "run_tests": Capability("run_tests", "python_executor", dangerous=True),
            "get_system_metrics": Capability("get_system_metrics", "current_affairs_fetcher"),
            "get_stock": Capability("get_stock", "stock_fetcher"),
            "get_news": Capability("get_news", "news_fetcher"),
            "get_weather": Capability("get_weather", "weather_fetcher"),
            "get_indian_market": Capability("get_indian_market", "indian_market_fetcher"),
            "get_person_info": Capability("get_person_info", "person_lookup_fetcher"),
            "run_code": Capability("run_code", "python_executor", dangerous=True),
        }

    def resolve_from_tool(self, tool_name: str) -> Capability | None:
        for cap in self._caps.values():
            if cap.tool_name == tool_name:
                return cap
        return None

