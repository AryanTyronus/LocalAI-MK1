"""System diagnostics snapshot tool."""

from __future__ import annotations

import os
import platform
import sys
from typing import Dict

try:
    import psutil
except Exception:
    psutil = None


class SystemDiagnosticsTool:
    def __init__(self, model_status_getter=None):
        self._model_status_getter = model_status_getter

    def execute(self, _params: Dict) -> Dict:
        cpu = float(psutil.cpu_percent(interval=None)) if psutil else 0.0
        ram = float(psutil.virtual_memory().percent) if psutil else 0.0
        return {
            "ok": True,
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_percent": cpu,
            "ram_percent": ram,
            "cwd": os.getcwd(),
            "model_status": self._model_status_getter() if callable(self._model_status_getter) else "unknown",
        }
