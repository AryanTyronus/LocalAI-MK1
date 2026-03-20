"""Guarded application launcher for local desktop automation."""

from __future__ import annotations

import platform
import subprocess
from typing import Dict


class OpenAppTool:
    def __init__(self, timeout_seconds: int = 5):
        self._timeout = max(1, int(timeout_seconds))

    def execute(self, params: Dict) -> Dict:
        app_name = str((params or {}).get("app_name", "")).strip()
        if not app_name:
            return {"ok": False, "error": "Missing app_name"}

        if platform.system().lower() != "darwin":
            return {"ok": False, "error": "open_app is currently supported only on macOS"}

        try:
            proc = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "app_name": app_name,
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[:800],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "app_name": app_name}
