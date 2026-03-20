"""Guarded shell command execution for low-risk diagnostics."""

from __future__ import annotations

import os
import subprocess
from typing import Dict

from core.config import Config


class SafeShellTool:
    def __init__(self, timeout_seconds: int = 3):
        cfg = Config()
        self._enabled = cfg.safe_shell_enabled
        self._timeout = timeout_seconds or cfg.safe_shell_timeout_seconds
        self._blocked = {
            "rm -rf",
            "sudo",
            "chmod 777",
            "chown",
            "curl ",
            "wget ",
            "pip install",
            "brew install",
            "apt-get",
            "yum ",
            "dnf ",
            "ssh ",
            "scp ",
            "docker",
            "kubectl",
        }

    def execute(self, params: Dict) -> Dict:
        if not self._enabled:
            return {"ok": False, "error": "Safe shell tool is disabled"}

        command = str((params or {}).get("command", "")).strip()
        if not command:
            return {"ok": False, "error": "Missing command"}

        lowered = command.lower()
        if any(token in lowered for token in self._blocked):
            return {"ok": False, "error": "Blocked command pattern"}
        if any(op in command for op in ["&&", "||", ";", "`", "$(", "|", ">", "<"]):
            return {"ok": False, "error": "Shell operators are not allowed"}

        try:
            proc = subprocess.run(
                command.split(),
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:1200],
                "stderr": (proc.stderr or "")[:1200],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
