"""
Reactive Python executor.

Lifecycle:
spawn -> execute -> return -> terminate
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Dict

from core.config import BASE_DIR


class PythonExecutor:
    """Execute short Python snippets with strict limits."""

    FORBIDDEN_TOKENS = (
        "import os",
        "import subprocess",
        "import socket",
        "import pty",
        "import signal",
        "__import__",
        "open(",
        "eval(",
        "exec(",
    )

    def __init__(self, timeout_seconds: int = 5, max_code_chars: int = 2000):
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_code_chars = max(200, int(max_code_chars))

    def _validate_code(self, code: str) -> str:
        if not code or not code.strip():
            raise ValueError("No Python code provided.")
        if len(code) > self.max_code_chars:
            raise ValueError(f"Code exceeds max length ({self.max_code_chars} chars).")
        lowered = code.lower()
        for token in self.FORBIDDEN_TOKENS:
            if token in lowered:
                raise ValueError(f"Blocked unsafe token: {token}")
        return code

    def execute(self, params: Dict) -> Dict:
        try:
            code = self._validate_code(str(params.get("code", "")))
        except Exception as exc:
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}
        process = subprocess.Popen(
            ["python3", "-I", "-S", "-c", code],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            stdout, stderr = process.communicate()
            return {
                "ok": False,
                "error": f"Execution timed out after {self.timeout_seconds}s",
                "stdout": (stdout or "")[:2000],
                "stderr": (stderr or "")[:2000],
            }

        return {
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": (stdout or "")[:4000],
            "stderr": (stderr or "")[:4000],
        }
