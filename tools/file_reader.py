"""
Safe file reader tool with workspace-only access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from core.config import BASE_DIR


class FileReader:
    """Read text files from project workspace with path controls."""

    def __init__(self, root: str = BASE_DIR, max_bytes: int = 20000):
        self.root = Path(root).resolve()
        self.max_bytes = max(1024, int(max_bytes))

    def _resolve_safe_path(self, filepath: str) -> Path:
        if not filepath:
            raise ValueError("No filepath provided.")
        candidate = (self.root / filepath).resolve() if not Path(filepath).is_absolute() else Path(filepath).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise PermissionError("Path is outside allowed workspace.")
        return candidate

    def execute(self, params: Dict) -> Dict:
        filepath = str(params.get("filepath", "")).strip()
        target = self._resolve_safe_path(filepath)

        if not target.exists():
            return {"ok": False, "error": f"File not found: {filepath}"}
        if target.is_dir():
            return {"ok": False, "error": "Target path is a directory."}

        size = target.stat().st_size
        if size > self.max_bytes:
            return {"ok": False, "error": f"File too large ({size} bytes > {self.max_bytes} bytes)."}

        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "filepath": str(target),
            "bytes": size,
            "content": content,
        }

