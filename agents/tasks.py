"""
Background agent tasks.
"""

from __future__ import annotations

import os
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

from core.config import BASE_DIR
from core.logger import logger


def system_monitor_task() -> None:
    cpu = float(psutil.cpu_percent(interval=None)) if psutil else 0.0
    ram = float(psutil.virtual_memory().percent) if psutil else 0.0
    logger.info(f"[Agent] System monitor cpu={cpu:.1f}% ram={ram:.1f}%")


def daily_summary_task() -> None:
    now = datetime.now().astimezone().isoformat()
    logger.info(f"[Agent] Daily summary heartbeat at {now}")


def repository_monitor_task() -> None:
    count = 0
    for root, _dirs, files in os.walk(BASE_DIR):
        count += len(files)
    logger.info(f"[Agent] Repository monitor files={count}")

