"""
Lightweight background task scheduler.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List

from core.logger import logger


@dataclass
class ScheduledTask:
    name: str
    interval_seconds: int
    func: Callable[[], None]
    last_run: float = 0.0


class AgentScheduler:
    def __init__(self):
        self._tasks: List[ScheduledTask] = []
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._started = False

    def add_task(self, name: str, interval_seconds: int, func: Callable[[], None]) -> None:
        with self._lock:
            self._tasks.append(
                ScheduledTask(
                    name=name,
                    interval_seconds=max(5, int(interval_seconds)),
                    func=func,
                )
            )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread = threading.Thread(target=self._run_loop, name="agent-scheduler", daemon=True)
        self._thread.start()
        logger.info("AgentScheduler started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("AgentScheduler stopped")

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                tasks = list(self._tasks)
            for task in tasks:
                if now - task.last_run < task.interval_seconds:
                    continue
                try:
                    task.func()
                except Exception as exc:
                    logger.warning(f"[Agent] task '{task.name}' failed: {exc}")
                finally:
                    task.last_run = now
            time.sleep(1.0)

