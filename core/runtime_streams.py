"""Runtime event/state and telemetry broadcasters."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Dict, Iterator, Optional


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


@dataclass
class RuntimeStateMachine:
    state: str = "idle"

    def transition(self, next_state: str) -> Dict[str, str]:
        self.state = next_state
        return {
            "type": "state",
            "timestamp": _iso_now(),
            "payload": {"state": self.state},
        }


class EventBroadcaster:
    def __init__(self, max_queue_size: int = 256):
        self._subs: Dict[int, Queue] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._max_queue_size = max_queue_size

    def subscribe(self) -> int:
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._subs[sid] = Queue(maxsize=self._max_queue_size)
            return sid

    def unsubscribe(self, sid: int) -> None:
        with self._lock:
            self._subs.pop(sid, None)

    def publish(self, event: Dict) -> None:
        serialized = json.dumps(event, ensure_ascii=True)
        with self._lock:
            items = list(self._subs.items())
        for _sid, q in items:
            try:
                q.put_nowait(serialized)
            except Full:
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(serialized)
                except Exception:
                    pass

    def iter_events(self, sid: int, heartbeat_seconds: float = 8.0) -> Iterator[str]:
        while True:
            q = self._subs.get(sid)
            if q is None:
                return
            try:
                payload = q.get(timeout=heartbeat_seconds)
                yield payload
            except Empty:
                yield json.dumps({"type": "heartbeat", "timestamp": _iso_now(), "payload": {}}, ensure_ascii=True)


class TelemetryWindow:
    def __init__(self, max_points: int = 200):
        self._points = deque(maxlen=max_points)
        self._lock = threading.Lock()

    def add(self, point: Dict) -> None:
        with self._lock:
            row = dict(point)
            row.setdefault("timestamp", _iso_now())
            self._points.append(row)

    def latest(self) -> Optional[Dict]:
        with self._lock:
            if not self._points:
                return None
            return dict(self._points[-1])


runtime_events = EventBroadcaster()
telemetry_events = EventBroadcaster()
tool_events = EventBroadcaster()
state_machine = RuntimeStateMachine()
telemetry_window = TelemetryWindow()
