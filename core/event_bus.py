# core/event_bus.py
# Emits time-based events for orchestrating 1m/5m logic without touching UI routines.
from __future__ import annotations
import threading
import time
from typing import Callable, List, Optional
from .time_sync import TimeSync

Callback = Callable[[float], None]  # epoch seconds (UTC)

class EventBus(threading.Thread):
    """
    A simple event loop that emits:
      - on_second_tick
      - on_top_of_minute (:00)
      - on_minute_close (when minute completes)
      - on_5min_close (when a 5-minute candle completes)
    Consumers can register callbacks. Thread-safe and stoppable.
    """
    def __init__(self, time_sync: TimeSync, sleep_resolution: float = 0.2):
        super().__init__(daemon=True)
        self.ts = time_sync
        self.sleep_resolution = max(0.05, float(sleep_resolution))
        self._stop = threading.Event()

        self._on_second: List[Callback] = []
        self._on_top_of_minute: List[Callback] = []
        self._on_minute_close: List[Callback] = []
        self._on_5min_close: List[Callback] = []

        # state
        self._last_minute = None
        self._last_5m_key = None
        self._last_second = None

    # registration
    def on_second(self, cb: Callback): self._on_second.append(cb)
    def on_top_of_minute(self, cb: Callback): self._on_top_of_minute.append(cb)
    def on_minute_close(self, cb: Callback): self._on_minute_close.append(cb)
    def on_5min_close(self, cb: Callback): self._on_5min_close.append(cb)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            now = self.ts.now_epoch()
            # per-second callbacks
            sec = int(now)
            if sec != self._last_second:
                self._last_second = sec
                for cb in self._on_second:
                    self._safe(cb, now)

            # detect top-of-minute (:00)
            rem = now % 60.0
            # consider top if within first 0.30s of a minute to be robust
            if rem < 0.30:
                # minute value
                m = int(now // 60)
                if m != self._last_minute:
                    # emit top-of-minute first
                    for cb in self._on_top_of_minute:
                        self._safe(cb, now)
                    # emit minute close (previous minute has just closed)
                    for cb in self._on_minute_close:
                        self._safe(cb, now)
                    self._last_minute = m

                    # emit 5m close if boundary
                    if (m % 5) == 0:
                        # create a unique key per 5m slot to avoid double-fire
                        key = int(now // 300)
                        if key != self._last_5m_key:
                            for cb in self._on_5min_close:
                                self._safe(cb, now)
                            self._last_5m_key = key

            time.sleep(self.sleep_resolution)

    def _safe(self, cb: Callback, now: float):
        try:
            cb(now)
        except Exception:
            # swallow to keep the bus alive; logging should be added by integrator
            pass
