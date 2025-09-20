# core/time_sync.py
# Lightweight time synchronization utilities (no external deps)
# Safe drop-in: no imports to project-internal modules.

from __future__ import annotations
import time
import datetime as _dt
from typing import Callable, Optional

class TimeSync:
    """
    Provides time utilities aligned to broker/server time if available.
    If no provider is given, uses local system time (UTC).
    """
    def __init__(self, server_time_provider: Optional[Callable[[], float]] = None):
        """
        server_time_provider: a callable returning current epoch seconds (float)
        in UTC based on the broker/server clock, or None to use local time.
        """
        self._server_time_provider = server_time_provider

    def now_epoch(self) -> float:
        if self._server_time_provider is not None:
            try:
                return float(self._server_time_provider())
            except Exception:
                # Fallback gracefully to local time if provider fails
                pass
        return time.time()

    def now_utc(self) -> _dt.datetime:
        return _dt.datetime.utcfromtimestamp(self.now_epoch()).replace(tzinfo=_dt.timezone.utc)

    def seconds_to_next_minute(self) -> float:
        t = self.now_epoch()
        return 60.0 - (t % 60.0)

    def seconds_to_next_5m_slot(self) -> float:
        """
        Seconds until next slot where minute % 5 == 0 and seconds == 0.
        """
        t = self.now_epoch()
        # compute seconds into current hour
        sec_into_hour = int(t) % 3600
        # next minute boundary
        rem = 60 - (sec_into_hour % 60)
        # move to next minute boundary and find first minute divisible by 5
        # conservative loop (max 5 iterations)
        for i in range(6):
            candidate = t + rem + i * 60
            dt = _dt.datetime.utcfromtimestamp(candidate)
            if dt.minute % 5 == 0 and dt.second == 0:
                return candidate - t
        # fallback
        return rem

    def on_top_of_minute_window(self, lead_time_sec: float) -> bool:
        """
        Returns True when we are within [0, lead_time_sec] seconds *before* the top of the next minute.
        Useful to prepare sending orders slightly before :00.
        """
        rem = self.seconds_to_next_minute()
        return rem <= lead_time_sec

    def sleep_until_top_of_minute(self, lead_time_sec: float = 0.0) -> None:
        """
        Sleeps so that we wake up `lead_time_sec` seconds *before* the next minute boundary.
        lead_time_sec >= 0. If lead_time_sec is 0 -> wake exactly at :00 boundary.
        """
        rem = self.seconds_to_next_minute() - lead_time_sec
        if rem > 0:
            time.sleep(rem)

    def sleep_until_next_5m_slot(self, lead_time_sec: float = 0.0) -> None:
        """
        Sleeps until `lead_time_sec` seconds before the next :00/:05/:10/... boundary.
        """
        rem = self.seconds_to_next_5m_slot() - lead_time_sec
        if rem > 0:
            time.sleep(rem)
