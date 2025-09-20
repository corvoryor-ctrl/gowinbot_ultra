# core/retry.py
# นโยบาย retry แบบรวมศูนย์ (exponential backoff + jitter)
from __future__ import annotations
import time, random, logging
from typing import Callable, TypeVar, Any, Iterable

T = TypeVar("T")

# จำแนกความผิดพลาดชั่วคราว (ปรับเพิ่มได้)
TRANSIENT_KEYWORDS = ("timeout", "temporarily", "temporarily unavailable", "connection reset", "closed", "unreachable")

def is_transient_error(err: BaseException) -> bool:
    s = str(err).lower()
    return any(k in s for k in TRANSIENT_KEYWORDS)

def jitter(base: float) -> float:
    return base * (0.7 + random.random() * 0.6)  # 0.7x - 1.3x

def retry_sync(fn: Callable[..., T], *args: Any, retries: int = 5, backoff: float = 0.5, max_backoff: float = 8.0, swallow: bool = False, **kwargs: Any) -> T:
    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt >= retries or (not is_transient_error(e)):
                if swallow:
                    logging.warning("[retry_sync] giving up after %d attempts: %s", attempt, e)
                    raise
                else:
                    raise
            logging.warning("[retry_sync] attempt %d/%d failed: %s (retry in %.2fs)", attempt, retries, e, delay)
            time.sleep(jitter(delay))
            delay = min(max_backoff, delay * 2.0)

__all__ = ["retry_sync", "is_transient_error"]
