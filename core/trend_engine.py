# core/trend_engine.py
# Simple H1/H4 trend/bias calculator with caching.
from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional

def _slope(values: List[float]) -> float:
    # simple linear slope proxy
    n = len(values)
    if n < 2: return 0.0
    # normalize x as 0..n-1
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n)) or 1.0
    return num / den

def _trend_from_slope(s: float, eps: float = 0.0) -> str:
    # eps threshold can be tuned
    th = eps if eps > 0.0 else 0.0
    if s > th: return "up"
    if s < -th: return "down"
    return "flat"

class TrendEngine:
    """
    Pulls H1/H4 candles via the provided candle_provider and computes a basic trend/bias.
    candle_provider(asset: str, tf_minutes: int, count: int) -> list of dicts with 'close' (float)
    """
    def __init__(self, candle_provider, cache_ttl_sec: int = 120):
        self._get_candles = candle_provider
        self.cache_ttl = max(30, int(cache_ttl_sec))
        self._cache: Dict[str, Dict[str, Tuple[float, dict]]] = {}
        # structure: {asset: {"H1": (ts, info), "H4": (ts, info)}}

    def _compute_for_tf(self, asset: str, tf: int, count: int = 20, eps: float = 0.0) -> dict:
        candles = self._get_candles(asset, tf, count) or []
        closes = [c.get("close") for c in candles if "close" in c]
        if len(closes) < 4:
            return {"trend": "flat", "slope": 0.0, "sample": len(closes)}
        s = _slope(closes)
        trend = _trend_from_slope(s, eps)
        return {"trend": trend, "slope": s, "sample": len(closes)}

    def refresh(self, assets: List[str]) -> None:
        now = time.time()
        for a in assets:
            entry = self._cache.get(a, {})
            # refresh H1/H4 if stale
            def need(t): return (t is None) or (now - t > self.cache_ttl)
            h1_t = entry.get("H1", (None, {}))[0]
            h4_t = entry.get("H4", (None, {}))[0]

            if need(h1_t):
                info = self._compute_for_tf(a, 60, 30, eps=0.0)
                entry["H1"] = (now, info)
            if need(h4_t):
                info = self._compute_for_tf(a, 240, 30, eps=0.0)
                entry["H4"] = (now, info)

            self._cache[a] = entry

    def get_bias(self, asset: str) -> dict:
        """
        Returns: {"H1": {"trend": up/down/flat, "slope": float, "sample": int},
                  "H4": {...}}
        Missing assets return "flat" defaults.
        """
        entry = self._cache.get(asset, {})
        def default(): return {"trend":"flat","slope":0.0,"sample":0}
        h1 = entry.get("H1", (None, default()))[1]
        h4 = entry.get("H4", (None, default()))[1]
        return {"H1": h1, "H4": h4}

    @staticmethod
    def pass_filter(bias: dict, direction: str, mode: str = "weak") -> bool:
        """
        direction: "call" (up) or "put" (down)
        mode:
          - "off": always True
          - "weak": need at least one TF aligned or flat; hard block only if both TF strongly opposite
          - "strict": both TF must align (or one align & other flat)
        """
        if mode == "off":
            return True
        want = "up" if direction == "call" else "down"
        h1 = bias.get("H1", {}).get("trend", "flat")
        h4 = bias.get("H4", {}).get("trend", "flat")

        if mode == "weak":
            # block only clear contradiction: both opposite
            if h1 not in (want, "flat") and h4 not in (want, "flat"):
                return False
            return True
        # strict
        # require both align or at least one align and the other flat
        ok = ((h1 == want and h4 in (want, "flat")) or (h4 == want and h1 in (want, "flat")))
        return ok
