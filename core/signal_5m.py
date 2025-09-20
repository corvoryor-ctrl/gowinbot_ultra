# core/signal_5m.py
# Wraps existing 5m strategies behind a stable interface. Supports optional micro-filter with 1m.
from __future__ import annotations
from typing import Callable, Dict, Any, Optional, Tuple, List
from .signal_1m import Signal1m

Signal = Dict[str, Any]
StrategyFn = Callable[[str, list], Signal]
CandleProvider = Callable[[str, int, int], list]

# -------------------- small utils (robust like signal_1m) --------------------

def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _clamp(n: int, lo: int, hi: int) -> int:
    try:
        return max(int(lo), min(int(n), int(hi)))
    except Exception:
        return int(lo)

def _normalize_candles(rows: list) -> list:
    """Ensure each candle has ts, open, high, low, close (floats)."""
    out = []
    for r in (rows or []):
        try:
            if isinstance(r, dict):
                ts = int(r.get("ts") or r.get("time") or r.get("t") or 0)
                o = _f(r.get("open", r.get("o", r.get("Open", 0.0))))
                h = _f(r.get("high", r.get("h", r.get("High", o))))
                l = _f(r.get("low",  r.get("l", r.get("Low", o))))
                c = _f(r.get("close",r.get("c", r.get("Close", o))))
            elif isinstance(r, (list, tuple)) and len(r) >= 5:
                ts = int(r[0]); o = _f(r[1]); h = _f(r[2]); l = _f(r[3]); c = _f(r[4])
            else:
                continue
            if h < max(o, c): h = max(o, c)
            if l > min(o, c): l = min(o, c)
            out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
        except Exception:
            continue
    return out

def _clip01(v: Optional[float]) -> Optional[float]:
    if v is None: return None
    try:
        vv = float(v)
        if vv != vv:  # NaN
            return None
        return max(0.0, min(1.0, vv))
    except Exception:
        return None

def _parse_strategy_output(raw: Any) -> Tuple[str, float, float]:
    """
    Normalize any strategy output to (dir, conf, gap).
    dir ∈ {"call","put","none"}, conf,gap ∈ [0,1]
    """
    direction: str = "none"
    conf: Optional[float] = None
    gap: Optional[float] = None

    if isinstance(raw, str):
        lo = raw.lower()
        if lo in ("call", "put"):
            direction = lo
    elif isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            a, b = raw[0], raw[1]
            if isinstance(a, str) and a.lower() in ("call", "put"):
                direction = a.lower()
                conf = _clip01(b if isinstance(b, (int, float)) else None)
            elif isinstance(a, (int, float)):
                p_up = _clip01(a)
                if p_up is not None:
                    direction = "call" if p_up >= 0.5 else "put"
                    conf = p_up if direction == "call" else (1.0 - p_up)
                    gap = abs(2 * p_up - 1.0)
        elif len(raw) == 1 and isinstance(raw[0], (int, float)):
            p_up = _clip01(raw[0])
            if p_up is not None:
                direction = "call" if p_up >= 0.5 else "put"
                conf = p_up if direction == "call" else (1.0 - p_up)
                gap = abs(2 * p_up - 1.0)
    elif isinstance(raw, dict):
        s = raw.get("dir") or raw.get("signal") or raw.get("side")
        if isinstance(s, str) and s.lower() in ("call", "put"):
            direction = s.lower()
        p = raw.get("p_up") or raw.get("prob_up") or raw.get("p")
        if p is not None:
            p_up = _clip01(p)
            if p_up is not None:
                if direction not in ("call", "put"):
                    direction = "call" if p_up >= 0.5 else "put"
                conf = p_up if direction == "call" else (1.0 - p_up)
                gap = abs(2 * p_up - 1.0)
        if conf is None and raw.get("conf") is not None:
            conf = _clip01(raw.get("conf"))
        if gap is None and raw.get("gap") is not None:
            gap = _clip01(raw.get("gap"))
    elif isinstance(raw, (int, float)):
        p_up = _clip01(raw)
        if p_up is not None:
            direction = "call" if p_up >= 0.5 else "put"
            conf = p_up if direction == "call" else (1.0 - p_up)
            gap = abs(2 * p_up - 1.0)

    direction = direction if direction in ("call", "put") else "none"
    conf = _clip01(conf) if conf is not None else 0.0
    gap  = _clip01(gap)  if gap  is not None else 0.0
    return direction, float(conf), float(gap)

# ---- dynamic min lookback (read indicator config with buffer) ----

def _bars_needed_for_indicator(name: str, params: Dict[str, Any]) -> int:
    name = (name or "").lower()
    p = params or {}
    def clamp(x, lo=1, hi=2000):
        try: return max(int(lo), min(int(x), int(hi)))
        except Exception: return int(lo)
    if name in ("ma", "moving_average"):
        length = int(p.get("length") or p.get("period") or 50)
        ma_type = str(p.get("type", "EMA")).upper()
        return clamp(length * (3 if ma_type == "EMA" else 1) + 10)
    if name == "macd":
        fast = int(p.get("fast", 12)); slow = int(p.get("slow", 26)); signal = int(p.get("signal", 9))
        return clamp(slow + signal + slow * 3)
    if name in ("ichimoku", "ichimoku_cloud"):
        t = int(p.get("tenkan", 9)); k = int(p.get("kijun", 26)); b = int(p.get("senkou_b", p.get("span_b", 52)))
        return clamp(max(b, k, t) + 26)
    if name == "rsi":
        n = int(p.get("length", 14)); return clamp(n + 5)
    if name in ("stochastic", "stoch", "stochastic_oscillator"):
        klen = int(p.get("k", 14)); dlen = int(p.get("d", 3)); return clamp(klen + dlen + 3)
    if name in ("bb", "bollinger", "bollinger_bands"):
        n = int(p.get("length", 20)); return clamp(n + 5)
    if name in ("atr", "average_true_range"):
        n = int(p.get("length", 14)); return clamp(n + 6)
    if name in ("obv", "on_balance_volume"):
        return clamp(120)
    if name in ("volume_profile", "vp", "vprofile"):
        win = int(p.get("window", p.get("period", 150))); return clamp(win)
    if name in ("price_action", "pa"):
        lb = int(p.get("lookback", 5)); return clamp(lb + 3)
    return 60

def _dynamic_min_lookback(tf_minutes: int) -> int:
    """Infer minimal bars from indicator configuration for the given timeframe."""
    try:
        from core.indicator_config import get_indicator_config  # type: ignore
        cfg = get_indicator_config(int(tf_minutes)) or {}
        ind = cfg.get("indicators") or cfg.get("custom") or cfg or {}
        needs: List[int] = []
        for k, v in ind.items():
            if isinstance(v, dict) and v.get("enabled"):
                needs.append(_bars_needed_for_indicator(k, v))
        if needs:
            req = int(max(needs) * 1.25) + 5
            return _clamp(req, 60, 1000)
    except Exception:
        pass
    return 120 if int(tf_minutes) == 1 else 60

# =============================================================================
# Signal5m
# =============================================================================

class Signal5m:
    """
    A thin adapter for 5m primary signals with optional 1m micro-filter.
    - candle_provider(asset, tf_minutes, count) -> candles
    - strategy_5m(asset, candles_5m) -> base signal (flexible shape)
    - (optional) signal_1m: Signal1m instance for micro confirmation
    """
    def __init__(self, candle_provider: CandleProvider, strategy_5m: StrategyFn, signal_1m: Signal1m | None = None):
        self._get_candles = candle_provider
        self._strategy_5m = strategy_5m
        self._sig1m = signal_1m
        # dynamic minima from indicator configs
        self._min5 = _dynamic_min_lookback(5)
        self._min1 = _dynamic_min_lookback(1)

    def check(self, asset: str, enable_micro_filter: bool = True, lookback_5m: int = 60, lookback_1m: int = 100) -> Signal:
        # Resolve effective lookbacks with dynamic minima
        lb5 = _clamp(lookback_5m, self._min5, 1000)
        lb1 = _clamp(lookback_1m, self._min1, 1000)

        # Fetch & normalize 5m candles
        try:
            candles5_raw = self._get_candles(asset, 5, lb5) or []
        except Exception as e:
            return {"asset": asset, "tf": 5, "dir": "none", "conf": 0.0, "gap": 0.0, "raw": {"error": f"provider_5m: {e}"}}
        candles5 = _normalize_candles(candles5_raw)

        # Base strategy (robust parse)
        base_raw: Any
        try:
            base_raw = self._strategy_5m(asset, candles5) or {}
        except Exception as e:
            return {"asset": asset, "tf": 5, "dir": "none", "conf": 0.0, "gap": 0.0, "raw": {"error": f"strategy_5m_error: {e}"}}

        base_dir, base_conf, base_gap = _parse_strategy_output(base_raw)

        # Micro filter (optional) — block only when strong opposite
        micro_ok = True
        micro = None
        MICRO_CONF_BLOCK_THR = 0.55
        MICRO_GAP_BLOCK_THR = 0.10

        if enable_micro_filter and self._sig1m is not None and base_dir in ("call", "put"):
            try:
                micro = self._sig1m.check(asset, lookback=lb1)
            except Exception as e:
                micro = {"dir": "none", "conf": 0.0, "gap": 0.0, "raw": {"error": f"signal_1m_error: {e}"}}

            m_dir = str(micro.get("dir", "none")).lower()
            m_conf = float(_clip01(micro.get("conf", 0.0)) or 0.0)
            m_gap  = float(_clip01(micro.get("gap", 0.0)) or 0.0)

            if m_dir not in ("none", base_dir):  # opposite
                if (m_conf > MICRO_CONF_BLOCK_THR) and (m_gap > MICRO_GAP_BLOCK_THR):
                    micro_ok = False

        return {
            "asset": asset,
            "tf": 5,
            "dir": base_dir if micro_ok else "none",
            "conf": base_conf,
            "gap": base_gap,
            "raw": {
                "base": {"dir": base_dir, "conf": base_conf, "gap": base_gap, "raw": base_raw},
                "micro": micro,
                "micro_ok": micro_ok,
                "micro_rules": {"conf_thr": MICRO_CONF_BLOCK_THR, "gap_thr": MICRO_GAP_BLOCK_THR},
                "lookbacks": {"m5": lb5, "m1": lb1},
            }
        }
