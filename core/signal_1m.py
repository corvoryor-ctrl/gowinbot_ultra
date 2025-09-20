# core/signal_1m.py
# Wraps existing 1m strategies behind a stable interface.
from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List, Tuple

Signal = Dict[str, Any]
StrategyFn = Callable[[str, list], Signal]
CandleProvider = Callable[[str, int, int], list]

# -------------------- tiny utils --------------------

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
            # รองรับ dict หรือ tuple/list รูปแบบ (ts, open, high, low, close) อย่างหยาบๆ
            if isinstance(r, dict):
                ts = int(r.get("ts") or r.get("time") or r.get("t") or 0)
                o = _f(r.get("open", r.get("o", r.get("Open", 0.0))))
                h = _f(r.get("high", r.get("h", r.get("High", o))))
                l = _f(r.get("low",  r.get("l", r.get("Low", o))))
                c = _f(r.get("close",r.get("c", r.get("Close", o))))
            elif isinstance(r, (list, tuple)) and len(r) >= 5:
                ts = int(r[0]); o = _f(r[1]); h = _f(r[2]); l = _f(r[3]); c = _f(r[4])
            else:
                # ไม่รู้ฟอร์แมต → ข้าม
                continue
            # กันกรณี high/low เพี้ยน
            if h < max(o, c): h = max(o, c)
            if l > min(o, c): l = min(o, c)
            out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
        except Exception:
            # candle แปลกมาก → ข้ามแท่งนั้น
            continue
    return out

def _parse_strategy_output(raw: Any) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Return (dir, conf, gap) after best-effort normalization.
    dir ∈ {"call","put","none"}; conf,gap ∈ [0,1] or None
    """
    direction: str = "none"
    conf: Optional[float] = None
    gap: Optional[float] = None

    def _clip01(v: Optional[float]) -> Optional[float]:
        if v is None: return None
        try:
            vv = float(v)
            if vv != vv:  # NaN
                return None
            return max(0.0, min(1.0, vv))
        except Exception:
            return None

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

    # Fill defaults
    conf = _clip01(conf) if conf is not None else 0.0
    gap  = _clip01(gap)  if gap  is not None else 0.0
    if direction not in ("call", "put"):
        direction = "none"

    return direction, conf, gap

# --- dynamic min lookback heuristic (aligned with the rest of the app) ---
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

def _dynamic_min_lookback() -> int:
    """Try to read indicator config for tf=1 and infer minimal bars required."""
    try:
        # optional import — if module not present, fall back
        from core.indicator_config import get_indicator_config  # type: ignore
        cfg = get_indicator_config(1) or {}
        ind = cfg.get("indicators") or cfg.get("custom") or cfg or {}
        needs = []
        for k, v in ind.items():
            if isinstance(v, dict) and v.get("enabled"):
                needs.append(_bars_needed_for_indicator(k, v))
        if needs:
            req = int(max(needs) * 1.25) + 5  # buffer
            return _clamp(req, 60, 1000)
    except Exception:
        pass
    return 120

class Signal1m:
    """
    A thin adapter. You pass in:
      - candle_provider(asset, tf_minutes, count) -> candles
      - strategy(asset, candles_1m) -> {"dir": "call/put/none", "conf": float, "gap": float}
    We do not depend on internal strategy modules to avoid UI breakage.
    """
    def __init__(self, candle_provider: CandleProvider, strategy: StrategyFn):
        self._get_candles = candle_provider
        self._strategy = strategy
        self._min_lb = _dynamic_min_lookback()  # ประมาณขั้นต่ำจากคอนฟิก (ถ้าทำได้)

    def check(self, asset: str, lookback: int = 100) -> Signal:
        # ปรับ lookback ให้พอเหมาะกับอินดิเคเตอร์ที่เปิดใช้อยู่ (หรือ fallback 120)
        lb = _clamp(lookback, self._min_lb, 1000)

        # ดึงเทียน 1m และ normalize
        try:
            candles_raw = self._get_candles(asset, 1, lb) or []
        except Exception as e:
            # provider พัง → คืน safe default
            return {"asset": asset, "tf": 1, "dir": "none", "conf": 0.0, "gap": 0.0, "raw": {"error": str(e)}}

        candles = _normalize_candles(candles_raw)

        # เรียกกลยุทธ์แบบกันพัง
        sig_raw: Any
        try:
            sig_raw = self._strategy(asset, candles) or {}
        except Exception as e:
            # กลยุทธ์ล้มเหลว → ไม่ทำให้ระบบล่ม
            return {
                "asset": asset, "tf": 1,
                "dir": "none", "conf": 0.0, "gap": 0.0,
                "raw": {"error": f"strategy_error: {e}"}
            }

        # normalize ผลลัพธ์ให้เป็นรูปมาตรฐาน
        direction, conf, gap = _parse_strategy_output(sig_raw)

        return {
            "asset": asset,
            "tf": 1,
            "dir": direction if direction in ("call", "put") else "none",
            "conf": float(conf or 0.0),
            "gap": float(gap or 0.0),
            "raw": sig_raw,
        }
