# strategies/break_retest_go_1m.py
# Break-Retest-Go (EMA20 Retest) — 1m with sensitivity scaling
from __future__ import annotations
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy

# ---------- helpers (เบา ๆ ไม่พึ่ง lib เพิ่ม) ----------
def _ema(arr: List[float], n: int) -> List[Optional[float]]:
    if n <= 1 or not arr: return [None]*len(arr)
    k = 2.0/(n+1.0)
    out = [None]*len(arr)
    s = sum(arr[:n])/n if len(arr)>=n else None
    if s is None: return out
    out[n-1] = s; prev = s
    for i in range(n, len(arr)):
        prev = prev + k*(arr[i]-prev); out[i] = prev
    return out

def _std(arr: List[float], n: int) -> List[Optional[float]]:
    out = [None]*len(arr)
    for i in range(n-1, len(arr)):
        w = arr[i-n+1:i+1]; m = sum(w)/n
        out[i] = (sum((x-m)**2 for x in w)/n) ** 0.5
    return out

def _bollinger(c: List[float], p: int, k: float):
    mid = _ema(c, p); sd = _std(c, p)
    up, lo = [None]*len(c), [None]*len(c)
    for i in range(len(c)):
        if mid[i] is not None and sd[i] is not None:
            up[i] = mid[i] + k*sd[i]; lo[i] = mid[i] - k*sd[i]
    return up, mid, lo

def _rolling_max(a: List[float], n: int, i: int): 
    return None if i-n+1<0 else max(a[i-n+1:i+1])
def _rolling_min(a: List[float], n: int, i: int): 
    return None if i-n+1<0 else min(a[i-n+1:i+1])

def _wick(o,h,l,c):
    rng = max(h-l, 1e-9)
    return (h-max(o,c))/rng, (min(o,c)-l)/rng  # upper, lower

# ---------- sensitivity mapping ----------
def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _lerp(a,b,t): return a + (b-a)*t
def _sens01(settings) -> float:
    try: s = float(settings.get("sensitivity", 50))
    except Exception: s = 50.0
    return _clamp(s, 0.0, 100.0)/100.0

def _sens_derived(settings) -> Dict[str, float]:
    """
    Baseline (sens=50) = ค่าปัจจุบันในไฟล์เดิม:
      consolidation_lookback=12, retest_tolerance_pct=0.06, wick_max=0.60
    ต่ำกว่า 50 → หลวม / สูงกว่า 50 → เข้ม
    """
    t = _sens01(settings)
    if t <= 0.5:
        # หลวม → lookback สั้นลง, tol กว้างขึ้น, wick ยาวขึ้น
        lookback = round(_lerp(8, 12, t/0.5))
        tol = _lerp(0.09, 0.06, t/0.5)   # %
        wick = _lerp(0.80, 0.60, t/0.5)
    else:
        # เข้ม → lookback ยาวขึ้น, tol แคบลง, wick สั้นลง
        lookback = round(_lerp(12, 18, (t-0.5)/0.5))
        tol = _lerp(0.06, 0.035, (t-0.5)/0.5)  # %
        wick = _lerp(0.60, 0.45, (t-0.5)/0.5)
    return {
        "consolidation_lookback": int(lookback),
        "retest_tolerance_pct": float(round(tol, 4)),
        "wick_max": float(round(wick, 3)),
    }

class BreakRetestGo1mStrategy(BaseStrategy):
    name = "Break-Retest-Go (EMA20) (1m)"
    tf = 1
    required_bars = 205

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__()
        s = settings or {}
        self.settings.update({
            "tf": 1, "required_bars": 205, "duration": 1,
            # core inputs
            "ema_fast": int(s.get("ema_fast", 9)),
            "ema_mid":  int(s.get("ema_mid", 20)),
            "ema_slow": int(s.get("ema_slow", 50)),
            "bb_period": int(s.get("bb_period", 20)),
            "bb_dev": float(s.get("bb_dev", 2.0)),

            # baseline thresholds (sens=50)
            "consolidation_lookback": int(s.get("consolidation_lookback", 12)),
            "retest_tolerance_pct": float(s.get("retest_tolerance_pct", 0.06)),
            "wick_top_ratio_max": float(s.get("wick_top_ratio_max", 0.60)),
            "wick_bot_ratio_max": float(s.get("wick_bot_ratio_max", 0.60)),

            # new: sensitivity (0..100), 50 = baseline
            "sensitivity": int(s.get("sensitivity", 50)),
        })
        self.last_meta = None

    def check_signal(self, candles) -> str:
        try:
            # --- derive dynamic thresholds from sensitivity ---
            d = _sens_derived(self.settings)
            # ถ้าผู้ใช้ตั้งค่าเองมาก่อน จะเคารพค่านั้น (ไม่เขียนทับ)
            cons_look = int(self.settings.get("consolidation_lookback") or d["consolidation_lookback"])
            tol_pct   = float(self.settings.get("retest_tolerance_pct") or d["retest_tolerance_pct"])
            wick_top  = float(self.settings.get("wick_top_ratio_max") or d["wick_max"])
            wick_bot  = float(self.settings.get("wick_bot_ratio_max") or d["wick_max"])

            # --- รองรับ pandas.DataFrame หรือ list[dict] ---
            if hasattr(candles, "empty"):
                if candles is None or candles.empty: self.last_meta={"reason":"empty"}; return "none"
                need = {"open","high","low","close"}
                cols = {c.lower() for c in candles.columns}
                if not need.issubset(cols): self.last_meta={"reason":"bad_cols"}; return "none"
                # แม็ปคอลัมน์แบบ ignore-case
                def col(name): return candles[[c for c in candles.columns if c.lower()==name][0]].astype(float).tolist()
                o, h, l, c = col("open"), col("high"), col("low"), col("close")
            else:
                o = [float(x.get("open", x.get("o"))) for x in candles]
                h = [float(x.get("high", x.get("h"))) for x in candles]
                l = [float(x.get("low",  x.get("l"))) for x in candles]
                c = [float(x.get("close",x.get("c"))) for x in candles]

            if len(c) < self.required_bars: self.last_meta={"reason":"not_enough"}; return "none"
            i, prev = len(c)-1, len(c)-2

            ema9  = _ema(c, int(self.settings["ema_fast"]))
            ema20 = _ema(c, int(self.settings["ema_mid"]))
            ema50 = _ema(c, int(self.settings["ema_slow"]))
            bb_up, bb_mid, bb_lo = _bollinger(c, int(self.settings["bb_period"]), float(self.settings["bb_dev"]))

            trend_up   = (ema20[i] and ema50[i] and ema20[i] >  ema50[i])
            trend_down = (ema20[i] and ema50[i] and ema20[i] <  ema50[i])
            if not (trend_up or trend_down): self.last_meta={"reason":"no_trend"}; return "none"

            cons_hi = _rolling_max(h, cons_look, prev); cons_lo = _rolling_min(l, cons_look, prev)
            if cons_hi is None or cons_lo is None: self.last_meta={"reason":"no_consolidation"}; return "none"

            # ----- logic: แตกกรอบ (prev) → รีเทสต์ EMA20/BB mid → ปิดกลับเหนือ/ใต้ EMA9 -----
            # UP
            if trend_up:
                broke = h[prev] > cons_hi
                ema_ref = ema20[i] if ema20[i] else c[i]
                ref = ema_ref if bb_lo[i] is None else max(ema_ref, bb_mid[i] or ema_ref)
                touched = (l[i] <= ref <= h[i]) or abs(c[i]-ref) <= max(ref,c[i])*(tol_pct/100.0)
                close_ok = c[i] > (ema9[i] or c[i]-1e-9)
                upper, _ = _wick(o[i],h[i],l[i],c[i])
                if broke and touched and close_ok and upper <= wick_top:
                    self.last_meta={"dir":"call","cons_hi":cons_hi,"ema20":ema20[i],"ema9":ema9[i],"lookback":cons_look,"tol_pct":tol_pct,"wick_top":upper}
                    return "call"

            # DOWN
            if trend_down:
                broke = l[prev] < cons_lo
                ema_ref = ema20[i] if ema20[i] else c[i]
                ref = ema_ref if bb_up[i] is None else min(ema_ref, bb_mid[i] or ema_ref)
                touched = (l[i] <= ref <= h[i]) or abs(c[i]-ref) <= max(ref,c[i])*(tol_pct/100.0)
                close_ok = c[i] < (ema9[i] or c[i]+1e-9)
                _, lower = _wick(o[i],h[i],l[i],c[i])
                if broke and touched and close_ok and lower <= wick_bot:
                    self.last_meta={"dir":"put","cons_lo":cons_lo,"ema20":ema20[i],"ema9":ema9[i],"lookback":cons_look,"tol_pct":tol_pct,"wick_bot":lower}
                    return "put"

            self.last_meta={"reason":"no_entry","lookback":cons_look,"tol_pct":tol_pct,"wick_top_max":wick_top,"wick_bot_max":wick_bot}
            return "none"
        except Exception as e:
            self.last_meta={"error":str(e)}
            return "none"

    def get_signal_meta(self): return self.last_meta
