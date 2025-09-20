# strategies/macd_ema_psar_5m.py
# Strategy: MACD(+scaled) + EMA + Parabolic SAR for 5m
# - พอร์ตจากเวอร์ชัน 1m โดยคงโครงสร้าง/เมธอดเดิมทั้งหมด
# - ทุกพารามิเตอร์ derive จาก "sensitivity" (0..100)
# - สเกลพารามิเตอร์ให้เหมาะกับ TF=5 นาที (ช้าลงเล็กน้อย, ลด noise)

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

try:
    import pandas as pd  # type: ignore
    _HAS_PD = True
except Exception:
    pd = None  # type: ignore
    _HAS_PD = False

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)

class MacdEmaPsar5mStrategy:
    name = "MACD+EMA12+PSAR (5m)"
    tf   = 5  # นาที

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        s = settings or {}
        self.sensitivity = _clamp(int(s.get("sensitivity", 50)), 0, 100)
        self._apply_sensitivity()

    # ----------------- Sensitivity → Params (scaled for 5m) -----------------
    def _apply_sensitivity(self):
        t = self.sensitivity / 100.0

        # EMA length: 34 → 12 (สูง = ไวกว่า) — ช้ากว่า 1m เล็กน้อยเพื่อลดสัญญาณหลอก
        self.ema_len      = int(round(_lerp(34, 12, t)))

        # MACD baseline: กว้างกว่า 1m เล็กน้อย
        # (1m: fast 6→3, slow 14→7, signal 6→3) → (5m: fast 8→5, slow 20→10, signal 8→5)
        self.macd_fast    = int(round(_lerp(8, 5, t)))
        self.macd_slow    = int(round(_lerp(20, 10, t)))
        self.macd_signal  = int(round(_lerp(8, 5, t)))

        # PSAR: ช้าลงเล็กน้อยเมื่อเทียบกับ 1m
        self.psar_af      = _lerp(0.010, 0.035, t)
        self.psar_af_step = _lerp(0.010, 0.035, t)
        self.psar_af_max  = _lerp(0.10,  0.18,  t)

        # Flip / slope windows: กว้างขึ้นนิด เพื่อตรวจความต่อเนื่องของเทรนด์
        self.lookback_flip = int(round(_lerp(6, 2, t)))
        self.slope_window  = int(round(_lerp(7, 2, t)))

        # Histogram gate & min confidence
        # (5m histogram มักผันผวนน้อยกว่า → เกณฑ์เริ่มต้นสูงขึ้นเล็กน้อย)
        self.min_hist_abs  = _lerp(0.0020, 0.0, t)
        self.min_conf      = round(_lerp(0.78, 0.52, t), 2)

    # ----------------- Candles extraction -----------------
    def _extract_ohlc(self, candles) -> Tuple[List[float], List[float], List[float]]:
        """
        คืน (closes, highs, lows) เป็น list[float]
        รองรับ:
          - list[dict] ที่มีคีย์ "close","high","low" (หรือ 'c','h','l' สำรอง)
          - pandas.DataFrame ที่มีคอลัมน์เดียวกัน
        """
        if candles is None:
            return [], [], []

        # pandas.DataFrame
        if _HAS_PD and isinstance(candles, pd.DataFrame):
            cols = {c.lower(): c for c in candles.columns}
            def pick(*names):
                for n in names:
                    if n in cols: return cols[n]
                return None
            ccol = pick("close", "c")
            hcol = pick("high", "h")
            lcol = pick("low", "l")
            if not (ccol and hcol and lcol):
                return [], [], []
            closes = [float(x) for x in candles[ccol].astype(float).tolist()]
            highs  = [float(x) for x in candles[hcol].astype(float).tolist()]
            lows   = [float(x) for x in candles[lcol].astype(float).tolist()]
            return closes, highs, lows

        # list[dict] (หรือ iterable ที่ให้ item เป็น dict)
        try:
            closes = []
            highs = []
            lows = []
            for r in candles:
                if not isinstance(r, dict):
                    try_get = lambda obj, a, b=None: obj.get(a) if isinstance(obj, dict) else getattr(obj, a, b)
                    c = try_get(r, "close", try_get(r, "c", None))
                    h = try_get(r, "high",  try_get(r, "h", None))
                    l = try_get(r, "low",   try_get(r, "l", None))
                else:
                    c = r.get("close", r.get("c"))
                    h = r.get("high",  r.get("h"))
                    l = r.get("low",   r.get("l"))
                if c is None or h is None or l is None:
                    continue
                closes.append(float(c))
                highs.append(float(h))
                lows.append(float(l))
            return closes, highs, lows
        except Exception:
            return [], [], []

    # ----------------- Indicators -----------------
    @staticmethod
    def _ema(arr: List[float], n: int) -> Optional[float]:
        if n <= 0 or len(arr) < n:
            return None
        k = 2.0 / (n + 1)
        e = sum(arr[:n]) / n
        for x in arr[n:]:
            e = x * k + e * (1 - k)
        return e

    def _ema_series(self, arr: List[float], n: int) -> List[float]:
        if n <= 0 or len(arr) < n:
            return []
        out: List[float] = []
        k = 2.0 / (n + 1)
        e = sum(arr[:n]) / n
        out.extend([math.nan] * (n - 1))
        out.append(e)
        for x in arr[n:]:
            e = x * k + e * (1 - k)
            out.append(e)
        return out

    def _macd(self, closes: List[float]) -> Dict[str, List[float]]:
        need = max(self.macd_slow + self.macd_signal + 5, self.ema_len + 2)
        if len(closes) < need:
            return {"line": [], "signal": [], "hist": []}

        def ema_series(series: List[float], n: int) -> List[float]:
            if n <= 0 or len(series) < n:
                return []
            out: List[float] = []
            k = 2.0 / (n + 1)
            e = sum(series[:n]) / n
            out.extend([math.nan] * (n - 1))
            out.append(e)
            for v in series[n:]:
                e = v * k + e * (1 - k)
                out.append(e)
            return out

        ef = ema_series(closes, self.macd_fast)
        es = ema_series(closes, self.macd_slow)
        line: List[float] = []
        for i in range(len(closes)):
            a = ef[i] if i < len(ef) else math.nan
            b = es[i] if i < len(es) else math.nan
            line.append((a - b) if (not math.isnan(a) and not math.isnan(b)) else math.nan)

        clean = [x for x in line if not math.isnan(x)]
        sig_series = ema_series(clean, self.macd_signal)
        signal: List[float] = [math.nan] * (len(line) - len(sig_series)) + sig_series
        hist: List[float] = []
        for i in range(len(line)):
            l = line[i]; s = signal[i]
            hist.append((l - s) if (not math.isnan(l) and not math.isnan(s)) else math.nan)
        return {"line": line, "signal": signal, "hist": hist}

    def _psar_series(self, highs: List[float], lows: List[float]) -> Dict[str, List[float]]:
        length = min(len(highs), len(lows))
        if length < 5:
            return {"psar": [], "bull": []}
        psar = [math.nan] * length
        bull = [False] * length

        af = self.psar_af
        ep = highs[1]
        is_bull = True
        psar[1] = lows[0]
        bull[1] = is_bull

        for i in range(2, length):
            prev = psar[i-1]
            if is_bull:
                psar[i] = prev + af * (ep - prev)
                psar[i] = min(psar[i], lows[i-1], lows[i-2])
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(self.psar_af_max, af + self.psar_af_step)
                if lows[i] < psar[i]:
                    is_bull = False
                    psar[i] = ep
                    ep = lows[i]
                    af = self.psar_af
            else:
                psar[i] = prev + af * (ep - prev)
                psar[i] = max(psar[i], highs[i-1], highs[i-2])
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(self.psar_af_max, af + self.psar_af_step)
                if highs[i] > psar[i]:
                    is_bull = True
                    psar[i] = ep
                    ep = highs[i]
                    af = self.psar_af
            bull[i] = is_bull

        return {"psar": psar, "bull": bull}

    # ----------------- Core logic -----------------
    def evaluate(self, asset: str, candles) -> Dict[str, Any]:
        # แปลง candles เป็น list ของ OHLC float ก่อนเสมอ
        closes, highs, lows = self._extract_ohlc(candles)
        need = max(self.ema_len + 3, self.macd_slow + self.macd_signal + 5, 30)

        if closes is None or highs is None or lows is None:
            return {"signal": None, "confidence": 0.0, "reason": "no_data"}
        if len(closes) < need or len(highs) < need or len(lows) < need:
            return {"signal": None, "confidence": 0.0, "reason": "insufficient_data"}

        # EMA trend + slope
        ema_series = self._ema_series(closes, self.ema_len)
        ema_now = ema_series[-1]
        back = max(1, self.slope_window)
        ema_prev = ema_series[-1 - back] if len(ema_series) > back else ema_series[-2]
        ema_up = (not math.isnan(ema_now) and not math.isnan(ema_prev) and ema_now > ema_prev)
        ema_dn = (not math.isnan(ema_now) and not math.isnan(ema_prev) and ema_now < ema_prev)

        price = closes[-1]
        above_ema = (not math.isnan(ema_now) and price > ema_now)
        below_ema = (not math.isnan(ema_now) and price < ema_now)

        # MACD
        macd = self._macd(closes)
        line, signal, hist = macd["line"], macd["signal"], macd["hist"]
        if (not hist) or math.isnan(hist[-1]) or math.isnan(line[-1]) or math.isnan(signal[-1]):
            return {"signal": None, "confidence": 0.0, "reason": "macd_nan"}

        bull_now  = (line[-1] > signal[-1]) and (hist[-1] > 0) and (abs(hist[-1]) >= self.min_hist_abs)
        bear_now  = (line[-1] < signal[-1]) and (hist[-1] < 0) and (abs(hist[-1]) >= self.min_hist_abs)
        bull_x    = (len(hist) >= 2 and hist[-2] <= 0 and hist[-1] > 0)
        bear_x    = (len(hist) >= 2 and hist[-2] >= 0 and hist[-1] < 0)

        # PSAR
        ps = self._psar_series(highs, lows)
        psar = ps["psar"]; bull = ps["bull"]
        if not psar or math.isnan(psar[-1]):
            return {"signal": None, "confidence": 0.0, "reason": "psar_nan"}
        psar_bull = bull[-1]
        psar_bear = not psar_bull
        psar_below = price > psar[-1]
        psar_above = price < psar[-1]

        # recent flip
        flipped_recent = False
        if len(bull) >= self.lookback_flip + 1:
            for i in range(1, self.lookback_flip + 1):
                if bull[-i] != bull[-i-1]:
                    flipped_recent = True
                    break

        # ---------------- Signals ----------------
        signal_out = None
        conf = 0.0
        reasons = []

        # CALL
        if above_ema and ema_up and psar_bull and psar_below and (bull_now or bull_x):
            signal_out = "CALL"
            conf = 0.60
            if bull_now and bull_x: conf += 0.15
            if flipped_recent: conf += 0.10
            if (not math.isnan(ema_now)) and price > ema_now * 1.0008: conf += 0.05
            reasons.append("ema_up+above, macd_bull, psar_bull")

        # PUT
        if below_ema and ema_dn and psar_bear and psar_above and (bear_now or bear_x):
            c2 = 0.60
            if bear_now and bear_x: c2 += 0.15
            if flipped_recent: c2 += 0.10
            if (not math.isnan(ema_now)) and price < ema_now * 0.9992: c2 += 0.05
            if (signal_out is None) or (c2 > conf):
                signal_out = "PUT"; conf = c2
            reasons.append("ema_dn+below, macd_bear, psar_bear")

        # ปรับตาม sensitivity (boost เล็กน้อย)
        conf = min(1.0, max(0.0, conf + (self.sensitivity/100.0)*0.05))

        # เกณฑ์ขั้นต่ำ
        if signal_out and conf < self.min_conf:
            signal_out = None

        return {
            "signal": signal_out,
            "confidence": round(conf, 2),
            "reason": ";".join(reasons) if reasons else "no_alignment",
            "tuned": {
                "sensitivity": self.sensitivity,
                "ema_len": self.ema_len,
                "macd_fast": self.macd_fast, "macd_slow": self.macd_slow, "macd_signal": self.macd_signal,
                "psar_af": round(self.psar_af, 4), "psar_af_step": round(self.psar_af_step, 4), "psar_af_max": round(self.psar_af_max, 4),
                "lookback_flip": self.lookback_flip, "slope_window": self.slope_window,
                "min_hist_abs": round(self.min_hist_abs, 6), "min_conf": self.min_conf
            }
        }

    def rank(self, asset: str, candles) -> float:
        r = self.evaluate(asset, candles)
        return float(r.get("confidence") or 0.0)

    def close(self) -> None:
        pass

