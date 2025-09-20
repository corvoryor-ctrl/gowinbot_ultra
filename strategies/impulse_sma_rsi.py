# strategies/impulse_sma_rsi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# ใช้ BaseStrategy จริงจากโปรเจกต์ (มี abstract check_signal)
from strategies.base_strategy import BaseStrategy

# --- Sensitivity helpers (0..100), baseline at 50 ---
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
def _lerp(a, b, t):
    return a + (b - a) * t
def _sens01(settings: dict) -> float:
    try:
        s = float((settings or {}).get('sensitivity', 50))
    except Exception:
        s = 50.0
    return _clamp(s, 0.0, 100.0) / 100.0
def _sens_derived(settings: dict) -> dict:
    t = _sens01(settings)
    if t <= 0.5:
        rsi_buy_min = _lerp(48.0, 50.0, t/0.5)
        rsi_buy_max = _lerp(67.0, 65.0, t/0.5)
        rsi_sell_min = _lerp(33.0, 35.0, t/0.5)
        rsi_sell_max = _lerp(52.0, 50.0, t/0.5)
        min_sma_slope = _lerp(0.0, 0.0, t/0.5)
        min_dist_atr  = _lerp(0.06, 0.10, t/0.5)
        max_dist_atr  = _lerp(3.00, 2.50, t/0.5)
        min_bb_width_pct = _lerp(0.10, 0.25, t/0.5)
    else:
        rsi_buy_min = _lerp(50.0, 52.0, (t-0.5)/0.5)
        rsi_buy_max = _lerp(65.0, 60.0, (t-0.5)/0.5)
        rsi_sell_min = _lerp(35.0, 40.0, (t-0.5)/0.5)
        rsi_sell_max = _lerp(50.0, 48.0, (t-0.5)/0.5)
        min_sma_slope = _lerp(0.0, 0.0, (t-0.5)/0.5)
        min_dist_atr  = _lerp(0.10, 0.14, (t-0.5)/0.5)
        max_dist_atr  = _lerp(2.50, 2.00, (t-0.5)/0.5)
        min_bb_width_pct = _lerp(0.25, 0.40, (t-0.5)/0.5)
    return {
        'rsi_buy_min': float(round(rsi_buy_min, 3)),
        'rsi_buy_max': float(round(rsi_buy_max, 3)),
        'rsi_sell_min': float(round(rsi_sell_min, 3)),
        'rsi_sell_max': float(round(rsi_sell_max, 3)),
        'min_sma_slope': float(round(min_sma_slope, 6)),
        'min_dist_atr': float(round(min_dist_atr, 3)),
        'max_dist_atr': float(round(max_dist_atr, 3)),
        'min_bb_width_pct': float(round(min_bb_width_pct, 3)),
    }


@dataclass
class ImpulseParams:
    sma_len: int = 30
    rsi_len: int = 14
    ema_len: int = 13
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # โซน RSI
    rsi_buy_min: float = 50.0
    rsi_buy_max: float = 65.0
    rsi_sell_min: float = 35.0
    rsi_sell_max: float = 50.0


class ImpulseSmaRsiStrategy(BaseStrategy):
    """
    BUY/CALL:
      1) แท่งล่าสุด 'ตัดขึ้น' SMA30 และปิดเขียวเหนือเส้น
      2) 50 < RSI(14) < 65
      3) Elder Impulse = เขียว (EMA13 ขึ้น และ MACD hist ขึ้น)

    SELL/PUT:
      1) แท่งล่าสุด 'ตัดลง' SMA30 และปิดแดงใต้เส้น
      2) 35 < RSI(14) < 50
      3) Elder Impulse = แดง (EMA13 ลง และ MACD hist ลง)
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(settings=settings or {})
        self.name = "Impulse SMA + RSI + Elder"
        # ให้พอสำหรับ SMA30, RSI14, MACD(12,26,9)
        self.required_bars: int = 100
        # ไม่ใช้ AI threshold ในกลยุทธ์นี้
        self.conf_thr: Optional[float] = None
        self.gap_thr: Optional[float] = None
        # hook สำหรับ meta/log
        self._meta_last_scan: Dict[str, Any] = {}

        # sensitivity (0..100); 50 = baseline
        self.sensitivity: int = int(self.settings.get('sensitivity', 50))

        # optional flags ที่ engine อาจอ่าน (เผื่อใช้ micro-5s ในภายหลัง)
        self.micro_5s_enable: bool = bool(self.settings.get("micro_5s_enable", False))
        self.micro_5s_window_sec: int = int(self.settings.get("micro_5s_window_sec", 15))

        # --- new filters (optional via settings) ---
        self.min_sma_slope     = float(self.settings.get("min_sma_slope", 0.0))   # >0 บังคับสโลปตามทิศ
        self.min_dist_atr      = float(self.settings.get("min_dist_atr", 0.10))   # ใกล้เส้น = เสี่ยง noise
        self.max_dist_atr      = float(self.settings.get("max_dist_atr", 2.50))   # ไกลเกิน = overextended
        self.min_bb_width_pct  = float(self.settings.get("min_bb_width_pct", 0.25)) # 0..1 ปิดสัญญาณช่วง squeeze
        self.adaptive_rsi      = bool(self.settings.get("adaptive_rsi", True))    # RSI ปรับตาม vol

    # --------- อินดิเคเตอร์หลัก ----------
    @staticmethod
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
        roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    @staticmethod
    def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(span=length, adjust=False).mean()

    def _elder_color(self, ema: pd.Series, hist: pd.Series) -> pd.Series:
        ema_slope = ema.diff()
        hist_slope = hist.diff()
        green = (ema_slope > 0) & (hist_slope > 0)
        red   = (ema_slope < 0) & (hist_slope < 0)
        return pd.Series(np.where(green, 1, np.where(red, -1, 0)), index=ema.index)  # 1=เขียว, -1=แดง, 0=กลาง

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        p = ImpulseParams()
        out = df.copy()
        out["sma"] = out["close"].astype(float).rolling(p.sma_len, min_periods=p.sma_len).mean()
        out["rsi"] = self._rsi(out["close"].astype(float), p.rsi_len)
        ema13 = self._ema(out["close"].astype(float), p.ema_len)
        _, _, hist = self._macd(out["close"].astype(float), p.macd_fast, p.macd_slow, p.macd_signal)
        out["impulse"] = self._elder_color(ema13, hist)
        out["atr"] = self._atr(out)
        out["bb_w"] = self._bb_width(out["close"].astype(float), length=20, k=2.0)
        return out

    def _bb_width(self, series: pd.Series, length: int = 20, k: float = 2.0) -> pd.Series:
        ma  = series.rolling(length, min_periods=length).mean()
        sd  = series.rolling(length, min_periods=length).std(ddof=0)
        up  = ma + k*sd
        lo  = ma - k*sd
        return (up - lo) / (ma.abs() + 1e-9)

    def _percentile_threshold(self, s: pd.Series, pct: float) -> float:
        s = s.dropna().tail(100)
        if len(s) < 20:
            return float("nan")
        return float(np.quantile(s.values, pct))

    # --------- กติกาหลัก ----------
    def _check_one(self, df: pd.DataFrame):
        if df is None or len(df) < self.required_bars:
            return None, {"reason": "insufficient_bars", "n": int(len(df) if df is not None else 0)}

        x = self._compute(df.tail(max(self.required_bars, 80)))
        if len(x) < 5 or x["sma"].isna().all():
            return None, {"reason": "na_sma"}

        c = float(x["close"].iloc[-1]); o = float(x["open"].iloc[-1])
        pclose = float(x["close"].iloc[-2])
        sma = float(x["sma"].iloc[-1]); psma = float(x["sma"].iloc[-2])
        rsi = float(x["rsi"].iloc[-1]); impulse = int(x["impulse"].iloc[-1])

        crossed_up = (pclose <= psma) and (c > sma) and (c > o)
        crossed_dn = (pclose >= psma) and (c < sma) and (c < o)

        d = _sens_derived(self.settings)
        rsi_call_ok = (rsi > d['rsi_buy_min']) and (rsi < d['rsi_buy_max'])
        rsi_put_ok  = (rsi > d['rsi_sell_min']) and (rsi < d['rsi_sell_max'])

        green = (impulse == 1)
        red   = (impulse == -1)

        meta = {
            "close": c, "open": o, "sma": sma, "psma": psma, "rsi": rsi, "impulse": impulse,
            "cross_up": bool(crossed_up), "cross_dn": bool(crossed_dn),
        }

        if crossed_up and rsi_call_ok and green:
            score = self._score_call(x)
            meta["score"] = float(score)
            return "call", meta

        if crossed_dn and rsi_put_ok and red:
            score = self._score_put(x)
            meta["score"] = float(score)
            return "put", meta

        meta["reason"] = "rules_not_met"

        # --- extras: slope/distance/bb/RSI adaptive ---
        sma_slope = (sma - psma)
        atr = float(x["atr"].iloc[-1] or 1e-6)
        dist_atr = abs(c - sma) / max(atr, 1e-9)
        bb_w_now = float(x["bb_w"].iloc[-1]) if "bb_w" in x.columns else float("nan")
        min_sma_slope = float(self.settings.get('min_sma_slope', d['min_sma_slope']))
        min_dist_atr  = float(self.settings.get('min_dist_atr',  d['min_dist_atr']))
        max_dist_atr  = float(self.settings.get('max_dist_atr',  d['max_dist_atr']))
        min_bb_width_pct = float(self.settings.get('min_bb_width_pct', d['min_bb_width_pct']))

        # 1) SMA slope ต้องหนุนทิศ (ถ้าตั้งค่า >0)
        if min_sma_slope > 0:
            if crossed_up and (sma_slope <= min_sma_slope):
                return None, {**meta, "reason": "slope_not_up"}
            if crossed_dn and (sma_slope >= -min_sma_slope):
                return None, {**meta, "reason": "slope_not_down"}

        # 2) ระยะจาก SMA ต้องไม่น้อย/ไม่มากเกินไป (กัน noise / overextended)
        if dist_atr < min_dist_atr:
            return None, {**meta, "reason": "too_close_to_sma"}
        if dist_atr > max_dist_atr:
            return None, {**meta, "reason": "too_far_from_sma"}

        # 3) ปิดสัญญาณช่วง squeeze (BB width ต่ำกว่าเปอร์เซ็นไทล์ที่กำหนด)
        if min_bb_width_pct > 0 and not np.isnan(bb_w_now):
            th = self._percentile_threshold(x["bb_w"], min_bb_width_pct)
            if th == th and bb_w_now < th:  # เช็คไม่ใช่ NaN
                return None, {**meta, "reason": "bb_squeeze"}

        # 4) RSI adaptive ตามความแรงของ vol (ต่ำ: เข้มขึ้น, สูง: ผ่อนลง)
        if self.adaptive_rsi:
            atr_vals = x["atr"].dropna().tail(100)
            if len(atr_vals) >= 20:
                # ลบ 'import numpy as np' ตรงนี้ออก
                med = float(np.median(atr_vals.values))
                if med == med and atr <= med:   # ช่วง vol ต่ำ → เข้ม
                    buy_min, buy_max = 52.0, 60.0
                    sell_min, sell_max = 40.0, 48.0
                else:                            # ช่วง vol สูง → ผ่อน
                    buy_min, buy_max = 50.0, 66.0
                    sell_min, sell_max = 34.0, 50.0
            else:
                buy_min, buy_max = 50.0, 65.0
                sell_min, sell_max = 35.0, 50.0

        rsi_call_ok = (rsi > buy_min) and (rsi < buy_max)
        rsi_put_ok  = (rsi > sell_min) and (rsi < sell_max)
        return None, meta

    # ใช้สำหรับจัดอันดับ (TopN)
    def _score_call(self, x: pd.DataFrame) -> float:
        rsi = float(x["rsi"].iloc[-1]); rsi_center = 57.5; rsi_half = 7.5
        rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center) / rsi_half)
        impulse_strength = float((x["impulse"].tail(3) == 1).mean())
        c = float(x["close"].iloc[-1]); sma = float(x["sma"].iloc[-1]); atr = float(x["atr"].iloc[-1] or 1e-6)
        dist = max(0.0, min(1.0, (c - sma) / (atr + 1e-6)))
        return 0.55*rsi_score + 0.30*impulse_strength + 0.15*dist

    def _score_put(self, x: pd.DataFrame) -> float:
        rsi = float(x["rsi"].iloc[-1]); rsi_center = 42.5; rsi_half = 7.5
        rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center) / rsi_half)
        impulse_strength = float((x["impulse"].tail(3) == -1).mean())
        c = float(x["close"].iloc[-1]); sma = float(x["sma"].iloc[-1]); atr = float(x["atr"].iloc[-1] or 1e-6)
        dist = max(0.0, min(1.0, (sma - c) / (atr + 1e-6)))
        return 0.55*rsi_score + 0.30*impulse_strength + 0.15*dist

    # --------- ✅ เมทอดที่ BaseStrategy ต้องการ (แก้ error abstract) ----------
    def check_signal(self, *args, **kwargs) -> str:
        """
        รองรับทั้งรูปแบบ:
            check_signal(asset: str, df: pd.DataFrame) -> "call"/"put"/"none"
            check_signal(df: pd.DataFrame) -> "call"/"put"/"none"
        """
        # ดึง df ไม่ว่าจะส่งมาแบบไหน
        df = None
        if len(args) >= 2 and isinstance(args[1], pd.DataFrame):
            df = args[1]
        elif len(args) >= 1 and isinstance(args[0], pd.DataFrame):
            df = args[0]
        else:
            df = kwargs.get("df", None)

        sig, meta = self._check_one(df) if isinstance(df, pd.DataFrame) else (None, {"reason": "no_df"})
        # เก็บ meta ล่าสุดไว้ให้ UI/Log เรียกดู
        try:
            self._meta_last_scan = {"single": meta}
        except Exception:
            pass
        return (sig or "none")

    # --------- Hooks สำหรับ engine ที่สแกนหลายคู่ ----------
    def check_signals_pair(self, df_map: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        scan_meta = []
        for asset, df in (df_map or {}).items():
            sig, meta = self._check_one(df)
            meta["asset"] = asset
            scan_meta.append(meta)
            if sig in ("call", "put"):
                out[asset] = sig
        self._meta_last_scan = {"scan": scan_meta, "best": (None if not out else list(out.items())[0])}
        return out

    def check_signals_rank(self, df_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        scan_meta = []
        for asset, df in (df_map or {}).items():
            sig, meta = self._check_one(df)
            meta["asset"] = asset
            scan_meta.append(meta)
            if sig == "call":
                ranked.append({"asset": asset, "signal": "call", "score": float(meta.get("score", 0.0))})
            elif sig == "put":
                ranked.append({"asset": asset, "signal": "put", "score": float(meta.get("score", 0.0))})
        ranked.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        self._meta_last_scan = {"scan": scan_meta, "best": (ranked[0] if ranked else None)}
        return ranked

    def get_signal_meta(self) -> Dict[str, Any]:
        return dict(self._meta_last_scan) if isinstance(self._meta_last_scan, dict) else {}
