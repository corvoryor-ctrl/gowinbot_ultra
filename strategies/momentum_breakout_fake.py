# backend/strategies/momentum_breakout_fake.py
from .base_strategy import BaseStrategy
import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Any
import logging

class MomentumBreakoutFakeStrategy(BaseStrategy):
    def __init__(self, settings: dict | None = None):
        super().__init__(settings or {})
        self.name = "Momentum Breakout (Fake - Counter)"

        # === 1) รับ sensitivity (ยอมรับค่า 0) ===
        s_in = self.settings.get("sensitivity")
        self.sensitivity: int = int(s_in) if s_in is not None else 50
        self.sensitivity = max(0, min(100, self.sensitivity))

        # === 2) ค่าพื้นฐาน (ไม่โยงกับ sensitivity) ===
        # defaults ที่ "ไม่" เปลี่ยนตาม sensitivity
        DEFAULT_MIN_CONF = 0.55
        DEFAULT_MIN_GAP  = 0.08

        # ถ้ามี override จาก UI ให้ใช้เลย ไม่งั้นใช้ default คงที่ด้านบน
        self.min_conf: float = float(self.settings.get("min_conf", DEFAULT_MIN_CONF))
        self.min_gap:  float = float(self.settings.get("min_gap",  DEFAULT_MIN_GAP))

        # === 3) lookback/atr_len เท่านั้นที่ผูกกับ sensitivity ===
        # ถ้ามีคนกำหนดตรง ๆ มาก็เคารพค่านั้น ไม่งั้นคำนวณจาก sensitivity
        if "lookback" in self.settings:
            self.lookback = int(self.settings["lookback"])
        else:
            self.lookback = int(10 + (self.sensitivity/100) * 50)   # 10 → 60

        if "atr_len" in self.settings:
            self.atr_len = int(self.settings["atr_len"])
        else:
            self.atr_len = int(7 + (self.sensitivity/100) * 21)     # 7 → 28

        logging.info(f"[MBF] params → sens={self.sensitivity}, "
                     f"lookback={self.lookback}, atr_len={self.atr_len}, "
                     f"min_conf={self.min_conf}, min_gap={self.min_gap}")

        self.last_meta: Optional[Dict[str, Any]] = None

    def _compute_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        N = self.lookback
        if df is None or df.empty or len(df) < N + 2:
            return None
        req = {"open","high","low","close"}
        if not req.issubset(df.columns):
            return None

        sub = df.iloc[-N - 1:-1].copy()
        recent_high = float(sub["high"].max())
        recent_low  = float(sub["low"].min())

        last = df.iloc[-1]
        prev = df.iloc[-2]

        atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.atr_len)
        atr = float(atr_series.iloc[-1]) if atr_series is not None else 0.0
        if not pd.notna(atr) or atr <= 0:
            atr = 1.0

        fake_up = (float(last["high"]) > recent_high) and (float(last["close"]) <= recent_high)
        fake_down = (float(last["low"]) < recent_low) and (float(last["close"]) >= recent_low)
        if not (fake_up or fake_down):
            return None

        last_open  = float(last["open"])
        last_close = float(last["close"])
        last_high  = float(last["high"])
        last_low   = float(last["low"])

        body = abs(last_close - last_open)
        upper_wick = last_high - max(last_close, last_open)
        lower_wick = min(last_close, last_open) - last_low
        wick_ratio = max(upper_wick, lower_wick) / max(1e-6, body + upper_wick + lower_wick)

        edge_price = recent_high if fake_up else recent_low
        dist_from_sr_atr = abs(last_close - edge_price) / max(1e-6, atr)

        # conf / gap คำนวณเหมือนเดิม (แต่ "เกณฑ์" ไม่ผูกกับ sensitivity แล้ว)
        conf = max(0.0, min(1.0, 0.5*wick_ratio + 0.5*min(1.0, dist_from_sr_atr)))
        overrun = (last_high - recent_high) if fake_up else (recent_low - last_low)
        gap = max(0.0, min(1.0, overrun / max(1e-6, atr)))

        direction = "put" if fake_up else "call"

        return {
            "direction": direction,
            "conf": float(conf),
            "gap": float(gap),
            "details": {
                "lookback": int(self.lookback),
                "atr_len": int(self.atr_len),
                "min_conf": float(self.min_conf),
                "min_gap": float(self.min_gap),
                "recent_high": recent_high,
                "recent_low": recent_low,
                "last_close": last_close,
                "atr": float(atr),
                "wick_ratio": float(wick_ratio),
                "dist_from_sr_atr": float(dist_from_sr_atr),
                "overrun_atr": float(overrun / max(1e-6, atr)),
                "fake_up": bool(fake_up),
                "fake_down": bool(fake_down)
            },
        }

    def check_signal(self, candles: pd.DataFrame) -> str:
        # กัน dataframe ว่าง/คอลัมน์ไม่ครบ
        if candles is None or candles.empty or not {"open","high","low","close"}.issubset(candles.columns):
            self.last_meta = {"reason": "bad_df", "cols": list(candles.columns) if candles is not None else None}
            return "none"

        sig = self._compute_signal(candles)
        if not sig:
            # บอกชัดว่าไม่มี fake
            self.last_meta = {"reason": "no_fake_breakout"}
            return "none"

        direction, conf, gap = sig["direction"], sig["conf"], sig["gap"]
        sig["reason"] = "threshold_check"
        self.last_meta = sig

        if conf >= self.min_conf and gap >= self.min_gap:
            return direction
        # บันทึกว่า “ตกเกณฑ์”
        self.last_meta["reason"] = "below_threshold"
        self.last_meta["min_conf"] = self.min_conf
        self.last_meta["min_gap"]  = self.min_gap
        return "none"

    def get_signal_meta(self) -> Optional[Dict[str, Any]]:
        return self.last_meta
