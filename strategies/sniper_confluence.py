# strategies/sniper_confluence.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import time

from .base_strategy import BaseStrategy


class SniperConfluence(BaseStrategy):
    """
    Sniper Confluence (multi-asset, no-hourly-fallback)
    ---------------------------------------------------
    - สแกนหลายคู่เงินพร้อมกัน (ผ่าน BotEngine เส้นทาง multi-asset)
    - ยิงเฉพาะเมื่อพบ "strong signal" เท่านั้น (ไม่มีการันตียิงทุกชั่วโมง)
    - รายงานสถานะการสแกน (meta) ให้ UI/log เห็นว่ากำลังทำอะไรอยู่

    settings ตัวอย่าง:
      {
        "assets": ["EURUSD","GBPUSD","USDJPY","EURJPY"],
        "sensitivity": 60,        # 0..100 (ยิ่งสูงยิ่งเข้ม)
        "lookback": 160,
        "ema_fast": 50, "ema_slow": 200,
        "rsi_len": 14, "bb_len": 20, "bb_k": 2.0,
        "atr_len": 14
      }
    """
    name = "Sniper Confluence (Multi)"
    multi_asset = True  # ให้ BotEngine ใช้เส้นทาง multi-asset

    # ====== lifecycle ======
    def __init__(self, settings: Optional[dict] = None):
        super().__init__(settings or {})
        s = self.settings

        # sensitivity → 0..1
        self.sens01 = float(max(0, min(100, int(s.get("sensitivity", 60)))))/100.0

        # indicators config
        self.ema_fast = int(s.get("ema_fast", 50))
        self.ema_slow = int(s.get("ema_slow", 200))
        self.rsi_len  = int(s.get("rsi_len", 14))
        self.bb_len   = int(s.get("bb_len", 20))
        self.bb_k     = float(s.get("bb_k", 2.0))
        self.atr_len  = int(s.get("atr_len", 14))
        self.lookback = int(s.get("lookback", 160))

        # Sniper thresholds (base + scaled by sensitivity)
        self.rsi_cross_lookback = max(2, int(5 - 3*self.sens01))        # 5..2 bars
        self.min_body_atr       = float(0.40 + 0.30*self.sens01)        # 0.40..0.70 ATR
        self.wick_ratio_max     = float(0.50 + 0.25*self.sens01)        # 0.50..0.75
        self.bb_squeeze_pctile  = float(35 + 25*self.sens01)            # 35..60
        self.breakout_dist_atr  = float(0.07 + 0.10*self.sens01)        # 0.07..0.17
        self.swing_lookback     = int(10)

        # state/meta
        self._scan_meta: Dict = {}

        # ให้ Engine ไม่ตัดด้วย prob threshold ภายนอก
        self.conf_thr = None
        self.gap_thr  = None

        # แจ้งจำนวนแท่งขั้นต่ำให้ Engine ดึงให้พอ
        self.min_required_bars = int(max(
            self.lookback,
            self.ema_slow + 5,
            self.bb_len + 5,
            self.atr_len + 5,
            160
        ))

    # ให้ Engine เรียกได้ถ้าต้องการจำนวนแท่งขั้นต่ำ
    def get_min_required_bars(self) -> int:
        return int(self.min_required_bars)

    # ====== abstract satisfy (single-asset ไม่ใช้งาน) ======
    def check_signal(self, df: pd.DataFrame) -> str:
        self._scan_meta = {"mode": "multi_only"}
        return "none"

    def get_signal_meta(self) -> Optional[dict]:
        return self._scan_meta or {}

    # ====== indicators ======
    def _ema(self, x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=n, adjust=False).mean()

    def _rsi(self, x: pd.Series, n: int) -> pd.Series:
        d = x.diff()
        up = d.clip(lower=0.0)
        dn = (-d).clip(lower=0.0)
        ma_up = up.ewm(alpha=1/n, adjust=False).mean()
        ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
        rs = ma_up / (ma_dn + 1e-12)
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
        pc = c.shift(1)
        tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()

    def _bb(self, x: pd.Series, n: int, k: float):
        ma = x.rolling(n).mean()
        sd = x.rolling(n).std(ddof=0)
        upper = ma + k*sd
        lower = ma - k*sd
        width = (upper - lower) / (ma.abs() + 1e-9)
        return ma, upper, lower, width

    # ====== feature & rules per asset ======
    def _features_and_rules(self, df: pd.DataFrame) -> Tuple[dict, dict]:
        feats, rules = {}, {}
        need = self.get_min_required_bars()
        if df is None or len(df) < need:
            feats["too_short"] = True
            feats["need"] = int(need)
            feats["n"] = 0 if df is None else int(len(df))
            return feats, rules

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        high  = pd.to_numeric(df["high"],  errors="coerce").astype(float)
        low   = pd.to_numeric(df["low"],   errors="coerce").astype(float)

        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        rsi14 = self._rsi(close, self.rsi_len)
        atr14 = self._atr(df, self.atr_len)
        ma, bb_u, bb_l, bb_w = self._bb(close, self.bb_len, self.bb_k)

        slope = float(ema_s.diff().iloc[-1])
        trend_up   = bool((close.iloc[-1] > ema_s.iloc[-1]) and (ema_f.iloc[-1] > ema_s.iloc[-1]) and (slope >= 0))
        trend_down = bool((close.iloc[-1] < ema_s.iloc[-1]) and (ema_f.iloc[-1] < ema_s.iloc[-1]) and (slope <= 0))

        # RSI cross 50 ภายในหน้าต่างล่าสุด
        def _cross_up_recent():
            tail = rsi14.tail(self.rsi_cross_lookback+2)
            return bool((tail.iloc[-1] >= 50) and (tail.iloc[:-1] < 50).any())

        def _cross_dn_recent():
            tail = rsi14.tail(self.rsi_cross_lookback+2)
            return bool((tail.iloc[-1] <= 50) and (tail.iloc[:-1] > 50).any())

        rsi_cross_up   = _cross_up_recent()
        rsi_cross_down = _cross_dn_recent()

        width_series = bb_w.dropna()
        if len(width_series) >= 30:
            pctile = (width_series.rank(pct=True).iloc[-2] if len(width_series) >= 2 else 0.5) * 100.0
        else:
            pctile = 50.0

        new_high = bool(close.iloc[-1] > high.rolling(self.swing_lookback).max().iloc[-2])
        new_low  = bool(close.iloc[-1] < low.rolling(self.swing_lookback).min().iloc[-2])

        atrv = float(atr14.iloc[-1] + 1e-9)
        dist_above = float((close.iloc[-1] - float(bb_u.iloc[-1])) / atrv)
        dist_below = float((float(bb_l.iloc[-1]) - close.iloc[-1]) / atrv)

        body = abs(float(close.iloc[-1] - float(df["open"].astype(float).iloc[-1])))
        body_atr = float(body / atrv)
        rng = float(high.iloc[-1] - low.iloc[-1] + 1e-9)
        upper_wick = float(high.iloc[-1] - max(float(df["open"].iloc[-1]), float(close.iloc[-1])))
        lower_wick = float(min(float(df["open"].iloc[-1]), float(close.iloc[-1])) - low.iloc[-1])
        up_wick_ratio = float(upper_wick / rng)
        dn_wick_ratio = float(lower_wick / rng)

        # strong rules
        call_strong = (
            trend_up and rsi_cross_up
            and (pctile <= self.bb_squeeze_pctile)
            and (new_high or dist_above >= self.breakout_dist_atr)
            and (body_atr >= self.min_body_atr)
            and (up_wick_ratio <= self.wick_ratio_max)
        )
        put_strong = (
            trend_down and rsi_cross_down
            and (pctile <= self.bb_squeeze_pctile)
            and (new_low or dist_below >= self.breakout_dist_atr)
            and (body_atr >= self.min_body_atr)
            and (dn_wick_ratio <= self.wick_ratio_max)
        )

        # scoring (เพื่อรายงาน/ดีบักเท่านั้น—ไม่ใช้ยิง fallback)
        def _score_for(side: str) -> float:
            trend = 1.0 if (trend_up if side=="call" else trend_down) else 0.0
            dist  = max(0.0, dist_above if side=="call" else dist_below) / max(0.15, self.breakout_dist_atr)
            dist  = min(1.5, dist)
            bodyn = min(1.5, body_atr / max(0.35, self.min_body_atr*0.8))
            sq    = max(0.0, (max(0.0, (self.bb_squeeze_pctile - pctile))) / max(1e-6, self.bb_squeeze_pctile))
            score = 0.38*trend + 0.32*dist + 0.22*bodyn + 0.08*sq
            return float(max(0.0, min(1.0, score)))

        feats.update(dict(
            need=self.get_min_required_bars(), n=int(len(df)),
            slope=slope, trend_up=trend_up, trend_down=trend_down,
            rsi=float(rsi14.iloc[-1]), rsi_cross_up=rsi_cross_up, rsi_cross_down=rsi_cross_down,
            bb_pctile_prev=float(pctile), dist_above_atr=float(dist_above), dist_below_atr=float(dist_below),
            new_high=new_high, new_low=new_low, body_atr=float(body_atr),
            wick_up_ratio=float(up_wick_ratio), wick_dn_ratio=float(dn_wick_ratio)
        ))
        rules.update(dict(
            call_strong=call_strong,
            put_strong=put_strong,
            score_call=_score_for("call"),
            score_put=_score_for("put"),
        ))
        return feats, rules

    # ====== MULTI-ASSET ENTRY ======
    def check_signals_pair(self, df_map: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        คืน {asset: 'call'|'put'} เฉพาะคู่ที่เป็น strong เท่านั้น
        ถ้าไม่มี strong เลย → คืน {} พร้อม meta รายงานการสแกน (mode='wait')
        """
        scan_log: List[dict] = []
        strong_entries: Dict[str, str] = {}
        best_candidate = {"asset": None, "side": None, "score": -1.0}

        for asset, df in (df_map or {}).items():
            if df is None or df.empty:
                scan_log.append({"asset": asset, "skip": "no_data"})
                continue

            feats, rules = self._features_and_rules(df)
            if feats.get("too_short"):
                scan_log.append({"asset": asset, "skip": "too_short", "n": feats.get("n", 0), "need": feats.get("need")})
                continue

            if rules.get("call_strong"):
                strong_entries[asset] = "call"
            elif rules.get("put_strong"):
                strong_entries[asset] = "put"

            # เก็บ best เพื่อรายงาน (ไม่ยิง fallback)
            sc_call = float(rules.get("score_call", 0.0))
            sc_put  = float(rules.get("score_put", 0.0))
            side = "call" if sc_call >= sc_put else "put"
            score = max(sc_call, sc_put)
            if score > best_candidate["score"]:
                best_candidate = {"asset": asset, "side": side, "score": score}

            scan_log.append({
                "asset": asset,
                "trend_up": feats.get("trend_up"),
                "trend_down": feats.get("trend_down"),
                "rsi": float(feats.get("rsi", 0)),
                "bb_pctile_prev": float(feats.get("bb_pctile_prev", 0)),
                "dist_above_atr": float(feats.get("dist_above_atr", 0)),
                "dist_below_atr": float(feats.get("dist_below_atr", 0)),
                "body_atr": float(feats.get("body_atr", 0)),
                "wick_up_ratio": float(feats.get("wick_up_ratio", 0)),
                "wick_dn_ratio": float(feats.get("wick_dn_ratio", 0)),
                "call_strong": bool(rules.get("call_strong")),
                "put_strong": bool(rules.get("put_strong")),
                "score_call": float(rules.get("score_call", 0)),
                "score_put": float(rules.get("score_put", 0)),
            })

        if strong_entries:
            self._scan_meta = {"mode": "strong", "hits": list(strong_entries.items()), "scan": scan_log}
            return strong_entries

        # ไม่มี strong → รายงานสถานะรอ
        self._scan_meta = {
            "mode": "wait",
            "best": best_candidate,
            "thresholds": {
                "min_body_atr": self.min_body_atr,
                "bb_squeeze_pctile": self.bb_squeeze_pctile,
                "breakout_dist_atr": self.breakout_dist_atr,
                "wick_ratio_max": self.wick_ratio_max,
                "rsi_cross_lookback": self.rsi_cross_lookback
            },
            "scan": scan_log
        }
        return {}
