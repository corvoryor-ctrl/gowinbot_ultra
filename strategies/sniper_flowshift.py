# strategies/sniper_flowshift.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import math, statistics, logging
log = logging.getLogger("SniperFlowShift")

from .base_strategy import BaseStrategy

class SniperFlowShift(BaseStrategy):
    """
    Sniper FlowShift (Rank)
    -----------------------
    แนวคิด: รักษาโมเมนตัมด้วย "ต่อรถหลังพัก" แทนการยิงที่ปลายหาง
    - จัดอันดับหลายคู่เงินพร้อมกัน (check_signals_rank)
    - เข้าหลัง pullback สั้นๆ กลับเข้าทิศทางหลัก (EMA8↗ EMA21↗ เหนือ EMA50 → CALL และกลับกันสำหรับ PUT)
    - Gate โดยภาวะ "คับแคบ→ขยาย" ของ Bollinger (หลีกเลี่ยงตลาดแกว่งไร้ทิศ)
    - ใช้ body/ATR และ wick ratio กรองแท่งสวย ๆ
    - ออกเป็นสัญญาณแบบจัดอันดับ score 0..1 เพื่อเลือกลำดับความเชื่อมั่น

    Settings (ค่าปริยายถูก map จาก sensitivity 0..100):
    {
      "sensitivity": 60,
      "lookback": 160,
      "ema_fast": 8, "ema_slow": 21, "ema_trend": 50,
      "rsi_len": 14,
      "bb_len": 20, "bb_k": 2.0,
      "atr_len": 14,
      "fallback_min_score": 0.55,
      "guarantee_enable": True,
      "guarantee_secs": 3600
    }
    """
    name = "Sniper FlowShift (Rank)"

    def __init__(self, settings: Optional[dict] = None):
        super().__init__(settings or {})
        s = self.settings

        # Sensitivity
        sens = s.get("sensitivity")
        self.sens01 = (float(sens)/100.0) if sens is not None else 0.60
        self.sens01 = max(0.0, min(1.0, self.sens01))

        # Core params
        self.lookback   = int(s.get("lookback", 160))
        self.ema_fast   = int(s.get("ema_fast", 8))
        self.ema_slow   = int(s.get("ema_slow", 21))
        self.ema_trend  = int(s.get("ema_trend", 50))
        self.rsi_len    = int(s.get("rsi_len", 14))
        self.bb_len     = int(s.get("bb_len", 20))
        self.bb_k       = float(s.get("bb_k", 2.0))
        self.atr_len    = int(s.get("atr_len", 14))
         # --- Micro 5s confirm/kill (default: kill-only) ---
        self.micro_5s_enable      = bool(s.get("micro_5s_enable", True))
        self.micro_5s_window_sec  = int(s.get("micro_5s_window_sec", 15))  # 10–20 แนะนำ 15
        self.micro_5s_policy      = str(s.get("micro_5s_policy", "kill")).lower()  # 'kill' | 'confirm_kill' (reserve)


        # Thresholds mapped by sensitivity
        self.min_body_atr     = float(0.35 + 0.35*self.sens01)     # 0.35..0.70
        self.wick_ratio_max   = float(0.60 + 0.20*self.sens01)     # 0.60..0.80
        self.bb_squeeze_pct   = float(40 + 20*self.sens01)         # 40..60
        self.rejoin_dist_atr  = float(0.06 + 0.10*self.sens01)     # 0.06..0.16
        self.min_slope_fast   = float(0.6 + 0.6*self.sens01)       # 0.6..1.2 * ATR/price
        self.rsi_gate         = float(50 + 8*self.sens01)          # 50..58

        # Fallback (การันตีอย่างสุภาพ)
        self.fallback_min_score = float(s.get("fallback_min_score", 0.60))
        self.guarantee_enable   = bool(s.get("guarantee_enable", True))
        self.guarantee_secs     = int(s.get("guarantee_secs", 3600))

        # ปล่อยให้ Engine ไม่บังคับ conf/gap
        self.conf_thr = None
        self.gap_thr  = None

        # แจ้ง min bars ให้ Engine
        self.required_bars = int(max(self.lookback, self.ema_trend + 20, self.bb_len + 20, self.atr_len + 20))

        self._last_scan_meta: Dict = {}

    # ---------- utils ----------
    def _ema(self, s: pd.Series, n: int) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").astype(float).ewm(span=n, adjust=False).mean()

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
        pc = c.shift(1)
        tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()

    def _rsi(self, x: pd.Series, n: int=14) -> pd.Series:
        s = pd.to_numeric(x, errors="coerce").astype(float)
        d = s.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        a = 1.0/n
        ma_up = up.ewm(alpha=a, adjust=False).mean()
        ma_dn = dn.ewm(alpha=a, adjust=False).mean()
        rs = ma_up / (ma_dn + 1e-12)
        return 100 - (100/(1+rs))

    def _bb(self, s: pd.Series, n: int, k: float):
        ma = s.rolling(n).mean()
        sd = s.rolling(n).std(ddof=0)
        upper = ma + k*sd
        lower = ma - k*sd
        width = (upper - lower) / (ma.abs() + 1e-9)
        return ma, upper, lower, width

    def _pctile(self, x: pd.Series, w: int=120) -> float:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        if len(x) < w+5:
            return 50.0
        cur = float(x.iloc[-1])
        win = x.iloc[-w:-1].to_numpy()
        rank = (win <= cur).mean()
        return float(100.0*rank)

    # ---------- core features ----------
    def _features(self, df: pd.DataFrame) -> Tuple[dict, dict]:
        feats, rules = {}, {}
        if df is None or len(df) < self.required_bars:
            feats.update({"too_short": True, "n": 0 if df is None else int(len(df)), "need": int(self.required_bars)})
            return feats, rules

        d = df.copy()
        d["ema8"]  = self._ema(d["close"], self.ema_fast)
        d["ema21"] = self._ema(d["close"], self.ema_slow)
        d["ema50"] = self._ema(d["close"], self.ema_trend)

        ema8 = d["ema8"].astype(float)
        slope = float(ema8.iloc[-1] - ema8.iloc[-5]) / max(1e-6, float(d["close"].iloc[-1]))
        atr = self._atr(d, self.atr_len)
        rsi = self._rsi(d["close"], self.rsi_len)
        ma, up, lo, width = self._bb(pd.to_numeric(d["close"], errors="coerce").astype(float), self.bb_len, self.bb_k)

        o2 = float(d["open"].iloc[-1]); c2 = float(d["close"].iloc[-1])
        hi = float(d["high"].iloc[-1]);  lo2 = float(d["low"].iloc[-1])
        body = abs(c2 - o2)
        body_atr = body / max(1e-9, float(atr.iloc[-1]))
        up_wick = max(0.0, hi - max(o2, c2))
        dn_wick = max(0.0, min(o2, c2) - lo2)
        px = c2
        up_wick_ratio = up_wick / max(1e-9, body + up_wick)
        dn_wick_ratio = dn_wick / max(1e-9, body + dn_wick)

        ema_up  = (d["ema8"].iloc[-1] > d["ema21"].iloc[-1] > d["ema50"].iloc[-1])
        ema_dn  = (d["ema8"].iloc[-1] < d["ema21"].iloc[-1] < d["ema50"].iloc[-1])
        above8  = (px > float(d["ema8"].iloc[-1]))
        below8  = (px < float(d["ema8"].iloc[-1]))
        near21  = abs(px - float(d["ema21"].iloc[-1])) / max(1e-9, float(atr.iloc[-1])) <= self.rejoin_dist_atr
        widen   = float(width.iloc[-1]) > float(width.iloc[-2])
        pctile  = self._pctile(width, w=140)

        rsi_val = float(rsi.iloc[-1])
        rsi_up  = rsi_val >= self.rsi_gate
        rsi_dn  = rsi_val <= (100 - self.rsi_gate)

        call_ok = (ema_up and slope >= (self.min_slope_fast * (atr.iloc[-1]/max(1e-9, px))) and near21 and above8 and rsi_up
                   and body_atr >= self.min_body_atr and dn_wick_ratio <= self.wick_ratio_max
                   and pctile <= self.bb_squeeze_pct and widen)

        put_ok  = (ema_dn and (-slope) >= (self.min_slope_fast * (atr.iloc[-1]/max(1e-9, px))) and near21 and below8 and rsi_dn
                   and body_atr >= self.min_body_atr and up_wick_ratio <= self.wick_ratio_max
                   and pctile <= self.bb_squeeze_pct and widen)

        def _score(side_up: bool) -> float:
            trend = 1.0 if (ema_up if side_up else ema_dn) else 0.0
            mom   = max(0.0, min(1.2, abs(slope) / max(1e-6, self.min_slope_fast*(atr.iloc[-1]/max(1e-9, px)))))
            sqz   = max(0.0, min(1.0, (self.bb_squeeze_pct - pctile) / max(1e-6, self.bb_squeeze_pct)))
            bodyn = max(0.0, min(1.2, body_atr / max(0.35, self.min_body_atr)))
            clean = 1.0 - (up_wick_ratio if side_up else dn_wick_ratio)
            clean = max(0.0, min(1.0, clean))
            return float(0.35*trend + 0.35*mom + 0.20*sqz + 0.10*0.5*(bodyn+clean))

        feats.update(dict(
            n=int(len(d)), need=int(self.required_bars),
            ema_up=bool(ema_up), ema_dn=bool(ema_dn),
            slope=float(slope), body_atr=float(body_atr),
            pctile=float(pctile), widen=bool(widen),
            rsi=float(rsi_val), rsi_gate=float(self.rsi_gate),
            wick_up_ratio=float(up_wick_ratio), wick_dn_ratio=float(dn_wick_ratio),
        ))
        rules.update(dict(call_ok=bool(call_ok), put_ok=bool(put_ok),
                          score_call=float(_score(True)), score_put=float(_score(False))))
        return feats, rules

    def check_signal(self, candles: pd.DataFrame) -> str:
        return "none"

    # --- PATCH 2: fallback ให้ทำงานแม้ไม่มีใครผ่านเกต + เกณฑ์ผ่อนเมื่อ sens ต่ำ ---
    def check_signals_rank(self, df_map):
        rank, scan_log = [], []
        best_gate = None   # ผู้ชนะที่ "ผ่านเกต"
        best_any  = None   # ผู้ชนะจาก proto score (มีเสมอ)

        for asset, df in (df_map or {}).items():
            feats, rules = self._features(df)
            if feats.get("too_short"):
                scan_log.append({"asset": asset, "skip": "too_short", "n": feats.get("n",0), "need": feats.get("need",0)})
                continue

            call_ok = bool(rules.get("call_ok"))
            put_ok  = bool(rules.get("put_ok"))
            side = "call" if call_ok else ("put" if put_ok else "none")

            score_call = float(rules.get("score_call", 0.0))
            score_put  = float(rules.get("score_put", 0.0))
            proto_side  = "call" if score_call >= score_put else "put"
            proto_score = max(score_call, score_put)

            meta = dict(asset=asset, side=side,
                        score=(score_call if side=="call" else score_put if side=="put" else 0.0),
                        proto_side=proto_side, proto_score=proto_score,
                        feats=feats, rules=rules)
            scan_log.append(meta)

            if side != "none":
                rank.append({"asset": asset, "signal": side, "score": meta["score"]})
                if (best_gate is None) or (meta["score"] > best_gate["score"]):
                    best_gate = {"asset": asset, "signal": side, "score": meta["score"]}

            if (best_any is None) or (proto_score > best_any["score"]):
                best_any = {"asset": asset, "signal": proto_side, "score": proto_score}

        rank.sort(key=lambda x: x["score"], reverse=True)

        # การันตีแบบสุภาพ: ยอมหยิบ "best_any" เมื่อไม่มีใครผ่านเกต แต่คะแนนถึงเกณฑ์
        if not rank and self.guarantee_enable and best_any is not None:
            # ยิ่ง sensitivity ต่ำ เกณฑ์ยิ่ง "ผ่อน" (เดิมของต้นกลับทิศ)
            thresh = self.fallback_min_score * (0.8 + 0.2*self.sens01)   # sens=0 -> 0.44x, sens=1 -> 0.55x (เมื่อ default=0.55)
            if best_any["score"] >= thresh:
                rank = [best_any]

        self._last_scan_meta = {"mode": "rank", "best": (best_gate or best_any), "scan": scan_log}
        return rank

    def get_signal_meta(self):
        return self._last_scan_meta



