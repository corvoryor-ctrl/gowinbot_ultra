# backend/strategies/strategy_r2s.py
from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import logging

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

class R2SStrategy(BaseStrategy):
    """
    R2S (RSI-to-Structure): RSI อ่านแรง + SMA เป็นจุดเข้า + โครงสร้างช่วยคัดกราฟ
    เหมาะกับ Binary 5 นาที: รอสัญญาณคม (แตะ SMA + RSI มุ่งหน้า/ครอส) แล้วค่อยเข้า

    🔧 ปรับง่ายด้วย sensitivity (0..100):
      - rsi_up  ≈ 60 + 16*sens
      - rsi_dn  ≈ 40 - 16*sens
      - ระยะห่างจาก SMA (max_dist_pct) จะคูณ (1 - 0.5*sens)
      - บังคับ RSI ครอสเส้น 50 เมื่อ sens ≥ 40%
      - ต้องมีขนาด body ขั้นต่ำ ≈ 2% * sens
      - บังคับ “เทรนด์สอดคล้อง” อัตโนมัติเมื่อ sens ≥ ~25%

    Settings (รองรับเดิม + เพิ่ม speak):
    {
        "rsi_len": 14,
        "rsi_up": 68,
        "rsi_dn": 32,
        "sma_entry_len": 4,
        "sma_trend_len": 20,
        "use_close_confirm": 1,
        "touch_mode": "wick",        # "wick" หรือ "body"
        "max_dist_pct": 0.004,       # 0.4%
        "need_trend_agree": 1,       # 0/1
        "min_body_bp": null,         # bp; ถ้า null จะ derive จาก sensitivity
        "sensitivity": 50,           # 0..100
        "debug_log": 0,              # 1 = พ่น META (dict)
        "speak": 1,                  # 1 = พูดสรุปอ่านง่าย, 0 = ปิด
        "speak_level": 1             # 1 = ย่อ, 2 = ละเอียด
    }
    """
    name = "R2S Strategy (RSI+SMA)"
    preferred_timeframe = "5m"
    prefer_binary = True

    def __init__(self, settings=None):
        s = settings or {}
        self.s = s
        # --- Base params / เดิมยังใช้ได้ ---
        self.rsi_len        = int(s.get("rsi_len", 14))
        self._user_rsi_up   = s.get("rsi_up", None)
        self._user_rsi_dn   = s.get("rsi_dn", None)
        self.sma_entry_len  = int(s.get("sma_entry_len", 4))
        self.sma_trend_len  = int(s.get("sma_trend_len", 20))
        self.use_close_conf = int(s.get("use_close_confirm", 1))
        self.touch_mode     = str(s.get("touch_mode", "wick")).lower()
        self.base_max_dist  = float(s.get("max_dist_pct", 0.004))   # 0.4%
        self._user_need_trend_ag = s.get("need_trend_agree", None)
        self._user_min_body_bp   = s.get("min_body_bp", None)

        self.sens           = float((s.get("sensitivity") or 50)) / 100.0  # 0..1
        self.debug_log      = bool(int(s.get("debug_log", 0)))

        # --- เพิ่ม ---
        self.speak          = int(s.get("speak", 1))   # 0=off, 1=brief, 2=verbose
        self.speak_level    = int(s.get("speak_level", 1))
        self.last_meta = {}

    # ---------------- Derived thresholds ----------------
    def _eff_rsi_up(self) -> float:
        if self._user_rsi_up is not None:
            try: return float(self._user_rsi_up)
            except: pass
        return 60.0 + 16.0 * self.sens

    def _eff_rsi_dn(self) -> float:
        if self._user_rsi_dn is not None:
            try: return float(self._user_rsi_dn)
            except: pass
        return 40.0 - 16.0 * self.sens

    def _need_cross50(self) -> bool:
        return self.sens >= 0.40

    def _eff_max_dist(self) -> float:
        return self.base_max_dist * (1.0 - 0.5 * self.sens)

    def _eff_need_trend_agree(self) -> bool:
        if self._user_need_trend_ag is not None:
            try: return bool(int(self._user_need_trend_ag))
            except: return bool(self._user_need_trend_ag)
        return self.sens >= 0.25

    def _eff_min_body_ratio(self) -> float:
        if self._user_min_body_bp is not None:
            try: return float(self._user_min_body_bp) / 10000.0
            except: pass
        return 0.02 * self.sens

    # ---------------- Helpers ----------------
    def _touched_entry_sma(self, row, touch_mode: str) -> bool:
        if pd.isna(row["sma_e"]): return False
        if touch_mode == "body":
            lo = min(row["open"], row["close"])
            hi = max(row["open"], row["close"])
            return lo <= row["sma_e"] <= hi
        return row["min"] <= row["sma_e"] <= row["max"]

    def _close_confirm_up(self, row, min_body_ratio: float) -> bool:
        if row["close"] <= row["open"]: return False
        body_ratio = abs(row["close"] - row["open"]) / (abs(row["close"]) + 1e-12)
        return body_ratio >= min_body_ratio

    def _close_confirm_dn(self, row, min_body_ratio: float) -> bool:
        if row["close"] >= row["open"]: return False
        body_ratio = abs(row["close"] - row["open"]) / (abs(row["close"]) + 1e-12)
        return body_ratio >= min_body_ratio

    def _near_enough(self, row, max_dist_pct: float) -> bool:
        dist = abs(row["close"] - row["sma_e"]) / (abs(row["close"]) + 1e-12)
        return dist <= max_dist_pct

    def _dist_pct(self, row) -> float:
        return abs(row["close"] - row["sma_e"]) / (abs(row["close"]) + 1e-12)

    def _body_ratio(self, row) -> float:
        return abs(row["close"] - row["open"]) / (abs(row["close"]) + 1e-12)

    def _trend_agree_up(self, prev_row, last_row) -> bool:
        if pd.isna(prev_row["sma_tr"]) or pd.isna(last_row["sma_tr"]): return True
        slope_ok = (last_row["sma_tr"] - prev_row["sma_tr"]) >= 0
        price_ok = last_row["close"] >= last_row["sma_tr"]
        return slope_ok or price_ok

    def _trend_agree_dn(self, prev_row, last_row) -> bool:
        if pd.isna(prev_row["sma_tr"]) or pd.isna(last_row["sma_tr"]): return True
        slope_ok = (last_row["sma_tr"] - prev_row["sma_tr"]) <= 0
        price_ok = last_row["close"] <= last_row["sma_tr"]
        return slope_ok or price_ok

    def _calc(self, df: pd.DataFrame):
        d = df.copy().reset_index(drop=True)
        d["rsi"]    = rsi(d["close"], self.rsi_len)
        d["sma_e"]  = d["close"].rolling(self.sma_entry_len).mean()
        d["sma_tr"] = d["close"].rolling(self.sma_trend_len).mean()
        return d

    # ---------------- Core ----------------
    def check_signal(self, df: pd.DataFrame) -> str:
        # reset meta
        self.last_meta = {}

        if df is None or df.empty:
            self._meta(reason="empty_df")
            return "none"

        need_base = {"open","close"}
        if not need_base.issubset(df.columns):
            self._meta(reason="missing_cols", need=list(need_base), have=list(df.columns))
            return "none"

        work = df.copy()
        if "min" not in work.columns and "low" in work.columns:
            work["min"] = work["low"]
        if "max" not in work.columns and "high" in work.columns:
            work["max"] = work["high"]

        need_full = {"open","close","min","max"}
        if not need_full.issubset(work.columns):
            self._meta(reason="missing_cols", need=list(need_full), have=list(work.columns))
            return "none"

        min_len = max(self.rsi_len + 3, self.sma_trend_len + 3, self.sma_entry_len + 3)
        if len(work) < min_len:
            self._meta(reason="not_enough_bars", need=min_len, got=len(work))
            return "none"

        d = self._calc(work)
        last = d.index[-1]; prev = last - 1

        rsi_prev = float(d.loc[prev, "rsi"])
        rsi_last = float(d.loc[last, "rsi"])
        row_prev = d.loc[prev]
        row_last = d.loc[last]

        eff_up   = self._eff_rsi_up()
        eff_dn   = self._eff_rsi_dn()
        need_crs = self._need_cross50()
        max_dist = self._eff_max_dist()
        need_tr  = self._eff_need_trend_agree()
        min_body = self._eff_min_body_ratio()

        touched = self._touched_entry_sma(row_last, self.touch_mode)
        if not touched:
            self._meta(reason="no_touch_entry_sma", touch_mode=self.touch_mode, sma=row_last["sma_e"])
            self._say_wait(d, row_prev, row_last, rsi_prev, rsi_last, eff_up, eff_dn, need_crs, max_dist, need_tr, min_body)
            return "none"

        dist_pct = self._dist_pct(row_last)
        if not self._near_enough(row_last, max_dist):
            self._meta(reason="too_far_from_entry_sma", dist_pct=dist_pct, max_dist_pct=max_dist)
            self._say_wait(d, row_prev, row_last, rsi_prev, rsi_last, eff_up, eff_dn, need_crs, max_dist, need_tr, min_body)
            return "none"

        # RSI conditions
        cross50_up_ok  = (rsi_prev < 50 <= rsi_last)
        cross50_dn_ok  = (rsi_prev > 50 >= rsi_last)
        rsi_up_move    = (rsi_last > rsi_prev)
        rsi_dn_move    = (rsi_last < rsi_prev)
        rsi_reach_up   = (rsi_last >= eff_up)
        rsi_reach_dn   = (rsi_last <= eff_dn)

        call_rsi_ok = rsi_up_move and ( rsi_reach_up or ((not need_crs) and rsi_up_move) or (need_crs and cross50_up_ok) )
        put_rsi_ok  = rsi_dn_move and ( rsi_reach_dn or ((not need_crs) and rsi_dn_move) or (need_crs and cross50_dn_ok) )

        body_ratio = self._body_ratio(row_last)
        call_body_ok = (not self.use_close_conf) or self._close_confirm_up(row_last, min_body)
        put_body_ok  = (not self.use_close_conf) or self._close_confirm_dn(row_last, min_body)

        trend_up_ok = (not need_tr) or self._trend_agree_up(row_prev, row_last)
        trend_dn_ok = (not need_tr) or self._trend_agree_dn(row_prev, row_last)

        # progress (กี่ข้อผ่านจาก checklist)
        checks_call = [
            True,                      # touched & near แล้วถึงได้มาถึงตรงนี้
            True,
            rsi_up_move,
            (rsi_reach_up or ((not need_crs) and rsi_up_move) or (need_crs and cross50_up_ok)),
            call_body_ok,
            trend_up_ok
        ]
        checks_put = [
            True,
            True,
            rsi_dn_move,
            (rsi_reach_dn or ((not need_crs) and rsi_dn_move) or (need_crs and cross50_dn_ok)),
            put_body_ok,
            trend_dn_ok
        ]
        prog_call = sum(bool(x) for x in checks_call)
        prog_put  = sum(bool(x) for x in checks_put)
        total_checks = 6

        base_meta = dict(
            rsi_prev=rsi_prev, rsi_last=rsi_last,
            eff_rsi_up=eff_up, eff_rsi_dn=eff_dn,
            need_cross50=need_crs,
            cross50_up_ok=cross50_up_ok, cross50_dn_ok=cross50_dn_ok,
            min_body_ratio=min_body, body_ratio=body_ratio,
            max_dist_pct=max_dist, dist_pct=dist_pct,
            need_trend_agree=need_tr,
            trend_up_ok=trend_up_ok, trend_dn_ok=trend_dn_ok,
            touched_entry_sma=bool(touched), touch_mode=self.touch_mode,
            progress_call=f"{prog_call}/{total_checks}",
            progress_put=f"{prog_put}/{total_checks}"
        )

        # สรุปผล
        if call_rsi_ok and call_body_ok and trend_up_ok:
            self._meta(dir="call", **base_meta)
            self._say_enter("CALL", base_meta)
            return "call"

        if put_rsi_ok and put_body_ok and trend_dn_ok:
            self._meta(dir="put", **base_meta)
            self._say_enter("PUT", base_meta)
            return "put"

        self._meta(reason="filters_failed", **base_meta)
        self._say_wait(d, row_prev, row_last, rsi_prev, rsi_last, eff_up, eff_dn, need_crs, max_dist, need_tr, min_body, extra=base_meta)
        return "none"

    # ---------------- Meta / Speak ----------------
    def _meta(self, **kw):
        self.last_meta = kw
        if self.debug_log:
            try:
                logging.info(f"[R2S] META: {kw}")
            except Exception:
                pass

    def _say_enter(self, direction: str, m: dict):
        if not self.speak: return
        msg = (f"✅ [R2S] ENTER → {direction} | "
               f"rsi {m.get('rsi_last'):.2f} "
               f"({'↑' if m.get('rsi_last',0) > m.get('rsi_prev',0) else '↓'} from {m.get('rsi_prev'):.2f}, "
               f"need {'≥'+str(round(m.get('eff_rsi_up'),2)) if direction=='CALL' else '≤'+str(round(m.get('eff_rsi_dn'),2))} ✓"
               f"{', cross50 ✓' if m.get('need_cross50') else ', cross50 n/a'}) • "
               f"body {m.get('body_ratio')*100:.2f}% "
               f"≥ {m.get('min_body_ratio')*100:.2f}% ✓ • "
               f"dist {m.get('dist_pct')*100:.2f}% ≤ {m.get('max_dist_pct')*100:.2f}% ✓ • "
               f"trend {'✓' if (m.get('trend_up_ok') if direction=='CALL' else m.get('trend_dn_ok')) else '✗'} • "
               f"touch {m.get('touch_mode','?')} ✓")
        logging.info(msg)

    def _say_wait(self, d, row_prev, row_last, rsi_prev, rsi_last, eff_up, eff_dn, need_crs, max_dist, need_tr, min_body, extra=None):
        if not self.speak: return
        try:
            dist_pct = self._dist_pct(row_last)
            body_rt  = self._body_ratio(row_last)

            cross_up_ok = (rsi_prev < 50 <= rsi_last)
            cross_dn_ok = (rsi_prev > 50 >= rsi_last)
            up_move     = rsi_last > rsi_prev
            dn_move     = rsi_last < rsi_prev

            # ประมาณ progress แบบคร่าว
            prog_parts_call = [
                True,  # touched
                dist_pct <= max_dist,
                up_move,
                (rsi_last >= eff_up) or ((not need_crs) and up_move) or (need_crs and cross_up_ok),
                (not self.use_close_conf) or self._close_confirm_up(row_last, min_body),
                (not need_tr) or self._trend_agree_up(d.loc[row_prev.name], row_last)
            ]
            prog_parts_put = [
                True,
                dist_pct <= max_dist,
                dn_move,
                (rsi_last <= eff_dn) or ((not need_crs) and dn_move) or (need_crs and cross_dn_ok),
                (not self.use_close_conf) or self._close_confirm_dn(row_last, min_body),
                (not need_tr) or self._trend_agree_dn(d.loc[row_prev.name], row_last)
            ]
            prog_call = sum(bool(x) for x in prog_parts_call)
            prog_put  = sum(bool(x) for x in prog_parts_put)

            # บอก “ยังขาดอะไร”
            lacks = []
            if dist_pct > max_dist:
                lacks.append(f"dist {dist_pct*100:.2f}%→≤{max_dist*100:.2f}%")
            if self.use_close_conf and body_rt < min_body:
                lacks.append(f"body {body_rt*100:.2f}%→≥{min_body*100:.2f}%")
            if need_tr:
                if not self._trend_agree_up(d.loc[row_prev.name], row_last) and not self._trend_agree_dn(d.loc[row_prev.name], row_last):
                    lacks.append("trend agree")
            if need_crs and not (cross_up_ok or cross_dn_ok):
                lacks.append("cross50")

            if not up_move and not dn_move:
                lacks.append("RSI move")

            # พูดแบบย่อ/ยาว
            if self.speak_level >= 2:
                msg = (f"🕒 [R2S] WAIT — CALL {prog_call}/6, PUT {prog_put}/6 | "
                       f"rsi {rsi_last:.2f} ({'↑' if up_move else ('↓' if dn_move else '•')}; prev {rsi_prev:.2f}), "
                       f"need CALL ≥{eff_up:.2f} / PUT ≤{eff_dn:.2f} "
                       f"{'(cross50 req)' if need_crs else ''} • "
                       f"body {body_rt*100:.2f}% / min {min_body*100:.2f}% • "
                       f"dist {dist_pct*100:.2f}% / max {max_dist*100:.2f}% • "
                       f"trend {'on' if need_tr else 'n/a'}")
            else:
                msg = (f"🕒 [R2S] WAIT — CALL {prog_call}/6, PUT {prog_put}/6 | "
                       f"rsi {rsi_last:.2f} ({'↑' if up_move else '↓' if dn_move else '•'}) • "
                       f"body {body_rt*100:.2f}%/≥{min_body*100:.2f}% • "
                       f"dist {dist_pct*100:.2f}%/≤{max_dist*100:.2f}% "
                       f"{'• cross50' if need_crs else ''}")

            if lacks:
                msg += " • need: " + ", ".join(lacks)

            logging.info(msg)
        except Exception:
            pass

    def get_signal_meta(self):
        return self.last_meta
