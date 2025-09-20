# core/orchestrator.py
# The central coordinator for slot-based entries, martingale discipline, and max-order arbitration.
from __future__ import annotations
import time
from typing import Dict, List, Any, Callable, Optional
from .trend_engine import TrendEngine

Signal = Dict[str, Any]
PlaceOrderFn = Callable[[str, str, float, Dict[str, Any]], bool]  # (asset, direction, amount, meta) -> success
PayoutFn = Callable[[str, int], float]
AvailFn = Callable[[str, int], bool]

class Orchestrator:
    def __init__(
        self,
        assets: List[str],
        trend_engine: TrendEngine,
        get_signal_1m: Callable[[str], Signal],
        get_signal_5m: Callable[[str, bool], Signal],
        place_order: PlaceOrderFn,
        get_payout: PayoutFn,
        is_available: AvailFn,
        settings: Dict[str, Any],
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.assets = assets
        self.trend = trend_engine
        self.get_signal_1m = get_signal_1m
        self.get_signal_5m = get_signal_5m
        self.place_order = place_order
        self.get_payout = get_payout
        self.is_available = is_available
        self.s = settings
        self.log = logger or (lambda msg: None)

        # runtime
        self.candidates: List[Signal] = []
        self.mg_state: Dict[str, Dict[str, Any]] = {}  # per-asset martingale info
        self.cooldown_ts: Dict[str, float] = {}
        self.trade_lock_until: float = 0.0

    # ==== Event hooks from EventBus ====
    def on_minute_close(self, now_epoch: float):
        # Refresh trends periodically (every minute is ok; engine caches internally)
        self.trend.refresh(self.assets)
        # Prepare candidates for 1m entries for the NEXT slot
        # (we will still re-validate at :00)
        if self.s.get("enable_1m", True):
            self._collect_candidates(tf=1, now_epoch=now_epoch)

    def on_5min_close(self, now_epoch: float):
        # Collect 5m primary signals at the 5m boundary
        if self.s.get("enable_5m", True):
            self._collect_candidates(tf=5, now_epoch=now_epoch)

    def on_top_of_minute(self, now_epoch: float):
        # At :00, arbitrate and fire respecting Max Order & Martingale discipline.
        self._execute_slot(now_epoch)

    # ==== Internal helpers ====
    def _collect_candidates(self, tf: int, now_epoch: float):
        enable_micro = bool(self.s.get("micro_filter_5m", True))
        trend_mode = self.s.get("trend_filter_mode", "weak")
        asset_cooldown = float(self.s.get("asset_cooldown_sec", 5.0))

        for a in self.assets:
            # 1) respect per-asset cooldown (ถ้ายังไม่หมดเวลา ข้ามรอบนี้ไปก่อน)
            if a in self.cooldown_ts and now_epoch < self.cooldown_ts[a]:
                self.log(f"[filter][cooldown] {a} skip until {int(self.cooldown_ts[a]-now_epoch)}s")
                continue

            # 2) ดึงสัญญาณตาม TF (กัน error ไว้ชั้นนอก)
            try:
                if tf == 1:
                    sig = self.get_signal_1m(a)               # expected: {'dir','conf','gap',...}
                else:
                    sig = self.get_signal_5m(a, enable_micro)  # expected: {'dir','conf','gap',...}
            except Exception as e:
                self.log(f"[signal-error] {a} tf={tf}m {e}")
                continue
            if not isinstance(sig, dict):
                continue

            # 3) normalize + validate direction
            d = (sig.get("dir") or "").lower()
            if d not in ("call", "put"):
                continue

            # 4) trend filter
            bias = self.trend.get_bias(a)
            if not TrendEngine.pass_filter(bias, d, trend_mode):
                self.log(f"[filter][trend:{trend_mode}] {a} dir={d} blocked H1={bias['H1']['trend']} H4={bias['H4']['trend']}")
                continue

            # 5) availability
            dur = 60 if tf == 1 else 300
            if not self.is_available(a, dur):
                self.log(f"[filter][available] {a} dur={dur}s not tradable now")
                continue

            # 6) enrich candidate (ใส่ asset ที่ขาด + payout ฯลฯ)
            cand = {
                "asset": a,
                "dir": d,
                "conf": float(sig.get("conf", 0.0)),
                "gap": float(sig.get("gap", 0.0)),
                "tf": tf,
                "bias": bias,
                "payout": float(self.get_payout(a, dur) or 0.0),
                "ts": now_epoch,
            }
            # แนบข้อมูลเดิมเผื่ออยากดู reason/debug ใน log ภายหลัง
            if "reason" in sig: cand["reason"] = sig["reason"]
            if "debug" in sig:  cand["debug"]  = sig["debug"]

            self.candidates.append(cand)
            self.log(f"[candidate][{tf}m] {a} dir={d} conf={cand['conf']:.2f} gap={cand['gap']:.3f} payout={cand['payout']:.2f}")


    def _execute_slot(self, now_epoch: float):
        # prevent double-fire within a short window
        lock_window = float(self.s.get("trade_lock_window_sec", 2.0))
        if now_epoch < self.trade_lock_until:
            return
        self.trade_lock_until = now_epoch + lock_window

        # Re-check martingale queue first (if any) – but still respect filters
        slot_candidates = list(self.candidates)
        self.candidates.clear()

        # Apply Max Order arbitration
        max_orders = int(self.s.get("max_orders_per_slot", 1))
        if not slot_candidates:
            self.log("[slot] no candidates")
            return

        chosen = self._arbitrate(slot_candidates, k=max_orders)
        self.log(f"[arb] chosen={[(c['asset'], c['dir']) for c in chosen]}")

        # Execute
        amount = float(self.s.get("base_amount", 1.0))
        mg_enabled = bool(self.s.get("mg_enabled", True))
        mg_mult = float(self.s.get("mg_multiplier", 2.2))
        mg_max = int(self.s.get("mg_steps", 0))

        for sig in chosen:
            a = sig["asset"]
            direction = sig["dir"]
            dur = 60 if sig["tf"] == 1 else 300
            # martingale state for asset
            st = self.mg_state.get(a, {"step": 0, "next_amount": amount, "active": False})
            use_amount = st["next_amount"] if st["active"] else amount

            meta = {
                "tf": sig["tf"],
                "conf": sig["conf"],
                "gap": sig["gap"],
                "payout": sig.get("payout", 0.0),
                "trend": sig.get("bias"),
                "mg_step": st["step"] if st["active"] else 0,
                "slot_epoch": now_epoch,
            }
            ok = self.place_order(a, direction, use_amount, meta)
            if ok:
                self.log(f"[ENTER][{sig['tf']}m] {a} {direction.upper()} ${use_amount:.2f} conf={sig['conf']:.2f} gap={sig['gap']:.3f} mg_step={meta['mg_step']}")
                # set cooldown per-asset (avoid immediate duplicates)
                self.cooldown_ts[a] = now_epoch + float(self.s.get("asset_cooldown_sec", 5.0))
            else:
                self.log(f"[ENTER-FAIL] {a} {direction} amount={use_amount}")

        # NOTE: Result handling (win/loss) should call `on_trade_result(asset, win)` externally.

    def on_trade_result(self, asset: str, win: bool):
        """
        Should be called by the main engine when a result comes back.
        Manages martingale state.
        """
        amount = float(self.s.get("base_amount", 1.0))
        mg_enabled = bool(self.s.get("mg_enabled", True))
        mg_mult = float(self.s.get("mg_multiplier", 2.2))
        mg_max = int(self.s.get("mg_steps", 0))

        st = self.mg_state.get(asset, {"step": 0, "next_amount": amount, "active": False})
        if win:
            st.update({"step": 0, "next_amount": amount, "active": False})
        else:
            if mg_enabled and st["step"] < mg_max:
                st["step"] += 1
                st["next_amount"] = round(st["next_amount"] * mg_mult, 2)
                st["active"] = True
            else:
                st.update({"step": 0, "next_amount": amount, "active": False})
        self.mg_state[asset] = st

    # ==== Arbitration ====
    def _arbitrate(self, cands: List[Signal], k: int = 1) -> List[Signal]:
        mode = self.s.get("arb_mode", "score_mix")
        res = sorted(cands, key=lambda c: self._score(c, mode), reverse=True)
        if self.s.get("dedupe_per_asset", False):
            seen = set()
            deduped = []
            for c in res:
                a = c.get("asset")
                if a in seen:
                    continue
                seen.add(a)
                deduped.append(c)
            res = deduped
        return res[:max(1, k)]

    def _score(self, c: Signal, mode: str) -> float:
        if mode == "best_conf":
            return float(c.get("conf", 0.0))
        # mixed score
        conf = float(c.get("conf", 0.0))
        gap = float(c.get("gap", 0.0))
        payout = float(c.get("payout", 0.0))
        # configurable weights could be added; for now, sane defaults
        return 0.6*conf + 0.3*gap + 0.1*payout
