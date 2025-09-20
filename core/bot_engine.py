#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/core/bot_engine.py — stable & main.py-aligned

Highlights:
- stop(keep_connection: bool = True): ถ้า True จะไม่ logout/stop_keepalive (คง session ให้ UI ใช้ต่อ)
- เพิ่ม MG state adapter: self.martingale.get_state()/load_state() + get_mg_state()/set_mg_state()
- คง behavior เดิม: lifecycle, MG, multi-asset modes, preview/confirm, AI hooks, resilient net

Drop-in: แทนไฟล์เดิมได้ทันที
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Project-level modules expected at root
from iq_wrapper import IQWrapper
from strategies.base_strategy import BaseStrategy
from database import trade_logger

# Optional modules
try:
    from core.cancel_token import CancelToken  # type: ignore
except Exception:  # pragma: no cover
    class CancelToken:  # minimal fallback
        def __init__(self) -> None:
            self._stop = False
        def cancel(self) -> None:
            self._stop = True
        @property
        def cancelled(self) -> bool:
            return self._stop

# Indicator config + calc
try:
    from .indicator_config import get_indicator_config  # type: ignore
except Exception:
    def get_indicator_config(_: int) -> Dict[str, Any]:
        return {}

try:
    from strategies.indicator_suite import stoch as stoch_calc  # type: ignore
except Exception:
    def stoch_calc(hi, lo, cl, k, d, smooth):  # naive fallback
        hi = np.asarray(hi, dtype=float)
        lo = np.asarray(lo, dtype=float)
        cl = np.asarray(cl, dtype=float)
        n = max(int(k), 1)
        roll_hi = pd.Series(hi).rolling(n, min_periods=1).max().to_numpy()
        roll_lo = pd.Series(lo).rolling(n, min_periods=1).min().to_numpy()
        denom = np.maximum(roll_hi - roll_lo, 1e-9)
        k_arr = 100.0 * (cl - roll_lo) / denom
        d_arr = pd.Series(k_arr).rolling(max(int(d), 1), min_periods=1).mean().to_numpy()
        return k_arr, d_arr

# Optional AI meta
try:
    import ai_meta  # type: ignore
except Exception:  # pragma: no cover
    ai_meta = None


# =============================================================================
# Bot Engine
# =============================================================================
class BotEngine:
    """Main loop: candles, filters, execution, result check, MG management."""

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None

        # Fast-stop event
        self._stop_evt = threading.Event()
        # Lifecycle lock + run generation
        self._lifecycle_lock = threading.RLock()
        self._run_id: int = 0
        # Phase for debug
        self._phase: str = "idle"
        self.iq: Optional[IQWrapper] = None
        self.strategy: Optional[BaseStrategy] = None
        self.settings: Dict[str, Any] = {}

        # Martingale / execution control
        self.martingale_state: Dict[str, int] = {}
        self.martingale_lock = threading.Lock()
        self.open_until: Dict[str, float] = {}          # lock timestamps per key
        self._last_slot_fired: Dict[str, int] = {}      # slot id per key → prevent double fire

        # Trend cache
        self._trend_cache: Dict[str, Dict[str, Any]] = {}

        # HUD / Session
        self.hud_max_step: int = 0
        self.reset_session_stats()

        # Preview (mid-candle scan)
        self._last_preview_slot: Optional[int] = None   # slot already previewed (avoid duplicate)
        self._preview_cache: Dict[str, str] = {}        # asset → side from latest preview

        # Anti-flip guard
        self.anti_flip_enable: bool = True
        self.anti_flip_body_atr: float = 0.15  # 0.15×ATR default

        self._trade_lock = threading.RLock()

        # Async tokens (future use)
        self._stop_token: Optional[CancelToken] = None
        self._task: Optional[asyncio.Task] = None
        self._phase: str = "idle"       # idle|running|stopping

        # MG adapter for endpoints that expect .martingale.get_state()/load_state()
        class _MGAdapter:
            def __init__(self, engine: "BotEngine") -> None:
                self._e = engine
            def get_state(self) -> Dict[str, int]:
                with self._e.martingale_lock:
                    return dict(self._e.martingale_state)
            def load_state(self, state: Dict[str, int]) -> None:
                if not isinstance(state, dict):
                    return
                with self._e.martingale_lock:
                    for k, v in state.items():
                        try:
                            self._e.martingale_state[str(k)] = int(v)
                        except Exception:
                            continue
        self.martingale = _MGAdapter(self)

    # ------------------------------------------------------------------ #
    # Session / HUD                                                      #
    # ------------------------------------------------------------------ #
    def reset_session_stats(self) -> None:
        self.session_win = 0
        self.session_lose = 0
        self.session_draw = 0
        self.session_profit = 0.0
        self.session_max_step = 0
        self.session_started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _bump_hud_steps(self, current_step: int) -> None:
        try:
            self.session_max_step = max(self.session_max_step, int(current_step))
            self.hud_max_step = max(self.hud_max_step, int(current_step))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Time Utilities                                                     #
    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        """Prefer broker server time when wrapper exposes it."""
        try:
            if self.iq and hasattr(self.iq, "server_timestamp"):
                ts = self.iq.server_timestamp()  # type: ignore[attr-defined]
                if ts:
                    return float(ts)
        except Exception:
            pass
        return time.time()

    def _secs_to_next_minute(self, now_ts: Optional[float] = None) -> float:
        now_ts = self._now() if now_ts is None else now_ts
        return 60 - (int(now_ts) % 60)

    def _secs_to_next_5m(self, now_ts: Optional[float] = None) -> float:
        now_ts = self._now() if now_ts is None else now_ts
        return 300 - (int(now_ts) % 300)

    def _align_to_closed_bar(self, period_sec: int, now_ts: Optional[float] = None) -> int:
        now_ts = self._now() if now_ts is None else now_ts
        return int(now_ts) - (int(now_ts) % int(period_sec))

    def _wait_until_preview(self, duration_min: int, lead: int, preview_sec: int) -> None:
        """Wait until we are inside the mid-candle preview window (1m only)."""
        if duration_min != 1 or preview_sec <= 0:
            return
        target_remain = max(int(lead) + int(preview_sec), int(lead) + 1)
        while self.is_running:
            try:
                remain = self._secs_to_next_minute()
                if remain <= target_remain:
                    break
                sleep_s = max(0.2, min(3.0, remain - target_remain))
            except Exception:
                sleep_s = 0.5
            self._wait(sleep_s)

    # ------------------------------------------------------------------ #
    # Slot Handling                                                      #
    # ------------------------------------------------------------------ #
    def _is_slot_window(self, duration_min: int, lead: int) -> bool:
        now_ts = self._now()
        if duration_min == 1:
            return self._secs_to_next_minute(now_ts) <= max(1, int(lead))
        return self._secs_to_next_5m(now_ts) <= max(1, int(lead))

    def _wait_until_slot(self, duration_min: int, lead: int) -> None:
        """Active-wait until last seconds of current minute/5m cycle."""
        while self.is_running and not self._is_slot_window(duration_min, lead):
            try:
                s = (self._secs_to_next_minute() if duration_min == 1 else self._secs_to_next_5m()) - lead - 0.1
                s = max(0.1, min(1.0, s))
            except Exception:
                s = 0.5
            self._wait(s)

    def _slot_id(self, period_sec: int, now_ts: Optional[float] = None) -> int:
        """Use last-closed bar time as slot id to ensure one trade per slot/key."""
        return self._align_to_closed_bar(period_sec, now_ts)

    # ------------------------------------------------------------------ #
    # Trend & Filters                                                    #
    # ------------------------------------------------------------------ #
    def _get_trend_dirs(self, asset: str) -> Tuple[str, str]:
        """Return (dir_5m, dir_15m) ∈ {"up","down","flat"} with ~30s cache."""
        try:
            cache = self._trend_cache.get(asset, {})
            if cache and (self._now() - cache.get("ts", 0.0)) < 30:
                return cache.get("d5", "flat"), cache.get("d15", "flat")

            rows5 = None
            rows15 = None
            try:
                rows5 = self.iq.get_candles_tf(asset, 5, count=40)  # type: ignore[union-attr]
            except TypeError:
                rows5 = self.iq.get_candles_tf(asset, 5, count=40)  # type: ignore[union-attr]
            try:
                rows15 = self.iq.get_candles_tf(asset, 15, count=40)  # type: ignore[union-attr]
            except TypeError:
                rows15 = self.iq.get_candles_tf(asset, 15, count=40)  # type: ignore[union-attr]

            rows5 = rows5 or []
            rows15 = rows15 or []

            def _dir(candles: List[Dict[str, Any]]) -> str:
                if not candles or len(candles) < 8:
                    return "flat"
                df = pd.DataFrame(candles).sort_values("ts")
                delta = float(df["close"].iloc[-1]) - float(df["close"].iloc[-8])
                if delta > 0:
                    return "up"
                if delta < 0:
                    return "down"
                return "flat"

            d5, d15 = _dir(rows5), _dir(rows15)
            self._trend_cache[asset] = {"ts": self._now(), "d5": d5, "d15": d15}
            return d5, d15
        except Exception:
            return "flat", "flat"

    def _trend_pass(self, asset: str, direction: str, mode: str) -> bool:
        """Trend gate with modes: off | weak | strict."""
        try:
            mode = (mode or "off").lower()
            if mode in ("off", "none", "0", "false"):
                return True
            d5, d15 = self._get_trend_dirs(asset)
            if mode == "weak":
                if direction == "call" and d5 == "down" and d15 != "up":
                    return False
                if direction == "put" and d5 == "up" and d15 != "down":
                    return False
                return True
            # strict
            if direction == "call":
                return d5 == "up" and d15 == "up"
            return d5 == "down" and d15 == "down"
        except Exception:
            return True

    def _micro_filter_ok(self, asset: str, direction: str) -> bool:
        """1m micro-filter: sum of last two 1m bodies aligned with direction (closed bars only)."""
        try:
            end = self._align_to_closed_bar(60)  # align to closed 1m bar
            try:
                rows = self.iq.get_candles_tf(asset, 1, count=3, to_ts=end)  # type: ignore[call-arg,union-attr]
            except TypeError:
                rows = self.iq.get_candles_tf(asset, 1, count=3)  # fallback
            rows = rows or []
            if len(rows) < 2:
                return True
            deltas = [(float(c.get("close", 0)) - float(c.get("open", 0))) for c in rows[-2:]]
            total = float(sum(deltas))
            return (total >= 0.0) if direction == "call" else (total <= 0.0)
        except Exception:
            return True

    def _micro_5s_filter_ok(self, asset: str, direction: str, window_sec: int = 15) -> bool:
        """Micro 5s kill-filter near entry. Block if momentum strongly opposes side."""
        try:
            import math
            import statistics

            end = self._align_to_closed_bar(5)  # close last 5s
            n = max(2, min(6, int(math.ceil(float(window_sec) / 5.0))))

            rows = None
            for call in (
                lambda: self.iq.get_candles(asset, 5, n, endtime=end),      # iqoptionapi classic
                lambda: self.iq.get_candles(asset, 5, count=n, to_ts=end),  # some wrappers
                lambda: self.iq.get_candles_sec(asset, 5, count=n, to_ts=end),
            ):
                try:
                    rows = call()
                    if rows:
                        break
                except Exception:
                    rows = None

            if not rows:
                return True  # no data → do not block

            if isinstance(rows, dict):
                rows = [rows[k] for k in sorted(rows.keys())]

            deltas: List[float] = []
            for r in rows[-n:]:
                o = float(r.get("open", 0.0)); c = float(r.get("close", 0.0))
                deltas.append(c - o)

            if not deltas:
                return True

            up = sum(1 for d in deltas if d > 0)
            dn = sum(1 for d in deltas if d < 0)
            net = float(sum(deltas))
            med = float(statistics.median(abs(d) for d in deltas)) if any(deltas) else 0.0
            need = int(math.ceil(n * (2.0 / 3.0)))
            strong = abs(net) >= (1.2 * (med if med > 0 else 1e-9))

            if direction == "call" and dn >= need and net < 0 and strong:
                logging.info(f"[{asset}] micro-5s kill → dn={dn}/{n}, net={net:.6f}, med={med:.6f}")
                return False
            if direction == "put" and up >= need and net > 0 and strong:
                logging.info(f"[{asset}] micro-5s kill → up={up}/{n}, net={net:.6f}, med={med:.6f}")
                return False
            return True
        except Exception as e:
            logging.debug(f"micro-5s filter error: {e}")
            return True

    def _stoch_filter_ok(self, df: pd.DataFrame, tf_min: int, signal: str) -> bool:
        """True = pass; False = blocked by stochastic config for tf_min."""
        try:
            cfg = get_indicator_config(int(tf_min)) or {}
            st = cfg.get("stoch", {}) or {}
            if not st.get("enabled", False):
                return True
            mode = str(st.get("mode", "filter")).lower()
            if mode in ("off", "none", "0", "false"):
                return True

            cl = df["close"].astype(float).to_numpy()
            hi = df["high"].astype(float).to_numpy()
            lo = df["low"].astype(float).to_numpy()
            k_arr, d_arr = stoch_calc(
                hi, lo, cl,
                int(st.get("k", 14)),
                int(st.get("d", 3)),
                int(st.get("smooth", 3)),
            )
            kv, dv = float(k_arr[-1]), float(d_arr[-1])
            ob, os = float(st.get("ob", 80)), float(st.get("os", 20))

            if signal == "call":
                return (kv < os) and (kv > dv)
            if signal == "put":
                return (kv > ob) and (kv < dv)
            return True
        except Exception:
            return True

    def get_recent_trades(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return recent trades as JSON-safe list (thread-safe snapshot)."""
        try:
            with self._trade_lock:
                rows: List[Dict[str, Any]] = []
                src = getattr(self, "trade_history", None) or getattr(self, "trades", None) or []
                data = list(src)[::-1]  # newest first
                sl = data[offset: offset + max(0, int(limit or 100))]
                for r in sl:
                    d = dict(r) if isinstance(r, dict) else {}
                    t = d.get("time") or d.get("ts") or d.get("created_at")
                    if hasattr(t, "isoformat"):
                        d["time"] = t.isoformat()
                    elif isinstance(t, (int, float, str)) and "time" not in d:
                        d["time"] = str(t)
                    # numpy → python
                    for k, v in list(d.items()):
                        try:
                            if isinstance(v, np.floating):
                                d[k] = float(v)
                            elif isinstance(v, np.integer):
                                d[k] = int(v)
                            elif isinstance(v, (set, tuple)):
                                d[k] = list(v)
                        except Exception:
                            pass
                    rows.append(d)
                return rows
        except Exception as e:
            raise RuntimeError(f"get_recent_trades failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Indicator Warmups (dynamic history need)                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bars_needed_for_indicator(name: str, params: Dict[str, Any]) -> int:
        """Heuristic min history for an indicator to stabilize."""
        name = (name or "").lower()
        p = params or {}

        def clamp(n: int, lo: int = 1, hi: int = 2000) -> int:
            return max(lo, min(int(n), hi))

        if name in ("ma", "moving_average"):
            length = int(p.get("length") or p.get("period") or 50)
            ma_type = (p.get("type") or "EMA").upper()
            return clamp(length * (3 if ma_type == "EMA" else 1) + 10)

        if name == "macd":
            fast = int(p.get("fast") or 12)
            slow = int(p.get("slow") or 26)
            signal = int(p.get("signal") or 9)
            return clamp(slow + signal + slow * 3)

        if name in ("ichimoku", "ichimoku_cloud"):
            t = int(p.get("tenkan") or 9)
            k = int(p.get("kijun") or 26)
            b = int(p.get("senkou_b") or p.get("span_b") or 52)
            return clamp(max(b, k, t) + 26)

        if name == "rsi":
            n = int(p.get("length") or p.get("period") or 14)
            return clamp(n + 5)

        if name in ("stochastic", "stoch", "stochastic_oscillator"):
            klen = int(p.get("k") or p.get("k_len") or 14)
            dlen = int(p.get("d") or p.get("d_len") or 3)
            return clamp(klen + dlen + 3)

        if name in ("bb", "bollinger", "bollinger_bands"):
            n = int(p.get("length") or p.get("period") or 20)
            return clamp(n + 5)

        if name in ("atr", "average_true_range"):
            n = int(p.get("length") or p.get("period") or 14)
            return clamp(n + 6)

        if name in ("obv", "on_balance_volume"):
            return clamp(120)

        if name in ("volume_profile", "vp", "vprofile"):
            win = int(p.get("window") or p.get("period") or 150)
            return clamp(win)

        if name in ("price_action", "pa"):
            lb = int(p.get("lookback") or 5)
            return clamp(lb + 3)

        return 60  # default safety

    @classmethod
    def _min_bars_for_cfg(cls, cfg: Dict[str, Any]) -> int:
        """Aggregate requirement of all enabled indicators; add buffer."""
        if not isinstance(cfg, dict):
            return 240
        ind = cfg.get("indicators") or cfg.get("custom") or cfg or {}
        needs: List[int] = []
        for k, v in ind.items():
            try:
                if isinstance(v, dict) and v.get("enabled"):
                    needs.append(cls._bars_needed_for_indicator(k, v))
            except Exception:
                continue
        required = max(needs) if needs else 240
        required = int(required * 1.25) + 5  # buffer
        return max(60, min(required, 1000))

    @classmethod
    def _dynamic_required_bars(cls, tf_min: int) -> int:
        """Dynamic warmup from indicator config for the given timeframe."""
        try:
            cfg = get_indicator_config(int(tf_min)) or {}
        except Exception:
            cfg = {}
        return cls._min_bars_for_cfg(cfg)

    # ------------------------------------------------------------------ #
    # Martingale Core                                                    #
    # ------------------------------------------------------------------ #
    def _mg_key(self, asset: str) -> str:
        scope = str(self.settings.get("martingale_scope", "Separate"))
        return "global" if scope == "Combined" else asset

    def _is_locked(self, key: str) -> bool:
        return self._now() < float(self.open_until.get(key, 0) or 0)

    def _try_lock(self, key: str, seconds: float) -> bool:
        """Atomic check-and-set lock to avoid double execution per slot/key."""
        with self.martingale_lock:
            now = self._now()
            until = float(self.open_until.get(key, 0) or 0)
            if now < until:
                return False
            self.open_until[key] = now + float(seconds)
            return True

    def _unlock(self, key: str) -> None:
        with self.martingale_lock:
            self.open_until[key] = 0

    def _get_martingale_step(self, asset: str) -> int:
        key = self._mg_key(asset)
        with self.martingale_lock:
            return int(self.martingale_state.get(key, 0))

    def _get_mg_max_steps(self) -> int:
        mode = str(self.settings.get("martingale_mode", "None"))
        if mode == "Custom":
            arr = self.settings.get("martingale_custom_amounts") or []
            try:
                return len(arr) or 1
            except Exception:
                return 1
        return int(self.settings.get("martingale_max_steps", 10))

    def _update_mg_step_after_result(self, key: str, result: str) -> None:
        with self.martingale_lock:
            mode = str(self.settings.get("martingale_mode", "None"))
            cur = int(self.martingale_state.get(key, 0))
            on_draw = str(self.settings.get("martingale_on_draw", "SAME")).upper()
            r = (result or "").upper()
            new_step = cur

            if mode == "None":
                if r in ("WIN", "LOSE") or (r in ("EQUAL", "DRAW") and on_draw == "RESET"):
                    new_step = 0
            else:
                if r == "WIN":
                    new_step = 0
                elif r == "LOSE":
                    new_step = cur + 1
                    if new_step >= self._get_mg_max_steps():
                        new_step = 0
                elif r in ("EQUAL", "DRAW"):
                    if on_draw == "RESET":
                        new_step = 0
                    elif on_draw == "NEXT":
                        new_step = cur + 1
                        if new_step >= self._get_mg_max_steps():
                            new_step = 0
                    else:
                        new_step = cur

            self.martingale_state[key] = new_step

    def _get_next_trade_amount(self, asset: str) -> float:
        mode = str(self.settings.get("martingale_mode", "None"))
        base = float(self.settings.get("base_amount", self.settings.get("amount", 1.0)) or 1.0)
        key = self._mg_key(asset)
        with self.martingale_lock:
            step = int(self.martingale_state.get(key, 0))

        if mode == "Custom":
            arr = self.settings.get("martingale_custom_amounts") or []
            try:
                return float(arr[min(step, len(arr) - 1)])
            except Exception:
                return base

        if mode == "Flat":
            return base

        # Standard geometric
        mult = float(self.settings.get("martingale_multiplier", 2.2) or 2.2)
        return float(base * (mult ** step))

    # Quick helpers for endpoints (optional but nice to have)
    def get_mg_state(self) -> Dict[str, int]:
        with self.martingale_lock:
            return dict(self.martingale_state)

    def set_mg_state(self, state: Dict[str, int]) -> None:
        if not isinstance(state, dict):
            return
        with self.martingale_lock:
            for k, v in state.items():
                try:
                    self.martingale_state[str(k)] = int(v)
                except Exception:
                    continue

    # ------------------------------------------------------------------ #
    # AI / Signal normalization                                          #
    # ------------------------------------------------------------------ #
    def _extract_p_up(self) -> Optional[float]:
        try:
            v = getattr(self.strategy, "p_up", None)  # type: ignore[union-attr]
            return None if v is None else float(v)
        except Exception:
            return None

    def _candidate_score(self, signal: str, p_up: Optional[float]) -> Optional[float]:
        """Return score if thresholds pass; None otherwise."""
        if signal not in ("call", "put"):
            return None

        conf_thr = getattr(self.strategy, "conf_thr", None)  # type: ignore[union-attr]
        gap_thr = getattr(self.strategy, "gap_thr", None)    # type: ignore[union-attr]

        # Confidence threshold
        if p_up is not None and conf_thr is not None:
            conf_ok = (p_up >= conf_thr) if signal == "call" else ((1.0 - p_up) >= conf_thr)
            if not conf_ok:
                return None

        # Gap threshold |2p-1|
        if p_up is not None and gap_thr is not None:
            gap_val = abs(2 * float(p_up) - 1.0)
            if gap_val < gap_thr:
                return None

        base = 0.0 if p_up is None else (p_up if signal == "call" else (1.0 - p_up))
        return float(base)

    def _ai_tag(self) -> str:
        try:
            name = str(getattr(self.strategy, "name", "")) or self.strategy.__class__.__name__  # type: ignore[union-attr]
            tf = int(getattr(self.strategy, "tf", int(self.settings.get("duration", 1))))
            return f"[AI {tf}m]" if "ai" in name.lower() else f"[{name}]"
        except Exception:
            return "[Strategy]"

    def _fmt3(self, x: Any) -> str:
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "None"

    def _log_ai_prob(self, p_up: Optional[float]) -> None:
        try:
            conf_thr = getattr(self.strategy, "conf_thr", None)  # type: ignore[union-attr]
            gap_thr = getattr(self.strategy, "gap_thr", None)    # type: ignore[union-attr]
            gap = None if p_up is None else abs(2 * float(p_up) - 1.0)
            logging.info(
                f"{self._ai_tag()} p_up={self._fmt3(p_up)} conf_thr={self._fmt3(conf_thr)} "
                f"gap={self._fmt3(gap)} gap_thr={self._fmt3(gap_thr)}"
            )
        except Exception:
            pass

    def _get_signal_universal(self, asset: str, candles_df: pd.DataFrame) -> Tuple[str, Optional[float]]:
        """
        Normalize strategy outputs to (signal, p_up).
        Tries: check_signal/get_signal/get_signal_asset/decide/evaluate/infer/predict,
        then predict_proba. Accepts: str | (signal,p) | (p,) | dict | number.
        """
        strat = self.strategy  # type: ignore[assignment]
        p_up: Optional[float] = None
        signal = "none"

        def _parse(out: Any) -> None:
            nonlocal signal, p_up
            if isinstance(out, str):
                lo = out.lower()
                if lo in ("call", "put", "none"):
                    signal = lo
            elif isinstance(out, (list, tuple)):
                if len(out) >= 2:
                    a, b = out[0], out[1]
                    if isinstance(a, str) and a.lower() in ("call", "put", "none"):
                        signal = a.lower()
                        if isinstance(b, (int, float)):
                            p_up = float(b)
                    elif isinstance(a, (int, float)):
                        p_up = float(a)
                elif len(out) == 1 and isinstance(out[0], (int, float)):
                    p_up = float(out[0])
            elif isinstance(out, dict):
                s = out.get("signal")
                if isinstance(s, str) and s.lower() in ("call", "put", "none"):
                    signal = s.lower()
                for k in ("p_up", "prob_up", "p"):
                    if out.get(k) is not None:
                        p_up = float(out[k])
                        break
            elif isinstance(out, (int, float)):
                p_up = float(out)

        for name in ("check_signal", "get_signal", "get_signal_asset", "decide", "evaluate", "infer", "predict"):
            m = getattr(strat, name, None)
            if callable(m):
                try:
                    out = m(asset, candles_df)                 # type: ignore[misc]
                except TypeError:
                    try:
                        out = m(candles_df)                     # type: ignore[misc]
                    except TypeError:
                        out = m(asset)                          # type: ignore[misc]
                _parse(out)
                if p_up is not None or signal in ("call", "put"):
                    break

        if p_up is None and signal == "none":
            m = getattr(strat, "predict_proba", None)
            if callable(m):
                try:
                    out = m(asset, candles_df)                 # type: ignore[misc]
                except TypeError:
                    try:
                        out = m(candles_df)                     # type: ignore[misc]
                    except TypeError:
                        out = m(asset)                          # type: ignore[misc]
                _parse(out)

        if signal not in ("call", "put"):
            signal = "call" if (isinstance(p_up, (int, float)) and p_up >= 0.5) else ("put" if isinstance(p_up, (int, float)) else "none")

        try:
            if p_up is not None:
                setattr(strat, "p_up", float(p_up))            # type: ignore[misc]
        except Exception:
            pass

        return signal, p_up

    # ------------------------------------------------------------------ #
    # Result Check (resilient)                                           #
    # ------------------------------------------------------------------ #
    def _check_and_update_trade_result(self, trade_id: int, db_id: int, duration_min: int, asset: str) -> None:
        try:
            wait_secs = 5 + duration_min * 60
            self._wait(wait_secs)
            # ถ้าถูกสั่งหยุด ก็เลิกเช็คต่อทันที
            if self._stop_evt.is_set() or not self.is_running:
                return

            # Retry while net can be flaky
            profit_amount: Optional[float] = None
            for _ in range(12):  # ~2 minutes
                if self._stop_evt.is_set() or not self.is_running:
                    return
                try:
                    profit_amount = self.iq.check_win_v3(trade_id)  # type: ignore[union-attr]
                except Exception:
                    profit_amount = None
                if profit_amount is not None:
                    break
                try:
                    self._iq_ensure()
                except Exception:
                    pass
                if self._wait(10):
                    return

            if profit_amount is None:
                logging.warning(f"[TradeChecker] Result unknown for Trade ID: {trade_id} after retries.")
                return

            result = "WIN" if profit_amount > 0 else ("LOSE" if profit_amount < 0 else "EQUAL")
            trade_logger.update_trade_result(db_id, result, float(profit_amount))

            # MG
            try:
                key = self._mg_key(asset)
                self._update_mg_step_after_result(key, result)
            except Exception as e:
                logging.warning(f"update mg error: {e}")

            # HUD
            try:
                self.session_profit += float(profit_amount)
                if result == "WIN":
                    self.session_win += 1
                elif result == "LOSE":
                    self.session_lose += 1
                else:
                    self.session_draw += 1
            except Exception:
                pass

            # TP/SL
            try:
                tp = self.settings.get("take_profit")
                sl = self.settings.get("stop_loss")
                if tp is not None and float(tp) > 0 and self.session_profit >= float(tp):
                    logging.info("[TP] Hit. Stopping bot.")
                    self.stop()
                if sl is not None and float(sl) > 0 and self.session_profit <= -float(sl):
                    logging.info("[SL] Hit. Stopping bot.")
                    self.stop()
            except Exception:
                pass

            # AI meta
            try:
                if ai_meta is not None and str(getattr(self.strategy, "name", "")).startswith("AI Model"):  # type: ignore[union-attr]
                    tf = int(getattr(self.strategy, "tf", duration_min))  # type: ignore[union-attr]
                    ai_meta.record_result(tf, result)
            except Exception as e:
                logging.warning(f"[AI] meta update error: {e}")

        except Exception as e:
            logging.error(f"[TradeChecker] Error checking win for Trade ID {trade_id}: {e}")

    # ------------------------------------------------------------------ #
    # Candle Fetch (with retries)                                        #
    # ------------------------------------------------------------------ #
    def _fetch_one_candles(
        self,
        asset: str,
        period_sec: int,
        limit: int,
        to_ts: Optional[int] = None,
    ) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch one asset's candles with retries and best-effort reconnect.
        to_ts=None  → latest/open bar possible; to_ts=ts → closed at ts.
        """
        last_err: Optional[str] = None
        for attempt in range(1, 4):
            try:
                try:
                    self._iq_ensure()
                except Exception:
                    pass

                rows: Any = None
                try:
                    if to_ts is None:
                        rows = self.iq.get_candles_tf(asset, period_sec // 60, count=limit)  # type: ignore[union-attr]
                    else:
                        rows = self.iq.get_candles_tf(asset, period_sec // 60, count=limit, to_ts=int(to_ts))  # type: ignore[call-arg,union-attr]
                except TypeError:
                    rows = self.iq.get_candles_tf(asset, period_sec // 60, count=limit)  # type: ignore[union-attr]

                rows = rows or []
                if isinstance(rows, dict):
                    rows = [rows[k] for k in sorted(rows.keys())]

                if not rows:
                    last_err = "empty"
                else:
                    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
                    return asset, df, None
            except Exception as e:
                last_err = str(e)

            self._wait(min(1.5 * attempt, 5.0))

        return asset, None, (last_err if isinstance(last_err, str) else "fetch_error")

    def _fetch_candles_batch(
        self,
        assets: List[str],
        period_sec: int,
        limit: int,
        to_ts: Optional[int] = None,
        workers: Optional[int] = None,
        align_to_closed: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        align_to_closed=True  → if to_ts=None align to last closed bar.
        align_to_closed=False → if to_ts=None allow open bar (preview).
        """
        # Early exit if stopping
        if (not self.is_running) or self._stop_evt.is_set():
            return {}

        effective_to_ts = (self._align_to_closed_bar(period_sec) if (align_to_closed and to_ts is None) else to_ts)
        workers = workers or int(self.settings.get("max_candle_workers", 6) or 6)
        results: Dict[str, pd.DataFrame] = {}
        if not assets:
            return results

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(self._fetch_one_candles, sym, period_sec, limit, effective_to_ts): sym for sym in assets}
            for fut in as_completed(futs):
                # Stop quickly if requested
                if (not self.is_running) or self._stop_evt.is_set():
                    for f in futs:
                        if not f.done():
                            f.cancel()
                    break
                sym = futs[fut]
                try:
                    asset, df, err = fut.result()
                    if err is None and df is not None:
                        results[asset] = df
                    else:
                        logging.warning(f"[{asset}] candle fetch error: {err}")
                except Exception as e:
                    logging.error(f"[{sym}] batch fetch exception: {e}")

        return results

    def _maybe_preview_scan(self, period_sec: int, assets: List[str], history_limit: int, preview_sec: int) -> bool:
        """Scan mid-candle for 1m to pre-pick ideas, without placing orders."""
        try:
            if period_sec != 60 or not assets or preview_sec <= 0:
                return False

            lead = int(self.settings.get("lead_time_sec", 5) or 5)
            remain = self._secs_to_next_minute()
            if not (lead < remain <= (lead + preview_sec)):
                return False

            next_slot = self._align_to_closed_bar(period_sec) + period_sec
            if self._last_preview_slot == next_slot:
                return False

            # Use open bar (do not align to closed)
            df_map = self._fetch_candles_batch(assets, period_sec, limit=history_limit, to_ts=None, align_to_closed=False)

            preview: Dict[str, str] = {}
            if hasattr(self.strategy, "check_signals_pair"):        # type: ignore[union-attr]
                pair_sigs = self.strategy.check_signals_pair(df_map) or {}  # type: ignore[union-attr]
                preview.update({a: s for a, s in pair_sigs.items()})
            elif hasattr(self.strategy, "check_signals_rank"):      # type: ignore[union-attr]
                rank = self.strategy.check_signals_rank(df_map) or []        # type: ignore[union-attr]
                for r in (rank[:8] if isinstance(rank, list) else []):
                    a, s = r.get("asset"), r.get("signal")
                    if a and s:
                        preview[a] = s

            self._preview_cache = preview
            self._last_preview_slot = next_slot

            try:
                meta = getattr(self.strategy, "get_signal_meta", lambda: {})()  # type: ignore[union-attr]
                if meta:
                    logging.info(f"[Preview {preview_sec}s] meta: {json.dumps(meta, ensure_ascii=False)[:2000]}")
            except Exception:
                pass

            logging.info(f"[Preview {preview_sec}s] picks: {preview}")
            return True
        except Exception as e:
            logging.warning(f"[Preview] error: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def start(self, iq_instance: IQWrapper, strategy_instance: BaseStrategy, settings: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lifecycle_lock:
            if self.is_running:
                return False, "Bot is already running."

            self.iq = iq_instance
            self.strategy = strategy_instance
            self.settings = settings or {}
            # เติมส่วนนี้ (best-effort sync thresholds สำหรับ AI)
            try:
                # เดาว่าเป็น AI จากชื่อ/คุณสมบัติ
                is_ai = str(getattr(self.strategy, "name", "")).lower().startswith("ai model")
                if is_ai:
                    tf = int(getattr(self.strategy, "tf", int(self.settings.get("duration", 1))))
                    # เลือก key ให้ถูกกับ TF
                    conf = self.settings.get("ai_conf")
                    gap  = self.settings.get("ai_gap")
                    if tf == 1:
                        conf = self.settings.get("ai1m_conf", conf)
                        gap  = self.settings.get("ai1m_gap",  gap)
                    elif tf == 5:
                        conf = self.settings.get("ai5m_conf", conf)
                        gap  = self.settings.get("ai5m_gap",  gap)
                    # อัปเดต threshold ถ้าผู้ใช้ส่งมา
                    if conf is not None:
                        try: self.strategy.conf_thr = float(conf)
                        except Exception: pass
                    if gap is not None:
                        try: self.strategy.gap_thr = float(gap)
                        except Exception: pass
            except Exception:
                pass
            # reflect anti-flip from settings (allow UI override)
            try:
                self.anti_flip_enable = bool(self.settings.get("anti_flip_enable", True))
                self.anti_flip_body_atr = float(self.settings.get("anti_flip_body_atr", 0.15))
            except Exception:
                pass

            # mark running and clear stop event for a fresh run
            self.is_running = True
            try:
                self._stop_evt.clear()
            except Exception:
                pass

            # Reset state for new session
            self.martingale_state.clear()
            self.open_until.clear()
            self._last_slot_fired.clear()
            self.reset_session_stats()

            # AI auto settings (best effort)
            try:
                if ai_meta is not None and str(getattr(self.strategy, "name", "")).startswith("AI Model"):  # type: ignore[union-attr]
                    tf = int(getattr(self.strategy, "tf", int(self.settings.get("duration", 1))))  # type: ignore[union-attr]
                    st = ai_meta.status(tf)
                    if st:
                        self.settings["ai_auto_retrain"] = bool(st.get("auto_enabled", True))
                        self.settings["ai_retrain_every_orders"] = int(st.get("auto_every", 40))
            except Exception as e:
                logging.warning(f"[AI] Unable to apply auto settings: {e}")

            # bump generation and phase
            self._run_id += 1
            curr_run = self._run_id
            self._phase = "running"

            # Start background loop with run_id guard
            self.thread = threading.Thread(target=self._run_loop, args=(curr_run,), daemon=True)
            self.thread.start()
            logging.info("BotEngine started.")
            return True, "Bot started successfully."

    def stop(self, keep_connection: bool = True) -> Tuple[bool, str]:
        """
        Stop loop quickly. ถ้า keep_connection=True (ค่าเริ่มต้น) จะไม่ไป logout()/stop_keepalive()
        เพื่อคง session ให้ UI เรียก /status, /indicators/live ต่อได้ทันที
        """
        with self._lifecycle_lock:
            if not self.is_running:
                return True, "Bot is not running."

            self._phase = "stopping"

            # 1) สั่งหยุดทันที (ให้ loop ทั้งหมดตรวจเจอ)
            try:
                self._stop_evt.set()
            except Exception:
                pass
            self.is_running = False

            # 2) bump run_id เพื่อให้ loop เก่า exit แม้หลุดจาก I/O กลับมา
            self._run_id += 1

            # 3) ถ้าไม่ต้องคงการเชื่อมต่อ → HARD-BREAK I/O
            if not keep_connection:
                try:
                    if self.iq:
                        try:
                            if hasattr(self.iq, "stop_keepalive"):
                                self.iq.stop_keepalive()
                        except Exception:
                            pass
                        try:
                            if hasattr(self.iq, "logout"):
                                self.iq.logout()
                        except Exception:
                            pass
                except Exception:
                    pass

        # 4) รอ thread จบแบบรวดเร็ว
        th = self.thread
        if th and th.is_alive():
            try:
                th.join(timeout=1.0)  # ตอบสนองไว
            except Exception:
                pass

        self.thread = None
        self._phase = "idle"
        logging.info("BotEngine stopped.")
        return True, "Bot stopped successfully."

    def _sleep_interruptible(self, seconds: float) -> None:
        t0 = self._now()
        while self.is_running and (self._now() - t0) < float(seconds):
            if self._wait(0.05):
                break

    def _wait(self, seconds: float) -> bool:
        """Wait up to 'seconds'. Returns True if stop was signaled during the wait."""
        try:
            return self._stop_evt.wait(timeout=max(0.0, float(seconds)))
        except Exception:
            time.sleep(max(0.0, float(seconds)))
            return False

    # --- Net Resilience Helpers --------------------------------------- #
    def _iq_has(self, name: str) -> bool:
        try:
            return callable(getattr(self.iq, name, None))
        except Exception:
            return False

    def _iq_ensure(self) -> bool:
        """Ensure connection using wrapper methods when available."""
        try:
            if self._iq_has("ensure_connected"):
                return bool(self.iq.ensure_connected())  # type: ignore[union-attr]
            if self._iq_has("connected"):
                return bool(self.iq.connected())         # type: ignore[union-attr]
        except Exception:
            return False
        return True

    def _ensure_connection_blocking(self) -> bool:
        """If disconnected, loop until reconnected (with short backoff)."""
        if not (self._iq_has("ensure_connected") or self._iq_has("connected")):
            return True
        backoff = 1.0
        t0: Optional[float] = None
        while self.is_running and not self._iq_ensure():
            if t0 is None:
                t0 = self._now()
                logging.warning("[NET] Disconnected. Auto-reconnect loop started…")
            self._wait(backoff)
            backoff = min(15.0, backoff * 1.6)
        if t0 is not None:
            dt = int(self._now() - t0)
            logging.info(f"[NET] Reconnected after {dt}s. Resuming main loop.")
        return True

    def _place_safe(
        self,
        amount: float,
        asset: str,
        direction: str,
        duration_min: int,
        retries: int = 3,
    ) -> Tuple[bool, Any]:
        """Place order with retries + best-effort reconnect. Returns (ok, info)."""
        last_err: Any = None
        for attempt in range(1, int(max(1, retries)) + 1):
            try:
                if self._iq_has("place"):
                    ok, trade_info = self.iq.place(amount, asset, direction, duration_min)  # type: ignore[union-attr]
                else:
                    ok, trade_info = self.iq.buy(amount, asset, direction, duration_min)    # type: ignore[union-attr]
                if ok:
                    return ok, trade_info
                last_err = trade_info
            except Exception as e:
                last_err = e
            # ensure and backoff
            try:
                self._iq_ensure()
            except Exception:
                pass
            self._wait(min(1.5 * attempt, 5.0))
        return False, last_err

    def _anti_flip_gate(self, df: pd.DataFrame, side: str, factor: float) -> bool:
        """
        Allow only when: last body >= factor * ATR(14) and bar color matches side.
        df: closed bars with open/high/low/close.
        """
        try:
            if df is None or len(df) < 20:
                return False
            c = float(df["close"].iloc[-1]); o = float(df["open"].iloc[-1])
            body = abs(c - o)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1] or 0.0)
            if not np.isfinite(atr) or atr <= 0:
                return False
            if side == "call" and not (c > o):
                return False
            if side == "put" and not (c < o):
                return False
            return body >= (factor * atr)
        except Exception as e:
            logging.warning(f"[AntiFlip] error: {e}")
            return False

    def _anti_whipsaw_gate(self, df: pd.DataFrame, side: str, wick_ratio_max: float = 0.65) -> bool:
        """Reject if last two bars have excessive wicks or reverse close."""
        try:
            if df is None or len(df) < 3:
                return True
            hi = df["high"].astype(float).tail(2).to_numpy()
            lo = df["low"].astype(float).tail(2).to_numpy()
            op = df["open"].astype(float).tail(2).to_numpy()
            cl = df["close"].astype(float).tail(2).to_numpy()
            rng = np.maximum(hi - lo, 1e-9)
            upper_wick = hi - np.maximum(op, cl)
            lower_wick = np.minimum(op, cl) - lo
            wick_ratio = (upper_wick + lower_wick) / rng
            if float(np.mean(wick_ratio)) > float(wick_ratio_max):
                return False
            if side == "call" and not (cl[-1] >= cl[-2]):
                return False
            if side == "put" and not (cl[-1] <= cl[-2]):
                return False
            return True
        except Exception:
            return True

    # --- Misc ---------------------------------------------------------- #
    def _is_block_time(self) -> bool:
        block_start = self.settings.get("block_start")
        block_end = self.settings.get("block_end")
        if not block_start or not block_end:
            return False
        try:
            now_time = datetime.now().time()
            start_time = datetime.strptime(block_start, "%H:%M").time()
            end_time = datetime.strptime(block_end, "%H:%M").time()
            return start_time <= now_time <= end_time
        except ValueError:
            logging.warning("Invalid block time format. Should be HH:MM.")
            return False

    # --- Main loop body ----------------------------------------------- #
    def _run_loop(self, run_id: int) -> None:
        my_id = int(run_id)

        try:
            strat_name = getattr(self.strategy, "name", self.strategy.__class__.__name__)  # type: ignore[union-attr]
            logging.info(f"Starting bot loop with strategy: {strat_name}")
        except Exception:
            logging.info("Starting bot loop.")

        duration = int(self.settings.get("duration", 1))
        period_sec = 60 * duration
        lead = int(self.settings.get("lead_time_sec", 5) or 5)
        micro_5m = bool(self.settings.get("micro_filter_5m", True))
        trend_mode = str(self.settings.get("trend_filter_mode", "weak")).lower()

        last_heartbeat = self._now()

        while self.is_running and not self._stop_evt.is_set():
            if my_id != self._run_id:
                logging.info("[Engine] run_id changed → exit old loop")
                break
            # Net resume (non-fatal)
            try:
                if self._iq_has("ensure_connected") or self._iq_has("connected"):
                    self._ensure_connection_blocking()
            except Exception:
                pass

            # Heartbeat
            if self._now() - last_heartbeat >= 60:
                logging.info("[Heartbeat] Bot running...")
                last_heartbeat = self._now()

            # Block window
            if self._is_block_time():
                logging.info("Inside block time. Waiting...")
                self._sleep_interruptible(60)
                continue

            # Per-cycle params
            assets_to_trade: List[str] = list(self.settings.get("assets", []))

            # Preview section (1m only)
            try:
                hist_ui_prev = int(self.settings.get("candles_history", 0) or 0)
            except Exception:
                hist_ui_prev = 0
            try:
                dyn_need_prev = int(self._dynamic_required_bars(int(self.settings.get("duration", 1))))
            except Exception:
                dyn_need_prev = 0
            try:
                strat_need_prev = int(getattr(self.strategy, "required_bars", 205) or 0)
            except Exception:
                strat_need_prev = 205
            preview_hist = max(120, min(1000, max(hist_ui_prev, dyn_need_prev, strat_need_prev)))

            raw_max = self.settings.get("max_orders_per_cycle", self.settings.get("max_orders", 0))
            try:
                maxN = max(0, int(raw_max or 0))
            except Exception:
                maxN = 0

            # -------- PREVIEW (mid-candle) --------
            try:
                preview_sec = int(self.settings.get("preview_sec", 30) or 30)
            except Exception:
                preview_sec = 30

            if duration == 1 and preview_sec > 0:
                self._wait_until_preview(duration, lead, preview_sec)
                try:
                    self._maybe_preview_scan(period_sec, assets_to_trade, preview_hist, preview_sec)
                except Exception as e:
                    logging.warning(f"[Preview] skip due to: {e}")

            # -------- CONFIRM (close candle) --------
            self._wait_until_slot(duration, lead)

            # Candle history limit  ➜ MAX(UI, dynamic, strategy)
            to_ts = self._align_to_closed_bar(period_sec)
            try:
                hist_ui = int(self.settings.get("candles_history", 0) or 0)
            except Exception:
                hist_ui = 0
            try:
                dyn_need = int(self._dynamic_required_bars(period_sec // 60) or 0)
            except Exception:
                dyn_need = 0
            try:
                strat_need = int(getattr(self.strategy, "required_bars", 205) or 0)
            except Exception:
                strat_need = 205
            history_limit = max(120, min(1000, max(hist_ui, dyn_need, strat_need)))

            # --- CANDLES DETAIL LOG ---
            ts_str = datetime.utcfromtimestamp(to_ts).isoformat(timespec="seconds") + "Z"
            assets_csv = ", ".join(assets_to_trade)
            logging.info(f"[Candles] TF={period_sec // 60}m, need≥{history_limit}, to={ts_str}, assets={len(assets_to_trade)} [{assets_csv}]")

            df_map = self._fetch_candles_batch(assets_to_trade, period_sec, history_limit, to_ts=to_ts)

            try:
                lens = {a: len(df) for a, df in df_map.items()}
                if lens:
                    lens_vals = sorted(lens.values())
                    minlen = lens_vals[0]
                    maxlen = lens_vals[-1]
                    medlen = lens_vals[len(lens_vals)//2]
                    missing = [a for a in assets_to_trade if a not in df_map]
                    logging.info(f"[Candles] fetched {len(df_map)}/{len(assets_to_trade)} ok, len[min/med/max]={minlen}/{medlen}/{maxlen}, missing={len(missing)} {missing}")
                else:
                    logging.info("[Candles] fetched 0 set(s).")
            except Exception:
                pass

            # ---------------- Multi-asset: pair checks ----------------
            if hasattr(self.strategy, "check_signals_pair") and getattr(self.strategy, "multi_asset", False):  # type: ignore[union-attr]
                placed = 0
                try:
                    pair_sigs = self.strategy.check_signals_pair(df_map) or {}  # type: ignore[union-attr]
                except Exception as e:
                    logging.error(f"[Multi] check_signals_pair error: {e}")
                    pair_sigs = {}

                # Optional meta preview
                try:
                    meta = getattr(self.strategy, "get_signal_meta", lambda: {})()  # type: ignore[union-attr]
                    if meta:
                        preview = json.dumps(meta, ensure_ascii=False)
                        logging.info(f"[Multi] scan meta: {preview[:2000] + '…' if len(preview) > 2000 else preview}")
                except Exception:
                    pass

                candidates: List[Dict[str, Any]] = []
                for asset, signal in pair_sigs.items():
                    key = self._mg_key(asset)
                    if self._is_locked(key):
                        logging.info(f"[{asset}] Locked (multi). Skip.")
                        continue
                    if not self._trend_pass(asset, signal, trend_mode):
                        logging.info(f"[{asset}] Trend filter ({trend_mode}) blocked {signal} (multi).")
                        continue
                    if duration == 5 and micro_5m and signal in ("call", "put"):
                        if not self._micro_filter_ok(asset, signal):
                            logging.info(f"[{asset}] 1m micro-filter blocked {signal} for 5m entry (multi).")
                            continue
                    p_up = self._extract_p_up()
                    self._log_ai_prob(p_up)
                    score = self._candidate_score(signal, p_up)
                    if score is None:
                        continue
                    candidates.append({"asset": asset, "signal": signal, "score": float(score), "p_up": p_up})

                candidates.sort(key=lambda x: x["score"], reverse=True)
                picks = candidates[:maxN] if maxN < len(candidates) else candidates

                for c in picks:
                    if placed >= maxN:
                        break
                    asset, signal, p_up = c["asset"], c["signal"], c["p_up"]
                    key = self._mg_key(asset)
                    if self._is_locked(key):
                        logging.info(f"[{asset}] Locked just now, skip.")
                        continue

                    current_mg_step = self._get_martingale_step(asset)
                    self._bump_hud_steps(current_mg_step)

                    # micro-5s (1m entry)
                    if duration == 1 and signal in ("call", "put") and getattr(self.strategy, "micro_5s_enable", False):
                        w = int(getattr(self.strategy, "micro_5s_window_sec", 15) or 15)
                        if not self._micro_5s_filter_ok(asset, signal, window_sec=w):
                            logging.info(f"[{asset}] micro-5s filter blocked {signal} ({w}s window).")
                            continue
                    # Anti-flip & whipsaw
                    base_df = df_map.get(asset)
                    if self.anti_flip_enable and base_df is not None:
                        if not self._anti_flip_gate(base_df, signal, self.anti_flip_body_atr):
                            logging.info(f"[{asset}] AntiFlip blocked {signal}.")
                            continue
                        if not self._anti_whipsaw_gate(base_df, signal, wick_ratio_max=0.65):
                            logging.info(f"[{asset}] AntiWhipsaw blocked {signal}.")
                            continue

                    amount = self._get_next_trade_amount(asset)
                    logging.info(f"[{asset}] (PICKED) Signal: {signal.upper()}, Score: {c['score']:.3f}, MG Step: {current_mg_step}, Amount: ${amount:.2f}")

                    slot = self._slot_id(period_sec)
                    if self._last_slot_fired.get(key) == slot:
                        logging.info(f"[{asset}] Already traded this slot ({slot}). Skip.")
                        continue
                    if not self._try_lock(key, duration * 60 + 7):
                        logging.info(f"[{asset}] Locked on execute phase, skip.")
                        continue

                    ok, trade_info = self._place_safe(amount, asset, signal, duration)
                    if ok and isinstance(trade_info, int):
                        self._last_slot_fired[key] = slot
                        trade_id = trade_info
                        logging.info(f"[{asset}] Trade placed. Trade ID: {trade_id}")

                        strategy_label = (
                            self.settings.get("strategy_name")
                            or getattr(self.strategy, "name", None)            # type: ignore[union-attr]
                            or self.strategy.__class__.__name__               # type: ignore[union-attr]
                        )
                        pred_prob = None if p_up is None else (float(p_up) if signal == "call" else float(1.0 - float(p_up)))

                        db_id = trade_logger.log_trade(
                            asset=asset,
                            direction=signal,
                            amount=amount,
                            duration=duration,
                            strategy=strategy_label,
                            mg_step=current_mg_step + 1,
                            pred_prob=pred_prob,
                        )

                        self._maybe_ai_autoretrain(duration)

                        th = threading.Thread(
                            target=self._check_and_update_trade_result,
                            args=(trade_id, db_id, duration, asset),
                            daemon=True,
                        )
                        th.start()
                        placed += 1
                    else:
                        logging.error(f"[{asset}] Failed to place trade: {trade_info}")
                        self._unlock(key)

                logging.info(f"[Multi] done. placed={placed}. Loop waiting {period_sec} sec…")
                self._sleep_interruptible(period_sec)
                continue

            # ---------------- Rank-all (Top-N) -------------------------
            if hasattr(self.strategy, "check_signals_rank"):  # type: ignore[union-attr]
                placed = 0
                try:
                    rank = self.strategy.check_signals_rank(df_map) or []  # type: ignore[union-attr]
                except Exception as e:
                    logging.error(f"[TopN] check_signals_rank error: {e}")
                    rank = []

                candidates: List[Dict[str, Any]] = []
                for r in (rank if isinstance(rank, list) else []):
                    a = r.get("asset"); s = (r.get("signal") or "none").lower()
                    if not a or s not in ("call", "put"):
                        continue
                    key = self._mg_key(a)
                    if self._is_locked(key):
                        logging.info(f"[{a}] Locked (rank). Skip.")
                        continue
                    if not self._trend_pass(a, s, trend_mode):
                        logging.info(f"[{a}] Trend filter ({trend_mode}) blocked {s} (rank).")
                        continue
                    if duration == 5 and micro_5m and s in ("call", "put"):
                        if not self._micro_filter_ok(a, s):
                            logging.info(f"[{a}] 1m micro-filter blocked {s} for 5m entry (rank).")
                            continue
                    p_up = self._extract_p_up()
                    self._log_ai_prob(p_up)
                    score = self._candidate_score(s, p_up)
                    if score is None:
                        continue
                    candidates.append({"asset": a, "signal": s, "score": float(score), "p_up": p_up})

                candidates.sort(key=lambda x: x["score"], reverse=True)
                picks = candidates[:maxN] if maxN < len(candidates) else candidates

                for c in picks:
                    if placed >= maxN:
                        break
                    a, s, p_up = c["asset"], c["signal"], c["p_up"]
                    key = self._mg_key(a)
                    if self._is_locked(key):
                        logging.info(f"[{a}] Locked just now, skip.")
                        continue

                    current_mg_step = self._get_martingale_step(a)
                    self._bump_hud_steps(current_mg_step)

                    # micro-5s (1m entry)
                    if duration == 1 and s in ("call", "put") and getattr(self.strategy, "micro_5s_enable", False):
                        w = int(getattr(self.strategy, "micro_5s_window_sec", 15) or 15)
                        if not self._micro_5s_filter_ok(a, s, window_sec=w):
                            logging.info(f"[{a}] micro-5s filter blocked {s} ({w}s window).")
                            continue
                    # Anti flip/whipsaw
                    base_df = df_map.get(a)
                    if self.anti_flip_enable and base_df is not None:
                        if not self._anti_flip_gate(base_df, s, self.anti_flip_body_atr):
                            logging.info(f"[{a}] AntiFlip blocked {s}.")
                            continue
                        if not self._anti_whipsaw_gate(base_df, s, wick_ratio_max=0.65):
                            logging.info(f"[{a}] AntiWhipsaw blocked {s}.")
                            continue

                    amount = self._get_next_trade_amount(a)
                    logging.info(f"[{a}] (PICKED) Signal: {s.upper()}, Score: {c['score']:.3f}, MG Step: {current_mg_step}, Amount: ${amount:.2f}")

                    slot = self._slot_id(period_sec)
                    if self._last_slot_fired.get(key) == slot:
                        logging.info(f"[{a}] Already traded this slot ({slot}). Skip.")
                        continue
                    if not self._try_lock(key, duration * 60 + 7):
                        logging.info(f"[{a}] Locked on execute phase, skip.")
                        continue

                    ok, trade_info = self._place_safe(amount, a, s, duration)
                    if ok and isinstance(trade_info, int):
                        self._last_slot_fired[key] = slot
                        trade_id = trade_info
                        logging.info(f"[{a}] Trade placed. Trade ID: {trade_id}")

                        strategy_label = (
                            self.settings.get("strategy_name")
                            or getattr(self.strategy, "name", None)            # type: ignore[union-attr]
                            or self.strategy.__class__.__name__               # type: ignore[union-attr]
                        )
                        pred_prob = None if p_up is None else (float(p_up) if s == "call" else float(1.0 - float(p_up)))

                        db_id = trade_logger.log_trade(
                            asset=a,
                            direction=s,
                            amount=amount,
                            duration=duration,
                            strategy=strategy_label,
                            mg_step=current_mg_step + 1,
                            pred_prob=pred_prob,
                        )

                        self._maybe_ai_autoretrain(duration)

                        th = threading.Thread(
                            target=self._check_and_update_trade_result,
                            args=(trade_id, db_id, duration, a),
                            daemon=True,
                        )
                        th.start()
                        placed += 1
                    else:
                        logging.error(f"[{a}] Failed to place trade: {trade_info}")
                        self._unlock(key)

                logging.info(f"[TopN] done. placed={placed}. Loop waiting {period_sec} sec…")
                self._sleep_interruptible(period_sec)
                continue

            # ---------------- Sequential (single/each) ------------------
            placed = 0
            for a in assets_to_trade:
                if placed >= maxN:
                    break
                key = self._mg_key(a)
                if self._is_locked(key):
                    logging.info(f"[{a}] Locked (sequential). Skip.")
                    continue
                base_df = df_map.get(a)
                if base_df is None or len(base_df) < history_limit:
                    logging.info(f"[{a}] no data ({len(base_df) if base_df is not None else 0}/{history_limit}).")
                    continue

                try:
                    sig, p_up = self._get_signal_universal(a, base_df)
                except Exception as e:
                    logging.error(f"[{a}] strategy error: {e}")
                    continue
                if sig not in ("call", "put"):
                    continue

                if not self._trend_pass(a, sig, trend_mode):
                    logging.info(f"[{a}] Trend filter ({trend_mode}) blocked {sig} (seq).")
                    continue

                if duration == 5 and micro_5m and sig in ("call", "put"):
                    if not self._micro_filter_ok(a, sig):
                        logging.info(f"[{a}] 1m micro-filter blocked {sig} for 5m entry (seq).")
                        continue

                self._log_ai_prob(p_up)
                score = self._candidate_score(sig, p_up)
                if score is None:
                    continue

                current_mg_step = self._get_martingale_step(a)
                self._bump_hud_steps(current_mg_step)

                if duration == 1 and sig in ("call", "put") and getattr(self.strategy, "micro_5s_enable", False):
                    w = int(getattr(self.strategy, "micro_5s_window_sec", 15) or 15)
                    if not self._micro_5s_filter_ok(a, sig, window_sec=w):
                        logging.info(f"[{a}] micro-5s filter blocked {sig} ({w}s window).")
                        continue

                if self.anti_flip_enable:
                    if not self._anti_flip_gate(base_df, sig, self.anti_flip_body_atr):
                        logging.info(f"[{a}] AntiFlip blocked {sig}.")
                        continue
                    if not self._anti_whipsaw_gate(base_df, sig, wick_ratio_max=0.65):
                        logging.info(f"[{a}] AntiWhipsaw blocked {sig}.")
                        continue

                amount = self._get_next_trade_amount(a)
                logging.info(f"[{a}] (EXECUTE@OPEN) {sig.upper()}, Score: {score:.3f}, MG Step: {current_mg_step}, Amount: ${amount:.2f}")

                slot = self._slot_id(period_sec)
                if self._last_slot_fired.get(key) == slot:
                    logging.info(f"[{a}] Already traded this slot ({slot}). Skip.")
                    continue
                if not self._try_lock(key, duration * 60 + 7):
                    logging.info(f"[{a}] Locked on execute phase, skip.")
                    continue

                ok, trade_info = self._place_safe(amount, a, sig, duration)
                if ok and isinstance(trade_info, int):
                    self._last_slot_fired[key] = slot
                    trade_id = trade_info
                    logging.info(f"[{a}] Trade placed. Trade ID: {trade_id}")

                    strategy_label = (
                        self.settings.get("strategy_name")
                        or getattr(self.strategy, "name", None)            # type: ignore[union-attr]
                        or self.strategy.__class__.__name__               # type: ignore[union-attr]
                    )
                    pred_prob = None if p_up is None else (float(p_up) if sig == "call" else float(1.0 - float(p_up)))

                    db_id = trade_logger.log_trade(
                        asset=a,
                        direction=sig,
                        amount=amount,
                        duration=duration,
                        strategy=strategy_label,
                        mg_step=current_mg_step + 1,
                        pred_prob=pred_prob,
                    )

                    self._maybe_ai_autoretrain(duration)

                    th = threading.Thread(
                        target=self._check_and_update_trade_result,
                        args=(trade_id, db_id, duration, a),
                        daemon=True,
                    )
                    th.start()
                    placed += 1
                else:
                    logging.error(f"[{a}] Failed to place trade: {trade_info}")
                    self._unlock(key)

            logging.info(f"[Seq] done. placed={placed}. Loop waiting {period_sec} sec…")
            self._sleep_interruptible(period_sec)

    # ------------------------------------------------------------------ #
    # AI auto-retrain bump                                               #
    # ------------------------------------------------------------------ #
    def _maybe_ai_autoretrain(self, duration_min: int) -> None:
        try:
            if ai_meta is None:
                return
            auto_enabled = bool(self.settings.get("ai_auto_retrain", False))
            every = int(self.settings.get("ai_retrain_every_orders", 40) or 40)
            if not auto_enabled or every <= 0:
                return
            tf = int(getattr(self.strategy, "tf", duration_min))  # type: ignore[union-attr]
            count, enabled, every = ai_meta.bump_on_place(tf)
            logging.info(f"[AI][{tf}m] bump via ai_meta → {count}/{every} (auto={enabled})")
            if enabled and count >= every:
                st = ai_meta.auto_check_and_run(tf)
                logging.info(f"[AI][{tf}m] auto retrain done → counter={st.get('orders_since_retrain')}")
        except Exception as e:
            logging.warning(f"[AI] auto-run error: {e}")


# Backward-compat alias for existing imports
bot_engine = BotEngine()
