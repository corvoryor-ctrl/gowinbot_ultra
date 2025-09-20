# iq_wrapper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from iqoptionapi.stable_api import IQ_Option
    import iqoptionapi.constants as iq_constants
except Exception as e:  # pragma: no cover
    IQ_Option = None  # type: ignore
    iq_constants = None  # type: ignore
    logging.error("iqoptionapi not available: %s", e)

Number = Union[int, float]


_CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "THB": "฿", "RUB": "₽", "BRL": "R$", "TRY": "₺",
    "IDR": "Rp", "INR": "₹", "VND": "₫", "AUD": "A$",
    "CAD": "C$", "NZD": "NZ$", "CHF": "CHF", "CNY": "¥"
}


class IQWrapper:
    """
    Thread-safe convenience wrapper around iqoptionapi.stable_api.IQ_Option

    จุดเด่น:
      • Lock ครอบทุกจุดที่แตะ API ป้องกัน race ใน multi-thread
      • ensure_connected() ช่วย reconnect อัตโนมัติเมื่อหลุด
      • ครอบคลุมฟังก์ชันยอดนิยมในโปรเจกต์นี้ + alias ให้ชื่อเดิมยังใช้ได้
      • รองรับหลายฟอร์กของ API (เช่น check_win_v3 / v2 / check_win)
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self, email: str, password: str, account_type: str = "PRACTICE"):
        self.email: str = email
        self.password: str = password
        self.account_type: str = (account_type or "PRACTICE").upper()
        self.api: Optional[IQ_Option] = IQ_Option(email, password) if IQ_Option else None

        # reentrant lock: ปลอดภัยแม้เมธอด call ซ้อนกันเอง
        self.lock = threading.RLock()

        # init-data cache
        self.last_init_data: Optional[Dict[str, Any]] = None
        self.last_init_data_time: float = 0.0

        # reconnect helpers
        self._relogin_lock = threading.Lock()
        self._last_connect_ts: float = 0.0
        self._backoff: float = 1.0  # seconds
        # keep-alive
        self._ka_thread: Optional[threading.Thread] = None
        self._ka_stop = threading.Event()
        self._ka_interval = 10.0

    # ให้ภายนอกเรียกหา client ได้ (เผื่อโค้ด legacy)
    @property
    def client(self) -> Optional[IQ_Option]:
        return self.api

    # ------------------------------------------------------------------
    # Connect / Session
    # ------------------------------------------------------------------
    def connected(self) -> bool:
        try:
            return bool(self.api and self.api.check_connect())
        except Exception:
            return False

    # --- Backward-compat alias (บางไฟล์เรียก is_connected) ---
    def is_connected(self) -> bool:
        return self.connected()

    # --------- helpers ใหม่สำหรับจัดการ WS/รีอินสแตนซ์ ----------
    def _ws_closed(self) -> bool:
        try:
            return (self.api is None) or (not self.api.check_connect())
        except Exception:
            return True

    def reset_api(self) -> None:
        """ปิดการเชื่อมต่อเดิม (ถ้ามี) แล้วล้าง self.api ให้พร้อมสร้างใหม่"""
        with self.lock:
            try:
                if self.api:
                    for name in ("stop_keepalive", "close", "close_connection", "logout"):
                        try:
                            fn = getattr(self.api, name, None)
                            if callable(fn):
                                fn()
                        except Exception:
                            pass
            finally:
                self.api = None
    # -----------------------------------------------------------

    def connect(self, force_new: bool = False, retry: int = 2, timeout: int = 10) -> Tuple[bool, Optional[str]]:
        """
        เชื่อมต่อกับ IQ Option:
        - ถ้า ws ปิด หรือ force_new=True → รีอินสแตนซ์ IQ_Option ใหม่
        - รีไทรอัตโนมัติ และเริ่ม keepalive เมื่อสำเร็จ
        คืน (ok:bool, reason:str|None)
        """
        if IQ_Option is None:
            logging.error("IQ_Option not available.")
            return False, "iqoptionapi not available"

        with self.lock:
            # สร้าง instance ใหม่ถ้าจำเป็น
            if force_new or self.api is None or self._ws_closed():
                try:
                    try:
                        self.stop_keepalive()
                    except Exception:
                        pass
                    self.reset_api()
                    self.api = IQ_Option(self.email, self.password)
                except Exception as e:
                    logging.error("init api failed: %s", e)
                    return False, str(e)

            last_err: Optional[str] = None
            for attempt in range(retry + 1):
                try:
                    logging.info("Connecting to IQ Option API (attempt %d/%d, account=%s)...",
                                attempt + 1, retry + 1, self.account_type)
                    ok, reason = self.api.connect()  # type: ignore[union-attr]
                    logging.info("api.connect() -> ok=%s, reason=%r", ok, reason)

                    if not ok:
                        last_err = str(reason or "connect() returned False")
                    else:
                        # รอจน websocket พร้อมจริง
                        ready = False
                        t0 = time.time()
                        for i in range(max(1, int(timeout / 0.5))):
                            try:
                                if self.api and self.api.check_connect():
                                    ready = True
                                    break
                            except Exception as e:
                                logging.debug("check_connect() error: %s", e)
                            time.sleep(0.5)
                        logging.info("check_connect() -> ready=%s (%.1fs)", ready, time.time() - t0)

                        if ready:
                            try:
                                self.set_account(self.account_type)
                            except Exception as e:
                                logging.warning("set_account(%s) warn: %s", self.account_type, e)
                            try:
                                self.get_all_init_data(force_refresh=True)
                            except Exception as e:
                                logging.warning("get_all_init_data warn: %s", e)
                            self._last_connect_ts = time.time()
                            self._backoff = 1.0
                            logging.info("Connection successful.")
                            self.start_keepalive(getattr(self, "_ka_interval", 7.0))
                            return True, None
                        else:
                            last_err = "connection check timed out"

                except Exception as e:
                    last_err = str(e)
                    logging.error("connect() attempt %d failed: %s", attempt + 1, e)

                # ถ้าไม่สำเร็จ → reset แล้วลองใหม่
                try:
                    self.reset_api()
                    self.api = IQ_Option(self.email, self.password)
                except Exception as e:
                    last_err = f"reinit failed: {e}"
                    break
                time.sleep(1.0)

            logging.error("Connection failed: %s", last_err)
            return False, last_err

    def ensure_connected(self, force: bool = False) -> bool:
        """Reconnect อัตโนมัติเมื่อหลุด; ถ้า force=True จะรีสร้าง session ใหม่"""
        # ถ้า WS ปิดจริง ให้ถือว่า force ทันที
        try:
            if self._ws_closed():
                force = True
        except Exception:
            pass

        if self.connected() and not force:
            return True
        with self._relogin_lock:
            if self.connected() and not force:
                return True

            # 1) ลอง reuse session เดิม (ไม่ force)
            try:
                if not force and self.api:
                    self.api.connect()  # type: ignore[union-attr]
                    if self.connected():
                        self._backoff = 1.0
                        logging.info("[IQ] reconnected (reuse session)")
                        return True
            except Exception:
                pass

            # 2) สร้าง session ใหม่
            try:
                if IQ_Option is None:
                    return False
                try:
                    self.stop_keepalive()
                except Exception:
                    pass
                self.reset_api()
                self.api = IQ_Option(self.email, self.password)
                ok, reason = self.api.connect()  # type: ignore[union-attr]
                if not ok:
                    raise RuntimeError(f"connect() failed: {reason}")
                if not self.connected():
                    raise RuntimeError("connect() did not establish connection")

                try:
                    self.set_account(self.account_type)
                except Exception:
                    pass
                try:
                    self.get_all_init_data(force_refresh=True)
                except Exception:
                    pass

                self._last_connect_ts = time.time()
                self._backoff = 1.0
                logging.info("[IQ] reconnected OK (new session)")
                self.start_keepalive(getattr(self, "_ka_interval", 7.0))
                return True
            except Exception as e:
                logging.error(f"[IQ] reconnect failed: {e}")
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 1.6, 15.0)

        return self.connected()

    def _switch_balance_safely(self, kind: str) -> None:
        """สลับบัญชี (REAL/PRACTICE) ให้ชัวร์: set_balance → change_balance(id)"""
        kind = (kind or "PRACTICE").upper()
        try:
            if not self.api:
                return

            # 1) เมธอดมาตรฐานใน stable_api: รับสตริง 'REAL'/'PRACTICE'
            fn_set = getattr(self.api, "set_balance", None)
            if callable(fn_set):
                try:
                    fn_set(kind)   # บางฟอร์กไม่คืนค่า แต่จะสลับสำเร็จ
                    return
                except Exception:
                    pass  # ตกไปลองวิธีถัดไป

            # 2) Fallback: change_balance(balance_id) ต้องส่ง "เลข id"
            fn_chg = getattr(self.api, "change_balance", None)
            if callable(fn_chg):
                bid = None
                try:
                    balances = self.api.get_balances()
                    if isinstance(balances, (list, tuple)):
                        for b in balances:
                            if not isinstance(b, dict):
                                continue
                            t = str(b.get("type") or b.get("name") or "").upper()
                            if t == kind:
                                bid = b.get("id") or b.get("balance_id") or b.get("id_v2")
                                break
                except Exception:
                    balances = None

                if bid is not None:
                    try:
                        ok = fn_chg(int(bid))
                        if not ok:
                            logging.warning("change_balance(id=%s) returned False for %s", bid, kind)
                    except Exception as e:
                        logging.warning("change_balance(id) raised for %s: %s", kind, e)
                else:
                    # เผื่อฟอร์กที่รองรับสตริงใน change_balance (ส่วนน้อย)
                    try:
                        fn_chg(kind)
                    except Exception:
                        pass

            else:
                logging.info("API has no set_balance/change_balance; skip")
        except Exception:
            logging.info("Switch balance raised; continue")

    def set_account(self, account_type: str) -> float:
        """เปลี่ยนประเภทบัญชี (REAL / PRACTICE) และคืนยอดเงินล่าสุด"""
        with self.lock:
            self.account_type = (account_type or "PRACTICE").upper()
            self._switch_balance_safely(self.account_type)

        # รอให้เซิร์ฟเวอร์อัปเดตเล็กน้อย แล้วลองอ่านซ้ำไม่เกิน ~2s
        for i in range(4):
            time.sleep(0.5 if i == 0 else 0.3)
            bal = self.get_balance()
            if bal > 0:                      # ส่วนใหญ่ REAL จะ > 0 ถ้าใช้งานจริง
                return bal
        return self.get_balance()

    def change_balance(self, kind: str) -> bool:
        """alias ภายนอก"""
        try:
            self.set_account(kind)
            return True
        except Exception:
            return False

    def logout(self) -> None:
        with self.lock:
            if not self.api:
                return
            try:
                fn = getattr(self.api, "close", None) or getattr(self.api, "logout", None)
                if callable(fn):
                    fn()
            except Exception:
                pass
            finally:
                self.api = None

    def close(self) -> None:
        self.logout()

    def start_keepalive(self, interval: float = 7.0):
        """เริ่มเธรด keep-alive (ไอดอมโพเทนต์)"""
        try:
            self._ka_interval = max(3.0, float(interval or 7.0))
        except Exception:
            self._ka_interval = 7.0

        t = getattr(self, "_ka_thread", None)
        if t and t.is_alive():
            # แค่ปรับ interval แล้วปล่อยให้เธรดเดิมทำงานต่อ
            logging.info("[IQ] keep-alive running (interval=%.1fs)", self._ka_interval)
            return

        self._ka_stop_evt = threading.Event()

        def _loop():
            logging.info("[IQ] keep-alive started (%.1fs)", self._ka_interval)
            try:
                while not self._ka_stop_evt.is_set():
                    try:
                        # เบาที่สุด: แค่เช็กสถานะ ไม่ยิงคำสั่งหนัก ๆ
                        _ = (not self._ws_closed())
                    except Exception as e:
                        logging.debug("[IQ] keep-alive tick error: %s", e)
                    # รอแบบ interruptible
                    self._ka_stop_evt.wait(self._ka_interval)
            finally:
                logging.info("[IQ] keep-alive stopped")

        self._ka_thread = threading.Thread(target=_loop, name="iq-keepalive", daemon=True)
        self._ka_thread.start()


    def stop_keepalive(self):
        """สั่งหยุดเธรด keep-alive (ไอดอมโพเทนต์, ไม่ log ซ้ำ)"""
        evt = getattr(self, "_ka_stop_evt", None)
        t = getattr(self, "_ka_thread", None)

        if evt is not None:
            try:
                evt.set()
            except Exception:
                pass

        if t and t.is_alive():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

        # เคลียร์รีเฟอเรนซ์
        self._ka_thread = None
        self._ka_stop_evt = None

    def _keepalive_loop(self) -> None:
        while not self._ka_stop.is_set():
            try:
                ok = self.ping()  # ping ภายในเรียก ensure_connected() ให้แล้ว
                if not ok:
                    self._ka_fail = int(getattr(self, "_ka_fail", 0) + 1)
                    # ครั้งแรก: soft ensure (ไม่ force)
                    if self._ka_fail == 1:
                        try:
                            self.ensure_connected(force=False)
                        except Exception:
                            pass
                    # ตั้งแต่ครั้งที่ 2 ขึ้นไป: force reconnect แล้วรีเซ็ตตัวนับ
                    else:
                        self.ensure_connected(force=True)
                        self._ka_fail = 0
                else:
                    self._ka_fail = 0
                    # 🔥 อุ่น init-data เบื้องหลัง ถ้าแก่กว่า 120s (ไม่บล็อกเมนลูป)
                    try:
                        if (time.time() - float(getattr(self, "last_init_data_time", 0))) > 120:
                            threading.Thread(
                                target=lambda: self.get_all_init_data(force_refresh=True),
                                daemon=True
                            ).start()
                    except Exception:
                        pass
            except Exception as e:
                logging.warning("[IQ] keep-alive tick error: %s", e)

            self._ka_stop.wait(self._ka_interval)


    # ------------------------------------------------------------------
    # Basics
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        try:
            if not self.ensure_connected():
                raise RuntimeError("not connected")
            with self.lock:
                if not self.api:
                    return 0.0
                bal = self.api.get_balance()
                return float(bal) if bal is not None else 0.0
        except Exception as e:
            logging.warning("get_balance failed: %s", e)
            return 0.0

    def get_server_time(self) -> int:
        with self.lock:
            if not self.api:
                return int(time.time())
            try:
                return int(self.api.get_server_timestamp())
            except Exception:
                return int(time.time())

    # ------------------------------------------------------------------
    # Profile / Currency helpers
    # ------------------------------------------------------------------
    def _probe_profile_dict(self) -> Dict[str, Any]:
        with self.lock:
            # 1) get_profile()
            if self.api and hasattr(self.api, "get_profile"):
                try:
                    p = self.api.get_profile()
                    if isinstance(p, dict):
                        return p
                except Exception:
                    pass

            # 2) get_profile_ansyc()  (สะกดแบบนี้จริงในบางฟอร์ก)
            if self.api and hasattr(self.api, "get_profile_ansyc"):
                try:
                    p = self.api.get_profile_ansyc()
                    if isinstance(p, dict):
                        return p
                except Exception:
                    pass

            # 3) self.api.profile
            if self.api and hasattr(self.api, "profile"):
                try:
                    p = self.api.profile  # type: ignore[attr-defined]
                    if isinstance(p, dict):
                        return p
                except Exception:
                    pass

            # 4) self.api.api.profile
            if self.api and hasattr(self.api, "api") and hasattr(self.api.api, "profile"):  # type: ignore[attr-defined]
                try:
                    p = self.api.api.profile  # type: ignore[attr-defined]
                    if isinstance(p, dict):
                        return p
                except Exception:
                    pass

            # 5) fallback ผ่าน balances
            try:
                if self.api and hasattr(self.api, "get_balances"):
                    balances = self.api.get_balances()
                    if isinstance(balances, (list, tuple)) and balances:
                        # pick the one matches our account type or first
                        pick = None
                        for b in balances:
                            if not isinstance(b, dict):
                                continue
                            t = str(b.get("type") or b.get("name") or "").upper()
                            if t == self.account_type:
                                pick = b; break
                        if pick is None:
                            pick = balances[0] if isinstance(balances[0], dict) else None
                        if isinstance(pick, dict):
                            return pick
            except Exception:
                pass

        return {}

    def get_currency_info(self) -> Dict[str, str]:
        """
        คืน {"code": "...", "symbol": "..."}
        """
        try:
            prof = self._probe_profile_dict()
            cands = [
                prof.get("currency"),
                prof.get("currency_code"),
                prof.get("user_currency"),
                prof.get("currency_id"),
                prof.get("balance_currency") if isinstance(prof.get("balance_currency"), str) else None,
            ]
            code = next((str(x).upper() for x in cands if x), None)
            symbol = prof.get("currency_char") if isinstance(prof.get("currency_char", None), str) else None

            if not code:
                code = "THB"
            if not symbol:
                symbol = _CURRENCY_SYMBOLS.get(code, "฿" if code == "THB" else "$")
            return {"code": code, "symbol": symbol}
        except Exception as e:
            logging.warning(f"[IQ] get_currency_info failed: {e}")
            return {"code": "THB", "symbol": "฿"}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def _resolve_active_id(self, asset: Union[str, int]) -> int:
        if isinstance(asset, int):
            return asset
        key = str(asset).upper().strip()
        if iq_constants and hasattr(iq_constants, "ACTIVES"):
            aid = iq_constants.ACTIVES.get(key)  # type: ignore[attr-defined]
        else:
            aid = None
        if aid is None:
            raise KeyError(f"Unknown asset '{key}'. Did you load actives.json?")
        return int(aid)

    def _normalize_candles(self, raw_list: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not raw_list:
            return []
        out: List[Dict[str, Any]] = []
        for c in raw_list:
            if not isinstance(c, dict):
                continue

            ts = c.get("from") or c.get("at") or c.get("timestamp") or c.get("to")
            try:
                ts = int(ts)
            except Exception:
                ts = None

            o = c.get("open")
            h = c.get("high", c.get("max"))
            l = c.get("low", c.get("min"))
            cl = c.get("close")
            v = c.get("volume", 0)

            if h is None and (o is not None and cl is not None):
                try:
                    h = max(float(o), float(cl))
                except Exception:
                    h = None
            if l is None and (o is not None and cl is not None):
                try:
                    l = min(float(o), float(cl))
                except Exception:
                    l = None

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            o = _to_float(o); h = _to_float(h); l = _to_float(l); cl = _to_float(cl)
            try:
                v = float(v)
            except Exception:
                v = 0.0

            if None in (o, h, l, cl):
                continue

            out.append({"ts": ts, "open": o, "high": h, "low": l, "close": cl, "volume": v})
        return out

    # --- เพิ่ม helper ด้านบนคลาส IQWrapper ---
    def _safe_int(x):
        try: return int(x)
        except Exception: return None

    # ภายในคลาส IQWrapper เพิ่ม helper นี้
    def _symbolize(self, asset: Union[str, int]) -> str:
        """รับทั้ง 'EURUSD-OTC' หรือ id:int แล้วคืน symbol เป็นสตริงเสมอ"""
        if isinstance(asset, int):
            # reverse-lookup (จาก iqoptionapi.constants.ACTIVES)
            try:
                if iq_constants and hasattr(iq_constants, "ACTIVES"):
                    for k, v in iq_constants.ACTIVES.items():
                        try:
                            if int(v) == int(asset):
                                return str(k)
                        except Exception:
                            continue
            except Exception:
                pass
            return str(asset)  # fallback
        return str(asset).upper().strip()

    def server_timestamp(self) -> Optional[float]:
        """อ่านเวลา server จากไลบรารี ถ้ามี"""
        with self.lock:
            try:
                if self.api and hasattr(self.api, "timesync"):
                    ts = getattr(self.api.timesync, "server_timestamp", None)
                    if ts is None:
                        return None
                    # เผื่อบาง fork ให้ ms
                    return float(ts / 1000.0) if ts > 1e12 else float(ts)
            except Exception:
                pass
        return None

    # --- ปรับ get_candles / get_candles_safe / get_candles_tf ให้รับ to_ts ---

    def get_candles(
        self,
        asset: Union[str, int],
        period: int = 60,
        count: int = 120,
        to_ts: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """ดึงข้อมูลแท่งเทียนย้อนหลัง (คืนค่าเป็น OHLCV มาตรฐานเสมอ)"""
        with self.lock:
            try:
                if not self.api:
                    return None
                end = int(to_ts or time.time())
                sym = self._symbolize(asset)
                raw = self.api.get_candles(sym, int(period), int(count), end)
                return self._normalize_candles(raw)
            except Exception as e:
                logging.error(f"Error getting candles for {asset}: {e}")
                return None

    def get_candles_safe(
        self,
        asset: Union[str, int],
        period: int = 60,
        count: int = 120,
        to_ts: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """เวอร์ชันกันหลุด + normalize เป็น OHLCV"""
        try:
            if not self.ensure_connected():
                raise RuntimeError("not connected")
            with self.lock:
                if not self.api:
                    return None
                end = int(to_ts or time.time())
                sym = self._symbolize(asset)
                raw = self.api.get_candles(sym, int(period), int(count), end)
                return self._normalize_candles(raw)
        except Exception as e:
            if "Connection is already closed" in str(e) or "websocket" in str(e).lower():
                logging.warning("[IQ] get_candles_safe: reconnect and retry once")
                if self.ensure_connected(force=True):
                    with self.lock:
                        if not self.api:
                            return None
                        end = int(to_ts or time.time())
                        sym = self._symbolize(asset)
                        raw = self.api.get_candles(sym, int(period), int(count), end)
                        return self._normalize_candles(raw)
            logging.error(f"get_candles_safe error for {asset}: {e}")
            return None

    def get_candles_tf(
        self,
        asset: Union[str, int],
        tf_min: int,
        count: int = 120,
        to_ts: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """เรียกด้วย tf นาที + รองรับ to_ts"""
        period = int(tf_min) * 60
        return self.get_candles(asset, period=period, count=count, to_ts=to_ts)

    def get_last_closed_candle(self, asset: str, tf_min: int) -> Optional[Dict[str, Any]]:
        rows = self.get_candles_tf(asset, tf_min, count=2) or []
        return rows[-1] if rows else None

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------
    def buy(self, amount: Number, asset: str, direction: str, duration_min: int = 1) -> Tuple[bool, Union[int, str, None]]:
        """เปิดออเดอร์ (binary/turbo API)"""
        with self.lock:
            try:
                if not self.api:
                    return False, "not connected"
                return self.api.buy(float(amount), asset, direction.lower(), int(duration_min))
            except Exception as e:
                logging.error(f"Error placing buy order for {asset}: {e}")
                return False, str(e)

    # ป้องกัน WS หลุด
    def buy_safe(self, amount: Number, asset: str, direction: str, duration_min: int = 1) -> Tuple[bool, Union[int, str, None]]:
        try:
            if not self.ensure_connected():
                raise RuntimeError("not connected")
            return self.buy(amount, asset, direction, duration_min)
        except Exception as e:
            if "Connection is already closed" in str(e) or "websocket" in str(e).lower():
                logging.warning("[IQ] buy_safe: reconnect and retry once")
                if self.ensure_connected(force=True):
                    return self.buy(amount, asset, direction, duration_min)
            logging.error(f"buy_safe error for {asset}: {e}")
            return False, str(e)

    # บางโปรเจกต์ใช้ digital
    def buy_digital_spot(self, asset: str, amount: Number, direction: str, duration_min: int = 1) -> Tuple[Optional[int], Any]:
        """proxy ไป api.buy_digital_spot() ถ้ามี"""
        with self.lock:
            try:
                if not self.api:
                    return None, None
                fn = getattr(self.api, "buy_digital_spot", None)
                if not callable(fn):
                    return None, None
                return fn(asset, float(amount), direction.lower(), int(duration_min))
            except Exception as e:
                logging.error(f"buy_digital_spot error for {asset}: {e}")
                return None, None

    # alias ตามชื่อเดิมที่บางไฟล์ใช้
    def place_digital_option(self, asset: str, direction: str, amount: Number, duration_min: int = 1) -> Tuple[Optional[int], Any]:
        return self.buy_digital_spot(asset, amount, direction, duration_min)

    def place(self, amount: Number, asset: str, direction: str, duration_min: int = 1) -> Tuple[bool, Union[int, str, None]]:
        """เหมือน buy() แต่กันหลุดให้ (เรียกใช้ใน engine ได้เลย)"""
        return self.buy_safe(amount, asset, direction, duration_min)

    # ผลลัพธ์ออเดอร์ (WIN/LOSE/EQUAL) => คืนเป็น "กำไรสุทธิ" (float) หรือ None
    def check_win(self, trade_id: int) -> Optional[float]:
        with self.lock:
            try:
                if not self.api:
                    return None
                # โปรเจกต์เดิมเรียก v3 โดยตรง
                fn = getattr(self.api, "check_win_v3", None)
                if callable(fn):
                    return self._normalize_profit(fn(trade_id))
                # ถ้าไม่มี v3 ใช้ v2/check_win
                return self.check_win_v3(trade_id)
            except Exception as e:
                logging.error(f"Error checking win for trade ID {trade_id}: {e}")
                return None

    def check_win_v3(self, trade_id: int, timeout: int = 5) -> Optional[float]:
        """คืนกำไร (positive/negative) ถ้าทราบผลแล้ว, ถ้ายังไม่ทราบ -> None"""
        # keep alive
        if not self.ensure_connected():
            return None

        with self.lock:
            if not self.api:
                return None

            candidates = (
                "check_win_v3",
                "check_win_v2",
                "check_win",
                "get_win",
                "result",
                "get_closed_deal_profit",
            )
            for name in candidates:
                fn = getattr(self.api, name, None)
                if not callable(fn):
                    continue
                try:
                    try:
                        ret = fn(trade_id, timeout=timeout)
                    except TypeError:
                        ret = fn(trade_id)
                    val = self._normalize_profit(ret)
                    if val is not None:
                        return val
                except Exception:
                    continue
        return None

    @staticmethod
    def _normalize_profit(ret: Any) -> Optional[float]:
        """แปลงผลลัพธ์ให้เป็น float ถ้าทำได้ (dict/tuple/str/number)"""
        try:
            if ret is None:
                return None
            if isinstance(ret, (int, float)):
                return float(ret)
            if isinstance(ret, (list, tuple)):
                if not ret:
                    return None
                return float(ret[0])
            if isinstance(ret, dict):
                for k in ("profit", "pnl", "amount", "result", "win"):
                    if k in ret and ret[k] is not None:
                        try:
                            return float(ret[k])
                        except Exception:
                            continue
                txt = str(ret.get("win") or ret.get("status") or "").lower()
                amt = ret.get("amount") or ret.get("profit_amount")
                if txt and amt is not None:
                    if txt in ("win", "won", "equal"):
                        return float(amt)
                    if txt in ("loose", "lose", "loss"):
                        return -abs(float(amt))
                return None
            return float(ret)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Utilities for UI / endpoints
    # ------------------------------------------------------------------
    def get_all_init_data(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """ดึงข้อมูล init ทั้งหมด + cache 180s (ช่วยลดภาระบน VPS/เน็ตช้า)"""
        with self.lock:
            if not self.api:
                return None
            current_time = time.time()
            # TTL เดิม 60s → 180s
            if force_refresh or not self.last_init_data or (current_time - self.last_init_data_time > 180):
                try:
                    # บางฟอร์กใช้ get_all_init_v2, บางฟอร์กใช้ get_all_init
                    fn = getattr(self.api, "get_all_init_v2", None) or getattr(self.api, "get_all_init", None)
                    if callable(fn):
                        self.last_init_data = fn()
                    else:
                        self.last_init_data = {}
                except Exception as e:
                    logging.warning(f"get_all_init_data failed: {e}")
                    self.last_init_data = {}
                self.last_init_data_time = current_time
            return self.last_init_data

    def get_all_open_time(self) -> Dict[str, Any]:
        with self.lock:
            if not self.api:
                return {}
            try:
                return self.api.get_all_open_time()
            except Exception as e:
                logging.warning("get_all_open_time failed: %s", e)
                return {}

    # alias เพื่อความเข้ากันได้
    def get_open_time(self) -> Dict[str, Any]:
        return self.get_all_open_time()

    def get_all_profit(self) -> Dict[str, Any]:
        with self.lock:
            if not self.api:
                return {}
            try:
                return self.api.get_all_profit()
            except Exception as e:
                logging.warning("get_all_profit failed: %s", e)
                return {}

    def is_asset_open(self, asset: str) -> bool:
        """สะดวกเช็คว่าสินทรัพย์เปิดเทรดอยู่ไหม (รวม turbo/binary)"""
        try:
            det = self.get_asset_details(asset)  # ใช้ cache ใน wrapper เดิม
            return bool(det and det.get("is_open"))
        except Exception:
            return False

    def get_asset_details(self, asset: str) -> Dict[str, Any]:
        """
        ดึงข้อมูลสำคัญของสินทรัพย์: id, is_open, payout
        ใช้ init-data cache + iq_constants.ACTIVES
        """
        details = {"id": None, "is_open": False, "payout": 0.0}
        init_data = self.get_all_init_data()
        if not init_data or not iq_constants or not hasattr(iq_constants, "ACTIVES"):
            return details

        asset_id = iq_constants.ACTIVES.get(str(asset).upper())
        if not asset_id:
            return details
        details["id"] = asset_id

        for option_type in ("binary", "turbo"):
            try:
                asset_data = init_data[option_type]["actives"][str(asset_id)]
                is_open = asset_data.get("enabled", False) and not asset_data.get("is_suspended", True)
                if is_open:
                    details["is_open"] = True
                commission = asset_data.get("option", {}).get("profit", {}).get("commission", 100)
                payout = (100.0 - commission) / 100.0
                if payout > details["payout"]:
                    details["payout"] = payout
            except Exception:
                continue
        return details

    def ensure_asset_open_and_payout(self, asset: str, min_payout: float = 0.70) -> Tuple[bool, float]:
        det = self.get_asset_details(asset) or {}
        payout = float(det.get("payout", 0.0) or 0.0)
        ok = bool(det.get("is_open")) and payout >= float(min_payout)
        return ok, payout

    def ping(self) -> bool:
        try:
            if not self.ensure_connected():
                return False
            with self.lock:
                if not self.api:
                    return False
                _ = self.api.get_server_timestamp()
            return True
        except Exception as e:
            logging.warning(f"[IQ] ping failed: {e}")
            return False
