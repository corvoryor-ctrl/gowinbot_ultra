#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend/main.py — clean & stable edition (SSE + robust connect)
- /connect รับ email/password/account_type จาก UI ทั้งแบบ JSON และ form-data
  และลองเดาฟิลด์ยอดฮิต (username, user, pwd, pass, credentials.email ฯลฯ)
  พร้อมฟอลแบ็กอ่านจาก user_state.json / ENV ถ้า UI ไม่ส่งมา
  และจะบันทึก credentials ลง user_state.json ให้เองเมื่อ connect สำเร็จ
- กัน WS ปิดหลัง Stop: ใช้ iq_wrapper.connect(force_new=True) + reset_api()
- /status ไม่ฝืน reconnect เมื่อบอทหยุด
- SSE /events + sse_push() สำหรับอัปเดตสด ลดการสแปมเรียก API
- คงฟีเจอร์เดิมทั้งหมด: license/connect/status/start/stop/indicators/assets/trades/session/settings
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wintypes
import json
import asyncio
import logging
import os
import signal
import threading
import time
import webbrowser
import inspect  # PATCH: for safe signature checks
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException, Query, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from websocket._exceptions import WebSocketConnectionClosedException
from pydantic import BaseModel, Field

# ------------------------------- CONFIG ---------------------------------

import sys
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = Path(BASE_DIR)
INDEX_PATH = BASE_PATH / "index.html"

def _user_data_dir(app_name: str = "GowinbotUltra") -> Path:
    try:
        if os.name == "nt":
            root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(root) / app_name
        elif sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / app_name
        else:
            root = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
            return Path(root) / app_name
    except Exception:
        return Path.cwd() / app_name

MODELS_DIR = BASE_PATH / "models"
try:
    need_user_dir = getattr(sys, "frozen", False) or (not os.access(BASE_PATH, os.W_OK))
    if need_user_dir:
        MODELS_DIR = _user_data_dir() / "models"
except Exception:
    MODELS_DIR = _user_data_dir() / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MARKET_HOURS_PATH = BASE_PATH / "market_hours.json"
ACTIVES_PATH = BASE_PATH / "actives.json"
ORDER_SETTINGS_PATH = MODELS_DIR / "order_settings.json"

DATA_DIR = _user_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)
USER_STATE_PATH = DATA_DIR / "user_state.json"

LICENSE_PATH = DATA_DIR / "license.json"
_license_cache = {"ok": False, "raw": ""}

# ------------------------------ LOGGING ---------------------------------

try:
    from util.logging_config import setup_logging
except Exception:
    setup_logging = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
if callable(setup_logging):
    try:
        setup_logging(app_name="GowinBotUltra", level="INFO")
    except Exception:
        pass

# ------------------------------ IMPORTS ---------------------------------

from iq_wrapper import IQWrapper
from iq_bootstrap import ensure_mapping
from database import trade_logger
from core.license_checker import check_license
from core.bot_engine import bot_engine

# strategies (ตัวอย่างที่โปรเจกต์นี้ใช้)
from strategies.ai_model_1m import AIModel1mStrategy
from strategies.ai_model_5m import AIModel5mStrategy
from strategies.momentum_breakout_fake import MomentumBreakoutFakeStrategy
from strategies.strategy_r2s import R2SStrategy
from strategies.sniper_confluence import SniperConfluence
from strategies.sniper_flowshift import SniperFlowShift
from strategies.impulse_sma_rsi import ImpulseSmaRsiStrategy
from strategies.macd_ema_psar_1m import MacdEmaPsar1mStrategy
from strategies.macd_ema_psar_5m import MacdEmaPsar5mStrategy
from strategies.break_retest_go_1m import BreakRetestGo1mStrategy


try:
    import ai_trainer  # noqa: F401
except Exception:
    ai_trainer = None  # type: ignore

try:
    import machineid
except Exception:
    machineid = None  # type: ignore

from services.health import router as health_router

app = FastAPI(title="GowinBot Ultra Backend", version="1.1.1")
app.include_router(health_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# PATCH: always include ai_router whether it's backend.ai_service or ai_service
try:
    try:
        from backend.ai_service import router as ai_router
    except ImportError:
        from ai_service import router as ai_router
    app.include_router(ai_router)
except Exception as e:
    logging.warning(f"[AI Router] Not loaded: {e}")

iq_session: Optional[IQWrapper] = None
_start_lock = threading.Lock()

ensure_mapping(path=str(ACTIVES_PATH))

KEEPALIVE_SEC = int(os.getenv("IQ_KEEPALIVE_SEC", "7"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "9000"))
AUTO_OPEN_URL = f"http://{HOST}:{PORT}"

# --------------------------- MODELS / PAYLOADS --------------------------

class LicensePayload(BaseModel):
    license_key: str

class LoginPayload(BaseModel):
    email: str
    password: str
    account_type: Literal["PRACTICE", "REAL"] = "PRACTICE"

class BotStartPayload(BaseModel):
    strategy_name: str
    duration: Literal[1, 5]
    assets: List[str]
    amount: float = Field(..., gt=0)
    block_start: Optional[str] = None
    block_end: Optional[str] = None
    ai1m_conf: Optional[float] = None
    ai1m_gap: Optional[float] = None
    ai5m_conf: Optional[float] = None
    ai5m_gap: Optional[float] = None
    min_conf: Optional[float] = None
    min_gap: Optional[float] = None
    martingale_mode: Optional[str] = "None"
    martingale_scope: Optional[str] = "Separate"
    martingale_custom_amounts: Optional[List[float]] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    max_orders: Optional[int] = None
    max_orders_per_cycle: Optional[int] = None
    sensitivity: Optional[int] = 50
    ai_auto_retrain: Optional[bool] = True
    ai_retrain_every_orders: Optional[int] = 40
    lead_time_sec: Optional[int] = 5
    micro_filter_5m: Optional[bool] = True
    trend_filter_mode: Optional[Literal["off", "weak", "strict"]] = "weak"

class AIAutoPayload(BaseModel):
    tf: Literal[1, 5]
    enabled: bool
    every_orders: int = 40

class AIActionPayload(BaseModel):
    tf: Literal[1, 5]
    reason: Optional[str] = "manual"
    method: Optional[str] = "auto"

class LiveRequest(BaseModel):
    asset: str
    tf: int = 1
    n: int = 200

# --------------------------- INDICATOR CONFIG ---------------------------

INDICATOR_FILES = {
    1: MODELS_DIR / "indicators_1m.json",
    5: MODELS_DIR / "indicators_5m.json",
}

INDICATOR_DEFAULT: Dict[str, Any] = {
    "use": True,
    "ma":      {"enabled": False, "type": "EMA", "length": 50, "source": "close", "bias": "any", "slope": "any"},
    "macd":    {"enabled": False, "fast": 12, "slow": 26, "signal": 9, "mode": "confirm"},
    "ichimoku":{"enabled": False, "tenkan": 9, "kijun": 26, "senkou_b": 52, "mode": "trend"},
    "rsi":     {"enabled": False, "length": 14, "ob": 70, "os": 30, "mode": "filter"},
    "stoch":   {"enabled": False, "k": 14, "d": 3, "smooth": 3, "ob": 80, "os": 20, "mode": "filter"},
    "bb":      {"enabled": False, "length": 20, "k": 2.0, "mode": "squeeze"},
    "atr":     {"enabled": False, "length": 14, "min_atr": 0.0},
    "vwap":    {"enabled": False},
    "obv":     {"enabled": False, "mode": "confirm"},
    "volprof": {"enabled": False, "bins": 24},
    "price_action": {
        "enabled": False,
        "patterns": ["bullish_engulfing", "bearish_engulfing", "pin_bar", "inside_bar", "morning_star", "evening_star"],
        "lookback": 5,
        "min_body_ratio": 1.2,
        "min_wick_ratio": 2.0,
        "confirm": "none",
        "bias": "any"
    },
}

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            tmp = dict(out[k]); tmp.update(v); out[k] = tmp
        else:
            out[k] = v
    return out

def _ind_path(tf: int) -> Path:
    return INDICATOR_FILES[5 if int(tf) == 5 else 1]

def _ind_read(tf: int) -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = _ind_path(tf)
    if not p.exists():
        p.write_text(json.dumps(INDICATOR_DEFAULT, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        return _merge(INDICATOR_DEFAULT, json.loads(p.read_text(encoding="utf-8") or "{}"))
    except Exception:
        return dict(INDICATOR_DEFAULT)

def _ind_write(tf: int, cfg: Dict[str, Any]) -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = _ind_path(tf)
    data = _merge(INDICATOR_DEFAULT, cfg or {})
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

def _load_indicator_cfg(tf: int) -> Dict[str, Any]:
    for name in (f"indicators_{tf}m.json", f"indicators_tf{tf}.json"):
        candidate = BASE_PATH / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
    return _ind_read(tf)

TREND_BADGE_MAP = {
    "Call":    {"class": "text-bg-success", "hex": "#198754"},
    "Put":     {"class": "text-bg-danger",  "hex": "#dc3545"},
    "SideWay": {"class": "text-bg-warning", "hex": "#ffc107"},
}

def _align_close_ts(ts: int, tf_min: int) -> int:
    step = int(tf_min) * 60
    return int(ts - (ts % step))

def _trend_dirs(asset: str) -> tuple[str, str]:
    try:
        server_ts = int(getattr(iq_session, "get_server_time", lambda: int(time.time()))())
    except Exception:
        server_ts = int(time.time())

    def _dir(tf_min: int) -> str:
        try:
            to_ts = _align_close_ts(server_ts, tf_min)
            rows = iq_session.get_candles_tf(asset, tf_min, count=40, to_ts=to_ts) or []
            if len(rows) < 8:
                return "flat"
            rows = sorted(rows, key=lambda r: r.get("ts", 0))
            c_now  = float(rows[-1].get("close", 0.0))
            c_prev = float(rows[-8].get("close", 0.0))
            delta = c_now - c_prev
            if delta > 0:  return "up"
            if delta < 0:  return "down"
            return "flat"
        except Exception:
            return "flat"

    return _dir(5), _dir(15)

def _trend_label_color(d5: str, d15: str) -> tuple[str, dict]:
    if d5 == "up" and d15 == "up":
        return "Call", TREND_BADGE_MAP["Call"]
    if d5 == "down" and d15 == "down":
        return "Put", TREND_BADGE_MAP["Put"]
    return "SideWay", TREND_BADGE_MAP["SideWay"]

def _is_asset_open(asset: str) -> bool:
    try:
        # ถ้ามีเมธอด builtin ใน IQWrapper ก็ใช้เลย
        if hasattr(iq_session, "is_asset_open"):
            return bool(getattr(iq_session, "is_asset_open")(asset))
    except Exception:
        pass
    # ฟอลแบ็ก: ถ้าดึงเทียน 1m ล่าสุดได้ แปลว่าเปิด (แบบหยาบๆ)
    try:
        server_ts = int(getattr(iq_session, "get_server_time", lambda: int(time.time()))())
        to_ts = _align_close_ts(server_ts, 1)
        rows = iq_session.get_candles_tf(asset, 1, count=2, to_ts=to_ts) or []
        return len(rows) >= 1
    except Exception:
        return False

def _last_price(asset: str) -> tuple[float, float]:
    try:
        server_ts = int(getattr(iq_session, "get_server_time", lambda: int(time.time()))())
        to_ts = _align_close_ts(server_ts, 1)
        rows = iq_session.get_candles_tf(asset, 1, count=2, to_ts=to_ts) or []
        rows = sorted(rows, key=lambda r: r.get("ts", 0))
        c_now  = float(rows[-1].get("close", 0.0)) if rows else 0.0
        c_prev = float(rows[-2].get("close", 0.0)) if len(rows) >= 2 else c_now
        return c_now, c_prev
    except Exception:
        return 0.0, 0.0

@app.get("/asset/hud")
def api_asset_hud(asset: str = Query(..., min_length=3)):
    """
    คืน HUD รายคู่เงิน: เปิด/ปิด, ราคา, เทรนด์ 5m/15m, ป้ายสีสำหรับ UI
    """
    if iq_session is None:
        raise HTTPException(status_code=503, detail="Not connected")

    # เทรนด์ 5m/15m จากโค้ดเดิม
    d5, d15 = _trend_dirs(asset)               # up/down/flat  (5m, 15m)
    label, badge = _trend_label_color(d5, d15) # "Call"/"Put"/"SideWay" + class/hex

    is_open = _is_asset_open(asset)
    px, px_prev = _last_price(asset)

    try:
        server_ts = int(getattr(iq_session, "get_server_time", lambda: int(time.time()))())
    except Exception:
        server_ts = int(time.time())

    return {
        "asset": asset,
        "server_ts": server_ts,
        "is_open": bool(is_open),
        "price": float(px),
        "price_prev": float(px_prev),
        "trend_5m": d5,
        "trend_15m": d15,
        "trend_label": label,         # "Call" | "Put" | "SideWay"
        "trend_badge": badge,         # {"class": "...", "hex": "..."}
    }

# ------------------------------ HELPERS ---------------------------------

def _sma(arr: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(arr) < length: return None
    return sum(arr[-length:]) / float(length)

def _ema(arr: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(arr) < length: return None
    k = 2.0 / (length + 1)
    ema = sum(arr[:length]) / length
    for x in arr[length:]:
        ema = x * k + ema * (1 - k)
    return ema

def _rsi(closes: List[float], length: int = 14) -> Optional[float]:
    if len(closes) <= length: return None
    gains = losses = 0.0
    for i in range(1, length + 1):
        ch = closes[i] - closes[i - 1]
        if ch >= 0: gains += ch
        else:       losses += -ch
    avg_gain = gains / length
    avg_loss = losses / length
    rsi_list: List[float] = []
    for i in range(length + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gain = max(0.0, ch)
        loss = max(0.0, -ch)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        rsi_list.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi_list[-1] if rsi_list else None

def _macd(closes: List[float], f: int = 12, s: int = 26, sig: int = 9) -> Optional[Dict[str, float]]:
    if len(closes) < s + sig + 5: return None
    def ema(arr: List[float], n: int) -> float:
        k = 2.0 / (n + 1)
        e = sum(arr[:n]) / n
        for x in arr[n:]:
            e = x * k + e * (1 - k)
        return e
    line = ema(closes, f) - ema(closes, s)
    series: List[float] = []
    for i in range(s, len(closes) + 1):
        series.append(ema(closes[:i], f) - ema(closes[:i], s))
    if len(series) < sig:
        return {"line": line, "signal": 0.0, "hist": line}
    k = 2.0 / (sig + 1)
    e = sum(series[:sig]) / sig
    for x in series[sig:]:
        e = x * k + e * (1 - k)
    return {"line": line, "signal": e, "hist": line - e}

def _bb(closes: List[float], n: int = 20, k: float = 2.0):
    if len(closes) < n: return None
    m = sum(closes[-n:]) / n
    var = sum((x - m) ** 2 for x in closes[-n:]) / n
    std = var ** 0.5
    return {"mid": m, "upper": m + k * std, "lower": m - k * std}

# PATCH: rework stochastic to safe rolling windows and correct %D
def _stoch(highs: List[float], lows: List[float], closes: List[float], klen: int = 14, dlen: int = 3):
    if len(closes) < klen or len(highs) < klen or len(lows) < klen:
        return None
    def k_at(offset: int) -> float:
        # offset=0: current bar, 1: prev bar, ...
        hi = max(highs[-(klen+offset): -offset or None])
        lo = min(lows[-(klen+offset): -offset or None])
        cl = closes[-(1+offset)]
        return 100.0 * (cl - lo) / (hi - lo) if hi > lo else 50.0
    try:
        k_vals = [k_at(i) for i in range(0, max(1, dlen))]
        k = k_vals[0]
        d = sum(k_vals) / len(k_vals)
        return {"k": k, "d": d}
    except Exception:
        return None

def _atr(highs: List[float], lows: List[float], closes: List[float], n: int = 14):
    if len(closes) < n + 1: return None
    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs[-n:]) / n

def _obv(closes: List[float], volumes: List[float]):
    if not volumes or len(closes) != len(volumes): return None
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]: obv += volumes[i]
        elif closes[i] < closes[i - 1]: obv -= volumes[i]
    return obv

def _min_bars_for_cfg(cfg: Dict[str, Any]) -> int:
    if not isinstance(cfg, dict): return 240
    ind = cfg.get("indicators") or cfg.get("custom") or cfg
    needs: List[int] = []
    def _bars_needed_for_indicator(name: str, p: Dict[str, Any]) -> int:
        name = (name or "").lower(); p = p or {}
        def clamp(n, lo=1, hi=2000): return max(lo, min(int(n), hi))
        if name in ("ma", "moving_average"):
            length = int(p.get("length") or p.get("period") or 50)
            ma_type = (p.get("type") or "EMA").upper()
            return clamp(length * (3 if ma_type == "EMA" else 1) + 10)
        if name == "macd":
            fast = int(p.get("fast") or 12); slow = int(p.get("slow") or 26); signal = int(p.get("signal") or 9)
            return clamp(slow + signal + slow * 3)
        if name in ("ichimoku", "ichimoku_cloud"):
            t = int(p.get("tenkan") or 9); k = int(p.get("kijun") or 26); b = int(p.get("senkou_b") or p.get("span_b") or 52)
            return clamp(max(b, k, t) + 26)
        if name == "rsi":
            n = int(p.get("length") or p.get("period") or 14); return clamp(n + 5)
        if name in ("stochastic", "stoch", "stochastic_oscillator"):
            klen = int(p.get("k") or p.get("k_len") or 14); dlen = int(p.get("d") or p.get("d_len") or 3); return clamp(klen + dlen + 3)
        if name in ("bb", "bollinger", "bollinger_bands"):
            n = int(p.get("length") or p.get("period") or 20); return clamp(n + 5)
        if name in ("atr", "average_true_range"):
            n = int(p.get("length") or p.get("period") or 14); return clamp(n + 6)
        if name in ("obv", "on_balance_volume"):
            return clamp(120)
        if name in ("volume_profile", "vp", "vprofile"):
            win = int(p.get("window") or p.get("period") or 150); return clamp(win)
        if name in ("price_action", "pa"):
            lb = int(p.get("lookback") or 5); return clamp(lb + 3)
        return 60
    for k, v in ind.items():
        try:
            if isinstance(v, dict) and v.get("enabled"):
                needs.append(_bars_needed_for_indicator(k, v))
        except Exception:
            continue
    required = max(needs) if needs else 240
    required = int(required * 1.25) + 5
    return max(60, min(required, 1000))

# ------------------------- License helpers ------------------------------

def _load_saved_license() -> None:
    try:
        if LICENSE_PATH.exists():
            raw = LICENSE_PATH.read_text(encoding="utf-8")
            ok, _ = check_license(raw)
            _license_cache.update({"ok": bool(ok), "raw": raw})
        else:
            _license_cache.update({"ok": False, "raw": ""})
    except Exception:
        _license_cache.update({"ok": False, "raw": ""})

def _save_license(raw: str) -> None:
    LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_PATH.write_text(raw, encoding="utf-8")
    _license_cache.update({"ok": True, "raw": raw})

def _require_license() -> None:
    if os.getenv("DEV_LICENSE_ALLOW", "0") == "1":
        return
    if not _license_cache["ok"]:
        _load_saved_license()
    if not _license_cache["ok"]:
        raise HTTPException(status_code=401, detail="License required or invalid.")

# -------------------------- Credential storage --------------------------

def _win_protect(raw: bytes) -> str:
    if not raw: return ""
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    blob_in = DATA_BLOB(len(raw), ctypes.cast(ctypes.create_string_buffer(raw), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if hasattr(ctypes, "windll"):
        CryptProtectData = ctypes.windll.crypt32.CryptProtectData
        CryptProtectData.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.c_wchar_p, ctypes.POINTER(DATA_BLOB),
                                     ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                     ctypes.POINTER(DATA_BLOB)]
        if CryptProtectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
            try:
                out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
                return base64.b64encode(out).decode("ascii")
            finally:
                ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return ""

def _win_unprotect(s: str) -> bytes:
    if not s: return b""
    data = base64.b64decode(s.encode("ascii"))
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if hasattr(ctypes, "windll"):
        CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
        CryptUnprotectData.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.POINTER(ctypes.c_wchar_p),
                                       ctypes.POINTER(DATA_BLOB), ctypes.c_void_p, ctypes.c_void_p,
                                       wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
        if CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
            try:
                out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
                return out
            finally:
                ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return b""

def _enc_cred(email: str, password: str) -> Dict[str, str]:
    if os.name == "nt":
        return {"scheme": "dpapi",
                "email": _win_protect((email or "").encode("utf-8")),
                "password": _win_protect((password or "").encode("utf-8"))}
    else:
        return {"scheme": "plain", "email": email or "", "password": password or ""}

def _dec_cred(obj: Dict[str, str]) -> Dict[str, str]:
    if not obj:
        return {"email": "", "password": ""}
    if obj.get("scheme") == "dpapi" and os.name == "nt":
        return {
            "email": _win_unprotect(obj.get("email", "")).decode("utf-8", "ignore"),
            "password": _win_unprotect(obj.get("password", "")).decode("utf-8", "ignore"),
        }
    return {"email": obj.get("email", ""), "password": obj.get("password", "")}

def _load_credentials_from_disk() -> tuple[Optional[str], Optional[str], str]:
    acct = "PRACTICE"
    try:
        if USER_STATE_PATH.exists():
            data = json.loads(USER_STATE_PATH.read_text(encoding="utf-8") or "{}")
            creds = _dec_cred(data.get("credentials", {}))
            settings = data.get("settings", {}) or {}
            acct = str(settings.get("account_type", acct)).upper() if settings.get("account_type") else acct
            email = (creds or {}).get("email") or None
            password = (creds or {}).get("password") or None
            return email, password, acct
    except Exception as e:
        logging.warning(f"[cred] load failed: {e}")
    email = os.getenv("IQ_EMAIL")
    password = os.getenv("IQ_PASSWORD")
    acct = (os.getenv("IQ_ACCOUNT_TYPE") or acct).upper()
    return email, password, acct

def _persist_credentials(email: Optional[str], password: Optional[str], account_type: Optional[str]) -> None:
    if not email or not password:
        return
    try:
        data = {}
        if USER_STATE_PATH.exists():
            try:
                data = json.loads(USER_STATE_PATH.read_text(encoding="utf-8") or "{}")
            except Exception:
                data = {}
        data["credentials"] = _enc_cred(email, password)
        settings = data.get("settings", {}) or {}
        if account_type:
            settings["account_type"] = account_type.upper()
        data["settings"] = settings
        USER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning(f"[cred] persist failed: {e}")

def _apply_creds_to_session(iq: Optional[IQWrapper], email: Optional[str], password: Optional[str], account_type: Optional[str]):
    if iq is None:
        return
    if email:
        setattr(iq, "email", email)
    if password:
        setattr(iq, "password", password)
    if account_type:
        at = account_type.upper()
        try:
            if hasattr(iq, "set_account") and callable(iq.set_account):
                iq.set_account(at)
            else:
                setattr(iq, "account_type", at)
        except Exception:
            setattr(iq, "account_type", at)

# ----------------------------- LOCK DEPENDENCY --------------------------

START_LOCK = threading.Lock()
STOP_LOCK  = threading.Lock()

def require_start_lock():
    with START_LOCK:
        yield

def require_stop_lock():
    with STOP_LOCK:
        yield

# -------------------------------- ROUTES --------------------------------

@app.get("/")
def get_index():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(INDEX_PATH))

@app.get("/system/hwid")
def api_get_hwid():
    if not machineid:
        raise HTTPException(status_code=500, detail="machineid module not available")
    try:
        return {"status": "ok", "hwid": machineid.id()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve HWID: {e}")

@app.post("/license/check")
def api_check_license(payload: LicensePayload):
    is_valid, message = check_license(payload.license_key)
    if not is_valid:
        raise HTTPException(status_code=401, detail=message)
    _save_license(payload.license_key)
    return {"status": "ok", "message": message}

@app.on_event("startup")
def on_startup():
    trade_logger.init_db()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    try:
        _load_saved_license()
    except Exception:
        pass

# --------------------------- SSE pub/sub --------------------------------

_status_cache = {"ts": 0.0, "data": None}
_IND_CACHE: Dict[tuple[str, int], Dict[str, Any]] = {}
_IND_TTL = float(os.getenv("INDICATOR_TTL", "0.6"))

_sse_queues: "set[asyncio.Queue[str]]" = set()

def sse_push(event: str, **fields):
    payload = {"event": event, "ts": int(time.time()), **fields}
    msg = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    targets = list(_sse_queues)
    if loop and loop.is_running():
        for q in targets:
            loop.call_soon_threadsafe(q.put_nowait, msg)
    else:
        for q in targets:
            try:
                q.put_nowait(msg)
            except Exception:
                pass

@app.get("/events")
async def sse_events(request: Request):
    q: asyncio.Queue[str] = asyncio.Queue()
    _sse_queues.add(q)

    async def gen():
        try:
            q.put_nowait("data: {\"event\":\"hello\"}\n\n")
            while True:
                # Starlette 0.27+: request.is_disconnected() เป็น coroutine
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                    yield item
                except asyncio.TimeoutError:
                    # comment line for SSE keep-alive
                    yield ":\n\n"
        finally:
            _sse_queues.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")

# --------------------------- Connect & Status ---------------------------

async def _parse_any_credentials(request: Request) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    รับ credentials จาก JSON หรือ form + เดาฟิลด์ยอดนิยมให้:
    - email: email, username, user, login, iq_email, credentials.email
    - password: password, pwd, pass, iq_password, credentials.password
    - account_type: account_type, account, mode, balance, credentials.account_type
    """
    email = password = account_type = None

    # 1) พยายามอ่าน JSON
    try:
        data = await request.json()
        if isinstance(data, dict):
            # flat
            email = data.get("email") or data.get("username") or data.get("user") or data.get("login") or data.get("iq_email")
            password = data.get("password") or data.get("pwd") or data.get("pass") or data.get("iq_password")
            account_type = data.get("account_type") or data.get("account") or data.get("mode") or data.get("balance")
            # nested
            cred = data.get("credentials") or data.get("cred") or {}
            if isinstance(cred, dict):
                email = email or cred.get("email") or cred.get("username") or cred.get("user")
                password = password or cred.get("password") or cred.get("pwd") or cred.get("pass")
                account_type = account_type or cred.get("account_type") or cred.get("account") or cred.get("mode")
    except Exception:
        pass

    # 2) พยายามอ่าน form-data
    try:
        form = await request.form()
        def pick(keys: List[str]):
            for k in keys:
                if k in form and form.get(k):
                    return str(form.get(k))
            return None
        email = email or pick(["email","username","user","login","iq_email"])
        password = password or pick(["password","pwd","pass","iq_password"])
        account_type = account_type or pick(["account_type","account","mode","balance"])
    except Exception:
        pass

    if account_type:
        account_type = str(account_type).upper()

    return email, password, account_type

# PATCH: unify connect return to (ok, reason)
def _unwrap_connect_result(ret) -> Tuple[bool, str]:
    try:
        if isinstance(ret, (tuple, list)):
            ok = bool(ret[0])
            reason = str(ret[1]) if len(ret) > 1 else ""
            return ok, reason
        return bool(ret), ""
    except Exception as e:
        return False, str(e)

# PATCH: generic param checker (for /bot/stop quick reconnect)
def _has_param(func, name: str) -> bool:
    try:
        sig = inspect.signature(func)
        return name in sig.parameters
    except Exception:
        code = getattr(func, "__code__", None) or getattr(getattr(func, "__func__", None), "__code__", None)
        try:
            return bool(code and name in getattr(code, "co_varnames", ()))
        except Exception:
            return False

@app.post("/connect")
async def api_connect(request: Request):
    """
    รับ credentials จาก UI (JSON หรือ form ก็ได้)
    - ถ้าให้ creds มา → อัปเดตเข้า iq_session แล้ว connect(force_new=True) และบันทึกลง user_state.json
    - ถ้าไม่ให้มา → ฟอลแบ็กอ่านจาก user_state.json / ENV
    """
    global iq_session

    email_req, password_req, account_type_req = await _parse_any_credentials(request)

    # account type
    if account_type_req:
        account_type = account_type_req
    else:
        account_type = getattr(iq_session, "account_type", None) if iq_session else None
        if not account_type:
            _, _, acct_disk = _load_credentials_from_disk()
            account_type = acct_disk
        account_type = (account_type or "PRACTICE").upper()

    # เตรียม session
    if iq_session is None:
        if not email_req or not password_req:
            email_d, pwd_d, acct_d = _load_credentials_from_disk()
            email_req = email_req or email_d
            password_req = password_req or pwd_d
            account_type = account_type or acct_d or "PRACTICE"
            if not email_req or not password_req:
                # ส่ง 200 + รายละเอียดให้ UI อ่านง่าย
                return JSONResponse(
                    {"status": "error",
                     "connected": False,
                     "reason": "Missing credentials. Please include 'email' and 'password' in JSON or form."},
                    status_code=200
                )
        try:
            iq_session = IQWrapper(email_req, password_req, account_type=account_type)
        except TypeError:
            iq_session = IQWrapper(email_req, password_req)  # type: ignore[call-arg]
            _apply_creds_to_session(iq_session, None, None, account_type)
    else:
        _apply_creds_to_session(iq_session, email_req, password_req, account_type)

    # ต่อแบบฮาร์ดรีเซ็ต (รองรับทั้ง bool และ (ok, reason))
    try:
        ok, reason = _unwrap_connect_result(iq_session.connect(force_new=True, retry=2, timeout=10))
        if not ok:
            # ลองซ้ำอีกครั้ง
            try:
                iq_session.reset_api()
            except Exception:
                pass
            ok2, reason2 = _unwrap_connect_result(iq_session.connect(force_new=True, retry=1, timeout=10))

            # ตอบกลับแบบ 200 พร้อมเหตุผล เพื่อให้ UI แสดงข้อความได้แม่น
            if not ok2:
                logging.error("/connect failed: %s", reason2 or reason)
                return JSONResponse(
                    {"status": "error", "connected": False, "reason": str(reason2 or reason)},
                    status_code=200
                )

        # สำเร็จ → เก็บคริดบนดิสก์
        if email_req and password_req:
            _persist_credentials(email_req, password_req, account_type)

        try:
            sse_push("session:connected", connected=True)
        except Exception:
            pass

        return {"status": "ok", "connected": True, "account_type": getattr(iq_session, "account_type", account_type)}
    except Exception as e:
        logging.exception("/connect exception: %s", e)
        return JSONResponse({"status": "error", "connected": False, "reason": str(e)}, status_code=200)

@app.get("/status")
def api_get_status():
    _require_license()
    global iq_session, _status_cache
    now = time.time()
    if _status_cache["data"] is not None and (now - _status_cache["ts"] < 2.0):
        return _status_cache["data"]

    is_connected = False
    balance = 0.0
    account_type = None
    currency_code = None
    currency_symbol = None
    try:
        if iq_session and getattr(iq_session, "api", None):
            try:
                if hasattr(iq_session, "ensure_connected") and bool(getattr(bot_engine, "is_running", False)):
                    iq_session.ensure_connected()
            except Exception:
                pass

            if hasattr(iq_session.api, "check_connect"):
                is_connected = bool(iq_session.api.check_connect())
            else:
                is_connected = True

            if is_connected:
                if hasattr(iq_session, "get_balance"):
                    balance = float(iq_session.get_balance() or 0.0)
                account_type = getattr(iq_session, "account_type", None)
                if hasattr(iq_session, "get_currency_info"):
                    cur = iq_session.get_currency_info() or {}
                    currency_code = cur.get("code")
                    currency_symbol = cur.get("symbol")
    except Exception as e:
        logging.warning(f"/status error: {e}")

    resp = {
        "is_connected": is_connected,
        "is_bot_running": bool(getattr(bot_engine, "is_running", False)),
        "balance": balance,
        "account_type": account_type,
        "currency_code": currency_code,
        "currency_symbol": currency_symbol,
        "stats": {
            "win": int(getattr(bot_engine, "session_win", 0) or 0),
            "loss": int(getattr(bot_engine, "session_lose", 0) or 0),
            "draw": int(getattr(bot_engine, "session_draw", 0) or 0),
            "max_step": int(getattr(bot_engine, "session_max_step", 0) or 0),
        },
        "hud_max_step": int(getattr(bot_engine, "hud_max_step", 0) or 0),
    }
    _status_cache["ts"] = now
    _status_cache["data"] = resp
    try:
        sse_push("status", **resp)
    except Exception:
        pass
    return resp

# ------------------------- Settings SAVE/LOAD ---------------------------

@app.get("/settings/load")
def settings_load():
    if not USER_STATE_PATH.exists():
        return {"ok": True, "settings": {}, "credentials": {}, "mg_state": {}}
    try:
        data = json.loads(USER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    creds = _dec_cred(data.get("credentials", {}))
    return {"ok": True,
            "settings": data.get("settings", {}),
            "credentials": creds,
            "mg_state": data.get("mg_state", {})}

@app.post("/settings/save")
def settings_save(payload: dict = Body(...)):
    try:
        settings = payload.get("settings", {}) or {}
        cred_in  = payload.get("credentials", {}) or {}
        mg_state = payload.get("mg_state", {}) or {}
        data = {
            "settings": settings,
            "credentials": _enc_cred(cred_in.get("email",""), cred_in.get("password","")),
            "mg_state": mg_state,
        }
        USER_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(USER_STATE_PATH)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------------------- Indicator Endpoints --------------------------

@app.get("/indicators/schema")
def api_indicators_schema():
    return INDICATOR_DEFAULT

@app.get("/indicators/get")
def api_indicators_get(tf: int = Query(1, ge=1, le=5)):
    return _ind_read(tf)

class IndicatorSetPayload(BaseModel):
    tf: int = Field(1, ge=1, le=5)
    config: Dict[str, Any]

@app.post("/indicators/set")
def api_indicators_set(payload: IndicatorSetPayload):
    ok = _ind_write(payload.tf, payload.config)
    if not ok:
        raise HTTPException(status_code=500, detail="save failed")
    return {"status": "ok"}

@app.post("/indicators/live")
def api_indicators_live(req: LiveRequest):
    if iq_session is None or not getattr(iq_session, "is_connected", lambda: False)():
        raise HTTPException(status_code=400, detail="Not connected")

    cfg = _load_indicator_cfg(req.tf)
    needed = _min_bars_for_cfg(cfg)

    n = max(int(req.n or 0), needed)
    n = min(max(n, 60), 1000)

    key = (req.asset, int(req.tf))
    now = time.time()
    c = _IND_CACHE.get(key)
    if c and (now - c["ts"] < _IND_TTL):
        return c["data"]

    to_ts = int(time.time()); to_ts -= to_ts % int(req.tf * 60)
    candles = iq_session.get_candles_tf(req.asset, req.tf, n, to_ts=to_ts) or []
    if not candles or len(candles) < 30:
        resp = {"asset": req.asset, "tf": req.tf, "values": {}}
        _IND_CACHE[key] = {"ts": now, "data": resp}
        return resp

    closes = [c["close"] for c in candles if "close" in c]
    highs  = [c["high"]  for c in candles if "high" in c]
    lows   = [c["low"]   for c in candles if "low"  in c]
    last   = closes[-1] if closes else None

    values = {
        "ma":   {"type": "EMA", "length": 50, "value": _ema(closes, 50), "slope":
                 (lambda v, p: (v - p) if v is not None and p is not None else 0.0)(
                     _ema(closes, 50), _ema(closes[:-1], 50)
                 )},
        "macd": _macd(closes, 12, 26, 9),
        "ichimoku": None,
        "rsi": _rsi(closes, 14),
        "bb": _bb(closes, n=20, k=2.0),
        "stoch": _stoch(highs, lows, closes, klen=14, dlen=3),
        "atr": _atr(highs, lows, closes, n=14),
        "obv": _obv(closes, [c.get("volume", 0.0) for c in candles]),
        "pa": None,
        "last": last,
    }
    resp = {"asset": req.asset, "tf": req.tf, "last": last, "valuesPrice": last, "values": values}
    _IND_CACHE[key] = {"ts": now, "data": resp}
    return resp

# ------------------------------ Market Hours ----------------------------

@app.get("/market-hours")
def api_market_hours():
    if not MARKET_HOURS_PATH.exists():
        return {"timezone": "Asia/Bangkok", "assets": {}}
    try:
        return json.loads(MARKET_HOURS_PATH.read_text(encoding="utf-8-sig"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot load market_hours.json: {e}")

# --- Adapter: expose /market/hours for UI (alias of /market-hours) ---
def _hours_flat_to_ui(doc: dict) -> dict:
    """
    รับเอกสารแบบเดิมที่ /market-hours คืน (เช่น {"timezone": "...", "assets": {...}})
    แล้วแปลงเป็นรูปที่ UI ใช้: {"timezone": "...", "hours": {ASSET: {Mon:[...], ...}}}
    """
    if not isinstance(doc, dict):
        return {"timezone": "Asia/Bangkok", "hours": {}}

    tz = doc.get("timezone") or "Asia/Bangkok"

    # กรณี 1: ไฟล์เป็นรูปแบบ { "assets": { "EURUSD": [ { "dow": "Mon", "ranges":[["07:00","23:59"], ...] }, ... ] } }
    if "assets" in doc and isinstance(doc["assets"], dict):
        hours = {}
        for asset, daylist in doc["assets"].items():
            daymap = {}
            if isinstance(daylist, list):
                for d in daylist:
                    try:
                        dow = d.get("dow")
                        ranges = d.get("ranges") or []
                        merged = []
                        if isinstance(ranges, list):
                            for pair in ranges:
                                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    merged.append(f"{pair[0]}-{pair[1]}")
                        if dow:
                            daymap[dow] = merged
                    except Exception:
                        continue
            hours[str(asset)] = daymap
        return {"timezone": tz, "hours": hours}

    # กรณี 2: ไฟล์ที่พร้อมใช้แล้วเป็น {"timezone": "...", "hours": {...}} ก็ผ่านคืนเลย
    if "hours" in doc and isinstance(doc["hours"], dict):
        return {"timezone": tz, "hours": doc["hours"]}

    # รูปอื่น ๆ → คืนว่างเพื่อไม่ให้ UI พัง
    return {"timezone": tz, "hours": {}}

@app.get("/market/hours")
def api_market_hours_ui():
    """
    UI ต้องการ /market/hours → รูปแบบ {"timezone": "...", "hours": {ASSET: {Mon:[...], ...}}}
    อะแดปต์จากไฟล์ market_hours.json ชุดเดียวกับ /market-hours
    """
    if not MARKET_HOURS_PATH.exists():
        return {"timezone": "Asia/Bangkok", "hours": {}}
    try:
        raw = json.loads(MARKET_HOURS_PATH.read_text(encoding="utf-8-sig"))
        return _hours_flat_to_ui(raw)
    except Exception as e:
        # อย่า 500 ให้ UI พัง; คืนว่างพอ
        logging.warning(f"/market/hours parse failed: {e}")
        return {"timezone": "Asia/Bangkok", "hours": {}}
# --- end adapter ---

# ------------------------------ Assets Details --------------------------

@app.post("/assets/details")
def api_assets_details(payload: Dict[str, Any] = Body(...)):
    if iq_session is None or not getattr(iq_session, "is_connected", lambda: False)():
        raise HTTPException(status_code=400, detail="Not connected")

    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise HTTPException(status_code=400, detail="assets must be a non-empty list")
    out: Dict[str, Any] = {}
    for a in assets:
        try:
            det = (iq_session.get_asset_details(a) or {})
            try:
                d5, d15 = _trend_dirs(a)
                t_label, t_badge = _trend_label_color(d5, d15)
            except Exception:
                d5, d15 = "flat", "flat"
                t_label, t_badge = "SideWay", TREND_BADGE_MAP["SideWay"]
            det["trend_5m"] = d5
            det["trend_15m"] = d15
            det["trend_label"] = t_label
            det["trend_badge"] = t_badge
            out[a] = det
        except Exception as e:
            out[a] = {"error": str(e)}
    return {"assets": out}

# ----------------------------- Bot Start/Stop ---------------------------

@app.get("/bot/mg_state")
def bot_get_mg_state():
    be = bot_engine
    try:
        if hasattr(be, "martingale") and hasattr(be.martingale, "get_state"):
            return {"ok": True, "mg_state": be.martingale.get_state()}
        if hasattr(be, "get_mg_state"):
            return {"ok": True, "mg_state": be.get_mg_state()}
    except Exception:
        pass
    return {"ok": True, "mg_state": {}}

@app.post("/bot/set_mg_state")
def bot_set_mg_state(payload: dict = Body(...)):
    be = bot_engine
    mg = (payload or {}).get("mg_state", {})
    try:
        if mg:
            if hasattr(be, "martingale") and hasattr(be.martingale, "load_state"):
                be.martingale.load_state(mg)
                return {"ok": True}
            if hasattr(be, "set_mg_state"):
                be.set_mg_state(mg)
                return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}

@app.post("/bot/start", dependencies=[Depends(require_start_lock)])
def api_start_bot(payload: BotStartPayload, background_tasks: BackgroundTasks):
    _require_license()

    if iq_session is None or not getattr(iq_session, "ensure_connected", lambda *a, **k: False)():
        raise HTTPException(status_code=400, detail="Not connected")

    strategy = _make_strategy(payload)

    settings: Dict[str, Any] = payload.dict()
    try:
        base = float(settings.get("base_amount", settings.get("amount", 1.0)) or 1.0)
    except Exception:
        base = 1.0
    settings["base_amount"] = base
    settings.pop("amount", None)

    if (settings.get("max_orders_per_cycle") in (None, 0)) and (settings.get("max_orders") is not None):
        try:
            settings["max_orders_per_cycle"] = int(settings["max_orders"])
        except Exception:
            pass

    if not _start_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Start is already in progress.")
    try:
        if getattr(bot_engine, "is_running", False):
            raise HTTPException(status_code=400, detail="Bot is already running.")

        def _do_start():
            try:
                bot_engine.start(iq_session, strategy, settings)
            finally:
                pass

        background_tasks.add_task(_do_start)
        sse_push("bot:start", is_running=True)
        return {"status": "ok", "message": "starting", "is_running": True}
    finally:
        _start_lock.release()

@app.post("/bot/stop")
def api_stop_bot():
    """
    Stop บอท แต่ 'อย่า' ทำให้หลุดการเชื่อมต่อ IQ Session
    ถ้า stop แล้วชั้นในไปปิด WS เราจะเช็ค/กู้คืน quick-reconnect ให้โดยอัตโนมัติ
    """
    global iq_session
    try:
        # ถ้า bot_engine.stop รองรับ keep_connection ให้สั่งไว้ก่อน (กันไว้ก่อน)
        try:
            ok, msg = bot_engine.stop(keep_connection=True)  # type: ignore
        except TypeError:
            ok, msg = bot_engine.stop()

        # ---- ค้ำประกันว่า session ยังต่ออยู่ ----
        try:
            if iq_session and getattr(iq_session, "api", None):
                chk = True
                if hasattr(iq_session.api, "check_connect"):
                    try:
                        chk = bool(iq_session.api.check_connect())
                    except Exception:
                        chk = False
                if not chk:
                    # quick reconnect ที่ 'ไม่' เปลี่ยนสถานะผู้ใช้
                    try:
                        if hasattr(iq_session, "reset_api"):
                            iq_session.reset_api()
                    except Exception:
                        pass
                    try:
                        connect_fn = getattr(iq_session, "connect", None)
                        if connect_fn:
                            if _has_param(connect_fn, "force_new"):
                                connect_fn(force_new=False, retry=1, timeout=6)
                            else:
                                connect_fn()  # type: ignore
                    except Exception as e:
                        logging.warning(f"[stop] quick reconnect failed: {e}")
                else:
                    # ยังต่ออยู่ ไม่ต้องทำอะไร
                    pass
        except Exception as e:
            logging.debug(f"[stop] keep-alive guard error: {e}")

        try:
            sse_push("bot:stop", is_running=False)
        except Exception:
            pass

        return {"status": "ok", "message": msg, "is_running": False}
    except Exception as e:
        logging.exception(f"stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"stop failed: {e}")


# ------------------------------- Trades ---------------------------------

def _load_trades_retry(limit: int, attempts: int = 5, base_sleep: float = 0.12):
    """
    ดึง trade ด้วย retry อัตโนมัติ กรณี sqlite 'database is locked' หรือ busy ชั่วคราว
    คืน [] แทนการโยน 500 เพื่อให้ UI ไม่เด้ง error
    """
    limit = max(10, min(int(limit or 100), 1000))
    last_err = None
    for i in range(max(1, attempts)):
        try:
            rows = trade_logger.get_all_trades(limit=limit) or []
            return rows
        except Exception as e:
            msg = str(e).lower()
            last_err = e
            if "locked" in msg or "busy" in msg:
                time.sleep(base_sleep * (i + 1))  # backoff นิด ๆ
                continue
            # error อื่น ๆ: ตัดจบ แต่ไม่โยน 500
            logging.warning(f"[trades] load failed (non-retry): {e}")
            return []
    logging.warning(f"[trades] load failed after retry: {last_err}")
    return []

@app.get("/trades")
def list_trades(limit: int = Query(100, ge=1, le=10000)):
    try:
        rows = _load_trades_retry(limit=limit)
        return {"trades": rows}
    except Exception as e:
        # ไม่โยน 500 — ส่ง [] กลับไปให้ UI แสดงแบบว่างแทน
        logging.warning(f"/trades error: {e}")
        return JSONResponse({"trades": [], "error": str(e)}, status_code=200)

@app.get("/trades/history")
def api_trades_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    คืนรูปแบบสม่ำเสมอเสมอ:
      { trades: [...], count: N, offset: X, error?: string }
    ต่อให้ชั้นในพัง ก็จะไม่โยน 500 เพื่อกัน UI เด้ง error
    """
    try:
        rows = []
        # ถ้ามี recent ใน memory ของบอท ใช้ก่อน (ไว)
        if bot_engine is not None and getattr(bot_engine, "get_recent_trades", None):
            try:
                rows = bot_engine.get_recent_trades(limit=limit, offset=offset) or []
            except Exception as e:
                logging.debug(f"[history] recent fallback error: {e}")
                rows = []

        # ถ้ายังว่าง → ลองอ่านจาก DB ด้วย retry
        if not rows:
            rows = _load_trades_retry(limit=limit)

        resp = {"trades": rows, "count": len(rows), "offset": offset}
        return JSONResponse(resp, status_code=200)
    except Exception as e:
        logging.warning(f"/trades/history error: {e}")
        # ไม่โยน 500
        return JSONResponse(
            {"trades": [], "count": 0, "offset": offset, "error": str(e)},
            status_code=200
        )

@app.post("/trades/export")
def api_export_trades():
    try:
        message = trade_logger.export_to_csv()
        return {"status": "ok", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- Session Stats ----------------------------

@app.get("/session/stats")
def api_session_stats():
    try:
        be = bot_engine
        return {
            "win": int(getattr(be, "session_win", 0) or 0),
            "lose": int(getattr(be, "session_lose", 0) or 0),
            "draw": int(getattr(be, "session_draw", 0) or 0),
            "pnl": float(getattr(be, "session_profit", 0.0) or 0.0),
            "max_step": int(getattr(be, "session_max_step", 0) or 0),
            "hud_max_step": int(getattr(be, "hud_max_step", 0) or 0),
            "started_at": getattr(be, "session_started_at", None),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/reset")
def api_session_reset():
    try:
        be = bot_engine
        if hasattr(be, "reset_session_stats"):
            be.reset_session_stats()
        else:
            be.session_win = 0
            be.session_lose = 0
            be.session_draw = 0
            be.session_profit = 0.0
            be.session_max_step = 0
            be.session_started_at = None
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------- Order Settings ----------------------------

class OrderSettings(BaseModel):
    base_amount: float = Field(1.0, gt=0)
    martingale_mode: Literal["None", "Flat", "Custom"] = "None"
    martingale_scope: Literal["Separate", "Combined"] = "Separate"
    martingale_on_draw: Literal["SAME", "RESET", "NEXT"] = "SAME"
    martingale_multiplier: float = Field(2.2, gt=0)
    martingale_custom_amounts: Optional[List[float]] = None
    max_orders_per_cycle: int = Field(1, ge=0)
    lead_time_sec: int = Field(5, ge=0, le=30)
    take_profit: Optional[float] = Field(None, ge=0)
    stop_loss: Optional[float] = Field(None, ge=0)

def _order_settings_read() -> OrderSettings:
    try:
        if ORDER_SETTINGS_PATH.exists():
            data = json.loads(ORDER_SETTINGS_PATH.read_text(encoding="utf-8"))
            return OrderSettings(**data)
    except Exception as e:
        logging.warning(f"[order] read failed: {e}")
    return OrderSettings()

def _order_settings_write(cfg: OrderSettings) -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ORDER_SETTINGS_PATH.write_text(json.dumps(cfg.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save settings: {e}")

@app.get("/order/settings")
def api_order_settings_get():
    return _order_settings_read().dict()

@app.post("/order/settings")
def api_order_settings_set(payload: OrderSettings):
    if payload.martingale_mode == "Custom":
        if not payload.martingale_custom_amounts or not len(payload.martingale_custom_amounts):
            raise HTTPException(status_code=400, detail="Custom ladder must have at least 1 amount.")
        if any(float(x) <= 0 for x in payload.martingale_custom_amounts):
            raise HTTPException(status_code=400, detail="Custom amounts must be positive.")
    _order_settings_write(payload)
    return {"status": "ok"}

class PreviewPayload(BaseModel):
    mg_step: int = Field(0, ge=0)
    settings: Optional[OrderSettings] = None

@app.post("/order/preview-amount")
def api_order_preview_amount(payload: PreviewPayload):
    s = payload.settings or _order_settings_read()
    step = int(payload.mg_step or 0)

    if s.martingale_mode == "Custom" and s.martingale_custom_amounts:
        idx = min(step, len(s.martingale_custom_amounts) - 1)
        amount = float(s.martingale_custom_amounts[idx])
    elif s.martingale_mode == "Flat":
        amount = float(s.base_amount)
    else:
        amount = float(s.base_amount) * (float(s.martingale_multiplier) ** step)

    return {
        "amount": round(amount, 2),
        "ladder": (
            s.martingale_custom_amounts
            if (s.martingale_mode == "Custom" and s.martingale_custom_amounts)
            else None
        )
    }

@app.post("/order/reset-mg")
def api_order_reset_mg():
    try:
        if hasattr(bot_engine, "martingale_state"):
            bot_engine.martingale_state.clear()
        if hasattr(bot_engine, "hud_max_step"):
            bot_engine.hud_max_step = 0
        if hasattr(bot_engine, "session_max_step"):
            bot_engine.session_max_step = 0
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------- Shutdown -------------------------------

@app.post("/system/shutdown")
def shutdown():
    logging.info("Shutdown request received. Terminating process.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "ok", "message": "Shutting down..."}

# ---------------------------- Strategy Picker ---------------------------

def _make_strategy(p: BotStartPayload):
    name = (p.strategy_name or "").lower()

    if "r2s" in name or "rsi+sma" in name:
        return R2SStrategy(p.dict() if hasattr(p, "dict") else {})

    # >>> สำคัญ: ส่ง settings เข้าไปตอนสร้าง AI <<<
    if "ai" in name and p.duration == 1:
        return AIModel1mStrategy(p.dict() if hasattr(p, "dict") else {})
    if "ai" in name and p.duration == 5:
        return AIModel5mStrategy(p.dict() if hasattr(p, "dict") else {})

    if "sniper confluence" in name:
        return SniperConfluence(p.dict() if hasattr(p, "dict") else {})
    if "sniper flow" in name:
        return SniperFlowShift(p.dict() if hasattr(p, "dict") else {})
    if "impulse" in name:
        return ImpulseSmaRsiStrategy(p.dict() if hasattr(p, "dict") else {})
    if "momentum breakout fake" in name:
        return MomentumBreakoutFakeStrategy()
    if "break-retest-go" in name or "retest-go" in name or "retest ema20" in name:
        return BreakRetestGo1mStrategy(p.dict() if hasattr(p, "dict") else {})

    # ✅ รวมไว้ที่เดียว และเลือก 1m/5m ตาม duration พร้อมส่ง settings
    if "macd+ema12+psar" in name:
        if p.duration == 5:
            return MacdEmaPsar5mStrategy(p.dict() if hasattr(p, "dict") else {})
        else:
            return MacdEmaPsar1mStrategy(p.dict() if hasattr(p, "dict") else {})

    # ------- Fallback: ไม่รู้จักชื่อกลยุทธ์ -> แจ้ง 400 ชัดเจน -------
    available = [
        "R2S Strategy (RSI+SMA)",
        "AI Model (1m)",
        "AI Model (5m)",
        "Sniper Confluence",
        "Sniper FlowShift (Rank)",
        "Impulse SMA+RSI",
        "Momentum Breakout Fake",
        "Break-Retest-Go (EMA20) (1m)",
        "MACD+EMA12+PSAR (1m)",
        "MACD+EMA12+PSAR (5m)",
    ]
    raise HTTPException(
        status_code=400,
        detail=f"Unknown strategy: {p.strategy_name}. Available: {', '.join(available)}"
    )

# ------------------------------ ENTRYPOINT ------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_config=None)
