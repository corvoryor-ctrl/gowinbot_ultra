# backend/ai_counters.py
# -*- coding: utf-8 -*-
import os, json, threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

APP_NAME = "GowinBotUltra"
_LOCK = threading.Lock()

# ----------------------------- PATHS (cross-platform, env-aware) -----------------------------
def _user_data_dir(app_name: str = APP_NAME) -> str:
    try:
        if os.name == "nt":
            root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
            return os.path.join(root, app_name)
        elif os.sys.platform == "darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", app_name)
        else:
            root = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
            return os.path.join(root, app_name)
    except Exception:
        return os.getcwd()

def _resolve_data_dir() -> str:
    # 1) ENV override
    v = os.environ.get("APP_DATA_DIR")
    if v:
        p = os.path.abspath(os.path.expanduser(v))
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            pass
    # 2) user data default
    ud = os.path.join(_user_data_dir(), "data")
    try:
        os.makedirs(ud, exist_ok=True)
        return ud
    except Exception:
        pass
    # 3) project ./data fallback
    try:
        base = os.path.abspath(os.path.dirname(__file__) or ".")
    except Exception:
        base = os.getcwd()
    proj = os.path.abspath(os.path.join(base, "..", "data"))
    try:
        os.makedirs(proj, exist_ok=True)
        return proj
    except Exception:
        return os.getcwd()

DATA_DIR = _resolve_data_dir()
STATE_PATH = os.path.join(DATA_DIR, "ai_state.json")

# ----------------------------- DEFAULT STATE -----------------------------
_DEFAULT_STATE: Dict[str, Any] = {
    "auto": {
        "1": {"enabled": True, "every_orders": 40},
        "5": {"enabled": True, "every_orders": 40},
    },
    "counters": {
        "1": {"orders_since_retrain": 0, "last_retrain_ts": None},
        "5": {"orders_since_retrain": 0, "last_retrain_ts": None},
    }
}

# ----------------------------- HELPERS -----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _merge_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(_DEFAULT_STATE))  # deep copy
    for k, v in (d or {}).items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    # ensure nested defaults exist
    for tf in ("1", "5"):
        out["auto"].setdefault(tf, {"enabled": True, "every_orders": 40})
        out["counters"].setdefault(tf, {"orders_since_retrain": 0, "last_retrain_ts": None})
    return out

def _load() -> Dict[str, Any]:
    with _LOCK:
        if not os.path.isfile(STATE_PATH):
            return json.loads(json.dumps(_DEFAULT_STATE))
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
        return _merge_defaults(d)

def _save(state: Dict[str, Any]) -> None:
    with _LOCK:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_merge_defaults(state), f, ensure_ascii=False, indent=2)

def _san_tf(tf_minutes: int | str | None) -> int:
    try:
        v = int(tf_minutes if tf_minutes is not None else 1)
    except Exception:
        v = 1
    return 5 if v == 5 else 1

def _is_ai_strategy(name: Optional[str]) -> bool:
    s = (name or "").lower().strip()
    # ยอมรับได้ทั้ง "ai model …", "aimodel…", "ai_model…", "ai 1m/5m"
    return ("ai model" in s) or s.startswith("aimodel") or s.startswith("ai_model") or s.startswith("ai ")

# ----------------------------- PUBLIC APIS -----------------------------
def on_new_order(tf_minutes: int, strategy_name: Optional[str]) -> None:
    """
    เรียกเมื่อมีการสร้างออเดอร์ใหม่
    - นับเฉพาะ TF 1 หรือ 5
    - นับเฉพาะกลยุทธ์ที่เป็น AI (ชื่อมี/ขึ้นต้นด้วย AI)
    """
    tf = _san_tf(tf_minutes)
    if not _is_ai_strategy(strategy_name):
        return
    st = _load()
    key = str(tf)
    c = st["counters"].get(key, {"orders_since_retrain": 0, "last_retrain_ts": None})
    c["orders_since_retrain"] = int(c.get("orders_since_retrain") or 0) + 1
    if not c.get("last_retrain_ts"):
        c["last_retrain_ts"] = _now_iso()
    st["counters"][key] = c
    _save(st)

def mark_retrain(tf_minutes: int) -> Dict[str, Any]:
    """รีเซ็ตตัวนับและอัปเดตเวลาล่าสุดเมื่อ retrain/calibrate เสร็จ"""
    tf = str(_san_tf(tf_minutes))
    st = _load()
    st["counters"][tf] = {"orders_since_retrain": 0, "last_retrain_ts": _now_iso()}
    _save(st)
    return st

def set_auto(tf_minutes: int, enabled: bool, every_orders: int) -> Dict[str, Any]:
    """ตั้งค่า auto; บังคับขั้นต่ำ every_orders = 10 เพื่อกันรีเทรนถี่เกินไป"""
    tf = str(_san_tf(tf_minutes))
    st = _load()
    st["auto"].setdefault(tf, {})
    st["auto"][tf]["enabled"] = bool(enabled)
    # clip ขั้นต่ำ 10; ไม่ไปยุ่งค่า max เพื่อคงพฤติกรรมเดิม
    try:
        eo = int(every_orders)
    except Exception:
        eo = 40
    st["auto"][tf]["every_orders"] = max(10, eo)
    _save(st)
    return st

def get_status(tf_minutes: int) -> Dict[str, Any]:
    tf = str(_san_tf(tf_minutes))
    st = _load()
    c = st["counters"].get(tf, {"orders_since_retrain": 0, "last_retrain_ts": None})
    a = st["auto"].get(tf, {"enabled": True, "every_orders": 40})
    return {
        "timeframe": int(tf),
        "orders_since_retrain": int(c.get("orders_since_retrain") or 0),
        "last_retrain_ts": c.get("last_retrain_ts"),
        "auto_enabled": bool(a.get("enabled", True)),
        "auto_every_orders": int(a.get("every_orders", 40)),
        # ช่องอื่น ๆ ให้ backend เติมได้เอง (เช่น model_name/version/ece ฯลฯ)
    }
