# core/indicator_config.py
# -*- coding: utf-8 -*-
"""
Indicator config I/O (robust & main-aligned)
- เลือกตำแหน่ง MODELS_DIR อย่างยืดหยุ่น:
    1) APP_MODELS_DIR หรือ MODELS_DIR จาก ENV ถ้ามี
    2) <root>/models ถ้าเขียนได้
    3) โฟลเดอร์ผู้ใช้ .../GowinbotUltra/models (กันกรณี run แบบ frozen/สิทธิ์เขียนน้อย)
- เติม DEFAULT keys ให้ครบเวลาอ่าน (deep-merge ระดับ 1 ชั้น)
- เวลา set_indicator_config จะ merge กับ DEFAULT ก่อนเซฟเสมอ
"""
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any

# ---------------- path resolve (compatible with main.py style) ----------------

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

def _resolve_models_dir() -> Path:
    # 1) ENV overrides
    for env_key in ("APP_MODELS_DIR", "MODELS_DIR"):
        v = os.environ.get(env_key)
        if v:
            p = Path(v).expanduser().resolve()
            try:
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception:
                pass

    # 2) project <root>/models
    try:
        base = Path(__file__).resolve().parent.parent  # .../core -> root
    except Exception:
        base = Path.cwd()
    proj_models = base / "models"
    try:
        proj_models.mkdir(parents=True, exist_ok=True)
        # ถ้าเขียนได้ ใช้ตรงนี้
        if os.access(proj_models, os.W_OK):
            return proj_models
    except Exception:
        pass

    # 3) user dir fallback (safe for frozen / limited perms)
    ud = _user_data_dir() / "models"
    try:
        ud.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return ud

MODELS_DIR = _resolve_models_dir()
CFG_1M = MODELS_DIR / "indicators_1m.json"
CFG_5M = MODELS_DIR / "indicators_5m.json"

# ---------------- default schema (aligned with main.py) -----------------

DEFAULT_CFG: Dict[str, Any] = {
    "use": True,
    "ma":      {"enabled": False, "type": "EMA", "length": 50, "source": "close", "bias": "any", "slope": "any"},
    "macd":    {"enabled": False, "fast": 12, "slow": 26, "signal": 9, "mode": "confirm"},   # confirm|contrarian
    "ichimoku":{"enabled": False, "tenkan": 9, "kijun": 26, "senkou_b": 52, "mode": "trend"},
    "rsi":     {"enabled": False, "length": 14, "ob": 70, "os": 30, "mode": "filter"},       # filter|entry
    "stoch":   {"enabled": False, "k": 14, "d": 3, "smooth": 3, "ob": 80, "os": 20, "mode": "filter"},
    "bb":      {"enabled": False, "length": 20, "k": 2.0, "mode": "squeeze"},                # squeeze|bandtouch
    "atr":     {"enabled": False, "length": 14, "min_atr": 0.0},
    "vwap":    {"enabled": False},
    "obv":     {"enabled": False, "mode": "confirm"},                                        # confirm|contrarian
    "volprof": {"enabled": False, "bins": 24},
    # เพิ่มให้ตรงกับ main.py เพื่อกัน key error ใน UI/กลไกอื่น
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

# ---------------- utils ----------------

def _ensure() -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for p in (CFG_1M, CFG_5M):
        if not p.exists():
            try:
                p.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                # พยายามอีกครั้งแบบเปิดไฟล์
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_CFG, f, ensure_ascii=False, indent=2)

def _merge_default(data: Dict[str, Any]) -> Dict[str, Any]:
    """merge DEFAULT_CFG เข้ากับ data (deep-merge แค่ 1 ชั้นก็พอสำหรับสคีมานี้)"""
    merged = dict(DEFAULT_CFG)
    for k, v in (data or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            tmp = dict(merged[k])
            tmp.update(v)
            merged[k] = tmp
        else:
            merged[k] = v
    return merged

# ---------------- public API ----------------

def schema() -> Dict[str, Any]:
    """คืนสคีมาค่าเริ่มต้น (ใช้วาด UI หรือรีเซ็ต)"""
    return DEFAULT_CFG

def get_indicator_config(tf: int) -> Dict[str, Any]:
    """
    อ่านคอนฟิกของ timeframe ที่ขอ (1 หรือ 5)
    - ถ้าไฟล์ไม่มีหรืออ่านพัง → คืน DEFAULT
    - ถ้ามีคีย์หาย → เติมให้ครบ (ไม่ทำให้ UI/บอทพัง)
    """
    _ensure()
    path = CFG_1M if int(tf) == 1 else CFG_5M
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_CFG
        return _merge_default(data)
    except Exception:
        return DEFAULT_CFG

def set_indicator_config(tf: int, cfg: Dict[str, Any]) -> bool:
    """
    เขียนคอนฟิก timeframe ที่ขอ โดยจะ merge กับ DEFAULT ก่อนเซฟ
    เพื่อให้ไฟล์คอนฟิกที่ได้ “ครบคีย์” เสมอ
    """
    _ensure()
    try:
        merged = _merge_default(cfg or {})
        tmp_path = (CFG_1M if int(tf) == 1 else CFG_5M).with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        # replace แบบอะตอมมิก (เท่าที่ระบบไฟล์รองรับ)
        tmp_path.replace(CFG_1M if int(tf) == 1 else CFG_5M)
        return True
    except Exception:
        # fallback เขียนตรงๆ
        with open(CFG_1M if int(tf) == 1 else CFG_5M, "w", encoding="utf-8") as f:
            json.dump(_merge_default(cfg or {}), f, ensure_ascii=False, indent=2)
        return True
