# backend/ai_meta.py
# Meta + counters + calibration with DB-fallback ECE
# -*- coding: utf-8 -*-
import os, json, threading, sqlite3, logging, sys
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

# ---------------------------------------------------------------------
# trade_logger (ถ้ามี) จะใช้เป็นแหล่ง metrics อันดับแรก
# รองรับโครง import หลายแบบ (database.* หรือ backend.database.*)
# ---------------------------------------------------------------------
try:
    from database import trade_logger  # expects compute_metrics(tf, last_n) -> (acc, ece, mean_conf)
except Exception:
    try:
        from backend.database import trade_logger  # alt layout
    except Exception:
        trade_logger = None  # type: ignore

# ---------------- Paths / Defaults ----------------
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_STATE_PATH = os.path.join(_BASE_DIR, "ai_meta.json")

_DEFAULT: Dict[str, Dict[str, Any]] = {
    "1": {
        "model_name": "AI Model (1m)", "timeframe": 1, "version": "1.0.1",
        "learning_rate": 0.001, "accuracy": None, "ece": None,
        "wins": 0, "trades": 0,
        "last_retrain_ts": None, "orders_since_retrain": 0,
        "auto_enabled": True, "auto_every_orders": 40,
    },
    "5": {
        "model_name": "AI Model (5m)", "timeframe": 5, "version": "1.0.1",
        "learning_rate": 0.001, "accuracy": None, "ece": None,
        "wins": 0, "trades": 0,
        "last_retrain_ts": None, "orders_since_retrain": 0,
        "auto_enabled": True, "auto_every_orders": 40,
    },
}

_LOCK = threading.Lock()
_STATE: Dict[str, Dict[str, Any]] = {}

# ---------------- Common helpers ----------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _tfkey(tf: int) -> str:
    return "1" if int(tf) <= 1 else "5"

def _ensure(d: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    for k in ("1", "5"):
        if k not in d:
            d[k] = dict(_DEFAULT[k])
        for subk, v in _DEFAULT[k].items():
            d[k].setdefault(subk, v)
    return d

def _load() -> Dict[str, Dict[str, Any]]:
    global _STATE
    if not _STATE:
        if os.path.exists(_STATE_PATH):
            try:
                with open(_STATE_PATH, "r", encoding="utf-8") as f:
                    _STATE = json.load(f)
            except Exception:
                _STATE = {}
        _STATE = _ensure(_STATE or {})
    return _STATE

def _save() -> None:
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(_ensure(_STATE), f, ensure_ascii=False, indent=2)

# ---------------- MODELS_DIR resolver (ให้ตรงกับ ai_model_*.py / ai_trainer.py) ----------------
APP_NAME = "GowinBotUltra"

def _user_data_dir(app_name: str = APP_NAME) -> str:
    try:
        if os.name == "nt":
            root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
            return os.path.join(root, app_name)
        elif sys.platform == "darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", app_name)
        else:
            root = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
            return os.path.join(root, app_name)
    except Exception:
        return os.getcwd()

def _resolve_models_dir() -> str:
    # 1) ENV override
    for key in ("APP_MODELS_DIR", "MODELS_DIR"):
        v = os.environ.get(key)
        if v:
            p = os.path.abspath(os.path.expanduser(v))
            try:
                os.makedirs(p, exist_ok=True)
                return p
            except Exception:
                pass
    # 2) project root /models  (ai_meta อยู่ใน backend/ → root คือ ..)
    try:
        proj_root = os.path.abspath(os.path.join(_BASE_DIR, ".."))
    except Exception:
        proj_root = os.getcwd()
    proj_models = os.path.join(proj_root, "models")
    try:
        os.makedirs(proj_models, exist_ok=True)
        if os.access(proj_models, os.W_OK):
            return proj_models
    except Exception:
        pass
    # 3) user data fallback
    ud = os.path.join(_user_data_dir(), "models")
    try:
        os.makedirs(ud, exist_ok=True)
    except Exception:
        pass
    return ud

MODELS_DIR = _resolve_models_dir()

def _calib_path(tf: int) -> str:
    """
    เขียน/อ่านไฟล์คาลิเบรตใน MODELS_DIR เดียวกับฝั่งโมเดล (ai_model_* จะมองที่นี่)
    """
    return os.path.join(MODELS_DIR, f"ai_{int(tf)}m.calib.json")

def _write_calibration(tf: int, A: float = 0.0, B: float = 0.0, T: float = 1.0) -> None:
    data = {"A": float(A), "B": float(B), "T": float(max(0.8, min(1.5, float(T))))}
    path = _calib_path(tf)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------- DB path resolver (สอดคล้อง trade_logger/ai_trainer) ----------------
def _resolve_db_path() -> str:
    # A) ถ้ามี trade_logger ให้ถามโดยตรง (ดีที่สุด)
    if trade_logger is not None:
        try:
            if hasattr(trade_logger, "get_db_path"):
                p = trade_logger.get_db_path()
                if p and os.path.isfile(p):
                    return p
        except Exception:
            pass
    # B) ENV override
    env_db = os.environ.get("APP_DB_PATH")
    if env_db and os.path.isfile(env_db):
        return env_db
    # C) user data default
    ud = os.path.join(_user_data_dir(), "data", "trade_log.sqlite3")
    if os.path.isfile(ud):
        return ud
    # D) project ./data
    proj = os.path.join(os.path.abspath(os.path.join(_BASE_DIR, "..")), "data", "trade_log.sqlite3")
    if os.path.isfile(proj):
        return proj
    # E) current dir fallback
    c2 = os.path.join(os.getcwd(), "trade_log.sqlite3")
    return c2

# ---------------- Public APIs ----------------
def get_status(tf: int) -> Dict[str, Any]:
    with _LOCK:
        s = _load()[_tfkey(tf)]
        wins = int(s.get("wins", 0))
        trades = int(s.get("trades", 0))
        win_rate = (wins / trades) if trades > 0 else s.get("accuracy")
        return {
            "model_name": s.get("model_name"),
            "timeframe": s.get("timeframe"),
            "version": s.get("version"),
            "learning_rate": s.get("learning_rate"),
            "accuracy": s.get("accuracy"),
            "win_rate": win_rate,
            "ece": s.get("ece"),
            "orders_since_retrain": int(s.get("orders_since_retrain", 0)),
            "auto_enabled": bool(s.get("auto_enabled", True)),
            "auto_every_orders": int(s.get("auto_every_orders", 40)),
            "last_retrain_ts": s.get("last_retrain_ts"),
        }

# alias ฝั่งเก่า
def status(tf: int) -> Dict[str, Any]:
    return get_status(tf)

def set_auto(tf: int, enabled: bool, every_orders: int) -> Dict[str, Any]:
    with _LOCK:
        st = _load()
        k = _tfkey(tf)
        st[k]["auto_enabled"] = bool(enabled)
        st[k]["auto_every_orders"] = max(10, int(every_orders or 40))
        _save()
        return get_status(tf)

def bump_on_place(tf: int) -> Tuple[int, bool, int]:
    """เรียกทันทีหลังวางไม้ (ยังไม่รู้ผล)"""
    with _LOCK:
        st = _load()
        k = _tfkey(tf)
        st[k]["orders_since_retrain"] = int(st[k].get("orders_since_retrain", 0)) + 1
        _save()
        return (
            int(st[k]["orders_since_retrain"]),
            bool(st[k]["auto_enabled"]),
            int(st[k]["auto_every_orders"]),
        )

# alias เก่า
bump_order_counter = bump_on_place

def record_result(tf: int, result: str) -> None:
    """เรียกเมื่อรู้ผล (win/lose) เพื่ออัปเดตตัวเลขพื้นฐาน"""
    r = (result or "").upper()
    with _LOCK:
        st = _load()
        k = _tfkey(tf)
        st[k]["trades"] = int(st[k].get("trades", 0)) + (1 if r in ("WIN", "LOSE") else 0)
        if r == "WIN":
            st[k]["wins"] = int(st[k].get("wins", 0)) + 1
        _save()

# ---------------- DB-fallback for ECE ----------------
# (เวอร์ชันใหม่ใช้ _resolve_db_path() แทนการล็อคไว้ที่ %APPDATA%)
def _pick_table(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = [r[0] for r in cur.fetchall()]
    for n in names:
        if "trade" in n.lower():
            return n
    return names[0] if names else None

def _compute_metrics_from_db(tf: int, last_n: int = 200) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    คืน (accuracy, ece, mean_conf) จาก SQLite ถ้ามีคอลัมน์ความเชื่อมั่น
    ยอมรับชื่อคอลัมน์: conf / p_up / prob / pred_conf / pred_prob
    """
    dbp = _resolve_db_path()
    if not os.path.isfile(dbp):
        return None, None, None
    conn = sqlite3.connect(dbp)
    try:
        tab = _pick_table(conn)
        if not tab:
            return None, None, None
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({tab})")
        cols = [r[1].lower() for r in cur.fetchall()]
        prob_cols = [c for c in ("conf", "p_up", "prob", "pred_conf", "pred_prob") if c in cols]
        if not prob_cols:
            return None, None, None
        pcol = prob_cols[0]

        tf_filter = ""
        if "tf" in cols:
            tf_filter = f"WHERE CAST(tf AS TEXT) LIKE '%{int(tf)}%'"

        q = f"SELECT result, {pcol} FROM {tab} {tf_filter} ORDER BY rowid DESC LIMIT ?"
        cur.execute(q, (int(last_n),))
        rows = cur.fetchall()
        if not rows:
            return None, None, None

        y, p = [], []
        for res, prob in rows:
            try:
                pr = float(prob)
            except Exception:
                continue
            if not (0.0 <= pr <= 1.0):
                continue
            r = str(res or "").lower()
            if r not in ("win", "lose"):  # ตัด draw
                continue
            y.append(1 if r == "win" else 0)
            p.append(pr)

        n = len(y)
        if n == 0:
            return None, None, None

        acc = sum(y) / n
        # Expected Calibration Error (10 bins)
        bins, ece = 10, 0.0
        mean_conf = sum(p) / n
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            idx = [i for i, pp in enumerate(p) if (pp >= lo and pp < hi) or (b == bins - 1 and pp == 1.0)]
            if not idx:
                continue
            conf_bin = sum(p[i] for i in idx) / len(idx)
            acc_bin  = sum(y[i] for i in idx) / len(idx)
            ece += (len(idx) / n) * abs(acc_bin - conf_bin)
        return float(acc), float(ece), float(mean_conf)
    finally:
        conn.close()

# ---------------- Retrain + Calibrate ----------------
def retrain_and_calibrate(tf: int, *, accuracy: Optional[float] = None, ece: Optional[float] = None, new_lr: Optional[float] = None) -> Dict[str, Any]:
    """
    อัปเดตเมตาเมื่อ retrain/calibrate
    - พยายามใช้ trade_logger ก่อน
    - ถ้ายังขาด (None) ใช้ fallback อ่านจาก DB มาคำนวณ ECE/accuracy เอง
    - รีเซ็ตตัวนับและเวลา
    - เขียนไฟล์คาลิเบรต (A,B,T) ไปที่ MODELS_DIR เดียวกับโมเดล
    """
    tf = 1 if int(tf) <= 1 else 5

    # 1) metrics จาก trade_logger (ถ้ามี)
    mean_conf = None
    if (accuracy is None or ece is None) and trade_logger is not None:
        try:
            acc_db, ece_db, mean_conf = trade_logger.compute_metrics(tf, last_n=200)
            if accuracy is None: accuracy = acc_db
            if ece is None:      ece      = ece_db
        except Exception as e:
            logging.warning(f"[ai_meta] trade_logger.compute_metrics failed: {e}")

    # 2) fallback จาก DB โดยตรง
    if accuracy is None or ece is None:
        acc2, ece2, mean_conf2 = _compute_metrics_from_db(tf, last_n=200)
        if accuracy is None: accuracy = acc2
        if ece is None:      ece      = ece2
        if mean_conf is None: mean_conf = mean_conf2

    # 3) เขียน state
    with _LOCK:
        st = _load()
        k = _tfkey(tf)
        if new_lr is not None:
            st[k]["learning_rate"] = float(new_lr)
        if accuracy is not None:
            st[k]["accuracy"] = float(accuracy)
        if ece is not None:
            st[k]["ece"] = float(ece)
        st[k]["orders_since_retrain"] = 0
        st[k]["last_retrain_ts"] = _now_iso()
        _save()

    # 4) คำนวณ T แบบอ่อน ๆ จาก gap ระหว่าง mean_conf กับ accuracy (ถ้ามี)
    T = 1.0
    if (mean_conf is not None) and (accuracy is not None):
        gap = float(mean_conf) - float(accuracy)  # >0 = over-confident
        T = 1.0 + max(-0.2, min(0.2, 0.8 * gap))
    _write_calibration(tf, A=0.0, B=0.0, T=T)
    return get_status(tf)

# ---------------- Auto check ----------------
def auto_check_and_run(tf: int) -> Dict[str, Any]:
    with _LOCK:
        st = _load()
        k = _tfkey(tf)
        count   = int(st[k].get("orders_since_retrain", 0))
        enabled = bool(st[k].get("auto_enabled", True))
        every   = int(st[k].get("auto_every_orders", 40))
    if enabled and count >= every:
        return retrain_and_calibrate(tf)
    return get_status(tf)
