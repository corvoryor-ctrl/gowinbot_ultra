# backend/database/trade_logger.py
# -*- coding: utf-8 -*-
"""
Trade logger (robust, path-aligned, and backward compatible)

ปรับปรุงหลัก:
- Path เลือกแบบฉลาดและสอดคล้องกับ ai_trainer/ai_model:
  ENV(APP_DATA_DIR) → <UserData>/GowinBotUltra/data → (legacy) %APPDATA%/GowinBotUltra/data → <project>/data
- SQLite: เปิด WAL + synchronous=NORMAL เพื่อรองรับอ่าน/เขียนซ้อน, timeout มากขึ้น
- Safety: clip pred_prob, sanitize result/direction, strategy name จาก .name
- Index: เติม index สำคัญแบบ idempotent
"""
from __future__ import annotations

import os
import csv
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple

APP_NAME = "GowinBotUltra"

# ----------------------------- PATHS (robust) -----------------------------
def _user_data_dir(app_name: str = APP_NAME) -> str:
    """โฟลเดอร์ข้อมูลผู้ใช้แบบข้ามแพลตฟอร์ม (Windows ใช้ LOCALAPPDATA)"""
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

def _resolve_data_dir() -> str:
    """
    ลำดับความสำคัญ:
      1) ENV: APP_DATA_DIR
      2) <UserData>/GowinBotUltra/data
      3) legacy: %APPDATA%/GowinBotUltra/data (ถ้ามีอยู่แล้ว)
      4) <project>/data
    """
    # 1) explicit override
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
        chosen = ud
    except Exception:
        chosen = None

    # 3) legacy APPDATA (รับเฉพาะกรณีมีอยู่แล้ว เพื่อไม่ทิ้งฐานเก่า)
    legacy = None
    appdata = os.environ.get("APPDATA")
    if appdata:
        legacy = os.path.join(appdata, APP_NAME, "data")
        if os.path.isdir(legacy) and any(
            os.path.isfile(os.path.join(legacy, fn)) for fn in ("trade_log.sqlite3", "trade_log_export.csv")
        ):
            chosen = legacy

    # 4) project ./data เป็นสำรองสุดท้าย
    if not chosen:
        base = os.path.abspath(os.path.dirname(__file__) or ".")
        proj = os.path.abspath(os.path.join(base, "..", "data"))
        try:
            os.makedirs(proj, exist_ok=True)
            chosen = proj
        except Exception:
            chosen = os.getcwd()

    return chosen

DATA_DIR = _resolve_data_dir()
DB_PATH = os.path.join(DATA_DIR, "trade_log.sqlite3")
CSV_PATH = os.path.join(DATA_DIR, "trade_log_export.csv")

# --- ai_meta แบบ optional: ถ้า import ไม่ได้ ให้ปล่อยผ่าน ---
try:
    import ai_meta  # type: ignore
except Exception:
    ai_meta = None

# ----------------------------- DB Utils -----------------------------
def _get_conn() -> sqlite3.Connection:
    """
    เปิด connection แบบทนทานขึ้น (WAL + timeout)
    หมายเหตุ: ใช้ autocommit (isolation_level=None) เพื่อให้ PRAGMA มีผลทันที
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass
    return conn

def init_db() -> None:
    """สร้างตารางถ้ายังไม่มี + ทำ migration เติมคอลัมน์และดัชนีสำคัญให้ครบ"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount REAL NOT NULL,
                duration INTEGER NOT NULL,
                strategy TEXT,
                mg_step INTEGER,
                result TEXT,
                profit REAL,
                trade_id INTEGER,
                pred_prob REAL
            )
            """
        )

        def _add_col(name: str, ddl: str):
            try:
                cur.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")
            except sqlite3.OperationalError:
                pass

        # idempotent columns
        _add_col("strategy", "TEXT")
        _add_col("mg_step", "INTEGER")
        _add_col("result", "TEXT")
        _add_col("profit", "REAL")
        _add_col("trade_id", "INTEGER")
        _add_col("pred_prob", "REAL")

        # indices (idempotent)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_duration ON trades(duration)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_tradeid  ON trades(trade_id)")
        conn.commit()

# ----------------------------- Helpers -----------------------------
def _strategy_to_name(strategy: Any) -> Optional[str]:
    """รับได้ทั้ง string หรือ object (พยายามอ่าน .name; ไม่ได้ก็ใช้ชื่อคลาส)"""
    if strategy is None:
        return None
    if isinstance(strategy, str):
        s = strategy.strip()
        return s or None
    try:
        name_attr = getattr(strategy, "name", None)
        if isinstance(name_attr, str) and name_attr.strip():
            return name_attr.strip()
        cls = getattr(strategy, "__class__", None)
        if cls and isinstance(getattr(cls, "__name__", None), str):
            return cls.__name__
    except Exception:
        pass
    s = str(strategy)
    return s if s else None

def _sanitize_result(result: Optional[str]) -> Optional[str]:
    if result is None:
        return None
    r = str(result).strip().upper()
    # รับหลายรูปแบบ แล้ว normalize
    if r in ("W", "WIN", "WON"):
        return "WIN"
    if r in ("L", "LOSE", "LOST"):
        return "LOSE"
    if r in ("E", "EQ", "EQUAL", "DRAW", "TIE"):
        return "EQUAL"
    # ไม่รู้จัก: เก็บค่าที่ส่งมา (อาจมี use case อื่น)
    return r

def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return max(0.0, min(1.0, v))
    except Exception:
        return None

# ----------------------------- Public APIs -----------------------------
def log_trade(
    asset: str,
    direction: str,
    amount: float,
    duration: int,
    strategy: Optional[Any] = None,
    mg_step: Optional[int] = None,
    trade_id: Optional[int] = None,
    pred_prob: Optional[float] = None,   # ความเชื่อมั่นของฝั่งที่สั่ง (0..1)
) -> int:
    """
    บันทึกออเดอร์ใหม่ (result/profit = NULL ก่อน) แล้วคืนค่า row id สำหรับอัปเดตผลภายหลัง
    """
    init_db()  # เผื่อคนเรียกยังไม่เคย init
    ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

    # sanitize / normalize fields
    strategy_name = _strategy_to_name(strategy)
    dir_str = (direction or "").strip().lower() or "none"
    amount_f = float(amount)
    duration_i = int(duration)
    mg_step_i = None if mg_step is None else int(mg_step)
    pp = _clip01(pred_prob)

    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades (timestamp, asset, direction, amount, duration, strategy, mg_step, result, profit, trade_id, pred_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (ts, asset, dir_str, amount_f, duration_i, strategy_name, mg_step_i, trade_id, pp),
        )
        db_id = cur.lastrowid
        conn.commit()
        return int(db_id)

def update_trade_result(db_id: int, result: str, profit: float) -> None:
    """อัปเดตผล (WIN/LOSE/EQUAL) และกำไร/ขาดทุน ของออเดอร์ที่ id = db_id"""
    r = _sanitize_result(result)
    p = float(profit)
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE trades SET result = ?, profit = ? WHERE id = ?", (r, p, int(db_id)))
        conn.commit()
    _maybe_update_ai_stats(db_id, r or "")

def update_trade_result_by_trade_id(trade_id: int, result: str, profit: float) -> Optional[int]:
    """
    อัปเดตผลด้วย trade_id (external id) — สะดวกเวลาไม่ทราบ db_id
    คืน db_id ที่ถูกอัปเดตหรือ None ถ้าไม่พบ
    """
    r = _sanitize_result(result)
    p = float(profit)
    with _get_conn() as conn:
        cur = conn.cursor()
        row = cur.execute("SELECT id FROM trades WHERE trade_id = ? ORDER BY id DESC LIMIT 1", (int(trade_id),)).fetchone()
        if not row:
            return None
        db_id = int(row["id"])
        cur.execute("UPDATE trades SET result = ?, profit = ? WHERE id = ?", (r, p, db_id))
        conn.commit()
    _maybe_update_ai_stats(db_id, r or "")
    return db_id

def get_all_trades(limit: Optional[int] = None) -> List[Dict]:
    """ดึงรายการเทรดทั้งหมด (ใหม่สุดก่อน)"""
    init_db()
    sql = "SELECT id, timestamp, asset, direction, amount, duration, strategy, mg_step, result, profit, trade_id, pred_prob FROM trades ORDER BY id DESC"
    if limit and isinstance(limit, int):
        sql += f" LIMIT {int(limit)}"
    with _get_conn() as conn:
        cur = conn.cursor()
        rows = cur.execute(sql).fetchall()
    return [dict(r) for r in rows]

def export_to_csv(path: Optional[str] = None) -> str:
    """Export เทรดทั้งหมดเป็น CSV"""
    init_db()
    if path is None:
        path = CSV_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = get_all_trades()
    headers = ["id", "timestamp", "asset", "direction", "amount", "duration", "strategy", "mg_step", "result", "profit", "trade_id", "pred_prob"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return f"Exported {len(rows)} rows to: {path}"

# ----------------- Metrics helpers for Auto Retrain/Calibration -----------------
def _maybe_update_ai_stats(db_id: int, result: str) -> None:
    """
    อ่านแถวที่เพิ่งอัปเดต แล้วอัปเดตสถิติไปที่ ai_meta:
    - เฉพาะกลยุทธ์ที่ชื่อมี 'AI Model' และ duration เป็น 1 หรือ 5
    """
    if ai_meta is None:
        return
    try:
        with _get_conn() as conn:
            cur = conn.cursor()
            row = cur.execute("SELECT duration, strategy FROM trades WHERE id = ?", (int(db_id),)).fetchone()
        if not row:
            return
        duration = int(row["duration"])
        sname = (row["strategy"] or "").lower()
        if duration in (1, 5) and "ai model" in sname:
            tf = 1 if duration == 1 else 5
            ai_meta.record_result(tf, result)  # type: ignore
    except Exception:
        # ไม่ให้สถิติทำให้ flow หลักพัง
        pass

def _fetch_ai_rows(tf: int, last_n: int = 200) -> List[sqlite3.Row]:
    """
    ดึงเฉพาะเทรดที่เป็น AI (duration = tf และ strategy ขึ้นต้น AI Model)
    เอาเฉพาะคอลัมน์ที่ต้องใช้สำหรับคำนวณ metric
    """
    init_db()
    with _get_conn() as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT result, pred_prob
            FROM trades
            WHERE duration = ? AND strategy LIKE 'AI Model (%'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(tf), int(last_n)),
        ).fetchall()
    return rows

def compute_metrics(tf: int, last_n: int = 200) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    คำนวณเมตริกจากเทรด AI ล่าสุด:
      - accuracy = wins / (wins + losses)  (ไม่นับ EQUAL)
      - ece = Expected Calibration Error (10 bins) ถ้ามี pred_prob พอ
      - mean_conf = ค่าเฉลี่ยความเชื่อมั่นของฝั่งที่สั่ง (0..1) ถ้ามี
    คืนค่าเป็น (accuracy, ece, mean_conf) — ถ้าข้อมูลไม่พอจะเป็น None
    """
    rows = _fetch_ai_rows(tf, last_n=last_n)
    if not rows:
        return None, None, None

    wins = losses = 0
    conf_list: List[float] = []
    correct_list: List[int] = []

    for r in rows:
        res = (r["result"] or "").upper() if r["result"] is not None else None
        p = r["pred_prob"]
        if res in ("WIN", "LOSE"):
            if res == "WIN":
                wins += 1
            else:
                losses += 1
        if p is not None:
            try:
                c = float(p)
                if 0.0 <= c <= 1.0:
                    conf_list.append(c)
                    if res in ("WIN", "LOSE"):
                        correct_list.append(1 if res == "WIN" else 0)
            except Exception:
                pass

    accuracy = None
    denom = wins + losses
    if denom > 0:
        accuracy = wins / denom

    ece = None
    mean_conf = None
    if len(conf_list) >= 30 and len(correct_list) >= 30 and len(conf_list) == len(correct_list):
        mean_conf = sum(conf_list) / len(conf_list)
        bins = 10
        bin_tot = [0] * bins
        bin_acc = [0.0] * bins
        bin_conf = [0.0] * bins
        for c, y in zip(conf_list, correct_list):
            b = min(bins - 1, int(c * bins))
            bin_tot[b] += 1
            bin_acc[b] += y
            bin_conf[b] += c
        e = 0.0
        total = len(conf_list)
        for i in range(bins):
            if bin_tot[i] > 0:
                acc_i = bin_acc[i] / bin_tot[i]
                conf_i = bin_conf[i] / bin_tot[i]
                e += (bin_tot[i] / total) * abs(acc_i - conf_i)
        ece = e

    return accuracy, ece, mean_conf

# ----------------- Debug helpers -----------------
def get_db_path() -> str:
    return DB_PATH

def get_data_dir() -> str:
    return DATA_DIR
