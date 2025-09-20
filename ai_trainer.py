# ai_trainer.py
# -*- coding: utf-8 -*-
"""
Offline trainer for AI 1m/5m models
- Robust MODELS_DIR (ENV → <proj>/models → user_data/GowinbotUltra/models)
- Robust DB path (ENV: APP_DB_PATH → user_data/data → ./data → .)
- Feature/label extraction resilient to missing columns
- Logistic + sigmoid calibration + temperature & AB fine-tune
"""
import os, json, sqlite3, logging, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
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
    # 1) ENV
    for key in ("APP_MODELS_DIR", "MODELS_DIR"):
        v = os.environ.get(key)
        if v:
            p = os.path.abspath(os.path.expanduser(v))
            try:
                os.makedirs(p, exist_ok=True)
                return p
            except Exception:
                pass
    # 2) project ./models
    try:
        base = os.path.abspath(os.path.dirname(__file__) or ".")
    except Exception:
        base = os.getcwd()
    proj = os.path.join(base, "models")
    try:
        os.makedirs(proj, exist_ok=True)
        if os.access(proj, os.W_OK):
            return proj
    except Exception:
        pass
    # 3) user data fallback
    ud = os.path.join(_user_data_dir(), "models")
    try:
        os.makedirs(ud, exist_ok=True)
    except Exception:
        pass
    return ud

def _resolve_db_path() -> str:
    # 1) explicit override
    env_db = os.environ.get("APP_DB_PATH")
    if env_db and os.path.isfile(env_db):
        return env_db
    # 2) user data default (…/GowinbotUltra/data/trade_log.sqlite3)
    ud = os.path.join(_user_data_dir(), "data", "trade_log.sqlite3")
    if os.path.isfile(ud):
        return ud
    # 3) ./data/trade_log.sqlite3
    c1 = os.path.join(os.getcwd(), "data", "trade_log.sqlite3")
    if os.path.isfile(c1):
        return c1
    # 4) ./trade_log.sqlite3
    c2 = os.path.join(os.getcwd(), "trade_log.sqlite3")
    if os.path.isfile(c2):
        return c2
    # 5) legacy (APPDATA on Windows)
    legacy = os.path.join(os.environ.get("APPDATA", ""), APP_NAME, "data", "trade_log.sqlite3")
    return legacy  # อาจไม่มีไฟล์: ผู้เรียกจะจัดการ error ต่อเอง

MODELS_DIR = _resolve_models_dir()
DB_PATH = _resolve_db_path()

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------
# DB I/O
# --------------------------------------------------------------------
def _pick_table(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = [r[0] for r in cur.fetchall()] or []
    if not names:
        raise FileNotFoundError("No tables in trade_log.sqlite3")
    # pick trade-like first
    for n in names:
        if "trade" in n.lower():
            return n
    return names[0]

def _read_df() -> pd.DataFrame:
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        tab = _pick_table(conn)
        df = pd.read_sql_query(f"SELECT * FROM {tab}", conn)
        return df
    finally:
        conn.close()

# --------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------
def _safe_float_series(s, default=0.0) -> pd.Series:
    try:
        out = pd.to_numeric(s, errors="coerce")
        return out.fillna(float(default)).astype(float)
    except Exception:
        return pd.Series([float(default)] * (len(s) if hasattr(s, "__len__") else 0), dtype=float)

def _choose_feature_cols(df: pd.DataFrame) -> list[str]:
    # Priority set
    pref = ["rsi", "momentum", "gap", "payout", "atr", "vol", "zscore", "conf"]
    cols = [c for c in pref if c in df.columns]
    if cols:
        return cols
    # fallback: amount + mg_step (as in original)
    fallback = [c for c in ["amount", "mg_step"] if c in df.columns]
    if fallback:
        return fallback
    # last-resort: create them
    return ["amount", "mg_step"]

def _label_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    y ∈ {0,1}; remove draws by default.
    Priority:
    - 'result' string: WIN/LOSE/EQUAL(DRAW) → map
    - else if 'profit' exists: >0 → 1, <0 → 0; ==0 → drop
    - else all zeros (should be avoided)
    """
    y = None
    if "result" in df.columns:
        s = df["result"].astype(str).str.lower()
        mask_valid = s.isin(["win", "lose"])
        y = np.where(s[mask_valid] == "win", 1, 0)
        # filter df to valid rows for caller: return y only; caller must mask X accordingly
        df._result_mask = mask_valid.to_numpy()
        return y

    if "profit" in df.columns:
        p = _safe_float_series(df["profit"])
        mask_valid = p != 0.0
        y = np.where(p[mask_valid] > 0.0, 1, 0)
        df._result_mask = mask_valid.to_numpy()
        return y

    # fallback: try payout if exists (rare)
    if "payout" in df.columns:
        p = _safe_float_series(df["payout"])
        mask_valid = p > 0.0
        y = np.where(mask_valid, 1, 0)[mask_valid]
        df._result_mask = mask_valid.to_numpy()
        return y

    # worst case: empty
    df._result_mask = np.zeros(len(df), dtype=bool)
    return np.array([], dtype=int)

def _build_Xy(df: pd.DataFrame, tf: int):
    """
    Build features/labels from trade log.
    - Filters rows to the requested timeframe if 'tf' column exists.
    - Drops DRAW/EQUAL automatically.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.empty((0, 0)), np.array([], dtype=int), []

    # If 'tf' exists, pre-filter rows
    if "tf" in df.columns:
        tf_mask = df["tf"].astype(str).str.contains(str(int(tf)), na=False)
        df = df.loc[tf_mask].reset_index(drop=True)

    # Compute y (and embed index mask into df._result_mask)
    y = _label_from_df(df)
    if y.size == 0:
        return np.empty((0, 0)), np.array([], dtype=int), []

    mask = getattr(df, "_result_mask", np.ones(len(df), dtype=bool))
    df_eff = df.loc[mask].reset_index(drop=True)

    cols = _choose_feature_cols(df_eff)
    # Ensure fallback columns exist
    for c in cols:
        if c not in df_eff.columns:
            df_eff[c] = 0.0
    X = df_eff[cols].apply(_safe_float_series).to_numpy(dtype=float)
    return X, y, cols

# --------------------------------------------------------------------
# Calibration helpers
# --------------------------------------------------------------------
def _temperature_search(logits: np.ndarray, y: np.ndarray) -> float:
    """Search T in [0.8, 1.5] that minimizes Brier score."""
    bestT, best = 1.0, 1e9
    for T in np.linspace(0.8, 1.5, 15):
        p = 1.0 / (1.0 + np.exp(-(logits / T)))
        b = brier_score_loss(y, p)
        if b < best:
            best, bestT = b, T
    return float(bestT)

# --------------------------------------------------------------------
# Train & Calibrate
# --------------------------------------------------------------------
def retrain_offline(tf: int = 1):
    """
    Train baseline Logistic + sigmoid calibration (via CalibratedClassifierCV).
    Save to models/ai_{tf}m.pkl and write a small meta file of feature columns.
    """
    df = _read_df()
    X, y, cols = _build_Xy(df, tf)
    if len(y) < 100 or X.shape[0] < 100:
        raise ValueError(f"not enough samples to retrain (>=100). got n={len(y)}")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Robust logistic (balanced classes); keep defaults friendly to most sklearn versions
    base = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)

    # Save model
    try:
        import joblib  # lazy import
    except Exception:
        import pickle as joblib  # type: ignore
    _ensure_dir(os.path.join(MODELS_DIR, "dummy"))
    out = os.path.join(MODELS_DIR, f"ai_{int(tf)}m.pkl")
    joblib.dump(clf, out)

    # Save meta (feature columns)
    meta_path = os.path.join(MODELS_DIR, f"ai_{int(tf)}m.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"features": cols, "n": int(X.shape[0]), "positive_rate": float(y.mean())}, f, ensure_ascii=False, indent=2)

    logging.info(f"[AI {tf}m] model saved → {out}")
    return out

def calibrate_from_log(tf: int = 1):
    """
    Create calibration A,B,T for the trained model using validation split.
    Saves ai_{tf}m.calib.json alongside the model.
    """
    # Load model
    try:
        import joblib
    except Exception:
        import pickle as joblib  # type: ignore

    model_path = os.path.join(MODELS_DIR, f"ai_{int(tf)}m.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    df = _read_df()
    X, y, _ = _build_Xy(df, tf)
    if len(y) < 100 or X.shape[0] < 100:
        raise ValueError(f"not enough samples to calibrate (>=100). got n={len(y)}")

    # split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)

    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xva)[:, 1]
    else:
        raise ValueError("model has no predict_proba")

    # logits
    eps = 1e-9
    logits = np.log((p + eps) / (1 - p + eps))

    # temperature
    T = _temperature_search(logits, yva)

    # small AB grid
    best = (0.0, 0.0, 1e9)
    for A in np.linspace(-0.5, 0.5, 11):
        for B in np.linspace(-0.5, 0.5, 11):
            px = 1.0 / (1.0 + np.exp(-((logits / T) + A + B)))
            b = brier_score_loss(yva, px)
            if b < best[2]:
                best = (float(A), float(B), float(b))
    A, B, _ = best

    calib = {"A": A, "B": B, "T": T}
    out = os.path.join(MODELS_DIR, f"ai_{int(tf)}m.calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    logging.info(f"[AI {tf}m] calibration saved → {out} :: {calib}")
    return out

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Example: python ai_trainer.py
    # Optional: set APP_DB_PATH / APP_MODELS_DIR via environment
    tfs = (1, 5)
    # Allow override via env: TF_LIST="1,5" or "1"
    env_tfs = os.environ.get("TF_LIST")
    if env_tfs:
        try:
            tfs = tuple(int(x.strip()) for x in env_tfs.split(",") if x.strip())
        except Exception:
            pass

    for tf in tfs:
        try:
            retrain_offline(tf)
            calibrate_from_log(tf)
        except Exception as e:
            logging.error(f"[AI {tf}m] pipeline error: {e}")
