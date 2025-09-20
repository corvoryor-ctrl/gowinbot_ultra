# strategies/ai_model_1m.py
# --- AI Model (1m) — robust paths, safer features, indicator filters ---
from __future__ import annotations
import os, json, logging, sys
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    import pickle as joblib

from .base_strategy import BaseStrategy

# ---- indicator config / suite ----
try:
    from core.indicator_config import get_indicator_config  # preferred
except Exception:  # pragma: no cover
    from indicator_config import get_indicator_config

from .indicator_suite import (
    ma, macd, ichimoku, rsi as rsi_fn, stoch,
    bollinger, atr as atr_fn, vwap as vwap_fn,
    obv as obv_fn, volume_profile
)

# --- helper: Series/ndarray/list -> ndarray (ไม่วนเรียกตัวเอง) ---
def to_np(x):
    try:
        return x.to_numpy()
    except AttributeError:
        return np.asarray(x)

# --------------------- robust model dir (align with indicator_config) ---------------------
def _user_data_dir(app_name: str = "GowinbotUltra") -> str:
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
    # 1) ENV overrides
    for env_key in ("APP_MODELS_DIR", "MODELS_DIR"):
        v = os.environ.get(env_key)
        if v:
            p = os.path.abspath(os.path.expanduser(v))
            try:
                os.makedirs(p, exist_ok=True)
                return p
            except Exception:
                pass
    # 2) project <root>/models
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../strategies -> root
    except Exception:
        base = os.getcwd()
    proj_models = os.path.join(base, "models")
    try:
        os.makedirs(proj_models, exist_ok=True)
        if os.access(proj_models, os.W_OK):
            return proj_models
    except Exception:
        pass
    # 3) user dir fallback
    ud = os.path.join(_user_data_dir(), "models")
    try:
        os.makedirs(ud, exist_ok=True)
    except Exception:
        pass
    return ud

MODELS_DIR = _resolve_models_dir()
MODEL_PATH = os.path.join(MODELS_DIR, "ai_1m.pkl")
CALIB_PATH = os.path.join(MODELS_DIR, "ai_1m.calib.json")

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# --------------------- math utils ---------------------
def _safe_sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def _load_calib():
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            c = json.load(f)
        A = float(c.get("A", 0.0))
        B = float(c.get("B", 0.0))
        T = float(c.get("T", 1.0))
        T = float(np.clip(T, 0.8, 1.5))
        return A, B, T
    except Exception:
        return 0.0, 0.0, 1.0

def _apply_calibration(logit, A, B, T):
    return float(_safe_sigmoid((logit / T) + A + B))

# --------------------- feature builder (safer) ---------------------
def _pad_to(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad 1D array to length n with edge values (if needed)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size >= n:
        return arr
    pad = np.full(n - arr.size, arr[0] if arr.size else 0.0, dtype=float)
    return np.r_[pad, arr]

def _build_features(candles: pd.DataFrame) -> dict:
    df = candles.copy()
    if "close" not in df.columns:
        raise ValueError("candles missing 'close'")

    close = df["close"].astype(float).to_numpy()
    need = 20  # ใช้ขั้นต่ำราว ๆ นี้สำหรับโมเมนตัม/ATR/RSI
    if close.size < need:
        close = _pad_to(close, need)

    if set(["high", "low"]).issubset(df.columns):
        high = df["high"].astype(float).to_numpy()
        low  = df["low"].astype(float).to_numpy()
    else:
        dif = np.abs(np.diff(np.r_[close[:1], close]))
        high = close + 0.5 * dif
        low  = close - 0.5 * dif

    # ให้ hi/lo “ยาวเท่า” close เสมอ (ป้องกัน shape mismatch)
    if high.size < close.size:
        high = _pad_to(high, close.size)
    if low.size < close.size:
        low = _pad_to(low, close.size)

    # Momentum windows
    m1 = close[-1] - close[-2]
    m3 = close[-1] - close[-4]
    m5 = close[-1] - close[-6] if close.size >= 6 else m3

    # Wilder-like TR mean (approx ATR)
    prev = np.r_[close[:1], close[:-1]]
    tr = np.maximum.reduce([
        high[-need:] - low[-need:],
        np.abs(high[-need:] - prev[-need:]),
        np.abs(low[-need:] - prev[-need:])
    ])
    atr = float(np.nanmean(np.nan_to_num(tr, nan=0.0)))

    # RSI(14) coarse
    diff = np.diff(close)
    up = np.clip(diff, 0, None)
    dn = -np.clip(diff, None, 0)
    eps = 1e-9
    rs = (up[-14:].mean() + eps) / (dn[-14:].mean() + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    x = np.array([m1, m3, m5, atr, rsi], dtype=float)
    return {"raw": {"m1": m1, "m3": m3, "m5": m5, "atr": atr, "rsi": rsi}, "x": x, "df": df}

# =============================================================================
# Core strategy
# =============================================================================
class _AIModel1mCore(BaseStrategy):
    name = "AI Model (1m)"
    tf = 1

    def __init__(self, settings: dict | None = None):
        super().__init__(settings or {})
        # thresholds (อ่านจาก settings ถ้ามี)
        self.conf_thr = float(self.settings.get("ai_conf", self.settings.get("ai1m_conf", 0.60)))
        self.gap_thr  = float(self.settings.get("ai_gap",  self.settings.get("ai1m_gap",  0.10)))

        _ensure_dir(MODEL_PATH)
        self.model = None
        self.has_model = False
        try:
            if os.path.isfile(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.has_model = True
        except Exception as e:  # pragma: no cover
            logging.warning(f"[AI 1m] load model failed: {e}")

        self.A, self.B, self.T = _load_calib()
        self.last_proba_up: float | None = None
        self.is_ai_model = True  # hint ให้ระบบอื่นรู้

        # indicator config (top-level keys: ma/macd/ichimoku/... ตามไฟล์ config)
        self.ind_cfg = get_indicator_config(1)
        self._last_meta = {}

    # ---- inference ----
    def _infer(self, feats: dict) -> float:
        x = feats["x"].reshape(1, -1)
        if self.has_model:
            try:
                if hasattr(self.model, "predict_proba"):
                    p_up = float(self.model.predict_proba(x)[0, 1])
                    logit = float(np.log((p_up + 1e-9) / (1 - p_up + 1e-9)))
                    p_up = _apply_calibration(logit, self.A, self.B, self.T)
                    # เบลนด์กับ prior เล็กน้อย กันค่าแกว่งเกิน
                    n_eff, k = 50.0, 10.0
                    p_up = float((n_eff * p_up + k * 0.5) / (n_eff + k))
                    return float(np.clip(p_up, 0.02, 0.98))
            except Exception as e:
                logging.warning(f"[AI 1m] infer with model failed: {e}")
        # fallback heuristic
        rsi = feats["raw"]["rsi"]
        m3  = feats["raw"]["m3"]
        rsi_n = (rsi - 50.0) / 50.0
        m3_n  = np.tanh(m3 * 5.0)
        z = 0.8 * rsi_n + 0.2 * m3_n
        return float(np.clip(0.5 + 0.5 * z, 0.02, 0.98))

    # ---- indicator filters (configurable) ----
    def _apply_indicator_filters(self, df: pd.DataFrame, signal: str) -> bool:
        if not self.ind_cfg.get("use", True):
            self._last_meta = {"badges": []}
            return True
        if signal not in ("call", "put"):
            self._last_meta = {"badges": []}
            return True

        cl  = to_np(df["close"]).astype(float)
        hi  = to_np(df["high"]).astype(float)   if "high"   in df else cl
        lo  = to_np(df["low"]).astype(float)    if "low"    in df else cl
        vol = to_np(df["volume"]).astype(float) if "volume" in df else np.zeros_like(cl)

        badges = []

        # MA
        MA = self.ind_cfg.get("ma", {})
        if MA.get("enabled", False):
            m = ma(cl, MA.get("length", 50), MA.get("type", "EMA"))
            last = float(m[-1])
            bias_ok = True
            if MA.get("bias") == "up"   and not (cl[-1] > last): bias_ok = False
            if MA.get("bias") == "down" and not (cl[-1] < last): bias_ok = False
            badges.append(f"MA{MA.get('length')} {MA.get('type')}@{last:.5f}")
            if not bias_ok:
                self._last_meta = {"badges": badges}
                return False

        # MACD
        MACD = self.ind_cfg.get("macd", {})
        if MACD.get("enabled", False):
            line, sig, hist = macd(cl, MACD.get("fast", 12), MACD.get("slow", 26), MACD.get("signal", 9))
            h = float(hist[-1])
            mode = str(MACD.get("mode", "confirm")).lower()
            if mode == "confirm":
                if signal == "call" and h <= 0: self._last_meta = {"badges": badges}; return False
                if signal == "put"  and h >= 0: self._last_meta = {"badges": badges}; return False
            elif mode == "contrarian":
                if signal == "call" and h >= 0: self._last_meta = {"badges": badges}; return False
                if signal == "put"  and h <= 0: self._last_meta = {"badges": badges}; return False
            badges.append(f"MACD {h:+.5f}")

        # Ichimoku (trend with cloud)
        ICH = self.ind_cfg.get("ichimoku", {})
        if ICH.get("enabled", False):
            conv, base, spa, spb = ichimoku(hi, lo, ICH.get("tenkan", 9), ICH.get("kijun", 26), ICH.get("senkou_b", 52))
            cloud_top = float(np.nanmax([spa[-1], spb[-1]]))
            cloud_bot = float(np.nanmin([spa[-1], spb[-1]]))
            price = float(cl[-1])
            mode = str(ICH.get("mode", "trend")).lower()
            ok = True
            if mode == "trend":
                if signal == "call" and not (price > cloud_top and conv[-1] > base[-1]): ok = False
                if signal == "put"  and not (price < cloud_bot and conv[-1] < base[-1]): ok = False
            if not ok:
                self._last_meta = {"badges": badges}
                return False
            badges.append("Ichimoku✓")

        # RSI
        RSI = self.ind_cfg.get("rsi", {})
        if RSI.get("enabled", False):
            r = float(rsi_fn(cl, RSI.get("length", 14))[-1])
            mode = str(RSI.get("mode", "filter")).lower()
            ob = float(RSI.get("ob", 70)); os = float(RSI.get("os", 30))
            if mode == "filter":
                if signal == "call" and r < os: pass
                elif signal == "put" and r > ob: pass
                else:
                    if signal == "call" and r < 50: self._last_meta = {"badges": badges}; return False
                    if signal == "put"  and r > 50: self._last_meta = {"badges": badges}; return False
            badges.append(f"RSI {r:.1f}")

        # Stochastic
        STO = self.ind_cfg.get("stoch", {})
        if STO.get("enabled", False):
            k, d = stoch(hi, lo, cl, STO.get("k", 14), STO.get("d", 3), STO.get("smooth", 3))
            kv = float(k[-1]); dv = float(d[-1])
            ob = float(STO.get("ob", 80)); os = float(STO.get("os", 20))
            mode = str(STO.get("mode", "filter")).lower()
            if mode == "filter":
                if signal == "call" and not (kv < os and kv > dv): self._last_meta = {"badges": badges}; return False
                if signal == "put"  and not (kv > ob and kv < dv): self._last_meta = {"badges": badges}; return False
            badges.append(f"STO {kv:.1f}/{dv:.1f}")

        # Bollinger
        BB = self.ind_cfg.get("bb", {})
        if BB.get("enabled", False):
            mid, up, lo_b, bw, sd = bollinger(cl, BB.get("length", 20), BB.get("k", 2.0))
            mode = str(BB.get("mode", "squeeze")).lower()
            if mode == "squeeze":
                if not (float(sd[-1]) <= np.nanmedian(to_np(sd)[-50:]) * 0.9):
                    self._last_meta = {"badges": badges}; return False
            elif mode == "bandtouch":
                if signal == "call" and not (cl[-1] <= lo_b[-1]): self._last_meta = {"badges": badges}; return False
                if signal == "put"  and not (cl[-1] >= up[-1]):  self._last_meta = {"badges": badges}; return False
            badges.append("BB✓")

        # ATR (vol floor)
        ATR = self.ind_cfg.get("atr", {})
        if ATR.get("enabled", False):
            a = float(atr_fn(hi, lo, cl, ATR.get("length", 14))[-1])
            if a < float(ATR.get("min_atr", 0.0)):
                self._last_meta = {"badges": badges}; return False
            badges.append(f"ATR {a:.6f}")

        # VWAP
        if self.ind_cfg.get("vwap", {}).get("enabled", False) and (vol != 0).any():
            v = float(vwap_fn(hi, lo, cl, vol)[-1])
            if signal == "call" and cl[-1] < v: self._last_meta = {"badges": badges}; return False
            if signal == "put"  and cl[-1] > v: self._last_meta = {"badges": badges}; return False
            badges.append(f"VWAP {v:.5f}")

        # OBV
        OBV = self.ind_cfg.get("obv", {})
        if OBV.get("enabled", False) and (vol != 0).any():
            o = obv_fn(cl, vol)
            slope = float(o[-1] - o[-4]) if len(o) >= 5 else 0.0
            mode = str(OBV.get("mode", "confirm")).lower()
            if mode == "confirm":
                if signal == "call" and slope <= 0: self._last_meta = {"badges": badges}; return False
                if signal == "put"  and slope >= 0: self._last_meta = {"badges": badges}; return False
            else:
                if signal == "call" and slope >= 0: self._last_meta = {"badges": badges}; return False
                if signal == "put"  and slope <= 0: self._last_meta = {"badges": badges}; return False
            badges.append("OBV✓")

        # Volume Profile (badge เท่านั้น)
        VP = self.ind_cfg.get("volprof", {})
        if VP.get("enabled", False) and (vol != 0).any():
            vp = volume_profile(cl, vol, VP.get("bins", 24))
            badges.append(f"POC {vp.get('poc', cl[-1]):.5f}")

        self._last_meta = {"badges": badges[-10:]}
        return True

    # ---- strategy API ----
    def check_signal(self, candles: pd.DataFrame) -> str:
        try:
            feats = _build_features(candles)
            p_up = self._infer(feats)
            self.last_proba_up = float(p_up)
            self.p_up = float(p_up)  # BaseStrategy field (ให้ engine อ่าน)
            p_dn = 1.0 - p_up
            gap = abs(p_up - 0.5) * 2.0
            logging.info(
                f"[AI 1m] p_up={p_up:.3f} conf_thr={self.conf_thr:.3f} gap={gap:.3f} gap_thr={self.gap_thr:.3f}"
            )

            sig = "none"
            if gap >= self.gap_thr:
                if p_up >= self.conf_thr:
                    sig = "call"
                elif p_dn >= self.conf_thr:
                    sig = "put"

            if sig in ("call", "put") and not self._apply_indicator_filters(feats["df"], sig):
                return "none"
            return sig
        except Exception as e:
            logging.error(f"[AI 1m] check_signal error: {e}")
            return "none"

    def predict_proba(self, *args, **kwargs):
        """
        ให้ engine ดึงความน่าจะเป็นได้ (บาง engine เรียกเมธอดนี้)
        คืน p_up ล่าสุด ถ้าไม่มีให้คำนวณจาก df สุดท้ายแบบเร็ว ๆ
        - คืนค่า float ตรง ๆ (engine ฝั่งเรา parse ได้)
        """
        try:
            if self.last_proba_up is not None:
                return float(self.last_proba_up)
        except Exception:
            pass
        try:
            df = kwargs.get("candles") or (args[1] if len(args) >= 2 else args[0])
            feats = _build_features(df)
            return float(self._infer(feats))
        except Exception:
            return 0.5

    def get_signal_meta(self):
        return {"indicators": self.ind_cfg, "badges": self._last_meta.get("badges", [])}

# Public aliases (เข้ากันกับโค้ดเดิม)
class AIModel1mStrategy(_AIModel1mCore): ...
class AIModel1M(_AIModel1mCore): ...
__all__ = ["AIModel1mStrategy", "AIModel1M"]
